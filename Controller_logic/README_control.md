# BEE_LAND controller

Bio-inspired visual landing controller: a companion-side ROS 2 node closes a
vision-only loop around PX4 to land on a vertically oscillating platform,
using nothing but a downward-facing camera once the visual handoff happens.
The core idea follows Herisse (2012) and de Croon (2016): regulate optical
**divergence** (the visual looming rate, D = -vz/h) to a constant setpoint D*,
which produces an exponential, self-slowing descent without ever estimating
height or velocity directly.

This document summarizes the architecture, the algorithms, and the design
decisions baked into each module. It is a map, not a substitute for the
module docstrings, which carry the full derivations and revision history.

---

## 1. Architecture

```
Gazebo SITL ──images/poses──▶ bee_node (ROS 2) ──attitude+thrust setpoints──▶ PX4
     ▲                          │        ▲                                     │
     └──── lockstep physics ────┘        └────────── state, body rates ────────┘
                                │
                        MavsdkWorker (thread) ──MAVLink: arm, takeoff, kill──▶ PX4
```

| Module | Responsibility |
|---|---|
| `bee_node.py` | The ROS 2 node: subscriptions, timers, the camera callback, phase orchestration, all wiring. Runs under the default single-threaded executor. |
| `px4_interface.py` | Turns (roll, pitch, yaw, thrust) into PX4 messages over uXRCE-DDS. No control logic. |
| `mavsdk_worker.py` | Async/threaded MAVLink side: guided takeoff and the terminal motor stop. Nothing else. |
| `target_acquisition.py` | NN-free target detection: centroid offset + ROI box + `fov_saturated` flag. |
| `optical_flow.py` | Farneback flow on the ROI + gradient-weighted affine divergence fit. |
| `derotation.py` | Gyro-only ego-rotation removal (currently disabled, see §3). |
| `control_law.py` | The visual-only controller: lateral PD + Herisse thrust law. |
| `mission_routine.py` | PROBE → gate → scheduled-gain DESCEND state machine. |
| `clock.py` | Single source of truth for time; enforces the three-clock-family discipline. |
| `platform_motion.py` | Diagnostics-only platform-relative motion (never feeds control). |
| `diagnostics_writer.py` | Per-frame CSV logging, including the vision-pipeline stage timings. |

Three architectural commitments worth knowing before reading anything else:

**Direct offboard, no PX4 position cascade.** The offboard heartbeat sets
`attitude=True` and everything else `False`; thrust is written straight into
`VehicleAttitudeSetpoint.thrust_body` and reaches the mixer directly. Roll
and pitch pass through only PX4's attitude-P and rate-PID inner loops. The
vertical axis — the one this project lives or dies on — has no hidden PX4
velocity/position controller underneath it. (Verified explicitly; this was
an early suspect for unexplained lag and it is not there.)

**Visual-only control after handoff.** Once the visual loop is active, the
control law consumes only target offset, normalized mean flow, and
divergence. PX4 state (`relative_z_m`, velocities, attitude) is logged for
diagnostics and used for the takeoff handoff, never inside the loop.

**MAVSDK is deliberately minimal.** It exists for exactly the two things PX4
does badly over raw offboard: the initial guided takeoff, and the terminal
disarm/kill after confirmed contact. Closed-loop setpoints never go through
it.

---

## 2. Clock discipline (`clock.py`)

The node straddles three clocks that must never be compared by absolute
value:

* **WALL** (epoch): every outgoing PX4 message is stamped on this clock,
  because the uXRCE-DDS agent's timesync references the agent's wall clock,
  not the ROS clock. Also the diagnostics `t_sec` origin.
* **SIM** (Gazebo seconds): carried on camera image stamps, and through them
  the flow, target, and control-dt timestamps. PX4 SITL advances in lockstep
  with it, but at an arbitrary absolute offset.
* **PX4** (hrt microseconds since boot): carried on incoming `/fmu/out`
  messages.

`TimeManager` estimates the cross-family offsets for observability only;
they are forbidden from feeding control. Practical corollary that mattered
repeatedly during tuning: any wall-clock *duration* measured inside a SITL
run that spans sim-driven events is inflated by 1/RTF (this project runs at
RTF ≈ 0.2, so ×5) and must never be used to size a real-hardware latency
budget. The per-stage compute timings (`timing_stage_*`) are exempt — they
measure genuine CPU time, which does not scale with RTF.

---

## 3. Vision pipeline

### Target acquisition (`target_acquisition.py`)

Blur → HSV saliency mask ∪ Canny edges → morphological cleanup → contour
scoring (area, compactness, centrality) → centroid offset normalized to
[-1, 1] + bounding box. Two robustness decisions:

* Large detections are **down-weighted, never rejected**: a near-full-frame
  target at touchdown is the success condition, not an outlier.
* A short temporal hold (`loss_grace_period_sec`) bridges single-frame
  dropouts, reusing the last good estimate with confidence decayed toward
  zero.

`fov_saturated` fires when the bounding box touches all four image borders:
past that point `area_fraction` and the box dimensions are frame-size
artifacts, not measurements, and consumers must treat them as uninformative.

### Optical flow and divergence (`optical_flow.py`)

Dense Farneback on the target ROI, then a **global affine fit** of the flow
field whose trace (a1 + b2) is the divergence. Deliberately a global fit
rather than a per-pixel median: once `fov_saturated` removes the textured
rim, only interior texture remains, and the fit is **weighted by per-pixel
image-gradient magnitude** so texture-poor patches are distrusted directly
instead of merely out-voted.

Performance decisions, each validated against a synthetic ground-truth
affine-warp test before being trusted:

* **ROI-adaptive downsampling**: `scale = clip(downsample_target_px /
  max(roi_w, roi_h), downsample_min_scale, 1.0)` — a bigger ROI gets shrunk
  *more*, keeping Farneback's search problem near a fixed working size. This
  replaced an earlier binary close-range parameter set that made the most
  critical regime (full-frame ROI near touchdown) also the most expensive.
  Defaults (96 px target, 0.5 floor) come from a measured bias sweep:
  downsampling systematically **underestimates** divergence (the dangerous
  direction — less commanded thrust exactly when more is needed), roughly
  -10% at these settings versus -77% at aggressive ones.
* **Fit directly on the downsampled field.** Profiling showed the
  least-squares fit, not Farneback, dominated the cost; the original design
  upsampled the flow field back to full resolution before fitting, throwing
  the savings away. `_fit_divergence_affine` takes a `pixel_scale` argument
  that widens the coordinate spacing so the fitted slope keeps physical
  units. The full-resolution reconstruction now happens only when derotation
  (which samples at full-image pixel coordinates) is active.
* **Normal-equations solve** instead of `np.linalg.lstsq`: three unknowns
  means a 3×3 system; forming it directly is ~5× cheaper and was verified
  numerically identical (to ~3e-5). `lstsq` remains as the fallback for a
  genuinely singular ROI.
* The control-facing fit runs a trim-and-refit outlier pass; the
  diagnostics-only pre-derotation fit uses `robust=False` (single pass) and
  the debug per-pixel divergence field is gated behind `store_debug`.

The divergence handed to control is an EMA (`divergence_smoothing = 0.7`,
old-value-weighted) blended 90/10 with the raw value
(`raw_divergence_weight = 0.10`). The α = 0.3 experiment is documented
history: lowering the smoothing to cut lag traded it for destabilizing
noise, because the smoothing constant never fed the stability gate's
dead-time in the first place — treat it as a noise knob, not a lag knob, and
if it is ever raised, its time constant (≈ dt/(1-α)) belongs in the
dead-time budget of §6.

`fit_quality` (weighted R²) is logged but **must not be used for gating**:
it stays high even when the flow is confidently wrong on a uniform looming
surface (measured correlation with actual divergence error: ~0.08). It
distinguishes "degenerate fit" from "clean fit", not "right" from "wrong".

### Derotation (`derotation.py`) — currently disabled

Gyro-only Longuet-Higgins rotational-field subtraction: the rotational
component of flow depends only on angular rate and image geometry, never on
depth, so it can be predicted from PX4 body rates and subtracted before the
divergence fit. The body→optical rotation matrix was **derived from the
mount chain** (FRD→FLU→camera joint→Gazebo camera convention→software
180° rotation) after the sign-permutation approach provably failed — a
global sign flip mirrored the induced error at the same magnitude instead of
shrinking it, the signature of an axis swap, which no diagonal matrix can
fix.

Disabled (`OpticalFlowEstimator(derotator=None)`) because with the platform
restricted to vertical-only oscillation the ego-rotation contamination is
small, and dropping it halves the per-frame fit count. Re-enabling requires
re-running the acceptance test in `derotation.py`'s docstring (hover wobble
with near-zero translation: derotated mean flow and divergence must collapse
toward zero) because downsampling changed the flow field's spatial
resolution since the matrix was last validated.

---

## 4. Control law (`control_law.py`)

Visual-only, constant-gain. The earlier LQR / per-area_fraction
gain-scheduling machinery was removed once this architecture validated in
flight.

**Lateral (roll, pitch)** — one fixed PD pair per axis:

```
u = -(kp * p_scale * offset  +  kd * d_scale * flow_norm)
```

The plant is effectively a double integrator (tilt → accel → velocity →
position), so kp sets closed-loop bandwidth and kd sets damping
(ζ ~ kd/√kp). The current gains were chosen after cross-spectral measurement
showed the previous pair gave only ~3.6% tracking at the platform's own
frequency with the tilt limit nowhere near binding — the bottleneck was
bandwidth, not authority. P and D scales are ramped independently by the
mission (they carry different historical scale factors, so one shared scale
cannot reproduce a validated pair).

**Thrust** — the Herisse (2012, eq. 32) accel-domain law:

```
a_cmd        = k * (D - D*)          [m/s²]
thrust_delta = hover_thrust * a_cmd / g
```

Sign **positive** on (D - D*): the identified plant has B < 0 (more thrust
reduces divergence), so arresting an approach requires more thrust. `k` is
supplied per-tick by the mission routine (§5). Commands pass through a
first-order low-pass, an optional slew limiter (off by default), and clamps
(`thrust_min = 0.57`, `thrust_max = 0.90`,
`max_visual_thrust_delta_from_hover = 0.18`, `hover_thrust = 0.73`).

---

## 5. Mission routine (`mission_routine.py`)

State machine: takeoff (MAVSDK) → **CENTER** → **PROBE** → gate →
**DESCEND** or **INFEASIBLE** → touchdown → kill.

**PROBE** holds D* = 0 — a true visual hover — for a fixed window and
records the peak commanded vertical acceleration `|a_cmd|` the loop needed
to hold it. Because the loop is regulating divergence to zero, every
acceleration it commands is a mirror of the platform-induced disturbance:
the vehicle measures the disturbance *through its own control effort*,
without a platform model, using the same sensor and loop that will fly the
descent (de Croon 2016's self-supervised idea).

From the probe's `peak_accel`:

```
k_min  = peak_accel / D*                      Herisse floor: below this gain
                                              the platform disturbance wins
h_crit = k_min * dt / (2 * safety)            de Croon ceiling: the height
                                              where any fixed gain must cross
                                              the instability boundary
gate:    feasible  iff  h_crit <= leg_clearance_m
```

`ceiling_safety_factor = 0.5` (a smaller safety holds the live gain further
below the ceiling, which makes `h_crit` larger and the gate stricter —
the conservative direction).

**DESCEND** runs a pre-committed, clock-driven schedule — no runtime height
estimate needed:

```
K(t) = clamp( k_explore * exp(-D* * t),  k_min,  k_explore )
```

with `k_explore = INITIAL_THRUST_GAIN` (hand-tuned, 6.5) and a raised-cosine
ramp on D* that fixed the PROBE→DESCEND transient. The lateral P/D scales
ride the same ramp normalized to K(t)/k_explore. `critical_time()` gives the
predicted instant K(t) hits the floor, usable as an open-loop descent
timeout. The open-loop height prediction `h(t) = h0·exp(-D*·t)` is logged
(`mission_h_pred_m`); in flight it tracked true height with correlation
~0.96.

**Probe-to-probe variance is real and large.** Each 15 s probe samples the
platform's oscillation phase differently; measured `peak_accel` has varied
by 6-7× between otherwise-identical runs, moving the feasibility verdict
with it. A single feasible/infeasible outcome is one sample, not a property
of the configuration.

---

## 6. The dead-time model — the load-bearing number

The gate's `dt` (`STABILITY_DT_SEC` in `bee_node.py`) is the most
consequential constant in the system: `h_crit` is linear in it. Its history,
each correction driven by flight evidence:

1. **1/30 s (one camera frame).** Gave `h_crit ≈ 0.08 m`; the descent then
   limit-cycled below ~0.5 m (thrust/divergence cross-correlation 0.93 at
   lag 0-1, ~1.7 Hz — a delay-induced relay oscillation, not sensor noise).
2. **`max(camera_period, publish_period) + vision_latency_budget`** — what
   the code currently ships. Recognizes that the 20 Hz setpoint publish
   timer, not the 30 Hz camera, bounds how often a fresh correction reaches
   the actuator, and that real CPU time between frame arrival and
   command-ready is additive on top.
3. **The corrected form (recommended, NOT yet in the code):**

   ```
   T = camera_period + vision_latency_p95 + publish_period      (straight sum)
   ```

   The camera and the publish timer are phase-independent clocks, so their
   worst-case waits stack; `max()` undercounts by one full period.

**Open items, stated plainly:** the shipped `bee_node.py` still has the
`max()` composition *and* `VISION_PROCESSING_LATENCY_BUDGET_SEC = 0.01` — a
deliberate test override used to force descents during the latency
investigation, not a measured value. All recent successful landings flew
under this permissive gate. Before trusting the gate again: switch to the
sum, and set the budget from the in-flight `timing_camera_cb_duration_ms`
p95 of a recent log (the `analyse_log.py` summary table produces exactly
this number — see the analysis README).

**Why in-flight, not offline:** the vision latency investigation established
that `optical_flow.update()` costs ~3 ms on real descend frames offline but
~10-13 ms median (p95 ~40 ms) inside the running node — after ruling out,
one by one: image content, cross-process CPU contention, MAVSDK poll
cadence, Python GC, and ROS callback preemption (structurally impossible
under the single-threaded executor). The residual is attributed to the
DDS/rmw layer's own background threads: a structural cost of living inside a
ROS 2 node. Offline benchmarks therefore systematically understate the
budget and must not be used to size it.

---

## 7. Touchdown chain

Contact is detected by a Gazebo contact sensor (physics-level, on the
platform's real collision geometry). On confirmation the node latches a
zero-thrust setpoint (`on_pre_motor_stop`) and `MavsdkWorker` stops the
motors. Three latency fixes live here, found via the touchdown-timing
investigation:

* An `asyncio.Event` wakes the worker the instant `request_motor_stop()`
  fires, replacing an up-to-50 ms poll (measured pickup latency: <1 ms).
* With `enable_kill_fallback` set (the SITL/moving-platform case), the
  `disarm()` attempt PX4's land detector is expected to refuse on a moving
  deck is skipped entirely — straight to `kill()`, saving a doomed MAVLink
  round-trip. With the flag off (real-hardware posture), behavior is
  unchanged disarm-only.
* Every MAVSDK telemetry stream is deterministically closed
  (`contextlib.aclosing`); they previously leaked, left to GC.

Combined effect, measured: apparent leg interpenetration at motor stop went
from ~19 cm to ~7 cm. The remaining sink is some mix of the setpoint-latch
publish gap (up to one 50 ms publish period) and compliant-contact
settling.

---

## 8. Geometry and reference-frame corrections

Two silent diagnostics bugs were found and fixed; neither ever affected
control (visual-only), but both corrupted every touchdown-height judgment:

* **Platform half-thickness** (`platform_motion.py`,
  `PLATFORM_TOP_SURFACE_OFFSET_M = 0.1`): the platform plugin reports its
  disc's geometric *center*; the landing surface is 0.1 m (half the 0.2 m
  cylinder length) above it. `relative_z` now references the top surface.
  The offset is constant, so `relative_vz` is unaffected.
* **True skid clearance is 0.227 m**, not the assumed 0.20: from
  `x500_base`'s unrotated skid-rail collision boxes (center z = -0.2195,
  half-height 0.0075). `leg_clearance_m` should reflect this.

---

## 9. Key constants, single-glance table

| Constant | Value | Where | Meaning |
|---|---|---|---|
| `hover_thrust` | 0.73 | control_law | Normalized thrust for hover |
| `thrust_min / thrust_max` | 0.57 / 0.90 | control_law | Hard thrust clamps |
| `max_visual_thrust_delta_from_hover` | 0.18 | control_law | Authority limit around hover |
| `raw_divergence_weight` | 0.10 | control_law | Raw-vs-filtered divergence blend |
| `divergence_smoothing` | 0.7 | optical_flow | EMA (old-value-weighted) |
| `downsample_target_px / min_scale` | 96 / 0.5 | optical_flow | ROI-adaptive downsample |
| `INITIAL_THRUST_GAIN` (k_explore) | 6.5 | bee_node | Descent-start thrust gain |
| `ceiling_safety_factor` | 0.5 | mission_routine | Gate conservatism |
| `descent D*` | 0.3 | mission config | Descent divergence setpoint |
| `PX4_SETPOINT_PERIOD_SEC` | 0.05 | bee_node | 20 Hz publish timer |
| `CAMERA_FRAME_PERIOD_SEC` | 1/30 | bee_node | Camera cadence at RTF=1 |
| `VISION_PROCESSING_LATENCY_BUDGET_SEC` | 0.01 (**test override**) | bee_node | Must be re-set from in-flight p95 |
| `PLATFORM_TOP_SURFACE_OFFSET_M` | 0.1 | platform_motion | Disc center → landing surface |
| skid-bottom clearance | 0.227 m | x500_base geometry | True `base_link` height at contact |

---

## 10. Known open items

* Fly a landing that the **honest gate** (sum-based T, measured budget)
  clears without an override — the most recent run's numbers put
  `h_crit ≈ 0.14 m` against 0.20+ m clearance, so this is within reach.
* Re-validate and re-enable **derotation** against the hover-wobble
  acceptance test if lateral platform motion returns.
* The **divergence schedule / touchdown overshoot** minimization work
  (queued next).
* The ~10 ms median in-process vision residual is accepted as structural;
  if it ever needs attacking, the remaining lever is executor/middleware
  configuration, not the vision algorithm.
