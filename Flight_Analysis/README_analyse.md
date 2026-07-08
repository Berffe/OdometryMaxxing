# BEE_LAND flight analysis (`analyse_log.py`)

Post-flight analysis of the diagnostics CSVs written by
`diagnostics_writer.py`. One command turns a log into a text summary plus a
set of high-signal plots; the default output is deliberately small, with the
legacy/detail plots behind `--full`.

---

## Usage

Typical single-log run (Windows):

```
py .\analyse_log.py .\logs\filename.csv --output-dir results\filename\test01\ --full
```

Other accepted forms:

```
# One CSV, output folder as the final positional argument
python analyse_log.py logs/bee_diagnostics_XXXXXXXX.csv results/test9

# A whole folder of CSVs -> one subfolder per log (results/test1, test2, ...)
python analyse_log.py logs

# Moving-platform test with a known oscillation frequency
python analyse_log.py logs/file.csv results/test9 --platform-frequency-hz 0.2
```

### Flags

| Flag | Default | What it does |
|---|---|---|
| `--output-dir` | `results` | Where plots and `summary.txt` are written. |
| `--full` | off | Adds the detailed/legacy plots (see below). |
| `--platform-frequency-hz` | 0.2 | Expected platform frequency for the reference fit in the spectrum plots. Pass 0 to disable. |
| `--time-base` | `auto` | X-axis time source. Auto prefers visual/flow timestamps, matching the controller's own timebase; PX4 epoch time is deliberately last (see below). |
| `--max-duration-sec` | none | Hard cutoff on wall-clock run time (`t_sec`) — drops rows after e.g. accidentally leaving the sim running. |
| `--image-width` / `--image-height` | 640 / 480 | Camera resolution for the FOV-reconstruction plot. Set to the actual camera config (this project flies 120×80 in some setups — mismatched values only distort `detection_boxes_fov.png`, nothing else). |
| `--max-boxes` | 120 | Cap on detection boxes drawn in the FOV plot. |

**Timestamp policy** (why `--time-base` exists): the controller runs on
image/visual timestamps (SIM family), while `t_sec` and `wall_timestamp` are
WALL family and PX4 stamps are their own epoch. The analyser never mixes
families by absolute value; auto-selection prefers `flow_timestamp_sec` →
`target_timestamp_sec` → `command_timestamp_sec` → `t_sec` → wall → PX4.
The practical consequence: on this SITL (RTF ≈ 0.2), durations on the
visual timebase are sim-seconds, ~5× shorter than the wall-clock run took.

---

## What you get

### `summary.txt`

The one-file digest, printed to the console and saved. Sections:

**Header** — row count, chosen timebase, time span, median dt, and the
measured sim/wall rate (the real-time factor).

**Signal ranges** — target-found and flow-valid ratios, FOV-saturation
count, and min/median/max for the command channels, divergence, and
relative z/vz.

**Platform motion** — z range, dominant frequency via FFT, and a sine fit at
the expected frequency (amplitude / offset / RMSE) — the quick check that
the platform plugin is doing what its SDF says.

**Mission section** — the probe/gate/descent numbers in one place:

* Probe `peak_accel`, the derived `k_min` and hand-tuned `k_explore`
* `h_crit` and the **feasibility verdict**
* Per-phase durations and the flown K(t) range
* Open-loop height-prediction error (`h_pred` vs |relative_z|): median, RMS,
  max — large or growing error means the gain schedule is running at the
  wrong height
* Divergence/kinematics decorrelation check: flags the onset time and height
  of any *sustained* mismatch between `flow_divergence` and the kinematic
  ground truth `relative_vz/|relative_z|` — the marker of a **sensing**
  breakdown (near-ground measurement collapse), explicitly distinguished
  from a gain problem

**Vision pipeline latency table** — the per-stage `on_camera` cost
breakdown, mean/median/p95/max in ms:

```
Vision pipeline latency (on_camera stage breakdown, ms)
--------------------------------------------------------
Computed over: 291 DESCEND-phase rows, pre-touchdown (command_thrust>0)
stage                                       mean  median     p95     max
cv_bridge conversion                        0.15    0.14    0.22    1.13
cv2.rotate                                  0.05    0.05    0.08    0.20
imshow/waitKey debug window                 0.01    0.01    0.01    0.12
body-rate buffer lookup                     0.11    0.10    0.16    0.53
target_acquisition.update()                 1.37    1.12    2.93   10.24
optical_flow.update()                      16.61   12.95   38.69   97.48
on_camera TOTAL                            18.34   14.69   40.45   98.75
```

Notes on reading it:

* It is computed over **DESCEND-phase, pre-touchdown rows only**
  (`command_thrust > 0`) when those exist — the regime that actually drives
  the stability gate. Far-field, on-ground, and post-touchdown frames are
  systematically cheaper and would understate the number that matters. The
  "Computed over" line states exactly which rows were used; it degrades
  gracefully to the whole log for pre-descent or older logs, and the section
  disappears entirely for logs that predate the instrumentation.
* **The `on_camera TOTAL` p95 from this table is the number to plug into
  `VISION_PROCESSING_LATENCY_BUDGET_SEC`** in `bee_node.py`. Use the
  in-flight figure, never an offline benchmark — the in-process residual
  (~10 ms median on this setup, attributed to the DDS/rmw layer) is real
  and does not appear offline.
* These stage timings are genuine CPU durations and are **not** inflated by
  the sim's real-time factor, unlike wall-clock period columns.
* A `(legacy, this log only)` interarrival-jitter line appears for logs
  captured during the scheduling-delay investigation; current logs no
  longer carry it.

### Default plots

| File | Shows |
|---|---|
| `target_detection_summary.png` | Detection confidence, offsets, area fraction over time. |
| `detection_boxes_fov.png` | The detection boxes reconstructed in the camera FOV — a spatial sanity check on target acquisition. |
| `lateral_control.png` | Roll/pitch commands vs the offsets and flow driving them. |
| `vertical_control.png` | Thrust command, divergence vs D* (the `--divergence-setpoint` line), the heart of the descent. |
| `flow_derotation.png` | Raw vs derotated mean flow and divergence. With derotation currently disabled the traces coincide — a flat "rotational component removed" line is expected, not a bug. |
| `gain_schedule.png` | The flown K(t) against the mission's schedule. |
| `divergence_consistency.png` | Vision divergence vs kinematic ground truth `relative_vz/|relative_z|` — the sensing-health plot; the summary's decorrelation verdict is read from this. |
| `height_prediction.png` | Open-loop `h_pred` vs true height. |
| `platform_motion_frequency.png` | Platform z spectrum vs the expected frequency (when `platform_z_m` is present). |
| `drone_platform_position_xyz.png` | Vehicle and platform positions per axis. |
| `closing_rate_spectrum.png` | Spectrum of the closing rate — where platform-induced modes and loop ringing show up. |

### Extra plots with `--full`

`vehicle_position_xyz.png`, `platform_position_xyz.png`,
`platform_velocity_xyz.png`, `relative_motion_xyz.png` — the per-axis
detail/legacy views.

---

## Reading a run in 60 seconds

1. `summary.txt` → feasibility verdict, `h_crit` vs leg clearance, phase
   durations. Remember the probe samples the platform's phase: `peak_accel`
   (and with it the verdict) has legitimately varied several-fold between
   identical configs, so treat one verdict as one sample.
2. `vertical_control.png` → divergence hugging D* without thrust touching
   the 0.57/0.90 rails is a healthy descent; flat-topped thrust with
   divergence swinging is the delay-induced limit cycle.
3. `divergence_consistency.png` + the decorrelation line in the summary →
   whether the last half-meter's misbehavior is sensing (expected below
   ~0.3-0.5 m once the target overfills the FOV) or control.
4. The latency table → is `on_camera TOTAL` p95 consistent with the budget
   the gate was configured with? If the pipeline changed since the budget
   was last set, this is where that shows up.
5. **Height caveats**: `relative_z_m` in logs predating the
   platform-top-surface fix reads 0.1 m low (distance to the disc's center,
   not its surface), and `base_link` at true skid contact sits at ~0.227 m,
   not 0.20 — both matter when judging "how high did we actually stop".

---

## Column families in the CSV (quick reference)

* `target_*` — detection outputs (found, offsets, box, area fraction,
  `fov_saturated`).
* `flow_*` — divergence (filtered + raw), normalized/pixel mean flow,
  `fit_quality` (diagnosis-only — high R² does **not** mean the divergence
  is right), derotation diagnostics.
* `command_*` — the attitude/thrust setpoints actually sent.
* `mission_*` — substate, probe results, `k_min`, `h_crit`, feasibility,
  K(t), `h_pred`.
* `vehicle_*`, `platform_*`, `relative_*` — kinematic ground truth
  (diagnostics only; the controller never sees these after handoff).
* `timing_*` — the per-stage vision pipeline costs (RTF-independent CPU
  time) plus wall-clock periods (RTF-inflated: `timing_control_period_wall`
  and the `*_age_*` columns mix clock families — do not use them to size
  real dead-time).
* `px4_*` — nav/arming/failsafe state; `px4_arming_state` flipping 2→1 is
  the disarm moment, the clean marker for "post-touchdown rows".
