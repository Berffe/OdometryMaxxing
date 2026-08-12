# Calculations and Visuals

Standalone plots for the theory the controller implements. Nothing here imports
`bee_control`, reads a log, or runs in flight — these are the derivations you
look at when you want to know *why* a gate is shaped the way it is, before
going to a flight log to see whether it behaved.

Each script runs on its own:

```bash
python3 vertical_plots.py
python3 lateral_plots.py
python3 gain_plot.py
python3 preLanding_Graphs.py
```

Only `numpy`, `matplotlib` and (for `preLanding_Graphs.py`) `scipy`.

---

## The scripts

### `vertical_plots.py` — sampled divergence loop

The vertical channel's stability analysis. Continuous relative dynamics,
exactly discretized with a zero-order hold, which is the honest model: the
controller sees the platform once per camera frame, not continuously.

Three figures:

1. **Divergence-residual frequency response** — how much platform motion
   survives into the divergence error, per input frequency.
2. **Drone/platform synchronization `a_d / a_p`** — how faithfully the vehicle
   reproduces the deck's acceleration. This is the quantity the *tracking*
   gate cares about, and it is genuinely different from the gain-margin
   question below.
3. **Discrete closed-loop root locus** — where the poles go as the gain rises,
   and where they leave the unit circle.

Tunables live in the `FrequencyConfig` dataclass at the top: `height_m` (2.0),
`sample_time_s` (0.05), `platform_frequency_hz` (0.4),
`platform_amplitude_m` (0.10), and the sweep range.

**`sample_time_s` is the parameter to respect.** It is the loop's dead time,
and it is what makes the ceiling a ceiling — the same gain that is comfortable
at 20 Hz is unstable at 10 Hz. In flight this quantity is measured, not
assumed: it comes from the vision delay tables and feeds
`CameraConfig.processing_latency_budget_sec`. If you change it here to make a
plot look better, you are drawing a different vehicle.

### `lateral_plots.py` — sampled lateral flow loop

The roll-axis analogue, for the regime after `CENTER` has already centred the
target. There the roll controller is not chasing position at all; it regulates
the lateral translational optical flow

```
omega = (v_platform - v_drone) / h
```

toward zero. With the small-angle approximation `a_drone = g·phi` and
`phi_cmd = k_D·omega`, the acceleration-equivalent gain is `K = g·k_D` — which
is exactly the conversion to use when porting a validated angle-domain gain
into the acceleration-domain control law.

Same three figures, same ZOH treatment. Config is `LateralConfig`; note
`height_m` defaults to **0.18 m**, i.e. this is deliberately posed at leg
clearance, the worst case, not at a comfortable cruise height.

The pitch axis is the same analysis with the appropriate sign convention, so
it isn't duplicated here.

### `gain_plot.py` — the authority/stability envelope

The one-page picture behind the whole feasibility argument: two bounds against
height.

* A **floor** `k_min`, from the authority requirement — the gain must be large
  enough to produce the acceleration the platform demands.
* A **ceiling** rising with height (`k ∝ z`), from the de Croon sampled-loop
  stability limit.

They cross at a **critical height** `h_crit`. Above it there is a usable gain
window; below it there is none, and no tuning recovers it. That crossing is
what the mission's vertical gate computes per flight from measured
`peak_accel` and measured dead time, instead of from the fixed numbers plotted
here.

Read this one first — the rest of `mission/gates.py` is much easier to follow
once the two-bounds picture is in your head.

### `preLanding_Graphs.py` — approach trajectories

The kinematics of a constant-divergence descent before any gate reasoning
enters: exponential height decay `z(t) = z₀·exp(λt)`, the resulting vertical
speed, and — the useful part — the family of trajectories obtained when
divergence is ramped with time constant `tau` rather than applied as a step.

`paw` is the leg-clearance offset, so the curves terminate at real contact
rather than at the mathematical `z = 0` an exponential never reaches. This is
why the mission ramps `D*` through a raised cosine instead of commanding it
directly.

---

## How these relate to the flown code

| Plot | Implemented in |
|---|---|
| Authority floor vs stability ceiling, `h_crit` | `mission/gates.py` — `compute_gate` (vertical), `compute_lateral_gate` |
| Drone/platform synchronization | `mission/visual_mismatch.py` — the `chi` bandwidth estimator, and `compute_tracking_gate` |
| `k(t)` decay and its asymptote | `mission/schedule.py` |
| `D*` ramping | the raised-cosine entry ramps in `mission/phases/` |
| `sample_time_s` | measured, then `CameraConfig.processing_latency_budget_sec` → `stability_dt_sec` |

The scripts use fixed, illustrative parameters. The flown code derives the same
quantities per flight from measurements, which is why a plot here and a verdict
in `summary.txt` can disagree — and when they do, the flight is right.

Figure titles and axis labels in `vertical_plots.py` and `lateral_plots.py` are
in French.