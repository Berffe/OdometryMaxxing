# BEE_LAND flight analysis (`analyse_log.py`)

Post-flight analysis of the two CSVs written by `diagnostics_writer.py`:
the **controller log** and the **Gazebo truth log**. One command pairs them,
writes a text summary and a set of plots.

The pairing is the point. The controller is vision-only and never sees height,
velocity or platform state in flight; truth exists solely so that afterwards
you can ask *was the measurement right* separately from *was the control
right*. Almost every plot here puts a commanded or measured quantity next to
the truth it should have matched.

---

## Usage

```bash
python3 analyse_log.py <controller.csv> <truth.csv> <output_dir> [--full] [--max-time SECONDS]
```

All three positionals are required. The two CSVs are **order-independent** —
each file is classified by its columns (`truth_sim_time_sec` +
`truth_drone_position_z_m` → truth; `flow_sim_timestamp_sec` +
`controller_phase` → controller) and swapped into place if needed. Passing two
controller logs, or two truth logs, is a hard error rather than a silently odd
plot:

```
Expected one controller CSV and one truth CSV. Detected a.csv=controller, b.csv=controller.
```

Examples:

```bash
# Typical run
python3 analyse_log.py logs/bee_controller_20260720_135806.csv \
                       logs/bee_truth_20260720_135806.csv \
                       results/test01

# Order doesn't matter, and --full adds the detail plots
python3 analyse_log.py logs/bee_truth_X.csv logs/bee_controller_X.csv results/test01 --full

# Stop everything 55 simulated seconds after the common start
python3 analyse_log.py logs/ctrl.csv logs/truth.csv results/test01 --max-time 55
```

| Flag | Default | What it does |
|---|---|---|
| `--full` | off | Adds `platform_motion`, `platform_angles`, `drone_angles`, `relative_motion`, `target_detection`. |
| `--max-time` | none | Cutoff in **simulated** seconds after the common plot start. Use it to drop the tail of a run left going after touchdown. |

### Timestamps: SIM only

The analyser works entirely in Gazebo SIM time and does not reconstruct a PX4
or platform clock, or differentiate positions across streams.

* Controller rows are placed by `flow_sim_timestamp_sec`, falling back to
  `command_source_sim_timestamp_sec`, then `contact_truth_sim_timestamp_sec`.
* Truth rows use `truth_sim_time_sec`, sorted and de-duplicated.
* Both are shifted to a common origin, so plot time starts at 0.

The practical consequence is unchanged from earlier versions: on this SITL the
real-time factor is well below 1, so **durations on the plots are sim-seconds
and are much shorter than the wall-clock run took**. The summary prints the
measured factor so you can convert.

---

## What you get

### `summary.txt`

Printed to the console and saved. In order:

**Header** — both file paths, analysed SIM span, controller and truth sample
counts, and estimated logged rates: processed camera fps in SIM time, PX4
publish rate in monotonic wall time.

**Clock health** — mean/median truth SIM period, effective truth rate, truth
receipt wall period, and the **effective Gazebo real-time factor (SIM/wall)**.
Divergence between the SIM-time and wall-time rates is the first place a
scheduling problem shows up.

**Mission samples by phase** — how many rows each `mission_substate` got. A
descent that spent three frames in `FINAL_PROBE` did not really probe.

**Contact** — true minimum pad clearance at first contact, and true closing
rate at confirmed contact. These are the two numbers that say whether the
landing was gentle, and they come from truth geometry, not from the
controller's opinion.

**Divergence truth comparison** — sample count, **bias (measured − truth)** and
**RMSE**, in 1/s. This is the headline vision-quality number: it is the
difference between "the estimator is fine and the loop is badly tuned" and the
reverse. Mind the convention — the controller's `lambda` is `c/h`, while the
fronto-parallel flow divergence is `2*c/h`.

**Infeasibility reason** — if the mission ended in `INFEASIBLE`, the structured
reason (axis + criterion) is echoed here rather than left in the CSV.

**Vision delay tables** — three nested breakdowns, mean/median/p95/max in ms,
computed over fresh controller/vision rows in the analysed interval.

1. *End to end*: `frame_to_result`, camera callback before queue, IPC inbound
   and outbound, `target_acquisition.update()`, `optical_flow.update()`,
   transport total, `frame_to_command`, control compute.
2. *Inside `optical_flow.update()`*: grayscale conversion, ROI setup, adaptive
   downsample resize, Farneback dense flow, flow scaling/upsample, derotation,
   mean-flow reductions, gradient weights, robust affine divergence fit,
   pre-derotation diagnostic fit, divergence filter, result and state.
3. *Inside the affine fit*: array/design setup, initial weighted solve,
   residual quantile, trimmed weighted refit.

Reading them:

* **Wall-clock durations are the delay that matters.**
  `optical_flow_total_cpu_ms` is reported separately and can exceed wall time,
  because OpenCV/BLAS use several native threads. It is a compute-load
  indicator, **not** additional serial delay — do not add it to a budget.
* The IPC legs exist because vision runs in a separate process. They are real
  delay and are the reason `frame_to_result` exceeds the sum of the two vision
  stages.
* **`frame_to_result` p95 is the measured counterpart of
  `CameraConfig.processing_latency_budget_sec`** in
  `bee_control/core/config.py` (default 0.02 s) — it covers the same leg:
  camera callback, both IPC hops, and the two vision stages. Mind that the
  config field is only one of four terms:

  ```
  stability_dt_sec = frame_period + processing_latency_budget
                   + px4_setpoint_period + smoothing_delay
  ```

  so do not compare `frame_to_command` against the budget directly — that path
  already includes the frame period and control compute, and you would
  double-count. `stability_dt_sec` is what sets every gain ceiling in the
  feasibility gate.

  The config comments the field as a *retained design target, not a
  measurement*: it does not update itself from logs. If this table has drifted
  above it, the gate is being sized against a loop that no longer exists, and
  the number needs changing by hand. Use the in-flight figure, never an offline
  benchmark — the in-process residual attributed to the DDS/rmw layer is real
  and does not appear offline.
* Stage timings are genuine CPU/wall durations inside one process and are
  **not** inflated by the real-time factor, unlike logged period columns.

### Default plots

| File | Shows |
|---|---|
| `vertical_divergence.png` | Divergence tracking vs `D*`, and measured vs truth. The heart of the descent, and the plot the divergence bias/RMSE numbers come from. |
| `vertical_descent.png` | Height, closing rate and thrust against truth, with the contact plane marked. |
| `gain_schedule.png` | Flown `k(t)` against the mission's schedule, with the floor and ceiling. |
| `probe_vertical.png`, `probe_roll.png`, `probe_pitch.png` | Per-axis probe internals: command-derived acceleration vs **true** relative acceleration, the residual, the rolling percentile, the leaky peak the gate consumes, and the scheduled-floor / stability-ceiling capacities. This is where a wrong verdict is diagnosed. |
| `visual_mismatch_z_probe.png`, `_x_`, `_y_` | The `chi` bandwidth estimator per axis, with the live leaky peak against the rejection limit. |
| `lateral_control.png` | Roll/pitch commands vs the offsets and flow driving them, against truth. |
| `lateral_optical_flow.png` | Lateral flow evolution — the signal the lateral gates are computed from. |
| `detections_boxes_fov.png` | Detection boxes reconstructed in the camera FOV, with FOV saturation marked. Spatial sanity check on target acquisition. |
| `drone_platform_position.png` | Drone and platform world positions from truth. |

### Extra plots with `--full`

`platform_motion.png`, `platform_angles.png`, `drone_angles.png` (true
attitude plus commanded Euler angles), `relative_motion.png`,
`target_detection.png`.

---

## Reading a run in 60 seconds

1. **`summary.txt` → divergence bias and RMSE.** If the estimator is biased,
   nothing downstream is interpretable.
2. **Mission samples by phase.** Confirm the probe phases got enough frames to
   mean anything — `ready` requires roughly one platform period of total
   probing, not just a full near-field hold.
3. **`probe_*.png` for the axis that failed.** The gate rests on one number
   (`peak_accel`); these plots show whether it came from real platform motion
   or from self-induced oscillation. Remember the probe samples whatever
   platform phase it caught: one verdict is one sample.
4. **`vertical_divergence.png` + `vertical_descent.png`.** Divergence hugging
   `D*` without thrust hitting its rails is a healthy descent; flat-topped
   thrust with divergence swinging is the delay-induced limit cycle.
5. **Contact numbers.** True minimum pad clearance and closing rate at
   confirmed contact are the actual landing quality metric.
6. **The delay tables** if anything above looks like dead time — is
   `frame_to_command` still consistent with the budget the gate was configured
   with?

**Height caveat:** true skid underside sits at `z = -0.227` m in `base_link`,
not `-0.20`. It matters when judging "how high did we actually stop".

---

## Column families in the CSVs

**Controller log — 265 columns** (8 base fields plus 257 contributed by the
registered telemetry sources; each subsystem owns its own names, so this list
grows from the producing module, never from the writer):

| Prefix | Count | What |
|---|---:|---|
| `mission_*` | 138 | Substate, the three probes, the gates, `k(t)`, `chi`, feasibility verdicts and reasons. |
| `timing_*` | 67 | Per-stage vision costs and IPC legs. RTF-independent CPU/wall time inside the process. |
| `flow_*` | 17 | Divergence (filtered + raw), mean flow, `fit_quality`, derotation diagnostics, SIM timestamp. |
| `target_*` | 8 | Detection outputs: found, offsets, box, area fraction, `fov_saturated`. |
| `px4_*` | 7 | Nav/arming/failsafe state. `px4_arming_state` flipping 2→1 is the disarm moment. |
| `contact_*` | 7 | The contact subset extracted from the truth packet — the only truth the node acts on. |
| `command_*` | 6 | The attitude/thrust setpoints actually sent. |
| `clock_*` | 5 | Time-manager state, including detected clock steps. |
| `log_*` | 4 | Wall and monotonic stamps, elapsed times. |
| `event_*` | 2 | Event rows: transitions and one-off notices. |
| `vision_*` | 2 | Worker health, dropped frames. |
| `controller_phase` | 1 | The **outer** sequencer phase. Not the same thing as `mission_substate`. |
| `diagnostics_schema_version` | 1 | Fingerprint of the assembled header; a renamed or lost column is detectable offline without diffing headers. |

**Truth log — 89 columns**, all `truth_*`, laid out by `truth_layout.py` and
mirrored in the C++ plugin enum. Drone and platform pose, orientation,
velocities and accelerations; per-skid distances; signed pad clearance;
closing rates; camera normal geometry; `truth_sim_time_sec` and
`truth_receipt_wall_timestamp_sec`.

Two traps worth naming:

* `fit_quality` is **diagnosis-only**. A high value does not mean the
  divergence is right — it means the affine model explained the flow field it
  was given. In pure translation it is legitimately near zero while `lambda` is
  perfectly correct.
* Wall-clock period and `*_age_*` columns mix clock families and are
  RTF-inflated. Do not use them to size real dead time; use the delay tables.

---

## `trajectory_gif_3panel.py`

Renders a 3-panel trajectory video (MP4, via FFMpeg) from the same pair of
logs, with the mission substate annotated:

```bash
python3 trajectory_gif_3panel.py \
  --controller-csv logs/bee_controller_X.csv \
  --truth-csv      logs/bee_truth_X.csv \
  --output-mp4     results/test01/trajectory.mp4
```

| Flag | Default | What |
|---|---|---|
| `--platform-radius` | 0.5 | Deck radius drawn in the 3-D view. |
| `--speed` | 1.0 | Playback multiplier. |
| `--trail-seconds` | 6.0 | Length of the motion trail. |
| `--video-fps` | 15.0 | Output frame rate. |
| `--no-stop-at-contact` | off | Keep rendering past touchdown. |
| `--elev` / `--azim` | 24 / −56 | 3-D camera angles. |
| `--dpi` | 110 | Output resolution. |

The defaults point at absolute paths from a past session — always pass the
three path flags explicitly.