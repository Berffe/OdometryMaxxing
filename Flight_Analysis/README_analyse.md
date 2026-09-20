# BEE_LAND log tools

Two scripts, one calling convention. Both read the CSVs written by
`diagnostics_writer.py` and both take **a run folder and an output folder**.

| Script | Produces |
|---|---|
| `analyse_log.py` | `summary.txt` + the diagnostic plots |
| `trajectory_video.py` | `trajectory.mp4`, the 3-panel animation |

Requirements: Python 3.9+, `pandas`, `numpy`, `matplotlib`. The video also
needs `ffmpeg` on the PATH.

---

## One run per folder

A run is a set of CSVs sharing the same timestamp suffix:

```
logs/run01/bee_controller_20260817_145438.csv
logs/run01/bee_truth_20260817_145438.csv
logs/run01/bee_wind_20260817_145438.csv
```

Point either tool at that folder. If a folder holds two complete runs, the
tools refuse to guess and tell you to split them.

`analyse_log.py` needs all three files. `trajectory_video.py` needs only
controller + truth and ignores the wind log.

---

## `analyse_log.py`

```bash
python3 analyse_log.py RUN_DIR OUTPUT_DIR [--wind] [--full] [--max-time SECONDS]
```

```bash
py analyse_log.py logs/valid results/valid
py analyse_log.py logs/windy results/windy --wind
py analyse_log.py logs/run01 results/run01 --full --max-time 55
```

Explicit files still work, in any order — each CSV is classified by its
columns:

```bash
python3 analyse_log.py bee_truth_X.csv bee_wind_X.csv bee_controller_X.csv results/run01
```

| Flag | Default | What it does |
|---|---|---|
| `--wind` | off | Adds `wind_biases`, `wind_lateral_commands`, `wind_acceleration_contribution`, plus a wind section in the summary. |
| `--wind-force-scale K_W` | 1.0 | Gazebo `force_approximation_scaling_factor` used by the `--wind` acceleration reconstruction. |
| `--full` | off | Adds `platform_motion`, `platform_angles`, `drone_angles`, `relative_motion`, `target_detection`. |
| `--max-time SECONDS` | none | Cuts everything this many **simulated** seconds after the common start. Use it to drop a tail left running after touchdown. |

Reusing an output folder is safe: plots belonging to a group you did not
request this time are deleted rather than left stale.

### Output

`summary.txt` (also printed) — file paths and sample counts, clock health and
the effective Gazebo real-time factor, mission samples per phase, contact
numbers, divergence bias/RMSE against truth, feasibility verdict and reason,
visual-mismatch gates, and the vision delay tables in ms.

Plots, always:

| Group | Files |
|---|---|
| Vertical | `vertical_divergence`, `vertical_descent`, `vertical_commands` |
| Lateral | `lateral_match`, `lateral_commands`, `lateral_decomposition`, `lateral_p_gain` |
| Probes | `probe_vertical`, `probe_roll`, `probe_pitch` |
| Bandwidth gates | `visual_mismatch_z_probe`, `_x_`, `_y_` |
| Vision / geometry | `detections_boxes_fov`, `drone_platform_position`, `gain_schedule` |

---

## `trajectory_video.py`

Same convention. 3-D, side and top views, with the mission substate written in
the title of every frame.

```bash
python3 trajectory_video.py RUN_DIR OUTPUT_DIR [options]
```

```bash
python3 trajectory_video.py logs/run01 results/run01
python3 trajectory_video.py logs/run01 results/run01 --speed 0.5 --max-time 55
python3 trajectory_video.py logs/run01 results/run01/slowmo.mp4 --speed 0.25
```

The output is a folder → `trajectory.mp4` is written inside it, or an explicit
`*.mp4` path. Two explicit CSVs in any order also work.

| Flag | Default | What it does |
|---|---|---|
| `--speed X` | 1.0 | Playback multiplier in SIM time. |
| `--max-time SECONDS` | none | Same convention as `analyse_log.py`. |
| `--no-stop-at-contact` | on | Keep rendering past first contact. |
| `--platform-radius M` | 0.5 | Deck radius drawn in the three views. |
| `--trail-seconds S` | 6.0 | Length of the fading motion trail. |
| `--video-fps FPS` | 15.0 | Output frame rate. |
| `--elev` / `--azim` | 24 / −56 | 3-D camera angles. |
| `--dpi` | 110 | Output resolution. |

---

## Reading a run in 60 seconds

1. `summary.txt` → **divergence bias and RMSE**. A biased estimator makes
   everything downstream uninterpretable.
2. **Mission samples by phase.** A probe that got three frames did not probe.
3. **`probe_*.png` for the axis that failed** — the gate rests on `peak_accel`;
   these show whether it came from platform motion or self-induced oscillation.
4. **`vertical_divergence` + `vertical_descent`.** Divergence hugging `D*`
   without thrust on its rails is healthy; flat-topped thrust with swinging
   divergence is the delay-induced limit cycle.
5. **Contact numbers** — true minimum pad clearance and closing rate.
6. **Delay tables** if anything above smells like dead time.

## Traps

- All plots are in **SIM time**. The real-time factor is well below 1, so a
  plot second is much shorter than a wall second. The summary prints the
  factor.
- `optical_flow_total_cpu_ms` is CPU across threads, **not** serial delay.
  Never add it to a latency budget.
- `fit_quality` is diagnosis-only. High does not mean the divergence is right;
  in pure translation it is legitimately near zero while `lambda` is correct.
- Divergence convention: the controller's `lambda` is `c/h`, fronto-parallel
  flow divergence is `2*c/h`.
- Wall-period and `*_age_*` columns mix clock families and are RTF-inflated.
  Use the delay tables for real dead time.
- The true skid underside sits at `z = -0.227` m in `base_link`, not `-0.20`.