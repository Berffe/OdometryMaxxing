# BEE_LAND

Vision-only landing on a moving platform: a PX4/Gazebo SITL quadrotor lands
on a platform oscillating in x, y and z using nothing but a downward camera —
divergence-based control (Herisse 2012 / de Croon 2016), no height or
velocity estimate, no platform model.

This page is a map. Each linked README goes deep on its own piece; this one
stays short on purpose.

This is a research repo. The controller has since been refactored into a
modular package with a documented architecture, so it is no longer the pile of
flat files it started as — but plenty of it is still "it gets the job done"
rather than beautiful, and we encourage you to optimize wherever you see the
possibility!!

---

## Start here

| Step | Guide |
|---|---|
| 1. Set up the environment (WSL2, PX4, Gazebo, ROS 2, MAVSDK) | [`BEELAND_InstallGuide.md`](./Guides_n_Tutorials/BEELAND_InstallGuide.md) |
| 2. Launch a simulation (5 terminals, PX4 → bridge → node) | [`BEELAND_LaunchGuide.md`](./Guides_n_Tutorials/BEELAND_LaunchGuide.md) |
| 3. Build the world: platform, plugins, drone model | [`README_gazebo.md`](./Gazebo_defs/README_gazebo.md) |
| 4. Understand what's flying, and find the file to edit | [`README_controller_logic.md`](./Controller_logic/README_controller_logic.md) |
| 5. Analyze a flight | [`README_analyse.md`](./Flight_Analysis/README_analyse.md) |

---

## Repo map

| Folder / file | What's there |
|---|---|
| `Controller_logic/` | The full vision + control + mission stack as the `bee_control` ROS 2 package. Layering, runtime structure, both state machines, the feasibility argument, and a task → file lookup table in [`README_controller_logic.md`](./Controller_logic/README_controller_logic.md). |
| `Gazebo_defs/worlds/` | `bee_platform.sdf` — the world, both custom plugins wired in, and the flower deck materials. `platform_motion_examples.sdf` holds ready-made oscillation profiles. |
| `Gazebo_defs/plugins/` | `oscillating_platform_controller/` drives the deck; `bee_landing_truth/` publishes the ground-truth packet. Build and install steps in [`README_gazebo.md`](./Gazebo_defs/README_gazebo.md). |
| `Gazebo_defs/models/bee_x500/` | The camera-equipped drone model — x500 plus a downward camera and skid contact sensors. |
| `Gazebo_defs/flower_generator.py` | Regenerates the deck texture and its normal map. The texture is a *sensor input*, not decoration — it is designed against measured gradient statistics. |
| `Gazebo_defs/truth_layout.py` | The truth-packet field layout. **Must stay identical** to `Controller_logic/bee_control/diagnostics/truth_layout.py`. |
| `Flight_Analysis/` | `analyse_log.py` pairs a controller log with its truth log and writes plots plus `summary.txt`; `trajectory_gif_3panel.py` renders a 3-panel trajectory video. See [`README_analyse.md`](./Flight_Analysis/README_analyse.md). |
| `Calc_n_Visuals/` | Standalone theory plots — the stability/authority envelope and the sampled-loop frequency analysis that the feasibility gate implements. See [`README_calc_vis.md`](./Calc_n_Visuals/README_calc_vis.md). |
| `Guides_n_Tutorials/` | Install and launch guides. |
| `bridge.sh` | The single Gazebo → ROS 2 bridge process: camera, platform pose, touchdown, the truth packet, and both skid contact sensors. |

---

## The controller at a glance

`bee_control` is ~11,700 lines across 46 files, layered so dependencies point
one way only:

```
bee_node.py     ROS 2 node: I/O, timers, wiring. Owns nothing conceptual.
core/           shared primitives — clock, config, dataclasses, live state
vision/         camera frames → measurements. Runs in a SEPARATE PROCESS.
mission/        the visual landing sequence (state machine + feasibility maths)
control/        measurements → attitude/thrust setpoint
interfaces/     talking to the vehicle: PX4 uORB, MAVSDK, the outer lifecycle
diagnostics/    the log schema contract and the two CSV writers
tests/          ROS-free contract tests + the visual debug harnesses
```

Three things worth knowing before you open any of it:

- **There are two state machines, deliberately separate.** `FlightSequencer`
  owns the outer lifecycle (takeoff → offboard → handoff → terminal);
  `MissionRoutine` owns the inner visual landing sequence and only runs inside
  the sequencer's `CLOSED_LOOP` phase. They are logged in separate columns
  (`controller_phase` vs `mission_substate`) and are not interchangeable.
- **Vision runs in its own process.** `rclpy` uses a single-threaded executor,
  and a 40–200 ms vision frame inline would stall the offboard setpoint stream
  long enough to trip PX4's `COM_OF_LOSS_T` failsafe.
- **Nothing declares a log column on someone else's behalf.** Each subsystem
  implements `telemetry_fields()` / `telemetry()`, and `DiagnosticsWriter`
  assembles the 265-column header from whoever is registered. Adding a column
  is a one-file change.

Adding a mission phase, a scheduled gain, or a log column should require **no
edit to `bee_node.py`**. If it does, the seam is in the wrong place — see the
task → file table in
[`README_controller_logic.md`](./Controller_logic/README_controller_logic.md).

---

## Tests and debug harnesses

Two different kinds, run from `Controller_logic/` (the directory containing
`bee_control/`). Neither needs PX4, Gazebo, or a sourced ROS environment —
only `numpy` and `opencv-python`.

**Contract tests** — 22 tests, no display, CI-friendly. They guard the seams
(telemetry schema, config validation, phase dispatch, sequencer transitions,
the `chi` gate, clock-step handling), not the numerics:

```bash
cd Controller_logic
python3 -m pytest bee_control/tests/test_contracts.py -q
```

**Visual harnesses** — interactive OpenCV windows, for checking the vision
stack on synthetic scenes where the right answer is known analytically:

```bash
python3 -m bee_control.vision.optical_flow              # divergence-reduction bench
python3 -m bee_control.vision.target_acquisition        # detector, six pipeline stages
python3 -m bee_control.tests._optFlow_targetAcqu_debug  # end-to-end detector → ROI → flow
```

The first scores seven ways of reducing one Farneback field to `lambda`
against analytic ground truth, reporting bias and RMS. It is the tool for
answering "is the divergence estimate actually good" with numbers instead of
impressions.

---

## Where the project stands

Repeatable smooth landings on the oscillating platform, a vision pipeline
optimized and validated stage by stage, and a self-measuring latency budget
feeding a de Croon-style feasibility gate before every descent. Ground truth
now comes from a Gazebo-native plugin on a single SIM clock, so post-flight
analysis compares measurement against truth directly instead of
reconstructing it.

The controller has been refactored from a monolith into the layered package
above, with the dependency rules, both state machines, and the feasibility
argument written down in
[`README_controller_logic.md`](./Controller_logic/README_controller_logic.md).

Next up: closing the loop on the gate's dead-time estimate, then minimizing
descent overshoot via the divergence gain schedule.