# Campaign automation

Six files that turn one manual four-terminal run into 360 unattended ones.

```
automation/
├── scenario.py        what a seed and a cell determine
├── make_world.py      generates a per-run world from bee_platform.sdf
├── run_once.sh        one run: five processes, readiness-gated, one session
├── teardown.sh        kill the group, sweep orphans, assert the machine is clean
├── campaign.py        define the matrix, fly it, resume where it died
├── scenario_test.py   four-run smoke test; run this before any overnight
├── diagnose_env.sh    why the launch failed, in seconds instead of timeouts
├── diagnose_camera.sh is the camera broken in gazebo, or in the bridge?
└── lib_env.sh         shared environment setup, sourced by both scripts
```

## Gazebo has to be resolved, not looked up

Sourcing `/opt/ros/<distro>/setup.bash` breaks `gz sim` in two independent
ways, and either alone is enough:

**PATH.** ROS prepends the vendored `gz` CLI (`gz_tools_vendor`), which has only
`log`, `msg`, `param`, `service`, `topic` — no `sim`, no `sdf`.

**`GZ_CONFIG_PATH`.** The `gz` CLI has no subcommands built in: it discovers
them from YAML files in the directory this variable names, and it holds a
**single path, not a list**. ROS points it at the vendored share directory,
which carries transport/msgs YAMLs but no `sim8.yaml` — so `/usr/share/gz` is
masked completely and *every* `gz` on PATH loses `sim`, including a perfectly
good `/usr/bin/gz` from a full Harmonic install.

The second is the nastier one, because the obvious fix — "call `/usr/bin/gz`
directly" — does not work. The binary is fine; it cannot see its own subcommand
definitions.

Both are invisible in manual use, because the terminal that launches PX4
typically has not sourced ROS. Only automation that sources ROS *and* calls
`gz sim` trips over them.

`lib_env.sh` therefore resolves a **(binary, config directory) pair**, probing
combinations of `$GZ_BIN` / each `gz` on PATH / the usual system locations
against `$GZ_CONFIG_PATH_OVERRIDE` / the inherited value / `/usr/share/gz` /
`/usr/local/share/gz` / unset. It selects the first whose `gz sim --versions`
returns something shaped like a version number — exit status alone is not
enough, since a CLI missing the subcommand prints help and still exits 0.

Every gz call in both scripts goes through the `bee_gz` wrapper, so `sim`,
`topic`, `sdf` and the orphan canary can never end up on different installs or
different config paths.

Overrides, if your layout is unusual:

```bash
export GZ_BIN=/path/to/gz
export GZ_CONFIG_PATH_OVERRIDE=/path/to/share/gz
```

## Before the overnight: the smoke test

```bash
python3 scenario_test.py --dry-run     # generate and check worlds, fly nothing
python3 scenario_test.py               # fly all four (~8 min)
```

Four runs — calm and severe, each gate-on and gate-off — at the nominal 0.50 m
deck, so a surprise is attributable to the automation rather than to a platform
size never tried before. It calls `campaign.run_one`, so the only difference
between it and the overnight is the length of the list.

It separates two kinds of result, and only the first can fail:

**Harness checks** must hold whatever the physics did — paired worlds identical,
every run left a strictly parseable record, `--gate off` actually reached the
controller, the three CSVs exist, the machine came back clean between runs. A
failure here means the overnight would produce garbage.

**Observations** are properties of the flight: whether the severe run refused,
whether its gate-off twin descended, rho, probed peak acceleration. These are
single samples of a stochastic process — the Gazebo README is explicit that
peak acceleration varies several-fold between identical configurations — so a
surprising value is information, not a failure. A test that fails on legitimate
physics variation gets ignored within two days.

Exit code is 0 only if every harness check passed. The summary is also written
to `smoke_report.txt` in the run directory.

Two checks worth knowing about:

- **The gate must not reach the world file.** If it did, gate-on and gate-off
  would be different scenarios and the ablation would compare two seas instead
  of two decisions — invalidating the paper's most important result while
  looking entirely normal in the logs.
- **The outcome record is parsed with `parse_constant` set to raise.** Python
  accepts bare `Infinity` by default, so a plain `json.loads` would pass a
  record every strict parser rejects. rho is legitimately `inf`, so this is the
  check that the writer's null-handling actually works.

## Running it

```bash
cd ~/PX4-Autopilot/BEE_LAND/controller/automation
export BEE_VENV=~/.control_venv          # if the controller lives in one

python3 campaign.py --campaign campaign_02 --base-seed 20260921 --dry-run
python3 campaign.py --campaign campaign_02 --base-seed 20260921 --limit 2
python3 campaign.py --campaign campaign_02 --base-seed 20260921
```

Output lands in `~/PX4-Autopilot/BEE_LAND/controller/logs/campaign_02/`:

```
campaign.json                 base seed, full case table, source world, start time
runs/ws1-ps2_r050_gateoff_rep00/
├── world.sdf                 the exact world this run flew
├── run.json                  the resolved scenario
├── bee_outcome_*.json        the verdict
├── bee_{controller,truth,wind}_*.csv
└── logs/{gz,agent,px4,bridge,node,run,teardown}.log
```

Re-running the same command skips finished runs. A campaign that dies at 3 am
costs only the runs it had not reached.

## Campaign matrix

360 runs: 6 wind cases × 6 platform cases × 5 deck radii × gate on/off, one
repetition per cell. The platform moves in **translation only** (surge, sway,
heave); roll and pitch are empty. Everything is defined in `campaign.py`
(`WIND_CASES`, `PLATFORM_CASES`, `PLATFORM_RADII_M`); `campaign.json` records
the full table for provenance.

Commands are under *Running it*. Expect ~3 min per run, so ~18 h: two nights. Flight order puts the nominal
0.50 m deck first — the first 72 runs are a complete 6 × 6 × 2 block — and
flies each gate-on/gate-off twin back to back, so machine drift (`measured_T`
under load) hits both arms of a pair equally. Run ids read
`<wind>-<platform>_r<radius cm>_gate<on|off>_rep<NN>`.

### The design rule: the bandwidth limit is a jerk budget

The gates evaluate three things at the FINAL_PROBE hold:

| Quantity | Condition | Numbers (current `config.py`) |
|---|---|---|
| Vertical authority | ρ_z = K_min/K_ceil(h_leg) ≤ 1, K_min = d_z/ω*_z | d_z ≤ 0.30 · 2.85 = 0.85 m/s² (probed, protected) |
| Lateral authority | ρ_xy = (c_max/κ + d_xy/ω_adm)/K_ceil,xy(h_leg) ≤ 1 | d_xy ≤ 0.38 m/s² |
| Synchronisation | χ ≤ χ_lim | χ_z,lim = 0.3, χ_xy,lim = 0.5 s⁻² |

with T_z = 76.7 ms, T_xy = 176.7 ms, K_z,probe = 3.13, K_D,probe = 1.12,
κ = 1/tan 40°, and the probe protection d = 1.15·Q95 + Δa.

From the loop model of `complete_math.tex`, χ_z = a_p·S(jω)/h and, well
below the loop bandwidth (1.25 Hz vertical, 0.53 Hz lateral at h_near),
|S| ≈ ωh/K. Hence

    χ ≈ j / K,      j = Σ a_i ω_i   (platform jerk amplitude)

independent of height, therefore of deck radius. The frequency limit is a
**jerk budget**: j_z ≤ χ_z,lim·K_z = 0.94 m/s³ and j_xy ≤ χ_xy,lim·K_D =
0.56 m/s³. Acceleration inside the gain region is not sufficient; weak cases
must also keep their frequency low enough to stay inside this budget.

The model reproduces both flights of the internship report: the "calm" sea
(landed) predicts χ_z = 0.37 × limit, the "aggressive" sea (refused on χ)
predicts 1.4 × limit.

Design targets:

- **weak** — ρ ≤ 0.5 on every axis, χ ≤ 0.4 × limit (the level of the
  validated calm flight). Leaves margin for estimator noise and for the real
  loop being slower than the model.
- **strong wind** — lateral region empty (ρ_xy > 1), χ_xy clean.
- **strong platform** — one case per corner of the (amplitude, frequency)
  plane, so refusal attribution separates the criteria.

### Platform cases

Components are (amplitude m, frequency Hz) with the nominal world's base
phases; the seed adds one phase offset per axis.

| Case | Heave A (m) | Heave f (Hz) | Surge A (m) | a_z (m/s²) | j_z (m/s³) | Predicted ρ_z / χ_z |
|---|---|---|---|---|---|---|
| pw1 | 0.08 / 0.20 / 0.03 | 0.06 / 0.10 / 0.14 | 0.030 / 0.025 / 0.010 | 0.11 | 0.07 | 0.21 / 0.08 |
| pw2 | 0.10 / 0.28 / 0.05 | 0.075 / 0.12 / 0.165 | 0.040 / 0.030 / 0.015 | 0.24 | 0.19 | 0.36 / 0.19 |
| pw3 | 0.10 / 0.18 / 0.06 | 0.10 / 0.16 / 0.22 | 0.050 / 0.040 / 0.020 | 0.34 | 0.37 | 0.49 / 0.37 |
| ps1 | 0.25 / 0.85 / 0.10 | 0.08 / 0.13 / 0.18 | nominal | 0.76 | 0.64 | 1.04 / 0.66 |
| ps2 | 0.02 / 0.07 / 0.02 | 0.23 / 0.36 / 0.50 | 0.004 / 0.004 / 0.002 | 0.60 | 1.49 | 0.77 / 1.45 |
| ps3 | 0.20 / 0.50 / 0.10 | 0.10 / 0.20 / 0.25 | nominal | 1.12 | 1.43 | 1.37 / 1.39 |

Sway is about a third of surge on the same frequencies. Lateral platform
acceleration stays ≤ 0.10 m/s² in every case, so a lateral refusal is
attributable to the wind.

- **pw3** is the "calm" sea of the report: a validated landing anchors the top
  of the weak range.
- **ps1, heavy swell** — sits on the authority boundary with χ clean. A
  vertical refusal on authority *alone* needs swell: raising a at fixed
  displacement raises frequency, hence jerk. With heave capped at 1.2 m (deck
  base at z = 2 m), ρ_z ≈ 1 is the best reachable. Expect a split of outcomes;
  that is the point — it tests the sharpness of the predicted boundary, and the
  gate-off twins test the predicted crossing height.
- **ps2, short chop** — inside the gain region but 1.5 × over the jerk budget:
  the χ-only refusal. High encounter frequencies are what a craft heading into
  a wind sea experiences (ω_e = ω₀ − ω₀²U cos μ / g; Fossen, *Handbook of
  Marine Craft Hydrodynamics and Motion Control*).
- **ps3** is the shipped `bee_platform.sdf` sea without tilt: both limits.

Wave frequencies of 0.1–0.25 Hz correspond to the Pierson–Moskowitz modal
frequency ω₀ = 0.4 √(g/H₁/₃) for sea states 3–5.

### Wind cases

Gazebo's `WindEffects` is scaled by 0.08, so the disturbance is 0.08 m/s² per
m/s of relative wind. The **mean** is absorbed by the static trim and never
reaches a gate; the **gust** is what the probe measures. Wind speeds are
therefore nominal: what the gates see is gust acceleration.

| Case | Mean (m/s) | Turbulence x: count, A (m/s) | Dominant gust (m/s, Hz) | σ_x (m/s) | Predicted outcome |
|---|---|---|---|---|---|
| ww1 | 2 | 3 × [0.10, 0.30] | — | 0.25 | ρ_xy ≤ 0.38 |
| ww2 | 4 | 3 × [0.20, 0.50] | — | 0.44 | ρ_xy ≤ 0.40 |
| ww3 | 6 | 3 × [0.40, 0.80] | — | 0.75 | ρ_xy ≤ 0.47 |
| ws1 | 6 | 3 × [0.40, 0.80] | x: [4.2, 5.0], [0.06, 0.10] | 3.3 | refused 95–100 % |
| ws2 | 3 | 3 × [0.60, 1.00] | x: [5.0, 5.8], [0.06, 0.10] | 3.9 | refused ≈ 100 % |
| ws3 | (3.5, 3.5) | 3 × [0.40, 0.80] on x and y | x and y: [3.6, 4.4], [0.06, 0.10] | 2.9 per axis | refused 96–97 % |

Turbulence frequencies are drawn in 0.03–0.12 Hz; the y axis carries a
smaller copy of the x turbulence except in ws3.

- **Band.** The Dryden low-altitude model (MIL-F-8785C, MIL-HDBK-1797) gives
  L_u = h/(0.177 + 0.000823 h)^1.2 ≈ 37 m at h ≈ 5 m, so the gust spectrum
  corners at V/L ≈ 0.03–0.05 Hz. The band covers the corner and the decade
  above it.
- **Intensity.** Weak cases stay below MIL "light" turbulence
  (σ_u ≈ 1.5 m/s at this height); strong cases fall between "moderate"
  (≈ 3.0 m/s) and "severe" (≈ 4.5 m/s).
- **Dominant gust.** MIL-F-8785C pairs continuous turbulence with a discrete
  gust; here one seeded sinusoid plays that role. Its 0.06–0.10 Hz band puts at
  least one full period inside the 20.1 s FINAL_PROBE window. Sums of random
  sines alone refused only ~75 % of strong runs, because the probe window
  often fell in a destructive beat.
- **Contrasts.** ws1 shares ww3's mean and background, so the pair isolates
  turbulence. ws2 has the lowest mean and the largest gust: refusal must follow
  the gust. ws3 is oblique, exercising both lateral gates.
- **Authority.** Worst-case wind load (mean plus every amplitude aligned) stays
  ≤ 1.25 m/s², i.e. ≤ 7.3° of the 11.5° tilt limit.

### Predicted attribution

| | weak platform | ps1 | ps2 | ps3 |
|---|---|---|---|---|
| weak wind | landed | region (vertical) | χ only | both |
| strong wind | region (lateral) | region | both | both |

The gate-off twins of the weak × weak quadrant measure the false-refusal rate
directly; the calm re-runs of the original plan are no longer needed.

### What the seed draws

The seed is derived from (wind case, platform case, radius, repetition) —
**never from the gate**, so a gate-on/gate-off pair flies a byte-identical
world. Draw order is fixed:

| Drawn per run | Fixed by case |
|---|---|
| One platform phase offset per axis (5 draws) | Amplitudes, frequencies, relative phasing |
| Spawn x/y jitter ±0.15 m, yaw ±0.35 rad | Spawn distance and height |
| Dominant gust amplitude, frequency, phase per lateral axis (always drawn) | Gust ranges |
| Plugin seed → turbulence amplitudes, frequencies, phases | Mean wind, turbulence ranges |

The gust draws are consumed even when a case has no gust, so every case keeps
the same draw sequence.

**Spawn distance** is max(1.70, r + 0.85) m from the deck centre. The drone
climbs vertically from its spawn, so it must start outside the deck footprint
plus the deck's lateral motion plus its own half-span. The earlier fixed
1.2 ± 0.25 m spawn put it under the deck for r ≥ 1.0 m. Worst-case clearance
is now 0.57 m. At r = 1.5 m the deck centre starts near the image edge; fly
one r = 1.5 run with `--limit` before committing the night.

**Wind components.** `WindSpec.gust_*` is written as explicit `<component>`
blocks beside `<synthesis>`; `WindController` sums both.
`max_wind_speed` is set per run to the conservative speed bound, so the
plugin's clipping warning never fires.

### Model caveats

The predictions come from the linear loop of `complete_math.tex` with the
stability periods as lumped delays, probed at h = 0.8·r. Two effects are
not in it: estimator noise on χ (hence the 0.4 × margin on weak cases), and a
real probe that reads lower than the model — the report's aggressive flight
showed a narrow margin where the model predicts ρ_z = 1.38. The ps1 boundary
is the case most exposed to the second.

## Why we start Gazebo, not PX4

`PX4_GZ_WORLD` is ignored when a simulation is already running, and the `gz sim`
server PX4 spawns is a sibling that outlives it. That is why your second run
never restarted the world.

`run_once.sh` starts the server itself and launches PX4 with
`PX4_GZ_STANDALONE=1` to attach. That gives teardown something it can actually
kill, a readiness signal to poll instead of a sleep, and a per-run world reached
by path. It also restores `PX4_GZ_MODEL_POSE`: the alias path
(`PX4_SIM_MODEL` with no `PX4_GZ_MODEL`) disables it, and the spawn pose is part
of the scenario.

## Launch order and readiness

| # | Process | Gated on |
|---|---|---|
| 1 | `gz sim -s -r world.sdf` | `/world/bee_platform/clock`, then `/bee_land/truth` |
| 2 | `MicroXRCEAgent udp4 -p 8888` | UDP 8888 bound |
| 3 | `px4` (standalone) | a **message** on `/fmu/out/vehicle_status*` |
| 3b | (still PX4) | exactly one vehicle, and **camera frames on the gz topic** |
| 4 | `bridge.sh` | messages on camera, truth and wind; gear contact topics present |
| 5 | `bee_node` | runs to its own terminal outcome |

Every wait is a poll with a deadline. PX4 precedes the bridge because the camera
and contact sensors only exist once the model is spawned.

The PX4 gate requires a received message, not merely a listed topic: a stale
DDS discovery record can advertise a topic with no publisher behind it.

Camera frames are checked on the **Gazebo** topic before the bridge starts, and
again on the ROS topic after. `gz topic -l` listing a camera only proves a
sensor advertised; it says nothing about whether images are being rendered, and
those two failures need completely different fixes. Checking before the bridge
exists means a rendering failure can never be misread as a bridge failure.
`diagnose_camera.sh` runs that same separation standalone, in about a minute.

## Two PX4 launch traps

**`PX4_SIM_MODEL` must be set even though `PX4_GZ_MODEL` is.** Autostart 4001
defaults `PX4_SIM_MODEL` to plain `x500`, and leaving that default makes PX4
spawn a *second* vehicle (`x500_0`) beside `bee_x500_0`. Both run, the bridge
keeps bridging, and PX4 binds to whichever it spawned last — so the camera and
the contacts can end up describing different aircraft. `run_once.sh` sets both
to `bee_x500`; `PX4_GZ_MODEL` is still needed because it is what lets
`PX4_GZ_MODEL_POSE` apply.

**PX4 needs `-d`.** Without it PX4 SITL starts its interactive `pxh>` console
and reads stdin. A backgrounded process that reads the terminal is stopped by
SIGTTIN and never becomes ready — it does not crash, it just hangs forever.
`run_once.sh` escaped this only because `setsid` detaches the controlling
terminal, which is an accident rather than a design. Every background process
in both scripts now also gets `</dev/null`.

**The vehicle-status topic is versioned.** Current PX4 publishes
`/fmu/out/vehicle_status_v4`; older builds use `_v1` or the unsuffixed name.
`VEHICLE_STATUS_TOPICS` in `run_once.sh` must stay in step with
`TopicsConfig.vehicle_status` in `core/config.py` — the controller subscribes to
all of them and takes whichever exists, so a launcher waiting on a narrower
list times out while the system is perfectly healthy.

## The orphan canary

`bridge.sh` and the truth plugin both hard-code `bee_x500_0`. That `_0` is PX4's
instance suffix, and it is `0` only when Gazebo started clean. If a stale server
survives, PX4 spawns `bee_x500_1`; every process keeps running, the bridge keeps
bridging, and gear contacts silently never fire. Nothing errors — the runs just
stop meaning anything.

`run_once.sh` asserts, before starting the bridge, that `bee_x500_0` exists
**and that it is the only vehicle in the world** — presence alone is not
enough, since the double-spawn above leaves `bee_x500_0` present and correct
next to a stock airframe. So this costs one run at launch instead of many at
analysis. `teardown.sh` asserts the
machine is clean after every run, and `campaign.py` stops the whole campaign if
it is not.

## Status vocabulary

The controller writes `landed`, `infeasible`, `aborted`, `crashed`. Two more
belong to the harness, because the node never reaches its own `close()`:

| Status | Written by | Means |
|---|---|---|
| `launch_failed` | `campaign.py` | a launch stage never became ready |
| `timeout` | `campaign.py` | the run exceeded `--timeout` (default 300 s) |

Both leave a record, so a failed run is evidence rather than an empty directory
the runner retries forever.

## Two things that bite in bash

**`set -u` and ROS.** `/opt/ros/<distro>/setup.bash` reads unbound variables as
a matter of course (`AMENT_TRACE_SETUP_FILES` among them), and so do most venv
activate scripts. `run_once.sh` drops `set -u` across the sourcing and restores
it immediately after. With it on, the launch aborts before a single process
starts.

**Teardown runs inside the group it is clearing.** `run_once.sh`'s exit trap
calls `teardown.sh "$$"`, and `$$` is the process-group leader that `setsid`
created — so a plain `kill -TERM -- -PGID` would signal the launcher and the
teardown script along with the simulation, the launcher's TERM trap would
re-enter, and the run would collapse into a loop of "Terminated".

Two defences, both needed:

- `teardown.sh` walks its own ancestry and removes itself and every ancestor
  from the target list before signalling anything.
- `run_once.sh`'s trap disarms itself on entry and guards with a flag, so a
  signal arriving mid-cleanup cannot re-enter it.

The target list is also snapshotted **once**. Re-enumerating the group on every
poll never terminates, because the `ps` doing the enumerating is itself a group
member — the wait would run its full duration on every run, which across 360
runs is about 37 minutes of doing nothing.

## Things deliberately not touched

- **`real_time_factor`** — asserted to be 1.0, never edited. The gate rests on a
  self-measured dead time and `K_max ∝ h/T`; a different RTF validates a
  different controller.
- **The world name `bee_platform`** — the platform model name, the truth
  plugin's `contact_target_substring` and both gear contact paths in `bridge.sh`
  are built from it.
- **PX4 parameters.** The campaign relies on the params from the Gazebo README
  already being saved in `build/px4_sitl_default/rootfs`. Nothing here sets
  them. A `make clean` mid-campaign would wipe them and every subsequent run
  would fail to arm — worth re-running the param block and `param save` before
  starting an overnight.

## Two known gaps

**`bee_platform.sdf` is not well-formed XML.** A comment contains
`f2 = 0.16 Hz  <-- dominant`, and `--` is illegal inside an XML comment.
Gazebo's TinyXML2 tolerates it, which is why it has always worked; strict
parsers reject the file outright. `make_world.py` repairs comments on load and
warns. The one-character fix in the source (`<--` → `<-`) is still worth making.

**The platform scale touches three places.** The collision cylinder is what the
skids touch, the visual cylinder is the rim, and the flower mesh is the texture
the controller tracks. `make_world.py` moves all three together; editing the
radius by hand and forgetting the mesh scale would give a target whose apparent
size does not match the surface it can land on — a systematic scale bias rather
than an obvious failure.
