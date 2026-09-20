# Campaign automation

Six files that turn one manual four-terminal run into ~225 unattended ones.

```
automation/
├── scenario.py        what a seed and a cell determine
├── make_world.py      generates a per-run world from bee_platform.sdf
├── run_once.sh        one run: five processes, readiness-gated, one session
├── teardown.sh        kill the group, sweep orphans, assert the machine is clean
├── campaign.py        plan the matrix, fly it, resume where it died
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

python3 campaign.py --campaign campaign_01 --base-seed 20260920 --dry-run
python3 campaign.py --campaign campaign_01 --base-seed 20260920 --limit 2
python3 campaign.py --campaign campaign_01 --base-seed 20260920
```

Output lands in `~/PX4-Autopilot/BEE_LAND/controller/logs/campaign_01/`:

```
campaign.json                 base seed, matrix size, source world, start time
runs/severe_r050_gateoff_rep07/
├── world.sdf                 the exact world this run flew
├── run.json                  the resolved scenario
├── bee_outcome_*.json        the verdict
├── bee_{controller,truth,wind}_*.csv
└── logs/{gz,agent,px4,bridge,node,run,teardown}.log
```

Re-running the same command skips finished runs. A campaign that dies at 3 am
costs only the runs it had not reached.

## The matrix

225 runs: calm gate-on (75), severe gate-on (75), severe gate-off (75), over
5 platform radii × 15 repetitions. The calm-refusal re-runs from §6.1 are not
planned in advance — they depend on which calm runs actually refuse.

**The seed excludes the gate flag.** A gate-on and a gate-off run at the same
cell produce a byte-identical world; only `--gate` differs. That pairing is the
ablation, and it is what makes the comparison measure the gate rather than
scenario variation. `campaign.py --dry-run` plus `diff` will demonstrate it.

## What the seed actually draws

| Drawn per run | Fixed by condition |
|---|---|
| Wind amplitudes, frequencies, phases (via the plugin's `<synthesis>`) | Mean wind, max wind, gust statistics |
| One platform phase offset per axis | Wave amplitudes, frequencies, relative phasing |
| Spawn position and yaw jitter | Spawn height |

Two of these fix problems the nominal world has for repeated runs.

**Wind.** The nominal world lists explicit `<component>` blocks, so its
`<seed>` controls nothing — every run gets the same gust. The generator
switches to the plugin's `<synthesis>` path, where the seed draws amplitudes,
frequencies and phases. Fifteen repetitions are now fifteen realisations of the
same statistics rather than the same wind fifteen times.

**Platform phase.** The oscillation plugin has no seed at all, so every run
starts the deck at the same point in its cycle. The generator adds one seeded
offset per axis to every component on that axis — rotating the axis through its
cycle while preserving the relative phasing between components, which is what
gives the deck its spectrum rather than three independent sines.

**Spawn jitter** exists because a fixed spawn makes CENTER solve the same
problem every time, and CENTER convergence is one of the things the campaign
reports.

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
member — the wait would run its full duration on every run, which across 225
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
