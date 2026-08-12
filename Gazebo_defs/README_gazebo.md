# Bee Environment

The simulated world: an oscillating flower deck, a camera-equipped x500, and
two custom Gazebo plugins — one that drives the platform, one that publishes
ground truth.

Nothing here is decoration. The deck texture is a sensor input, the skid
geometry is what the contact truth measures against, and the oscillation
profile is the disturbance the feasibility gate has to survive.

---

## What lives where

```text
Gazebo_defs/
├── worlds/
│   ├── bee_platform.sdf              # the world; both custom plugins wired in
│   ├── platform_motion_examples.sdf  # ready-made oscillation profiles
│   └── materials/                    # flower deck mesh + textures
├── plugins/
│   ├── oscillating_platform_controller/   # drives platform_link
│   └── bee_landing_truth/                 # publishes /bee_land/truth
├── models/bee_x500/                  # x500 + downward camera + skid contact sensors
├── flower_generator.py               # regenerates the deck texture
└── truth_layout.py                   # truth packet field layout (mirrored copy)
```

**`truth_layout.py` exists twice on purpose** — here and at
`Controller_logic/bee_control/diagnostics/truth_layout.py`. The two are byte-
identical field lists (89 fields) and are the shared contract between the C++
plugin and the Python logger. The C++ enum in `BeeLandingTruth.hpp` is the
third copy. **Change one, change all three**, and bump
`TRUTH_SCHEMA_VERSION` — a silent mismatch shifts every column in the truth
CSV without erroring.

---

## Syncing to the PX4 tree

Gazebo will not see any of this until it is linked into PX4's own directories.
The repo is the source of truth; `~/PX4-Autopilot/BEE_LAND/` is a working copy.

Set a variable once so the commands below are copy-pasteable:

```bash
REPO=/mnt/c/path/to/OdometryMaxxing        # adjust to your checkout
BEE=~/PX4-Autopilot/BEE_LAND
```

Windows → Linux (run in the Linux terminal):

```bash
cp -r $REPO/Gazebo_defs/*                 $BEE/
cp -r $REPO/Gazebo_defs/worlds/*          $BEE/worlds/
cp -r $REPO/Controller_logic/bee_control  $BEE/controller/
cp    $REPO/bridge.sh                     $BEE/
```

Note the controller line: `bee_control` is a **package directory** now, not a
handful of loose `.py` files, so `cp Controller_logic/*.py` no longer copies
anything useful.

Pulling logs back, Linux → Windows:

```bash
cp -r $BEE/logs/* $REPO/Flight_Analysis/logs/
```

And the plugins, after editing them on the Linux side (run in the Windows
terminal):

```bash
scp -r ~/PX4-Autopilot/BEE_LAND/plugins/oscillating_platform_controller <repo>\Gazebo_defs\plugins\
scp -r ~/PX4-Autopilot/BEE_LAND/plugins/bee_landing_truth               <repo>\Gazebo_defs\plugins\
```

Then symlink the world and both plugins into the trees Gazebo and PX4 read:

```bash
ln -s $BEE/worlds/bee_platform.sdf  ~/PX4-Autopilot/Tools/simulation/gz/worlds/bee_platform.sdf
ln -s $BEE/worlds/materials         ~/PX4-Autopilot/Tools/simulation/gz/worlds/

ln -s $BEE/plugins/oscillating_platform_controller \
      ~/PX4-Autopilot/src/modules/simulation/gz_plugins/oscillating_platform_controller

ln -s $BEE/plugins/bee_landing_truth \
      ~/PX4-Autopilot/src/modules/simulation/gz_plugins/bee_landing_truth
```

Register both in `src/modules/simulation/gz_plugins/CMakeLists.txt`, below the
other inclusions:

```cmake
add_subdirectory(oscillating_platform_controller)
add_subdirectory(bee_landing_truth)
```

Rebuild, and check both `.so` files exist:

```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/PX4-Autopilot/build/px4_sitl_default/src/modules/simulation/gz_plugins:$GZ_SIM_SYSTEM_PLUGIN_PATH
cd ~/PX4-Autopilot
make px4_sitl

find build/px4_sitl_default -name 'libOscillatingPlatformController.so'
find build/px4_sitl_default -name 'libBeeLandingTruth.so'
```

`bee_platform.sdf` loads both by filename, so a missing `.so` shows up as the
platform sitting still or `/bee_land/truth` never appearing — not as a build
error.

---

## The two custom plugins

### `custom::OscillatingPlatformController` — model plugin

Drives `platform_link` along a prescribed trajectory using a position servo,
so the deck moves *through* the physics rather than being teleported.

### `custom::BeeLandingTruth` — world plugin

Reads drone, camera, platform and contact state, computes geometry and
velocities on **one SIM clock**, and publishes a single fixed-layout
`gz.msgs.Float_V` packet on `/bee_land/truth`.

It is a **world** plugin, not a model plugin — it needs both models. Details,
geometry conventions, and the reasoning behind the `Float_V` transport are in
[`plugins/bee_landing_truth/README.md`](./plugins/bee_landing_truth/README.md).

Two conventions from that file are worth repeating here, because they are the
ones that bite during analysis:

- **Skid distance is the minimum over four physical bottom corners** per skid,
  so roll and pitch are handled. True skid underside sits at `z = -0.227` m in
  `base_link`, not `-0.20`.
- **Normal expansion truth is `c/h`** (camera normal closing rate over camera
  normal distance). The fronto-parallel *flow divergence* expectation is
  `2*c/h`. The controller's `lambda` is the former. Comparing a measured
  `lambda` against the `2*c/h` field is the classic factor-of-two error.

---

## Changing the oscillations

The deck motion is defined inside the plugin block in `bee_platform.sdf`. Each
axis takes **any number of `<component>` entries**, summed — this is how the
platform gets a non-trivial spectrum instead of a single sine the controller
could trivially track:

```xml
<plugin filename="libOscillatingPlatformController.so" name="custom::OscillatingPlatformController">
  <link_name>platform_link</link_name>

  <axis_x>
    <component><amplitude>0.20</amplitude><frequency>0.180</frequency><phase>1.5</phase></component>
    <component><amplitude>0.05</amplitude><frequency>0.320</frequency><phase>0.0</phase></component>
  </axis_x>
  <axis_y>
    <component><amplitude>0.30</amplitude><frequency>0.100</frequency><phase>0.0</phase></component>
    <component><amplitude>0.06</amplitude><frequency>0.310</frequency><phase>2.1</phase></component>
  </axis_y>
  <axis_z>
    <component><amplitude>0.40</amplitude><frequency>0.150</frequency><phase>0.7</phase></component>
    <component><amplitude>0.25</amplitude><frequency>0.310</frequency><phase>4.0</phase></component>
  </axis_z>

  <position_gain>20.0</position_gain>
  <max_linear_velocity>5.0</max_linear_velocity>
  <angular_enabled>false</angular_enabled>
</plugin>
```

Amplitudes are metres, frequencies Hz, phases radians.

| Element | Effect |
|---|---|
| `<component>` | One sine on that axis. Sum as many as you like. |
| `position_gain` | Servo stiffness tracking the prescribed position. Too low and the deck lags its own trajectory, so the *commanded* profile stops being the *actual* one. |
| `max_linear_velocity` | Saturation. If the profile demands more, the deck silently under-travels — check `platform_motion.png` from the analyser before trusting the amplitude you typed. |
| `angular_enabled` | Deck tilt. Off by default; the truth plugin handles roll/pitch geometry either way. |

Ready-made profiles live in `worlds/platform_motion_examples.sdf` — copy a
block across rather than retuning from scratch.

**The probe samples whatever phase it happens to catch.** Peak acceleration,
and with it the feasibility verdict, legitimately varies several-fold between
identical configs. One verdict is one sample, not a property of the profile.

---

## The deck texture is a sensor input

`flower_generator.py` regenerates `flower_top.png` and its normal map. It
exists because the original texture was measured to be a poor optical-flow
target: inside the disc it was dark and narrow-range (grayscale mean ≈ 50,
std ≈ 14), **88.5% of pixels had near-zero gradient**, and its dot pattern had
a single spatial frequency (~30 px ≈ 5.9 cm).

Farneback has nothing to lock onto in a flat region, and a single pitch stops
being resolvable at one specific height — producing a divergence collapse that
looks like a control problem and is not.

The replacement is multi-octave noise, histogram-equalized, spanning ~25 cm
features down to near-texel scale, so some resolvable frequency exists at every
camera distance. If you regenerate it, re-run the texture through the
`_optFlow_targetAcqu_debug` harness (see the controller README) before flying
it — the poor-texture toggle exists precisely to compare.

---

## Launching

Platform alone:

```bash
gz sim Tools/simulation/gz/worlds/bee_platform.sdf
```

Platform with the bee model:

```bash
PX4_SYS_AUTOSTART=4001 \
PX4_SIMULATOR=gz \
PX4_SIM_MODEL=bee_x500 \
PX4_GZ_MODEL_POSE="0,0,2.4,0,0,0" \
PX4_GZ_WORLD=bee_platform \
./build/px4_sitl_default/bin/px4
```

`bee_x500` is an x500 modification carrying the downward camera and the skid
contact sensors. `PX4_GZ_MODEL_POSE` is `(x, y, z, roll, pitch, yaw)`.

Check the truth packet is live before flying:

```bash
gz topic -l | grep /bee_land/truth
gz topic -e -t /bee_land/truth
```

Then start the bridge (`bridge.sh` from the repo root), which carries the
camera, platform pose, touchdown flag, the truth packet, and both skid contact
sensors in one process.

---

## PX4 parameters for SITL

Once the simulation is running:

```bash
param set SYS_HAS_BARO 1
param set SYS_HAS_MAG 1

param set EKF2_BARO_CTRL 1
param set EKF2_HGT_REF 0

param set EKF2_MAG_TYPE 1
param set EKF2_MAG_CHECK 0
param set COM_ARM_MAG_STR 0

param set EKF2_OF_CTRL 0
param set EKF2_RNG_CTRL 0

param set COM_RC_IN_MODE 4
param set COM_RCL_EXCEPT 31
param set NAV_RCL_ACT 0

param set NAV_DLL_ACT 0
param set COM_DLL_EXCEPT 0

param set COM_ARM_WO_GPS 1
param set COM_ARM_ODID 0

param set COM_CPU_MAX -1
param set COM_RAM_MAX -1
param set COM_POWER_COUNT 0
param set CBRK_SUPPLY_CHK 894281
param set CBRK_USB_CHK 197848

param save
```

| Group | Why |
|---|---|
| `SYS_HAS_BARO`, `SYS_HAS_MAG` | Keep simulated baro/mag enabled. |
| `EKF2_*` | Use baro/GPS/mag cleanly; disable optical-flow and range fusion, since those sensors don't exist on this airframe. **The controller never reads EKF height anyway** — it is vision-only. |
| `COM_RC_IN_MODE`, `COM_RCL_EXCEPT`, `NAV_RCL_ACT` | Operate without RC. |
| `NAV_DLL_ACT`, `COM_DLL_EXCEPT` | Operate without QGroundControl. |
| `COM_ARM_*` | Relax GPS/OpenDroneID/mag arming blockers. |
| `COM_CPU_MAX`, `COM_RAM_MAX`, `COM_POWER_COUNT`, `CBRK_*` | Remove SITL/WSL system-health blockers. |

One parameter this project does **not** relax: `COM_OF_LOSS_T`, the offboard-
loss failsafe. The controller's whole timer and clock design exists to keep
the setpoint stream inside it. If you see unexplained failsafes, that is the
symptom to look up in the controller README, not a parameter to widen.