# BEE_LAND Simulation Launch Guide

This file summarizes the terminals needed to launch the current BEE_LAND simulation stack:

- PX4 SITL + Gazebo world
- `bee_x500` custom drone model
- Micro XRCE-DDS Agent for PX4 ↔ ROS 2 communication
- One Gazebo → ROS 2 bridge process for camera, platform pose, and touchdown
- `bee_node.py` ROS 2 node for PX4 state, camera image, platform pose, and touchdown handling

---

## Commands to remember

To copy files from Windows to Linux (in the Linux terminal):

```bash
cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Gazebo_defs/* ~/PX4-Autopilot/BEE_LAND/

cp -r ~/PX4-Autopilot/BEE_LAND/logs/* /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/OpenLoop_Test/logs

cp -r ~/PX4-Autopilot/BEE_LAND/logs/* /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Flight_Analysis/logs

cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Gazebo_defs/worlds/* ~/PX4-Autopilot/BEE_LAND/worlds/

cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Gazebo_defs/models/* ~/PX4-Autopilot/BEE_LAND/models/

cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Controller_logic/*.py ~/PX4-Autopilot/BEE_LAND/controller/

cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/OpenLoop_Test/*.py ~/PX4-Autopilot/BEE_LAND/openloop/
```

```bash
cp -r ~/PX4-Autopilot/BEE_LAND/logs/* /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Flight_Analysis/logs_new/
```

Sometimes the environment needs to be rebuilt:

```bash
cd ~/PX4-Autopilot

make px4_sitl gz_x500 # or
make px4_sitl_default
```

---

## Terminal 1 — Launch PX4 SITL + Gazebo with `bee_x500`

```bash
unset GZ_CONFIG_PATH
unset GZ_VERSION
export PATH=/usr/bin:/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:$PATH
hash -r

cd ~/PX4-Autopilot

source build/px4_sitl_default/rootfs/gz_env.sh

PX4_SYS_AUTOSTART=4001 \
PX4_SIMULATOR=gz \
PX4_SIM_MODEL=bee_x500 \
PX4_GZ_MODEL_POSE="1.5,1.5,0.4,0,0,0" \
PX4_GZ_WORLD=bee_platform \
./build/px4_sitl_default/bin/px4
```

This launches PX4 SITL and spawns the custom `bee_x500` model inside the `bee_platform` Gazebo world.

Important elements:

```text
PX4_SYS_AUTOSTART=4001       x500 quadrotor airframe
PX4_SIMULATOR=gz             selects Gazebo as the simulator
PX4_SIM_MODEL=bee_x500       custom drone model with camera
PX4_GZ_WORLD=bee_platform    custom oscillating platform world
PX4_GZ_MODEL_POSE=...        initial drone position and attitude
```

---

## Terminal 2 — Start Micro XRCE-DDS Agent

```bash
MicroXRCEAgent udp4 -p 8888
```

This creates the communication bridge between PX4 uORB topics and ROS 2 DDS topics.

PX4 publishes topics such as:

```text
/fmu/out/vehicle_local_position_v1
/fmu/out/vehicle_attitude
/fmu/out/vehicle_status
```

Your ROS 2 nodes can subscribe to these topics through `px4_msgs`.

---

## Terminal 3 — Start all Gazebo → ROS 2 bridges

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /bee_x500/camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /platform/pose@geometry_msgs/msg/Pose[gz.msgs.Pose \
  /bee_platform/touched@std_msgs/msg/Bool[gz.msgs.Boolean
```

This single persistent process bridges the three Gazebo topics needed by the controller:

```text
/bee_x500/camera/image      -> sensor_msgs/msg/Image
/platform/pose              -> geometry_msgs/msg/Pose
/bee_platform/touched       -> std_msgs/msg/Bool
```

Direction is Gazebo → ROS 2 only. The `[` syntax is intentional.

Use this bridge for:

- camera input to `bee_node.py`
- platform pose diagnostics and relative-motion logging
- touchdown detection after platform contact

Before starting the bridge, Gazebo must already be publishing the topics. Check if needed:

```bash
gz topic -l | grep -E "camera|platform|touched"
gz topic -e -t /platform/pose
gz topic -e -t /bee_platform/touched
```

Notes:

- `/platform/pose` is published directly by `OscillatingPlatformController`.
- `/bee_platform/touched` requires the contact sensor + `TouchPlugin` in `bee_platform.sdf`.
- If either topic is missing, rebuild/reinstall the relevant Gazebo plugin or check that the updated world file is the one being loaded.

---

## Terminal 4 — Run the BEE_LAND ROS node

```bash
cd ~/PX4-Autopilot/BEE_LAND

ros_bee # Defined in ~/.bashrc

python3 -m controller.bee_node
```

For open-loop calibration instead:

```bash
python3 -m openloop.calibration_node
```

The node currently:

- subscribes to PX4 local position through ROS 2
- subscribes to the bridged Gazebo camera image
- subscribes to the bridged platform pose
- subscribes to the bridged touchdown signal
- converts ROS images using `cv_bridge`
- displays the camera feed with OpenCV
- computes target detection, optical flow, and visual control commands
- logs diagnostics to CSV
- enters landed mode and requests motor stop after confirmed touchdown

Expected output includes messages like:

```text
bee_land_node started.
Waiting for required streams: local_position and camera.
First platform pose received on /platform/pose: ...
Touchdown detection enabled: listening on /bee_platform/touched ...
```

---

## Terminal 5 — Optional checks

Check if ROS sees the camera topic:

```bash
ros_bee
ros2 topic list | grep camera
```

Expected:

```text
/bee_x500/camera/image
```

Check camera publishing rate:

```bash
ros2 topic hz /bee_x500/camera/image
```

Inspect topic information and QoS:

```bash
ros2 topic info --verbose /bee_x500/camera/image
```

Optional visual check:

```bash
rqt_image_view
```

Then select:

```text
/bee_x500/camera/image
```

Check if ROS sees the platform pose topic:

```bash
ros2 topic list | grep platform
ros2 topic info /platform/pose -v
ros2 topic hz /platform/pose
```

Expected:

```text
Publisher count: >= 1
```

and a steady rate from `ros2 topic hz`.

Check touchdown:

```bash
ros2 topic echo /bee_platform/touched
ros2 topic echo /bee_land/touchdown
```

Expected after contact:

```text
data: true
```

If `platform_*` / `relative_*` diagnostics columns are empty, check:

```bash
gz topic -l | grep platform
gz topic -e -t /platform/pose
ros2 topic list | grep platform
ros2 topic hz /platform/pose
```

If touchdown is not detected, check:

```bash
gz topic -l | grep touched
gz topic -e -t /bee_platform/touched
ros2 topic list | grep touched
ros2 topic echo /bee_platform/touched
```

---

## Current Architecture

```text
Gazebo camera
    |
    v
/bee_x500/camera/image  [gz.msgs.Image]
    |
    | ros_gz_bridge, Terminal 3
    v
/bee_x500/camera/image  [sensor_msgs/msg/Image]
    |
    v
bee_node.py
```

```text
PX4 SITL
    |
    | uXRCE-DDS client
    v
MicroXRCEAgent udp4 -p 8888
    |
    | ROS 2 DDS
    v
/fmu/out/... topics
    |
    v
bee_node.py
```

```text
OscillatingPlatformController
    |
    v
/platform/pose  [gz.msgs.Pose]
    |
    | ros_gz_bridge, Terminal 3
    v
/platform/pose  [geometry_msgs/msg/Pose]
    |
    v
bee_node.py / calibration_node.py
    |
    v
PlatformState diagnostics only
```

```text
Platform contact sensor + TouchPlugin
    |
    v
/bee_platform/touched  [gz.msgs.Boolean]
    |
    | ros_gz_bridge, Terminal 3
    v
/bee_platform/touched  [std_msgs/msg/Bool]
    |
    v
bee_node.py
    |
    v
PHASE_LANDED + /bee_land/touchdown + MAVSDK disarm/kill fallback
```

---

## Minimal Launch Order

Use this order:

```text
1. Terminal 1: PX4 SITL + Gazebo + bee_x500
2. Terminal 2: MicroXRCEAgent
3. Terminal 3: one Gazebo → ROS 2 bridge command
4. Terminal 4: bee_node.py
5. Terminal 5: optional checks / rqt_image_view
```

If something fails, test in this order:

```text
1. gz topic -l | grep camera
2. gz topic -l | grep platform
3. gz topic -l | grep touched
4. ros2 topic list | grep image
5. ros2 topic list | grep platform
6. ros2 topic list | grep touched
7. ros2 topic hz /bee_x500/camera/image
8. ros2 topic hz /platform/pose
9. ros2 topic echo /bee_platform/touched
10. python3 -m controller.bee_node
```

---

## Notes

- `bee_x500` is spawned through `PX4_SIM_MODEL=bee_x500`.
- The camera image comes from Gazebo, not directly from PX4.
- PX4 state comes through Micro XRCE-DDS and `px4_msgs`.
- Camera, platform pose, and touchdown now share one `ros_gz_bridge` process.
- The PX4 DDS bridge is still separate and requires `MicroXRCEAgent`.
- `control_law.py` remains visual-only; touchdown is handled in `bee_node.py` as a terminal mission event.