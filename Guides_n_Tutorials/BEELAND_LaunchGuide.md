# BEE_LAND Simulation Launch Guide

This file summarizes the terminals needed to launch the current BEE_LAND simulation stack:

- PX4 SITL + Gazebo world
- `bee_x500` custom drone model
- Micro XRCE-DDS Agent for PX4 ↔ ROS 2 communication
- Gazebo camera bridge to ROS 2
- `bee_node.py` ROS 2 node for PX4 state + camera image access

---
## Commandes à retenir
Pour copier les fichiers de Windows à Linux (dans le terminal Linux!) :

```bash
cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Gazebo_defs/* ~/PX4-Autopilot/BEE_LAND/

cp -r ~/PX4-Autopilot/BEE_LAND/logs/* /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/OpenLoop_Test/logs

cp -r ~/PX4-Autopilot/BEE_LAND/logs/* /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Flight_Analysis/logs

cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Gazebo_defs/worlds/* ~/PX4-Autopilot/BEE_LAND/worlds/

cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Controller_logic/*.py ~/PX4-Autopilot/BEE_LAND/controller/

cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/OpenLoop_Test/*.py ~/PX4-Autopilot/BEE_LAND/openloop/
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
PX4_GZ_MODEL_POSE="0,0,2.4,0,0,0" \
PX4_GZ_WORLD=bee_platform \
./build/px4_sitl_default/bin/px4
```

This launches PX4 SITL directly and spawns the custom `bee_x500` model inside the `bee_platform` Gazebo world.

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

## Terminal 3 — Bridge Gazebo Camera Image to ROS 2

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /bee_x500/camera/image@sensor_msgs/msg/Image@gz.msgs.Image
```

This bridges the Gazebo camera topic:

```text
/bee_x500/camera/image
```

from Gazebo Transport into ROS 2 as:

```text
sensor_msgs/msg/Image
```

This allows regular ROS 2 nodes, `rqt_image_view`, and `cv_bridge` to access the simulated camera image.


---

## Terminal 4 — Run the BEE_LAND ROS Node

```bash
cd ~/PX4-Autopilot/BEE_LAND

ros_bee # Defined in /.bashrc

python3 -m controller.bee_node
python3 -m openloop.calibration_node

```

This node currently:

- subscribes to PX4 local position through ROS 2
- subscribes to the bridged Gazebo camera image
- converts ROS images using `cv_bridge`
- displays the camera feed with OpenCV
- logs basic vehicle position and image information

Expected output includes messages like:

```text
bee_land_node started.
Waiting for PX4 local position on /fmu/out/vehicle_local_position_v1
Waiting for camera images on /bee_x500/camera/image
image #30: 640x480, encoding=rgb8
local position: x=..., y=..., z=...
```
---

## Terminal 5 — Optional Camera and ROS Checks

Check if ROS sees the camera topic:

```bash
ros2 topic list | grep image
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

---

## Current Architecture

```text
Gazebo world
    |
    | publishes camera image
    v
/bee_x500/camera/image  [gz.msgs.Image]
    |
    | ros_gz_bridge
    v
/bee_x500/camera/image  [sensor_msgs/msg/Image]
    |
    | bee_node.py
    v
OpenCV image display / future optical flow processing
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
    | bee_node.py
    v
Vehicle state access / future landing controller
```

---

## Minimal Launch Order

Use this order:

```text
1. Terminal 1: PX4 SITL + Gazebo + bee_x500
2. Terminal 2: MicroXRCEAgent
3. Terminal 3: Gazebo camera bridge
4. Terminal 4: bee_node.py
5. Terminal 5: optional checks / rqt_image_view
```

If something fails, test in this order:

```text
1. gz topic -l | grep camera
2. ros2 topic list | grep image
3. ros2 topic hz /bee_x500/camera/image
4. rqt_image_view
5. python3 bee_node.py
```

---

## Notes

- `bee_x500` is spawned through `PX4_SIM_MODEL=bee_x500`.
- The camera image comes from Gazebo, not directly from PX4.
- PX4 state comes through Micro XRCE-DDS and `px4_msgs`.
- The camera bridge and PX4 DDS bridge are separate communication paths.
- The current `bee_node.py` is a first integration node; later it can host optical flow, platform-motion estimation, and landing-control logic.
