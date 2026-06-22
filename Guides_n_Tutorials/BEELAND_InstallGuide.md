# BEE_LAND — Full Environment Setup Guide

Complete installation reference for the PX4 SITL + Gazebo (oscillating platform) + ROS 2 + MAVSDK environment. Target system: **WSL2, Ubuntu 24.04**. Steps are ordered — later steps assume earlier ones are done.

---

## 1. PX4-Autopilot

Assumes you already have the repo cloned at `~/PX4-Autopilot`. If starting fresh:

```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive ~/PX4-Autopilot
cd ~/PX4-Autopilot
bash ./Tools/setup/ubuntu.sh   # installs PX4's own build dependencies, incl. Gazebo
```

Build once to confirm the base toolchain works before adding anything custom:
```bash
make px4_sitl gz_x500
```
(`Ctrl+C` once you see `Ready for takeoff!` — this is just a sanity check.)

---

## 2. Project structure (`BEE_LAND`)

Custom assets live in a dedicated folder and get **symlinked** into the locations PX4/Gazebo actually scan, keeping your work separable from upstream PX4.

```bash
mkdir -p ~/PX4-Autopilot/BEE_LAND/{worlds,plugins,models,controller}
```

```bash
# World file → must live under Tools/simulation/gz/worlds/
ln -s ~/PX4-Autopilot/BEE_LAND/worlds/bee_platform.sdf \
      ~/PX4-Autopilot/Tools/simulation/gz/worlds/bee_platform.sdf

# Custom Gazebo plugin source → must live under src/modules/simulation/gz_plugins/
ln -s ~/PX4-Autopilot/BEE_LAND/plugins/oscillating_platform_controller \
      ~/PX4-Autopilot/src/modules/simulation/gz_plugins/oscillating_platform_controller

# Custom drone model (bee_x500, with camera/IMU) → must live under Tools/simulation/gz/models/
ln -s ~/PX4-Autopilot/BEE_LAND/models/bee_x500 \
      ~/PX4-Autopilot/Tools/simulation/gz/models/bee_x500
```

---

## 3. Custom Gazebo plugin — `OscillatingPlatformController`

Source lives in `BEE_LAND/plugins/oscillating_platform_controller/` (`.cpp`, `.hpp`, `CMakeLists.txt`).

Wire it into the build by adding one line to the parent CMake file:
```bash
echo "add_subdirectory(oscillating_platform_controller)" \
  >> ~/PX4-Autopilot/src/modules/simulation/gz_plugins/CMakeLists.txt
```

Rebuild PX4 — this compiles the plugin as part of the normal build:
```bash
cd ~/PX4-Autopilot
make px4_sitl
```
Confirm the shared library was produced:
```bash
find build/px4_sitl_default -name 'libOscillatingPlatformController.so'
```

The plugin is referenced from inside `bee_platform.sdf` via a `<plugin filename="OscillatingPlatformController" name="custom::OscillatingPlatformController">` block on the platform model — no `server.config` changes needed, since it's a world-level (not global) plugin.

**Two SDF settings the plugin depends on:**
- `<static>true</static>` on the platform model — required for `SetWorldPoseCmd` to actually move it (a dynamic/non-static body fights the teleport with physics and the platform just falls).
- `<spherical_coordinates>` at the world level — gives Gazebo's synthetic GPS a realistic MSL altitude reference, which EKF2 needs to mark altitude valid.

---

## 4. Gazebo plugin path (env var)

Needed only when launching `gz sim` **directly** (bypassing `make px4_sitl`, which sets this automatically):
```bash
echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH="$HOME/PX4-Autopilot/build/px4_sitl_default/src/modules/simulation/gz_plugins:$GZ_SIM_SYSTEM_PLUGIN_PATH"' \
  >> ~/.bashrc
```

---

## 5. PX4 SITL parameters

These relax hardware-safety checks that don't make sense in a custom SITL world (synthetic GPS/mag, no RC transmitter, no real baro source of truth). Set once inside the PX4 shell (`pxh>`), then `param save` persists them across restarts:

```
param set COM_RCL_EXCEPT 4        # no failsafe on RC/GCS loss in auto/offboard modes
param set COM_RC_IN_MODE 4        # no physical RC transmitter required
param set COM_DL_LOSS_T 5         # 5s grace period before requiring GCS heartbeat
param set COM_ARM_MAG_STR 0       # disable hard arming block on mag strength mismatch
param set EKF2_MAG_CHECK 0        # disable EKF2 magnetometer consistency check
param set SENS_MAG_AUTOCAL 0      # disable in-flight mag auto-calibration
param set EKF2_HGT_REF 0          # use GPS (not baro) as primary height reference
param set EKF2_BARO_CTRL 0        # disable barometer fusion in EKF2
param set EKF2_GPS_CTRL 7         # full GPS fusion (position + velocity + height)
param set EKF2_GPS_CHECK 0        # disable strict GPS quality gates (SITL GPS fails them)
param set EKF2_GPS_V_NOISE 0.5    # noise floor so EKF2 doesn't reject "too-perfect" SITL GPS
param set EKF2_GPS_P_NOISE 0.5
param set SYS_HAS_BARO 0          # tell PX4 no baro is expected (avoids health failure)
param set COM_ARM_WO_GPS 0        # deny arming without GPS (GPS is now the sole height ref)
param save
```

You also need a GCS heartbeat for arming to be allowed at all — either run `MAVProxy`/QGroundControl, or send one manually (see §7).

---

## 6. Python control environment

```bash
python3 -m venv ~/.control_venv --system-site-packages
source ~/.control_venv/bin/activate

pip install mavsdk
pip install opencv-python
pip install "numpy<2"   # required: ROS's cv_bridge binary is built against NumPy 1.x ABI
```

`--system-site-packages` is required so this venv can also see `rclpy` and `cv_bridge` once ROS 2 is installed (§8–9) — those are apt-installed into the system Python, not pip-installable.

---

## 7. (Optional) GCS heartbeat without QGroundControl

Only needed if you don't run a full GCS and want to satisfy PX4's "GCS connected" arming check manually:
```bash
pip install pymavlink
```
A minimal heartbeat sender is enough — see your `controller/` scripts.

---

## 8. ROS 2 Jazzy

Ubuntu 24.04 → ROS 2 **Jazzy Jalisco** (LTS, supported through 2029):

```bash
sudo apt install software-properties-common -y
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-jazzy-ros-base -y   # NOT ros-jazzy-desktop — conflicts with python3-paraview on 24.04
sudo apt install ros-dev-tools -y

echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```

> `ros-jazzy-desktop` currently has an unmet-dependency conflict between `python3-paraview` and `python3-vtk9` on Ubuntu 24.04. `ros-base` avoids it entirely and is all that's needed for a headless rclpy pipeline (no RViz).

---

## 9. Gazebo ↔ ROS 2 bridge

```bash
sudo apt install ros-jazzy-ros-gz-bridge -y
sudo apt install ros-jazzy-cv-bridge -y
```

---

## 10. PX4 ↔ ROS 2 bridge (uXRCE-DDS)

**Micro-XRCE-DDS-Agent** (build from source — use `v2.4.3`, not `v2.4.2`, which fails to clone its pinned Fast-DDS tag):
```bash
git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git ~/Micro-XRCE-DDS-Agent
cd ~/Micro-XRCE-DDS-Agent
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig /usr/local/lib/
```

**`px4_msgs`** (ROS 2 message definitions — branch should match your PX4 firmware version):
```bash
cd ~/PX4-Autopilot && git describe --tags   # check your firmware version first

mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/PX4/px4_msgs.git -b release/1.15   # adjust to match
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

> If your PX4 firmware is well ahead of the `px4_msgs` branch (e.g. running a `main`/alpha build against `release/1.15` messages), some topics will only have live publishers on the version-suffixed name (`/fmu/out/<topic>_v1`) — check with `ros2 topic info <topic> --verbose` and look at `Publisher count` to confirm which name is actually live before assuming.

---

## 11. Display output (for camera viewing via OpenCV)

WSL2 on Windows 11 ships **WSLg** — check first:
```bash
ls /mnt/wslg
```
If present, export the display variables (and persist in `~/.bashrc`):
```bash
export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export PULSE_SERVER=/mnt/wslg/PulseServer
```
If `/mnt/wslg` doesn't exist, update WSL from **Windows PowerShell** (not inside Ubuntu):
```powershell
wsl --update
wsl --shutdown
```
then reopen Ubuntu completely and re-check.

Test independently of the full pipeline:
```bash
python3 -c "
import cv2, numpy as np
img = np.zeros((300,300,3), dtype=np.uint8)
cv2.putText(img, 'Hello WSLg', (40,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.imshow('test', img); cv2.waitKey(3000); cv2.destroyAllWindows()
"
```

---

## 12. Shell environment consolidation (`~/.bashrc`)

```bash
cat >> ~/.bashrc << 'EOF'

# ── BEE_LAND project environment ────────────────────────────────────────────
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HOME/PX4-Autopilot/build/px4_sitl_default/src/modules/simulation/gz_plugins:$GZ_SIM_SYSTEM_PLUGIN_PATH"
export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export PULSE_SERVER=/mnt/wslg/PulseServer

ros_bee() {
    if [ -n "$VIRTUAL_ENV" ]; then deactivate; fi
    source /opt/ros/jazzy/setup.bash
    source "$HOME/ros2_ws/install/setup.bash"
    source "$HOME/.control_venv/bin/activate"
    echo "ROS 2 Jazzy + ros2_ws + .control_venv active."
}

alias killall_sim='sudo kill -9 $(sudo lsof -t -i UDP:14540) 2>/dev/null; pkill -9 -f px4; pkill -9 -f "gz sim"; pkill -9 -f ruby; pkill -9 -f mavsdk_server; echo "Sim processes cleared."'
alias beeland='cd ~/PX4-Autopilot/BEE_LAND'
alias beectl='cd ~/PX4-Autopilot/BEE_LAND/controller'

EOF

source ~/.bashrc
```

---

## Quick verification checklist

| Check | Command |
|---|---|
| PX4 build OK | `make px4_sitl gz_x500` reaches `Ready for takeoff!` |
| Plugin compiled | `find build/px4_sitl_default -name 'libOscillatingPlatformController.so'` |
| ROS 2 sourced | `ros2 --help` |
| px4_msgs built | `ls ~/ros2_ws/install/setup.bash` |
| Micro-XRCE-DDS-Agent installed | `MicroXRCEAgent --version` |
| Python stack OK | `python3 -c "from mavsdk import System; import rclpy; from cv_bridge import CvBridge; print('OK')"` |
| Display OK | the "Hello WSLg" test above shows a window |

If all of these pass, `ros_bee` + the four standard terminals (sim, agent, camera bridge, your node) should bring up the full environment cleanly.