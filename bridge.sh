#!/usr/bin/env bash
# Current BEE_LAND bridge plus the new atomic Gazebo truth packet.
set -euo pipefail

BEE_WORLD="bee_platform"
BEE_MODEL_INSTANCE="bee_x500_0"

GEAR_LEFT_GZ="/world/${BEE_WORLD}/model/${BEE_MODEL_INSTANCE}/link/bee_gear_contact_link/sensor/landing_gear_contact_sensor_left/contact"
GEAR_RIGHT_GZ="/world/${BEE_WORLD}/model/${BEE_MODEL_INSTANCE}/link/bee_gear_contact_link/sensor/landing_gear_contact_sensor_right/contact"

exec ros2 run ros_gz_bridge parameter_bridge \
  /bee_x500/camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /platform/pose@geometry_msgs/msg/Pose[gz.msgs.Pose \
  /bee_platform/touched@std_msgs/msg/Bool[gz.msgs.Boolean \
  /bee_land/truth@ros_gz_interfaces/msg/Float32Array[gz.msgs.Float_V \
  "${GEAR_LEFT_GZ}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts" \
  "${GEAR_RIGHT_GZ}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts" \
  --ros-args \
  -r "${GEAR_LEFT_GZ}:=/bee/gear_contacts/left" \
  -r "${GEAR_RIGHT_GZ}:=/bee/gear_contacts/right"
