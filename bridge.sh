#!/usr/bin/env bash
# Single ros_gz_bridge invocation for the BEE_LAND stack.
#
# WHY THIS EXISTS: a hand-typed multi-line `ros2 run ros_gz_bridge
# parameter_bridge ...` command has twice lost a line silently -- once the
# gear-contact topics when the platform-pose line was still separate, then the
# platform-pose line itself when the gear-contact remap args were added. A
# missing bridge line produces no error, just a topic with zero publishers that
# looks identical to every other kind of "nothing is arriving" failure -- see
# bee_node's on-startup warnings for /platform/pose and the touchdown topics,
# which exist precisely because this class of failure has no other symptom.
# One script, one place to add a topic, is the fix for the CLASS of bug, not
# just this instance of it.
#
# Adjust BEE_MODEL_INSTANCE below if you ever spawn more than one vehicle
# (PX4_INSTANCE != 0) -- the gear-contact sensor topics are gz-sim's
# auto-generated per-sensor path, which embeds the model instance name and is
# NOT stable across instance indices. See model.sdf's landing_gear_contact_sensor
# comments and bee_node.py's GEAR_CONTACTS_*_TOPIC comment for why an SDF
# <topic> override could not be used instead (confirmed by live testing:
# contact-type sensors in this gz-sim version do not publish real data on an
# SDF <topic> override, so the auto path + a ROS-side remap is the only path
# that has actually been observed to carry data).
set -euo pipefail

BEE_WORLD="bee_platform"
BEE_MODEL_INSTANCE="bee_x500_0"

GEAR_LEFT_GZ="/world/${BEE_WORLD}/model/${BEE_MODEL_INSTANCE}/link/bee_gear_contact_link/sensor/landing_gear_contact_sensor_left/contact"
GEAR_RIGHT_GZ="/world/${BEE_WORLD}/model/${BEE_MODEL_INSTANCE}/link/bee_gear_contact_link/sensor/landing_gear_contact_sensor_right/contact"

exec ros2 run ros_gz_bridge parameter_bridge \
  /bee_x500/camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /platform/pose@geometry_msgs/msg/Pose[gz.msgs.Pose \
  /bee_platform/touched@std_msgs/msg/Bool[gz.msgs.Boolean \
  "${GEAR_LEFT_GZ}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts" \
  "${GEAR_RIGHT_GZ}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts" \
  --ros-args \
  -r "${GEAR_LEFT_GZ}:=/bee/gear_contacts/left" \
  -r "${GEAR_RIGHT_GZ}:=/bee/gear_contacts/right"
