"""
Everything that talks to PX4 over the uXRCE-DDS bridge: the
OffboardControlMode heartbeat, VehicleAttitudeSetpoint publishing, and
the arm / switch-to-offboard command handshake.

This class has no control logic of its own — it only knows how to turn
(roll, pitch, yaw, thrust) into the right PX4 messages. The decision of
*when* to call these methods (heartbeat rate vs control rate, when to
arm) lives in bee_node.py.
"""

import math

from px4_msgs.msg import (
	OffboardControlMode,
	VehicleAttitudeSetpoint,
	VehicleCommand,
)


class PX4Interface:

	# px4_msgs VehicleCommand command IDs / PX4 custom mode values.
	# Verify these against your px4_msgs version with:
	#   ros2 interface show px4_msgs/msg/VehicleCommand
	_VEHICLE_CMD_DO_SET_MODE = 176
	_VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
	_PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

	def __init__(self, node, qos_profile):
		self._node = node

		self.offboard_control_mode_pub = node.create_publisher(
			OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
		self.attitude_setpoint_pub = node.create_publisher(
			VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)
		self.vehicle_command_pub = node.create_publisher(
			VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

	def publish_heartbeat(self):
		"""
		Tells PX4 "I intend to control attitude+thrust in offboard mode".
		Call this on every tick of the fast heartbeat timer, regardless
		of whether a new setpoint was just computed — PX4 drops out of
		offboard mode if this stream stops for ~500ms.
		"""
		msg = OffboardControlMode()
		msg.timestamp = self._timestamp_us()
		msg.position = False
		msg.velocity = False
		msg.acceleration = False
		msg.attitude = True
		msg.body_rate = False
		self.offboard_control_mode_pub.publish(msg)

	def publish_attitude_setpoint(self, roll: float, pitch: float, yaw: float, thrust: float):
		"""
		roll, pitch, yaw: desired attitude [rad]
		thrust: normalized collective thrust in [0, 1]

		thrust_body = [0, 0, -thrust] because PX4 setpoints use the body
		FRD frame (Z down), so upward thrust is along -Z body.
		"""
		msg = VehicleAttitudeSetpoint()
		msg.timestamp = self._timestamp_us()
		msg.q_d = self._euler_to_quaternion(roll, pitch, yaw)
		msg.thrust_body = [0.0, 0.0, -float(thrust)]
		self.attitude_setpoint_pub.publish(msg)

	def arm(self):
		self._publish_vehicle_command(
			self._VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

	def engage_offboard_mode(self):
		self._publish_vehicle_command(
			self._VEHICLE_CMD_DO_SET_MODE,
			param1=1.0,
			param2=float(self._PX4_CUSTOM_MAIN_MODE_OFFBOARD),
		)

	def _publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0):
		msg = VehicleCommand()
		msg.timestamp = self._timestamp_us()
		msg.param1 = param1
		msg.param2 = param2
		msg.command = command
		msg.target_system = 1
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		self.vehicle_command_pub.publish(msg)

	def _timestamp_us(self) -> int:
		return int(self._node.get_clock().now().nanoseconds / 1000)

	@staticmethod
	def _euler_to_quaternion(roll: float, pitch: float, yaw: float):
		cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
		cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
		cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

		qw = cr * cp * cy + sr * sp * sy
		qx = sr * cp * cy - cr * sp * sy
		qy = cr * sp * cy + sr * cp * sy
		qz = cr * cp * sy - sr * sp * cy
		return [qw, qx, qy, qz]
