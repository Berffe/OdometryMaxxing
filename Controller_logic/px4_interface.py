"""Pure PX4 command adapter for BEE_LAND."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from px4_msgs.msg import OffboardControlMode, VehicleAttitudeSetpoint, VehicleCommand


@dataclass(frozen=True)
class PublishReceipt:
	sequence: int
	wall_timestamp_sec: float
	monotonic_timestamp_sec: float
	px4_timestamp_us: int


class PX4Interface:
	OFFBOARD_CONTROL_MODE_TOPIC = "/fmu/in/offboard_control_mode"
	VEHICLE_ATTITUDE_SETPOINT_TOPIC = "/fmu/in/vehicle_attitude_setpoint_v1"
	VEHICLE_COMMAND_TOPIC = "/fmu/in/vehicle_command"

	_VEHICLE_CMD_DO_SET_MODE = 176
	_VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
	_PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

	def __init__(self, node, qos_profile, time_manager=None):
		self._time = time_manager
		self._sequence = 0
		self.offboard_control_mode_pub = node.create_publisher(
			OffboardControlMode, self.OFFBOARD_CONTROL_MODE_TOPIC, qos_profile)
		self.attitude_setpoint_pub = node.create_publisher(
			VehicleAttitudeSetpoint, self.VEHICLE_ATTITUDE_SETPOINT_TOPIC, qos_profile)
		self.vehicle_command_pub = node.create_publisher(
			VehicleCommand, self.VEHICLE_COMMAND_TOPIC, qos_profile)

	def publish_cycle(self, setpoint) -> PublishReceipt:
		wall = self._wall_sec()
		mono = time.monotonic()
		stamp_us = int(wall * 1_000_000)
		self.publish_heartbeat(stamp_us)
		self.publish_attitude_setpoint(
			setpoint.roll, setpoint.pitch, setpoint.yaw, setpoint.thrust, stamp_us)
		receipt = PublishReceipt(self._sequence, wall, mono, stamp_us)
		self._sequence += 1
		return receipt

	def publish_heartbeat(self, timestamp_us=None):
		msg = OffboardControlMode()
		msg.timestamp = self._resolve_timestamp_us(timestamp_us)
		msg.position = False
		msg.velocity = False
		msg.acceleration = False
		msg.attitude = True
		msg.body_rate = False
		self.offboard_control_mode_pub.publish(msg)

	def publish_attitude_setpoint(self, roll, pitch, yaw, thrust, timestamp_us=None):
		msg = VehicleAttitudeSetpoint()
		msg.timestamp = self._resolve_timestamp_us(timestamp_us)
		msg.q_d = self._euler_to_quaternion(roll, pitch, yaw)
		msg.thrust_body = [0.0, 0.0, -float(thrust)]
		self.attitude_setpoint_pub.publish(msg)

	def arm(self, timestamp_us=None):
		self._publish_vehicle_command(
			self._VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0,
			timestamp_us=timestamp_us)

	def engage_offboard_mode(self, timestamp_us=None):
		self._publish_vehicle_command(
			self._VEHICLE_CMD_DO_SET_MODE, param1=1.0,
			param2=float(self._PX4_CUSTOM_MAIN_MODE_OFFBOARD),
			timestamp_us=timestamp_us)

	def _publish_vehicle_command(self, command, param1=0.0, param2=0.0, timestamp_us=None):
		msg = VehicleCommand()
		msg.timestamp = self._resolve_timestamp_us(timestamp_us)
		msg.param1 = float(param1)
		msg.param2 = float(param2)
		msg.command = int(command)
		msg.target_system = 1
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		self.vehicle_command_pub.publish(msg)

	def _wall_sec(self):
		return float(self._time.wall_sec()) if self._time is not None else time.time()

	def _resolve_timestamp_us(self, timestamp_us):
		return int(timestamp_us) if timestamp_us is not None else int(self._wall_sec() * 1_000_000)

	@staticmethod
	def _euler_to_quaternion(roll, pitch, yaw):
		cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
		cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
		cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
		return [
			sr * cp * cy - cr * sp * sy,
			cr * sp * cy + sr * cp * sy,
			cr * cp * sy - sr * sp * cy,
			cr * cp * cy + sr * sp * sy,
		]
