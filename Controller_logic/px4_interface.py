"""
Everything that talks to PX4 over the uXRCE-DDS bridge: the
OffboardControlMode heartbeat, VehicleAttitudeSetpoint publishing, and
the arm / switch-to-offboard command handshake.

This class has no control logic of its own -- it only knows how to turn
(roll, pitch, yaw, thrust) into the right PX4 messages. The decision of
*when* to call these methods (heartbeat rate vs control rate, when to
arm) lives in bee_node.py.

Timestamps: every outgoing message is stamped on the WALL clock supplied
by a TimeManager (see clock.py), NOT node.get_clock(). The uXRCE-DDS
agent's timesync references the wall clock, so a wall stamp keeps the
bridge's PX4<->companion sync valid regardless of use_sim_time -- the
previous node.get_clock().now() path silently moved to sim time whenever
use_sim_time was true and desynced the offboard stream. The publish
methods also accept an explicit timestamp_us so the caller can sample one
instant per cycle and share it across the heartbeat+setpoint pair.
"""

import math
import time

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

	# PX4 uXRCE-DDS INPUT topic names. These are VERSIONED INDEPENDENTLY per
	# message on recent PX4 builds -- do not assume they share a suffix. On this
	# build (confirmed with `ros2 topic list | grep fmu/in`):
	#   - offboard_control_mode   : UNVERSIONED  (nav_state=OFFBOARD proved it works)
	#   - vehicle_attitude_setpoint: _v1         (PX4 subscribes ONLY to _v1; the
	#         unversioned name had NO subscriber, so every attitude/thrust setpoint
	#         published there was silently dropped -> vehicle held level and sank
	#         while armed+offboard. This mismatch was THE 'commands ignored' bug.)
	#   - vehicle_command          : UNVERSIONED (arm/mode switches worked)
	# If a future PX4 bumps a version, re-check with `ros2 topic list | grep fmu/in`
	# and update the matching constant here.
	OFFBOARD_CONTROL_MODE_TOPIC = '/fmu/in/offboard_control_mode'
	VEHICLE_ATTITUDE_SETPOINT_TOPIC = '/fmu/in/vehicle_attitude_setpoint_v1'
	VEHICLE_COMMAND_TOPIC = '/fmu/in/vehicle_command'

	def __init__(self, node, qos_profile, time_manager=None):
		self._node = node
		# TimeManager is the single source of the WALL clock used for PX4
		# stamps. Kept optional so this class can still be constructed in
		# isolation; the fallback is the same epoch clock, just not shared.
		self._time = time_manager

		self.offboard_control_mode_pub = node.create_publisher(
			OffboardControlMode, self.OFFBOARD_CONTROL_MODE_TOPIC, qos_profile)
		self.attitude_setpoint_pub = node.create_publisher(
			VehicleAttitudeSetpoint, self.VEHICLE_ATTITUDE_SETPOINT_TOPIC, qos_profile)
		self.vehicle_command_pub = node.create_publisher(
			VehicleCommand, self.VEHICLE_COMMAND_TOPIC, qos_profile)

	def publish_heartbeat(self, timestamp_us: int = None):
		"""
		Tells PX4 "I intend to control attitude+thrust in offboard mode".
		Call this on every tick of the fast heartbeat timer, regardless
		of whether a new setpoint was just computed -- PX4 drops out of
		offboard mode if this stream stops for ~500ms.

		timestamp_us: WALL-clock stamp in microseconds. Pass the value you
		also give publish_attitude_setpoint() this cycle so the pair shares
		one instant. If None, one is sampled here.
		"""
		msg = OffboardControlMode()
		msg.timestamp = self._resolve_timestamp_us(timestamp_us)
		msg.position = False
		msg.velocity = False
		msg.acceleration = False
		msg.attitude = True
		msg.body_rate = False
		self.offboard_control_mode_pub.publish(msg)

	def publish_attitude_setpoint(
		self,
		roll: float,
		pitch: float,
		yaw: float,
		thrust: float,
		timestamp_us: int = None,
	):
		"""
		roll, pitch, yaw: desired attitude [rad]
		thrust: normalized collective thrust in [0, 1]
		timestamp_us: WALL-clock stamp in microseconds; share it with the
		    heartbeat published the same cycle (see publish_heartbeat).

		thrust_body = [0, 0, -thrust] because PX4 setpoints use the body
		FRD frame (Z down), so upward thrust is along -Z body.
		"""
		msg = VehicleAttitudeSetpoint()
		msg.timestamp = self._resolve_timestamp_us(timestamp_us)
		msg.q_d = self._euler_to_quaternion(roll, pitch, yaw)
		msg.thrust_body = [0.0, 0.0, -float(thrust)]
		self.attitude_setpoint_pub.publish(msg)

	def arm(self, timestamp_us: int = None):
		self._publish_vehicle_command(
			self._VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0,
			timestamp_us=timestamp_us)

	def engage_offboard_mode(self, timestamp_us: int = None):
		self._publish_vehicle_command(
			self._VEHICLE_CMD_DO_SET_MODE,
			param1=1.0,
			param2=float(self._PX4_CUSTOM_MAIN_MODE_OFFBOARD),
			timestamp_us=timestamp_us,
		)

	def _publish_vehicle_command(
		self,
		command: int,
		param1: float = 0.0,
		param2: float = 0.0,
		timestamp_us: int = None,
	):
		msg = VehicleCommand()
		msg.timestamp = self._resolve_timestamp_us(timestamp_us)
		msg.param1 = param1
		msg.param2 = param2
		msg.command = command
		msg.target_system = 1
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		self.vehicle_command_pub.publish(msg)

	def _resolve_timestamp_us(self, timestamp_us) -> int:
		"""Use the caller's stamp if given, else sample the WALL clock once.

		Routes through TimeManager when available so the whole node shares one
		definition of the PX4 wall clock; falls back to time.time() otherwise.
		"""
		if timestamp_us is not None:
			return int(timestamp_us)
		if self._time is not None:
			return self._time.px4_tx_timestamp_us()
		return int(time.time() * 1e6)

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