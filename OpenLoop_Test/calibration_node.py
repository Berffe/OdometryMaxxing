"""
Open-loop calibration node.

This is a sibling to bee_node.py, not a replacement: same camera and
PX4 plumbing, same vision pipeline, same diagnostics CSV — but
ControlLaw is not used. Instead, a StepSequence (calibration_sequence.py)
drives roll, pitch, and thrust through a known step train, one axis at
a time, while the other axes are held at trim. That's what makes the
resulting log usable for identifying control_law.py's per-axis discrete
model (e[k+1] = a*e[k] + b*u[k]): the command has to be independent of
the measured state for the regression to be valid, and it isn't, if the
closed-loop controller is the thing producing it.

How to use this:

	1. In Gazebo, position the vehicle at the altitude/area_fraction
	   operating point you want to identify (this node does not manage
	   altitude — see the safety note below on why).
	2. Run this node instead of bee_node.py for the duration of the test
	   sequence (TEST_SETTLE_SEC + 3 * TEST_REPEATS * 4 * TEST_HOLD_SEC
	   seconds with the defaults below; printed at startup).
	3. Stop the node once it logs that the sequence is finished, and
	   feed the resulting CSV to fit_axis_models.py.
	4. Reposition to the next operating point and repeat.

Safety note: the thrust step train commands real deviations from
hover_thrust, so the vehicle will actually climb/descend during that
part of the sequence — make sure there's enough clearance below (and
above) wherever you run it in Gazebo. Roll/pitch steps stay small
(TEST_*_AMPLITUDE_RAD, well inside ROLL_LIMIT_RAD/PITCH_LIMIT_RAD by
default) and shouldn't move the vehicle far, but they're not zero
either.

Run:

	ros2 run bee_control calibration_node

or directly:

	python -m bee_control.calibration_node
"""

import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (
	QoSProfile,
	QoSReliabilityPolicy,
	QoSDurabilityPolicy,
	QoSHistoryPolicy,
)

from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleLocalPosition
from cv_bridge import CvBridge, CvBridgeError

from .state import VehicleState, AttitudeSetpoint, TargetEstimate
from .optical_flow import OpticalFlowEstimator
from .target_acquisition import TargetAcquisition
from .px4_interface import PX4Interface
from .diagnostics_writer import DiagnosticsWriter
from .calibration_sequence import build_calibration_sequence


HEARTBEAT_PERIOD_SEC = 0.1
TEST_PERIOD_SEC = 0.5  # must match the dt control_law.py's local models assume
ARM_AFTER_HEARTBEATS = 10
SHOW_CAMERA = True

# Vehicle trim. Keep this matched to ControlLaw's hover_thrust — the
# thrust step train is defined as a deviation from it.
HOVER_THRUST = 0.49

# Step-train shape. Keep the roll/pitch amplitudes comfortably inside
# ControlLaw's roll_limit/pitch_limit, and the thrust amplitude
# comfortably inside thrust_min/thrust_max, so the identified model
# stays valid over the range the real controller will actually command.
ROLL_TEST_AMPLITUDE_RAD = 0.04
PITCH_TEST_AMPLITUDE_RAD = 0.04
THRUST_TEST_AMPLITUDE = 0.05
TEST_HOLD_SEC = 2.0
TEST_REPEATS = 3
TEST_SETTLE_SEC = 2.0
TEST_AXES = ("roll", "pitch", "thrust")

# Defensive clamps applied to whatever the sequence produces, mirroring
# ControlLaw's own limits, in case the amplitudes above are ever set
# too large by mistake.
ROLL_LIMIT_RAD = 0.10
PITCH_LIMIT_RAD = 0.10
THRUST_MIN = 0.35
THRUST_MAX = 0.65


class CalibrationNode(Node):
	def __init__(self):
		super().__init__("bee_calibration_node")

		self.bridge = CvBridge()

		self._last_position_log_time = 0.0
		self._position_log_period_sec = 1.0

		self._image_count = 0
		self._last_image_log_time = 0.0
		self._image_log_period_sec = 1.0

		self._vehicle_state = VehicleState()

		self._latest_flow = None
		self._latest_frame = None
		self._latest_target = TargetEstimate()
		self._latest_setpoint = AttitudeSetpoint(
			roll=0.0, pitch=0.0, yaw=0.0, thrust=HOVER_THRUST
		)

		self._heartbeat_count = 0
		self._offboard_engaged = False

		self._test_start_time = None
		self._sequence_finished_logged = False

		self.optical_flow = OpticalFlowEstimator()
		self.target_acquisition = TargetAcquisition()

		self.sequence = build_calibration_sequence(
			hover_thrust=HOVER_THRUST,
			roll_amplitude=ROLL_TEST_AMPLITUDE_RAD,
			pitch_amplitude=PITCH_TEST_AMPLITUDE_RAD,
			thrust_amplitude=THRUST_TEST_AMPLITUDE,
			hold_sec=TEST_HOLD_SEC,
			repeats=TEST_REPEATS,
			settle_sec=TEST_SETTLE_SEC,
			axes=TEST_AXES,
		)

		date_str = time.strftime("%Y%m%d_%H%M%S")
		self.diagnostics = DiagnosticsWriter(
			output_dir="logs",
			filename=f"calibration_{date_str}.csv",
			flush_every_row=True,
		)

		self.get_logger().info(
			f"Calibration diagnostics CSV: {self.diagnostics.filepath}"
		)
		self.get_logger().info(
			f"Test sequence duration: {self.sequence.total_duration:.1f} s "
			f"(axes: {TEST_AXES})"
		)

		px4_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)

		camera_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.VOLATILE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=5,
		)

		self.create_subscription(
			VehicleLocalPosition,
			"/fmu/out/vehicle_local_position_v1",
			self.on_local_position,
			px4_qos,
		)

		self.create_subscription(
			Image,
			"/bee_x500/camera/image",
			self.on_camera,
			camera_qos,
		)

		self.px4 = PX4Interface(self, px4_qos)

		self.create_timer(HEARTBEAT_PERIOD_SEC, self.on_heartbeat_timer)
		self.create_timer(TEST_PERIOD_SEC, self.on_test_timer)

		if SHOW_CAMERA:
			cv2.namedWindow("Bee Calibration - Camera", cv2.WINDOW_NORMAL)

		self.get_logger().info("bee_calibration_node started.")
		self.get_logger().info("Waiting for PX4 local position on /fmu/out/vehicle_local_position_v1")
		self.get_logger().info("Waiting for camera images on /bee_x500/camera/image")

	def on_camera(self, msg: Image):
		self._image_count += 1

		now = time.time()

		if now - self._last_image_log_time >= self._image_log_period_sec:
			self._last_image_log_time = now

			self.get_logger().info(
				f"image #{self._image_count}: "
				f"{msg.width}x{msg.height}, encoding={msg.encoding}"
			)

		try:
			src = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		except CvBridgeError as exc:
			self.get_logger().error(f"cv_bridge conversion failed: {exc}")
			return

		if SHOW_CAMERA:
			frame = cv2.rotate(src, cv2.ROTATE_180)
			cv2.imshow("Bee Calibration - Camera", frame)
			cv2.waitKey(1)

		stamp = time.time()

		# Vision pipeline still runs as normal: this is what we're
		# measuring the response of. Only the control law is skipped.
		target = self.target_acquisition.update(frame, timestamp=stamp)
		flow = self.optical_flow.update(frame, stamp, target=target)

		self._latest_frame = frame
		self._latest_target = target
		self._latest_flow = flow

	def on_local_position(self, msg: VehicleLocalPosition):
		now = time.time()

		if now - self._last_position_log_time >= self._position_log_period_sec:
			self._last_position_log_time = now

			self.get_logger().info(
				f"local position: "
				f"x={msg.x:.2f} m, y={msg.y:.2f} m, z={msg.z:.2f} m"
			)

		self._vehicle_state = VehicleState(
			timestamp=now,
			x=msg.x,
			y=msg.y,
			z=msg.z,
			vx=msg.vx,
			vy=msg.vy,
			vz=msg.vz,
			yaw=msg.heading,
		)

	def on_heartbeat_timer(self):
		self.px4.publish_heartbeat()

		self.px4.publish_attitude_setpoint(
			self._latest_setpoint.roll,
			self._latest_setpoint.pitch,
			self._latest_setpoint.yaw,
			self._latest_setpoint.thrust,
		)

		self._heartbeat_count += 1

		if not self._offboard_engaged and self._heartbeat_count == ARM_AFTER_HEARTBEATS:
			self.px4.arm()
			self.px4.engage_offboard_mode()
			self._offboard_engaged = True

	def on_test_timer(self):
		if self._latest_flow is None or self._latest_frame is None:
			return

		if self._test_start_time is None:
			self._test_start_time = time.time()
			self.get_logger().info("First camera frame received, starting test sequence.")

		elapsed = time.time() - self._test_start_time

		roll, pitch, thrust = self.sequence.command_at(elapsed)

		roll = self._clamp(roll, -ROLL_LIMIT_RAD, ROLL_LIMIT_RAD)
		pitch = self._clamp(pitch, -PITCH_LIMIT_RAD, PITCH_LIMIT_RAD)
		thrust = self._clamp(thrust, THRUST_MIN, THRUST_MAX)

		self._latest_setpoint = AttitudeSetpoint(
			timestamp=self._latest_target.timestamp,
			roll=roll,
			pitch=pitch,
			yaw=0.0,
			thrust=thrust,
		)

		self.diagnostics.write(
			wall_timestamp=time.time(),
			target=self._latest_target,
			flow=self._latest_flow,
			setpoint=self._latest_setpoint,
			vehicle_state=self._vehicle_state,
		)

		if self.sequence.is_finished(elapsed) and not self._sequence_finished_logged:
			self._sequence_finished_logged = True
			self.get_logger().info(
				"Test sequence finished; holding at trim. "
				"Stop the node and run fit_axis_models.py on the CSV above."
			)

	@staticmethod
	def _clamp(value: float, lower: float, upper: float) -> float:
		return max(lower, min(upper, float(value)))


def main(args=None):
	rclpy.init(args=args)

	node = CalibrationNode()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.diagnostics.close()
		node.destroy_node()

		if SHOW_CAMERA:
			cv2.destroyAllWindows()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
