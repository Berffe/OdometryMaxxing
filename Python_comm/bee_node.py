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
from .control_law import ControlLaw
from .px4_interface import PX4Interface
from .diagnostics_writer import DiagnosticsWriter


HEARTBEAT_PERIOD_SEC = 0.1
CONTROL_PERIOD_SEC = 0.5
ARM_AFTER_HEARTBEATS = 10
SHOW_CAMERA = True


class BeeLandNode(Node):
	def __init__(self):
		super().__init__("bee_land_node")

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
		self._latest_setpoint = AttitudeSetpoint()

		self._heartbeat_count = 0
		self._offboard_engaged = False

		self.optical_flow = OpticalFlowEstimator()
		self.target_acquisition = TargetAcquisition()
		self.control_law = ControlLaw()

		self.diagnostics = DiagnosticsWriter(
			output_dir="logs",
			filename=None,
			flush_every_row=True,
		)

		self.get_logger().info(
			f"Diagnostics CSV: {self.diagnostics.filepath}"
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
		self.create_timer(CONTROL_PERIOD_SEC, self.on_control_timer)

		if SHOW_CAMERA : 
			cv2.namedWindow("Bee Land - Camera", cv2.WINDOW_NORMAL)

		self.get_logger().info("bee_land_node started.")
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

		if SHOW_CAMERA : 
			frame = cv2.rotate(src, cv2.ROTATE_180)
			cv2.imshow("Bee Land - Camera", frame)
			cv2.waitKey(1)

		# Same timestamp for target and flow.
		stamp = time.time()

		# Target first, because optical flow now uses the target box as ROI.
		target = self.target_acquisition.update(frame, timestamp=stamp)

		# Flow second, using the detected target ROI.
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

	def on_control_timer(self):
		if self._latest_flow is None or self._latest_frame is None:
			return

		self._latest_setpoint = self.control_law.compute(
			self._latest_target,
			self._latest_flow,
			CONTROL_PERIOD_SEC,
		)
	
		self.diagnostics.write(
			wall_timestamp=time.time(),
			target=self._latest_target,
			flow=self._latest_flow,
			setpoint=self._latest_setpoint,
			vehicle_state=self._vehicle_state,
		)


def main(args=None):
	rclpy.init(args=args)

	node = BeeLandNode()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.diagnostics.close()
		node.destroy_node()

		if SHOW_CAMERA : 
			cv2.destroyAllWindows()
		rclpy.shutdown()


if __name__ == "__main__":
	main()