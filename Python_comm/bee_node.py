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


class BeeLandNode(Node):
	def __init__(self):
		super().__init__('bee_land_node')

		self.bridge = CvBridge()

		self._last_position_log_time = 0.0
		self._position_log_period_sec = 1.0

		self._image_count = 0
		self._last_image_log_time = 0.0
		self._image_log_period_sec = 1.0

		# PX4 uXRCE-DDS QoS
		px4_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)

		# Gazebo camera through ros_gz_bridge.
		# BEST_EFFORT avoids silent incompatibility with camera streams.
		camera_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.VOLATILE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=5,
		)

		self.create_subscription(
			VehicleLocalPosition,
			'/fmu/out/vehicle_local_position_v1',
			self.on_local_position,
			px4_qos,
		)

		self.create_subscription(
			Image,
			'/bee_x500/camera/image',
			self.on_camera,
			camera_qos,
		)

		cv2.namedWindow('Bee Land - Camera', cv2.WINDOW_NORMAL)

		self.get_logger().info('bee_land_node started.')
		self.get_logger().info('Waiting for PX4 local position on /fmu/out/vehicle_local_position_v1')
		self.get_logger().info('Waiting for camera images on /bee_x500/camera/image')

	def on_camera(self, msg: Image):
		self._image_count += 1

		now = time.time()
		if now - self._last_image_log_time >= self._image_log_period_sec:
			self._last_image_log_time = now
			self.get_logger().info(
				f'image #{self._image_count}: '
				f'{msg.width}x{msg.height}, encoding={msg.encoding}'
			)

		try:
			frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
		except CvBridgeError as exc:
			self.get_logger().error(f'cv_bridge conversion failed: {exc}')
			return

		cv2.imshow('Bee Land - Camera', frame)
		cv2.waitKey(1)

	def on_local_position(self, msg: VehicleLocalPosition):
		now = time.time()

		if now - self._last_position_log_time >= self._position_log_period_sec:
			self._last_position_log_time = now

			self.get_logger().info(
				f'local position: '
				f'x={msg.x:.2f} m, y={msg.y:.2f} m, z={msg.z:.2f} m'
			)


def main(args=None):
	rclpy.init(args=args)

	node = BeeLandNode()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		cv2.destroyAllWindows()
		rclpy.shutdown()


if __name__ == '__main__':
	main()