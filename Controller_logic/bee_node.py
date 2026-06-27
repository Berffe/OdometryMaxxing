import asyncio
import contextlib
import math
import os
import signal
import subprocess
import threading
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

try:
	from mavsdk import System
	from mavsdk.offboard import Attitude as MavsdkAttitude, OffboardError
except ImportError:
	System = None
	MavsdkAttitude = None
	OffboardError = Exception

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from px4_msgs.msg import VehicleLocalPosition
from cv_bridge import CvBridge, CvBridgeError

from .state import VehicleState, PlatformState, AttitudeSetpoint, TargetEstimate
from .optical_flow import OpticalFlowEstimator
from .target_acquisition import TargetAcquisition
from .control_law import ControlLaw
from .diagnostics_writer import DiagnosticsWriter


CONTROL_PERIOD_SEC = 0.5
MISSION_PERIOD_SEC = 0.1
MAVSDK_OFFBOARD_PERIOD_SEC = 0.05

SHOW_CAMERA = True
VERBOSE_STREAM_LOGS = False

# Start the attempt already airborne. 5 m corresponds to the cleanest far-range
# calibration operating point (area_fraction around 0.066 in the last batch).
TAKEOFF_ALTITUDE_M = 5.0
EKF2_SETTLE_TIME = 5.0
MAVSDK_SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
MAVSDK_PORT_TO_FREE = 14540
MAVSDK_HOLD_CURRENT_YAW = True

MAVSDK_CONNECT_TIMEOUT_SEC = 15.0
MAVSDK_HEALTH_TIMEOUT_SEC = 30.0
MAVSDK_TAKEOFF_ALTITUDE_TIMEOUT_SEC = 130.0
OFFBOARD_PRESTREAM_SEC = 2.0
OFFBOARD_START_TIMEOUT_SEC = 5.0

# Runtime safety for the first closed-loop tests. After the MAVSDK handoff,
# nominal commands are visual-only. If the visual target/flow is lost, keep
# streaming a neutral visual-hover setpoint instead of using PX4 velocity/altitude
# feedback. PX4 local state remains diagnostics-only after handoff.
LOST_TARGET_TIMEOUT_SEC = 2.0
ENABLE_INERTIAL_SAFETY_ABORTS = False
SAFETY_VZ_LIMIT = 1.0
SAFETY_LATERAL_VELOCITY_LIMIT = 2.0

# Bridged (ros_gz_bridge) ROS2 topic for the platform's exact world pose.
# Published directly by OscillatingPlatformController itself (see
# MovingPlatformController.cpp's publishPose) as a plain gz.msgs.Pose, on
# its own dedicated single-entity topic -- not via gz-sim's generic pose
# broadcasting, which two earlier approaches both confirmed unreliable here:
# a PosePublisher SDF plugin only emitted a one-shot static snapshot for
# this <static>true</static> model, and SceneBroadcaster's pose/info (a
# Pose_V of every entity) bridges through ros_gz_bridge into
# tf2_msgs/msg/TFMessage with every entity's name left empty -- confirmed
# directly via this node's own "Entity names seen" log -- so there was no
# way to pick this entity back out on the ROS side. Publishing our own
# topic sidesteps both problems entirely, back to the simple message type:
#   ros2 run ros_gz_bridge parameter_bridge \
#       /platform/pose@geometry_msgs/msg/Pose@gz.msgs.Pose
# Same topic/bridge as calibration_node.py -- keep these in sync.
# Diagnostics-only, same as vehicle_state: the control law never sees this
# (see control_law.py's module docstring). Set PLATFORM_POSE_TOPIC to None
# to disable (e.g. a stationary-platform run); diagnostics rows just get
# empty platform_*/relative_* fields either way.
PLATFORM_POSE_TOPIC = "/platform/pose"

# Real pose telemetry is noisy/jittery sample-to-sample; smooth the finite-
# differenced velocity the same way OpticalFlowEstimator smooths divergence
# (see optical_flow.py's module docstring for the same underlying argument).
PLATFORM_VELOCITY_SMOOTHING = 0.5

# Touchdown bridge. The Gazebo side is published by TouchPlugin in
# bee_platform.sdf. Bridge it with:
#   ros2 run ros_gz_bridge parameter_bridge \
#       /bee_platform/touched@std_msgs/msg/Bool@gz.msgs.Boolean
TOUCHDOWN_TOPIC = "/bee_platform/touched"
TOUCHDOWN_STATUS_TOPIC = "/bee_land/touchdown"
ENABLE_TOUCHDOWN_MOTOR_STOP = True
# SITL fallback: if PX4 refuses a normal disarm because its internal land
# detector does not recognize the moving-platform touchdown yet, kill() stops
# the simulated motors. Keep this False for real hardware.
ENABLE_TOUCHDOWN_KILL_FALLBACK = True

PHASE_WAITING_FOR_STREAMS = "waiting_for_streams"
PHASE_MAVSDK_TAKEOFF = "mavsdk_takeoff"
PHASE_PRESTREAM = "prestream_offboard"
PHASE_WAIT_OFFBOARD = "wait_offboard"
PHASE_CLOSED_LOOP = "closed_loop"
PHASE_LANDED = "landed"
PHASE_ABORTED = "aborted"


@contextlib.contextmanager
def suppress_stderr_fd(enabled: bool = True):
	"""Temporarily suppress native stderr spam from OpenCV/Qt while keeping video."""
	if not enabled:
		yield
		return
	try:
		stderr_fd = 2
		saved_fd = os.dup(stderr_fd)
		with open(os.devnull, "w") as devnull:
			os.dup2(devnull.fileno(), stderr_fd)
			yield
	finally:
		try:
			os.dup2(saved_fd, stderr_fd)
			os.close(saved_fd)
		except Exception:
			pass


class BeeLandNode(Node):
	def __init__(self):
		super().__init__("bee_land_node")

		self._node_start_time = time.time()
		self.bridge = CvBridge()
		self._vehicle_state = VehicleState()

		# Platform pose (dedicated bridge -> on_platform_pose): exact
		# world-frame position each message, finite-differenced into a
		# smoothed velocity (see PLATFORM_VELOCITY_SMOOTHING). None until the
		# first message arrives, or forever if PLATFORM_POSE_TOPIC is None --
		# diagnostics rows just get empty platform_*/relative_* fields.
		self._platform_state = None
		self._prev_platform_pose_t = None
		self._prev_platform_pose_xyz = None
		self._platform_velocity_filtered = (0.0, 0.0, 0.0)
		self._has_filtered_platform_velocity = False
		self._platform_pose_count = 0
		self._platform_pose_stall_logged = False

		self._latest_flow = None
		self._latest_frame = None
		self._latest_target = TargetEstimate()

		self.control_law = ControlLaw()
		self._latest_setpoint = AttitudeSetpoint(thrust=self.control_law.hover_thrust)

		self._have_local_position = False
		self._image_count = 0
		self._last_image_log_time = 0.0
		self._last_position_log_time = 0.0

		self._mission_phase = PHASE_WAITING_FOR_STREAMS
		self._phase_start_time = time.time()
		self._streams_ready_logged = False
		self._closed_loop_logged = False
		self._lost_target_since = None

		# Visual/control time bookkeeping. Target acquisition, optical flow, and
		# control_law.compute() must use one clock family. The source of truth is
		# the camera Image.header.stamp when ros_gz_bridge provides it. PX4 time is
		# kept only as a fallback for missing image stamps and as diagnostics.
		# Do NOT compare image/Gazebo timestamps and PX4 timestamps by absolute
		# value: in this setup they can live in different epochs. Only deltas inside
		# one clock family are meaningful.
		self._prev_control_flow_timestamp = None
		self._control_dt_fallback_logged = False
		self._image_stamp_fallback_logged = False

		self._mavsdk_thread = None
		self._mavsdk_takeoff_started = False
		self._mavsdk_takeoff_done = False
		self._mavsdk_takeoff_error = None
		self._mavsdk_offboard_start_requested = False
		self._mavsdk_offboard_started = False
		self._mavsdk_offboard_error = None
		self._mavsdk_stop_requested = False

		# Touchdown is a mission-level terminal event, not a visual-control input.
		# The contact signal comes from Gazebo/TouchPlugin through ros_gz_bridge.
		self._touchdown_detected = False
		self._touchdown_time = None
		self._touchdown_message_count = 0

		self._mavsdk_motor_stop_requested = False
		self._mavsdk_motor_stop_attempted = False
		self._mavsdk_motor_stop_done = False
		self._mavsdk_motor_stop_error = None

		self.optical_flow = OpticalFlowEstimator()
		self.target_acquisition = TargetAcquisition()

		self.diagnostics = DiagnosticsWriter(output_dir="logs", filename=None, flush_every_row=True)
		self.get_logger().info(f"Diagnostics CSV: {self.diagnostics.filepath}")

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
		touchdown_status_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.RELIABLE,
			durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)

		self.create_subscription(
			VehicleLocalPosition,
			"/fmu/out/vehicle_local_position_v1",
			self.on_local_position,
			px4_qos,
		)
		self.create_subscription(Image, "/bee_x500/camera/image", self.on_camera, camera_qos)

		if PLATFORM_POSE_TOPIC:
			self.create_subscription(
				Pose,
				PLATFORM_POSE_TOPIC,
				self.on_platform_pose,
				camera_qos,
			)
			self.get_logger().info(
				f"Platform pose tracking enabled: listening on {PLATFORM_POSE_TOPIC}. "
				"If platform_*/relative_* diagnostics columns stay empty, the bridge "
				"(ros_gz_bridge) for this topic likely isn't running, or the topic name "
				"doesn't match what Gazebo actually publishes -- see this node's warning "
				"after a few seconds with no messages, and PLATFORM_POSE_TOPIC's comment "
				"for how to check both."
			)
		else:
			self.get_logger().info("Platform pose tracking disabled (PLATFORM_POSE_TOPIC is None).")

		self._touchdown_status_pub = self.create_publisher(
			Bool,
			TOUCHDOWN_STATUS_TOPIC,
			touchdown_status_qos,
		)
		self.create_subscription(
			Bool,
			TOUCHDOWN_TOPIC,
			self.on_touchdown,
			camera_qos,
		)
		self._publish_touchdown_status(False)
		self.get_logger().info(
			f"Touchdown detection enabled: listening on {TOUCHDOWN_TOPIC}. "
			f"Latched status is republished on {TOUCHDOWN_STATUS_TOPIC}."
		)

		self.create_timer(MISSION_PERIOD_SEC, self.on_mission_timer)
		self.create_timer(CONTROL_PERIOD_SEC, self.on_control_timer)

		if SHOW_CAMERA:
			with suppress_stderr_fd(True):
				cv2.namedWindow("Bee Land - Camera", cv2.WINDOW_NORMAL)

		self.get_logger().info("bee_land_node started.")
		self.get_logger().info("Waiting for required streams: local_position and camera.")

	def on_camera(self, msg: Image):
		self._image_count += 1
		now = time.time()
		if VERBOSE_STREAM_LOGS and now - self._last_image_log_time >= 1.0:
			self._last_image_log_time = now
			self.get_logger().info(f"image #{self._image_count}: {msg.width}x{msg.height}, encoding={msg.encoding}")

		try:
			src = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		except CvBridgeError as exc:
			self.get_logger().error(f"cv_bridge conversion failed: {exc}")
			return

		# Keep camera orientation independent of whether the debug window is open.
		frame = cv2.rotate(src, cv2.ROTATE_180)
		if SHOW_CAMERA:
			with suppress_stderr_fd(True):
				cv2.imshow("Bee Land - Camera", frame)
				cv2.waitKey(1)

		stamp = self._image_timestamp_sec(msg)
		target = self.target_acquisition.update(frame, timestamp=stamp)
		flow = self.optical_flow.update(frame, stamp, target=target)

		self._latest_frame = frame
		self._latest_target = target
		self._latest_flow = flow

	def on_platform_pose(self, msg: Pose):
		"""
		Exact platform world pose, published directly by
		OscillatingPlatformController on its own dedicated topic (see
		PLATFORM_POSE_TOPIC and MovingPlatformController.cpp's publishPose) --
		no entity matching needed, since every message on this topic IS the
		platform, by construction. Position is exact; Pose carries no
		velocity, so velocity is finite-differenced against the previous
		message using this callback's own receipt time (same time.time()
		pattern as on_camera/on_local_position), then smoothed -- raw
		frame-to-frame differencing of real, slightly-jittery pose telemetry
		amplifies noise the same way it would for optical flow (see
		optical_flow.py's module docstring for the same argument). Stored in
		the SDF world's own ENU convention; platform_motion.relative_motion()
		handles the NED conversion when this is logged alongside
		vehicle_state. Diagnostics-only -- never read by control_law.
		"""
		now = time.time()
		x, y, z = msg.position.x, msg.position.y, msg.position.z

		self._platform_pose_count += 1
		if self._platform_pose_count == 1:
			self.get_logger().info(
				f"First platform pose received on {PLATFORM_POSE_TOPIC}: "
				f"x={x:.3f} y={y:.3f} z={z:.3f} (SDF world/ENU). "
				"Platform tracking is live."
			)

		if self._prev_platform_pose_t is not None:
			dt = now - self._prev_platform_pose_t
			if dt > 1e-3:
				px, py, pz = self._prev_platform_pose_xyz
				raw_v = ((x - px) / dt, (y - py) / dt, (z - pz) / dt)

				alpha = PLATFORM_VELOCITY_SMOOTHING
				if not self._has_filtered_platform_velocity:
					self._platform_velocity_filtered = raw_v
					self._has_filtered_platform_velocity = True
				else:
					fv = self._platform_velocity_filtered
					self._platform_velocity_filtered = tuple(
						alpha * fv[i] + (1.0 - alpha) * raw_v[i] for i in range(3)
					)

		self._prev_platform_pose_t = now
		self._prev_platform_pose_xyz = (x, y, z)

		vx, vy, vz = self._platform_velocity_filtered
		self._platform_state = PlatformState(
			timestamp=now, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
		)

	def on_touchdown(self, msg: Bool):
		"""Gazebo/TouchPlugin contact event bridged from /bee_platform/touched.

		The event is latched: after the first True sample, the mission is considered
		landed even if the contact signal later drops because the platform moves or
		the vehicle bounces. TouchPlugin's <time> parameter in the SDF already filters
		out single-frame grazes before this callback receives True.
		"""
		self._touchdown_message_count += 1
		if not bool(msg.data):
			return

		if self._touchdown_detected:
			return

		self._touchdown_detected = True
		self._touchdown_time = time.time()
		self.get_logger().warning(
			"Gazebo touchdown detected: platform contact is stable."
		)
		self._publish_touchdown_status(True)

		if self._mission_phase == PHASE_CLOSED_LOOP:
			self._enter_landed_phase("touchdown contact event")

	def on_local_position(self, msg: VehicleLocalPosition):
		now = time.time()
		self._have_local_position = True
		if VERBOSE_STREAM_LOGS and now - self._last_position_log_time >= 1.0:
			self._last_position_log_time = now
			self.get_logger().info(f"local position: x={msg.x:.2f} m, y={msg.y:.2f} m, z={msg.z:.2f} m")

		self._vehicle_state = VehicleState(
			timestamp=now,
			x=msg.x,
			y=msg.y,
			z=msg.z,
			vx=msg.vx,
			vy=msg.vy,
			vz=msg.vz,
			yaw=msg.heading,
			px4_timestamp_sec=msg.timestamp / 1e6,
		)

	def on_mission_timer(self):
		now = time.time()

		if (
			PLATFORM_POSE_TOPIC
			and self._platform_pose_count == 0
			and not self._platform_pose_stall_logged
			and now - self._node_start_time >= 10.0
		):
			self._platform_pose_stall_logged = True
			self.get_logger().warning(
				f"No platform pose received on {PLATFORM_POSE_TOPIC} after "
				f"{now - self._node_start_time:.0f}s. diagnostics will log empty "
				"platform_*/relative_* fields until this is fixed. Check, in order: "
				f"(1) `gz topic -l` shows {PLATFORM_POSE_TOPIC} and `gz topic -e -t "
				"<that topic>` shows live data, not just a topic name that exists -- if "
				"not, OscillatingPlatformController's new publisher may need the plugin "
				"rebuilt/reinstalled, or the .so may be stale; "
				"(2) the ros_gz_bridge process for this topic is actually running; "
				f"(3) `ros2 topic info {PLATFORM_POSE_TOPIC} -v` WHILE this node is "
				"still running (not after stopping it) shows this node's own name as "
				"a subscriber, not just ros_gz_bridge's internal pub/sub pair."
			)

		if self._mission_phase == PHASE_LANDED:
			# Keep the terminal state latched and keep publishing an explicit zero-
			# thrust command until the MAVSDK worker confirms motor stop or the node
			# is shut down by the user.
			self._latest_setpoint = self._landed_zero_thrust_setpoint()
			self._publish_touchdown_status(True)
			return

		if self._mission_phase == PHASE_ABORTED:
			# Keep streaming a safe inertial hold setpoint instead of simply
			# stopping MAVSDK offboard. Stopping the stream can trigger a PX4
			# offboard failsafe while the vehicle still has velocity.
			self._latest_setpoint = self._neutral_visual_hold_setpoint()
			return

		if self._mission_phase == PHASE_WAITING_FOR_STREAMS:
			if self._ready_to_start():
				self._enter_phase(PHASE_MAVSDK_TAKEOFF)
			return

		if self._mission_phase == PHASE_MAVSDK_TAKEOFF:
			self._ensure_mavsdk_worker_started()
			if self._mavsdk_takeoff_error is not None:
				self._abort(f"MAVSDK takeoff failed: {self._mavsdk_takeoff_error}")
				return
			if self._mavsdk_takeoff_done:
				self.get_logger().info("MAVSDK takeoff complete. Starting offboard prestream.")
				self._enter_phase(PHASE_PRESTREAM)
			return

		if self._mission_phase == PHASE_PRESTREAM:
			# Let the MAVSDK worker stream a stable hover setpoint before offboard.start().
			self._latest_setpoint = AttitudeSetpoint(
				timestamp=getattr(self._latest_target, "timestamp", 0.0),
				roll=0.0,
				pitch=0.0,
				yaw=0.0,
				thrust=self.control_law.hover_thrust,
			)
			if now - self._phase_start_time >= OFFBOARD_PRESTREAM_SEC:
				self.get_logger().info("Requesting MAVSDK attitude offboard start.")
				self._mavsdk_offboard_start_requested = True
				self._enter_phase(PHASE_WAIT_OFFBOARD)
			return

		if self._mission_phase == PHASE_WAIT_OFFBOARD:
			if self._mavsdk_offboard_error is not None:
				self._abort(f"MAVSDK offboard start failed: {self._mavsdk_offboard_error}")
				return
			if self._mavsdk_offboard_started:
				self.control_law.reset_visual_integrators()
				self.get_logger().info(
					"MAVSDK attitude offboard started. Closed-loop visual controller is now active."
				)
				self._enter_phase(PHASE_CLOSED_LOOP)
				return
			if now - self._phase_start_time >= OFFBOARD_START_TIMEOUT_SEC:
				self._abort("timed out waiting for MAVSDK offboard.start()")
			return

		if self._mission_phase == PHASE_CLOSED_LOOP:
			if self._touchdown_detected:
				self._enter_landed_phase("latched touchdown flag")
				return

			if not self._closed_loop_logged:
				self._closed_loop_logged = True
				self.get_logger().info("Closed-loop visual landing/hover attempt running.")
			return

	def on_control_timer(self):
		now = time.time()

		if self._latest_flow is None or self._latest_frame is None:
			return

		if self._mission_phase == PHASE_LANDED:
			self._latest_setpoint = self._landed_zero_thrust_setpoint()
			self._publish_touchdown_status(True)
			self._write_diagnostics_row()
			return

		if self._mission_phase == PHASE_ABORTED:
			self._latest_setpoint = self._neutral_visual_hold_setpoint()
			self._write_diagnostics_row()
			return

		if self._mission_phase != PHASE_CLOSED_LOOP:
			return

		if self._touchdown_detected:
			self._enter_landed_phase("latched touchdown flag")
			self._write_diagnostics_row()
			return

		# Optional diagnostic-only safety aborts. Disabled by default because the
		# project constraint is that PX4 state must not participate in the visual
		# control logic after handoff. When disabled, these states are logged only.
		if ENABLE_INERTIAL_SAFETY_ABORTS:
			if abs(self._vehicle_state.vz) > SAFETY_VZ_LIMIT:
				self._abort(f"vertical velocity safety limit exceeded: vz={self._vehicle_state.vz:.3f} m/s")
				self._latest_setpoint = self._neutral_visual_hold_setpoint()
				self._write_diagnostics_row()
				return

			if (
				abs(self._vehicle_state.vx) > SAFETY_LATERAL_VELOCITY_LIMIT
				or abs(self._vehicle_state.vy) > SAFETY_LATERAL_VELOCITY_LIMIT
			):
				self._abort(
					"lateral velocity safety limit exceeded: "
					f"vx={self._vehicle_state.vx:.3f} m/s, vy={self._vehicle_state.vy:.3f} m/s"
				)
				self._latest_setpoint = self._neutral_visual_hold_setpoint()
				self._write_diagnostics_row()
				return

		target_ok = bool(getattr(self._latest_target, "found", False))
		flow_ok = bool(getattr(self._latest_flow, "valid", False))

		if not (target_ok and flow_ok):
			if self._lost_target_since is None:
				self._lost_target_since = now
			elif now - self._lost_target_since >= LOST_TARGET_TIMEOUT_SEC:
				self._abort(
					f"target/flow lost for >= {LOST_TARGET_TIMEOUT_SEC:.1f}s "
					f"(target_found={target_ok}, flow_valid={flow_ok})"
				)
				self._latest_setpoint = self._neutral_visual_hold_setpoint()
				self._write_diagnostics_row()
				return
		else:
			self._lost_target_since = None

		self._latest_setpoint = self.control_law.compute(
			self._latest_target,
			self._latest_flow,
			self._control_dt_sec(),
		)
		self._write_diagnostics_row()

	def _control_dt_sec(self) -> float:
		"""Return the control step in the same clock as optical flow.

		OpticalFlowEstimator converts px/frame into px/s using FlowResult.timestamp,
		which comes from the camera image timestamp. Therefore the control law's
		integral and slew-rate terms must use the delta between consecutive visual
		timestamps, not a PX4 timestamp and not wall-clock time. This keeps the
		units of flow/divergence [1/s] and controller dt [s] consistent.
		"""
		current = float(getattr(self._latest_flow, "timestamp", 0.0))
		previous = self._prev_control_flow_timestamp
		self._prev_control_flow_timestamp = current

		if current <= 0.0 or previous is None:
			return CONTROL_PERIOD_SEC

		delta = current - previous

		# Same frame / paused sim / duplicate image timestamp: do almost nothing,
		# but keep control_law.compute() numerically happy.
		if 0.0 <= delta <= 1e-6:
			return 1e-3

		# Reject true timestamp glitches without mixing in PX4/wall clocks.
		if not (0.0 < delta <= 10.0 * CONTROL_PERIOD_SEC):
			if not self._control_dt_fallback_logged:
				self._control_dt_fallback_logged = True
				self.get_logger().warning(
					f"Implausible visual timestamp dt ({delta:.4f}s, nominal "
					f"{CONTROL_PERIOD_SEC:.4f}s). Falling back to the fixed "
					"control period for this tick."
				)
			return CONTROL_PERIOD_SEC

		return delta

	def _image_timestamp_sec(self, msg: Image) -> float:
		"""Return the timestamp used by target acquisition and optical flow.

		Preferred source: sensor_msgs/Image.header.stamp, normally filled by
		ros_gz_bridge from Gazebo simulation time. If it is missing/zero, fall back
		to PX4's simulated timestamp if available. Wall-clock is only a last-resort
		startup fallback. The returned value is only compared to previous image
		timestamps, never to PX4/wall timestamps by absolute value.
		"""
		stamp = getattr(getattr(msg, "header", None), "stamp", None)
		stamp_sec = self._ros_stamp_to_sec(stamp)
		if stamp_sec > 0.0:
			return stamp_sec

		px4_time = float(getattr(self._vehicle_state, "px4_timestamp_sec", 0.0))
		if px4_time > 0.0:
			if not self._image_stamp_fallback_logged:
				self._image_stamp_fallback_logged = True
				self.get_logger().warning(
					"Camera Image.header.stamp is zero; using PX4 timestamp as the "
					"vision timestamp. Prefer fixing the camera bridge so images carry "
					"Gazebo sim time."
				)
			return px4_time

		if not self._image_stamp_fallback_logged:
			self._image_stamp_fallback_logged = True
			self.get_logger().warning(
				"Camera Image.header.stamp and PX4 timestamp are unavailable; "
				"temporarily using wall-clock for vision timestamps."
			)
		return time.time()

	@staticmethod
	def _ros_stamp_to_sec(stamp) -> float:
		if stamp is None:
			return 0.0
		try:
			return float(stamp.sec) + 1e-9 * float(stamp.nanosec)
		except AttributeError:
			return 0.0

	def _write_diagnostics_row(self):
		self.diagnostics.write(
			wall_timestamp=time.time(),
			target=self._latest_target,
			flow=self._latest_flow,
			setpoint=self._latest_setpoint,
			vehicle_state=self._vehicle_state,
			platform_state=self._platform_state,
		)

	def _ready_to_start(self) -> bool:
		if not self._have_local_position:
			return False
		if self._latest_frame is None:
			return False
		if not self._streams_ready_logged:
			self._streams_ready_logged = True
			self.get_logger().info("Required streams are available; starting automatic climb.")
		return True

	def _ensure_mavsdk_worker_started(self):
		if self._mavsdk_takeoff_started:
			return
		if System is None or MavsdkAttitude is None:
			self._mavsdk_takeoff_error = "mavsdk/offboard is not installed in this Python environment"
			return
		self._mavsdk_takeoff_started = True
		self.get_logger().info(f"Starting MAVSDK takeoff to {TAKEOFF_ALTITUDE_M:.2f} m.")
		self._mavsdk_thread = threading.Thread(target=self._run_mavsdk_worker_thread, name="mavsdk_takeoff_offboard", daemon=True)
		self._mavsdk_thread.start()

	def _run_mavsdk_worker_thread(self):
		try:
			asyncio.run(self._mavsdk_worker_async())
		except Exception as exc:
			if not self._mavsdk_takeoff_done:
				self._mavsdk_takeoff_error = repr(exc)
			else:
				self._mavsdk_offboard_error = repr(exc)

	async def _mavsdk_worker_async(self):
		self._free_mavsdk_port(MAVSDK_PORT_TO_FREE)
		drone = System()
		await drone.connect(system_address=MAVSDK_SYSTEM_ADDRESS)

		self.get_logger().info("MAVSDK: waiting for drone connection...")
		await self._wait_for_condition(
			drone.core.connection_state(),
			lambda state: state.is_connected,
			MAVSDK_CONNECT_TIMEOUT_SEC,
			"MAVSDK connection",
		)
		self.get_logger().info("MAVSDK: connected.")

		self.get_logger().info("MAVSDK: waiting for global/home/local position estimates...")
		await self._wait_for_condition(
			drone.telemetry.health(),
			lambda h: h.is_global_position_ok and h.is_home_position_ok and h.is_local_position_ok,
			MAVSDK_HEALTH_TIMEOUT_SEC,
			"global/home/local position health",
		)
		self.get_logger().info("MAVSDK: all position estimates OK.")

		await asyncio.sleep(EKF2_SETTLE_TIME)
		home_position = await self._wait_for_condition(
			drone.telemetry.position(),
			lambda position: True,
			MAVSDK_CONNECT_TIMEOUT_SEC,
			"initial position reading",
		)
		home_baro_offset = home_position.relative_altitude_m
		self.get_logger().info(f"MAVSDK: home altitude offset {home_baro_offset:.2f} m.")

		self.get_logger().info(f"MAVSDK: setting MIS_TAKEOFF_ALT={TAKEOFF_ALTITUDE_M:.2f} m.")
		await drone.param.set_param_float("MIS_TAKEOFF_ALT", float(TAKEOFF_ALTITUDE_M))
		await asyncio.sleep(0.5)

		self.get_logger().info("MAVSDK: arming.")
		await drone.action.arm()
		self.get_logger().info("MAVSDK: takeoff command.")
		await drone.action.takeoff()

		await self._wait_for_condition(
			drone.telemetry.position(),
			lambda p: (p.relative_altitude_m - home_baro_offset) >= TAKEOFF_ALTITUDE_M - 0.20,
			MAVSDK_TAKEOFF_ALTITUDE_TIMEOUT_SEC,
			"takeoff altitude reached",
			progress_fn=lambda p, elapsed: self.get_logger().info(
				f"MAVSDK: still climbing after {elapsed:.0f}s -- altitude={p.relative_altitude_m - home_baro_offset:.2f} m"
			),
			progress_interval=5.0,
		)
		self.get_logger().info("MAVSDK: reached takeoff altitude; hovering.")
		self._mavsdk_takeoff_done = True

		while not self._mavsdk_offboard_start_requested and not self._mavsdk_stop_requested:
			await asyncio.sleep(0.05)
		if self._mavsdk_stop_requested:
			return

		for _ in range(10):
			await self._send_mavsdk_attitude_setpoint(drone)
			await asyncio.sleep(MAVSDK_OFFBOARD_PERIOD_SEC)

		try:
			await drone.offboard.start()
		except OffboardError as exc:
			self._mavsdk_offboard_error = repr(exc)
			return

		self._mavsdk_offboard_started = True
		while not self._mavsdk_stop_requested:
			if self._mavsdk_motor_stop_requested and not self._mavsdk_motor_stop_done:
				await self._try_mavsdk_motor_stop(drone)
				if self._mavsdk_motor_stop_done:
					break

			await self._send_mavsdk_attitude_setpoint(drone)
			await asyncio.sleep(MAVSDK_OFFBOARD_PERIOD_SEC)

	async def _try_mavsdk_motor_stop(self, drone):
		"""Stop motors after confirmed Gazebo touchdown.

		Normal disarm is tried first. In SITL, kill() is used as a fallback when PX4's
		internal land detector refuses the disarm on the moving platform. This method
		attempts motor stop only once to avoid spamming MAVSDK/PX4 with repeated
		commands if both methods fail.
		"""
		if self._mavsdk_motor_stop_attempted:
			return

		self._mavsdk_motor_stop_attempted = True
		self._latest_setpoint = self._landed_zero_thrust_setpoint()

		# Push one zero-thrust setpoint before asking PX4 to stop the motors.
		try:
			await self._send_mavsdk_attitude_setpoint(drone)
		except Exception as exc:
			self.get_logger().warning(
				f"MAVSDK: failed to send final zero-thrust setpoint before disarm: {repr(exc)}"
			)

		try:
			self.get_logger().warning("MAVSDK: touchdown confirmed, requesting disarm.")
			await drone.action.disarm()
			self.get_logger().warning("MAVSDK: disarm accepted after touchdown.")
			self._mavsdk_motor_stop_done = True
			self._mavsdk_stop_requested = True
			return
		except Exception as exc:
			self._mavsdk_motor_stop_error = repr(exc)
			self.get_logger().error(
				f"MAVSDK: disarm failed after touchdown: {self._mavsdk_motor_stop_error}"
			)

		if not ENABLE_TOUCHDOWN_KILL_FALLBACK:
			self.get_logger().error(
				"MAVSDK: kill fallback disabled; keeping zero-thrust offboard stream alive."
			)
			return

		try:
			self.get_logger().error(
				"MAVSDK: using kill fallback after confirmed Gazebo contact. SITL only."
			)
			await drone.action.kill()
			self.get_logger().warning("MAVSDK: kill accepted after touchdown.")
			self._mavsdk_motor_stop_done = True
			self._mavsdk_stop_requested = True
		except Exception as kill_exc:
			self._mavsdk_motor_stop_error = repr(kill_exc)
			self.get_logger().error(
				f"MAVSDK: kill fallback failed: {self._mavsdk_motor_stop_error}"
			)

	async def _send_mavsdk_attitude_setpoint(self, drone):
		sp = self._latest_setpoint
		yaw_rad = sp.yaw
		if MAVSDK_HOLD_CURRENT_YAW and self._vehicle_state.timestamp > 0.0:
			yaw_rad = self._vehicle_state.yaw
		await drone.offboard.set_attitude(
			MavsdkAttitude(
				math.degrees(sp.roll),
				math.degrees(sp.pitch),
				math.degrees(yaw_rad),
				self._clamp(sp.thrust, 0.0, 1.0),
			)
		)

	@staticmethod
	async def _wait_for_condition(async_iterable, condition, timeout: float, label: str, progress_fn=None, progress_interval: float = 5.0):
		async def _inner():
			loop = asyncio.get_event_loop()
			start = loop.time()
			last_progress = start
			async for item in async_iterable:
				if condition(item):
					return item
				if progress_fn is not None:
					now = loop.time()
					if now - last_progress >= progress_interval:
						last_progress = now
						progress_fn(item, now - start)
		try:
			return await asyncio.wait_for(_inner(), timeout=timeout)
		except asyncio.TimeoutError:
			raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {label}")

	@staticmethod
	def _free_mavsdk_port(port: int):
		try:
			result = subprocess.run(["lsof", "-t", f"-i:UDP:{port}"], capture_output=True, text=True, check=False)
		except FileNotFoundError:
			return
		for pid in result.stdout.strip().split():
			try:
				os.kill(int(pid), signal.SIGKILL)
			except ProcessLookupError:
				pass

	def _publish_touchdown_status(self, value: bool):
		msg = Bool()
		msg.data = bool(value)
		self._touchdown_status_pub.publish(msg)

	def _landed_zero_thrust_setpoint(self) -> AttitudeSetpoint:
		"""Terminal setpoint after confirmed touchdown.

		This is deliberately separate from _neutral_visual_hold_setpoint(): abort keeps
		hover thrust, while a successful landing commands zero thrust until PX4/MAVSDK
		accepts disarm or kill.
		"""
		return AttitudeSetpoint(
			timestamp=getattr(self._latest_target, "timestamp", 0.0),
			roll=0.0,
			pitch=0.0,
			yaw=0.0,
			thrust=0.0,
		)

	def _enter_landed_phase(self, reason: str):
		if self._mission_phase == PHASE_LANDED:
			return

		self.get_logger().warning(f"LANDING COMPLETE: {reason}. Entering landed phase.")
		self._latest_setpoint = self._landed_zero_thrust_setpoint()
		self._publish_touchdown_status(True)
		if ENABLE_TOUCHDOWN_MOTOR_STOP:
			self._mavsdk_motor_stop_requested = True
		else:
			self.get_logger().warning(
				"Touchdown motor stop disabled; landed phase will only stream zero thrust."
			)
		self._enter_phase(PHASE_LANDED)

	def _neutral_visual_hold_setpoint(self) -> AttitudeSetpoint:
		"""Neutral visual-hover setpoint used after abort/target loss.

		This deliberately does NOT use PX4 local position or velocity. After
		handoff, PX4 state is diagnostics-only; this fallback simply keeps the
		MAVSDK offboard stream alive with zero roll/pitch and nominal hover thrust
		until the user stops the node.
		"""
		return AttitudeSetpoint(
			timestamp=getattr(self._latest_target, "timestamp", 0.0),
			roll=0.0,
			pitch=0.0,
			yaw=0.0,
			thrust=self.control_law.hover_thrust,
		)


	def _enter_phase(self, phase: str):
		if phase != self._mission_phase:
			self.get_logger().info(f"Mission phase: {self._mission_phase} -> {phase}")
		self._mission_phase = phase
		self._phase_start_time = time.time()

	def _abort(self, reason: str):
		if self._mission_phase != PHASE_ABORTED:
			self.get_logger().error(f"ABORTING bee_land_node: {reason}")
		self._mission_phase = PHASE_ABORTED
		# If offboard was never started, there is no stream to maintain.
		# Once offboard is active, keep MAVSDK streaming _neutral_visual_hold_setpoint()
		# until the user stops the node.
		if not self._mavsdk_offboard_started:
			self._mavsdk_stop_requested = True

	@staticmethod
	def _clamp(value: float, lower: float, upper: float) -> float:
		return max(lower, min(upper, float(value)))


def main(args=None):
	rclpy.init(args=args)
	node = BeeLandNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node._mavsdk_stop_requested = True
		node.diagnostics.close()
		node.destroy_node()
		if SHOW_CAMERA:
			with suppress_stderr_fd(True):
				cv2.destroyAllWindows()
		try:
			if rclpy.ok():
				rclpy.shutdown()
		except Exception:
			pass


if __name__ == "__main__":
	main()