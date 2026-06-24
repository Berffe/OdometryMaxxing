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
from px4_msgs.msg import VehicleLocalPosition
from cv_bridge import CvBridge, CvBridgeError

from .state import VehicleState, AttitudeSetpoint, TargetEstimate
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

PHASE_WAITING_FOR_STREAMS = "waiting_for_streams"
PHASE_MAVSDK_TAKEOFF = "mavsdk_takeoff"
PHASE_PRESTREAM = "prestream_offboard"
PHASE_WAIT_OFFBOARD = "wait_offboard"
PHASE_CLOSED_LOOP = "closed_loop"
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

		self.bridge = CvBridge()
		self._vehicle_state = VehicleState()
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

		self._mavsdk_thread = None
		self._mavsdk_takeoff_started = False
		self._mavsdk_takeoff_done = False
		self._mavsdk_takeoff_error = None
		self._mavsdk_offboard_start_requested = False
		self._mavsdk_offboard_started = False
		self._mavsdk_offboard_error = None
		self._mavsdk_stop_requested = False

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

		self.create_subscription(
			VehicleLocalPosition,
			"/fmu/out/vehicle_local_position_v1",
			self.on_local_position,
			px4_qos,
		)
		self.create_subscription(Image, "/bee_x500/camera/image", self.on_camera, camera_qos)

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

		stamp = time.time()
		target = self.target_acquisition.update(frame, timestamp=stamp)
		flow = self.optical_flow.update(frame, stamp, target=target)

		self._latest_frame = frame
		self._latest_target = target
		self._latest_flow = flow

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
		)

	def on_mission_timer(self):
		now = time.time()
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
			if not self._closed_loop_logged:
				self._closed_loop_logged = True
				self.get_logger().info("Closed-loop visual landing/hover attempt running.")
			return

	def on_control_timer(self):
		now = time.time()

		if self._latest_flow is None or self._latest_frame is None:
			return

		if self._mission_phase == PHASE_ABORTED:
			self._latest_setpoint = self._neutral_visual_hold_setpoint()
			self._write_diagnostics_row()
			return

		if self._mission_phase != PHASE_CLOSED_LOOP:
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
			CONTROL_PERIOD_SEC,
		)
		self._write_diagnostics_row()

	def _write_diagnostics_row(self):
		self.diagnostics.write(
			wall_timestamp=time.time(),
			target=self._latest_target,
			flow=self._latest_flow,
			setpoint=self._latest_setpoint,
			vehicle_state=self._vehicle_state,
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
			await self._send_mavsdk_attitude_setpoint(drone)
			await asyncio.sleep(MAVSDK_OFFBOARD_PERIOD_SEC)

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
