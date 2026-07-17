"""Lean BEE_LAND ROS 2 controller node.

Live dependencies after takeoff:
- camera images -> vision worker -> mission/control;
- Gazebo truth contact confirmation -> terminal motor stop;
- PX4 receives attitude/thrust setpoints only.

No PX4 state, platform pose, clock fitting, position reconstruction or physical
truth enters the controller. Full Gazebo truth is logged by a separate process.
"""
from __future__ import annotations

import inspect
import multiprocessing as mp
import queue
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from ros_gz_interfaces.msg import Float32Array

from .clock import TimeManager
from .control_law import ControlLaw
from .diagnostics_writer import DiagnosticsWriter
from .mavsdk_worker import MavsdkWorker
from .mission_routine import MissionRoutine, INFEASIBLE as MISSION_INFEASIBLE
from .px4_interface import PX4Interface
from .state import AttitudeSetpoint, ContactState, TargetEstimate
from .truth_layout import decode_truth_array
from .vision_worker import run_vision_worker

# Scheduling
CONTROL_PERIOD_SEC = 0.01
PX4_SETPOINT_PERIOD_SEC = 0.03
OFFBOARD_PRESTREAM_SEC = 2.0
VISION_INPUT_QUEUE_MAX = 2
LOST_TARGET_TIMEOUT_SEC = 2.0
SHOW_CAMERA = False

# Mission tuning retained from the previous node
DESCENT_DIVERGENCE_SETPOINT = 0.30
APPROACH_DIVERGENCE_SETPOINT = 0.12
D_STAR_RAMP_IN_SEC = 5.0
FINAL_PROBE_ENTRY_RAMP_SEC = 1.5
FOV_NEAR_AREA_FRACTION = 0.85
CENTER_TO_PROBE_LATERAL_RAMP_SEC = 0.1
CENTER_LATERAL_P_SCALE = 1.0
CENTER_LATERAL_D_SCALE = 1.0
PROBE_LATERAL_P_SCALE = 1.0
PROBE_LATERAL_D_SCALE = 1.0
PROBE_DESIGN_PERIOD_SEC = 6.7
FAR_PROBE_WINDOW_SEC = 1.5 * PROBE_DESIGN_PERIOD_SEC
FAR_PROBE_DECAY_TAU_SEC = 1.5 * PROBE_DESIGN_PERIOD_SEC
FAR_PROBE_HIGHPASS_TAU_SEC = 4.0 * PROBE_DESIGN_PERIOD_SEC
NEAR_PROBE_WINDOW_SEC = 0.6 * PROBE_DESIGN_PERIOD_SEC
NEAR_PROBE_DECAY_TAU_SEC = PROBE_DESIGN_PERIOD_SEC
NEAR_PROBE_HIGHPASS_TAU_SEC = 2.0 * PROBE_DESIGN_PERIOD_SEC
PROBE_MIN_DURATION_SEC = 3.0 * PROBE_DESIGN_PERIOD_SEC
FINAL_PROBE_DURATION_SEC = 2.0 * PROBE_DESIGN_PERIOD_SEC
CEILING_SAFETY_FACTOR = 0.5
DESCENT_CEILING_MARGIN = 0.8
NEAR_FIELD_HEIGHT_M = 0.4
CAMERA_FRAME_PERIOD_SEC = 1.0 / 30.0
VISION_PROCESSING_LATENCY_BUDGET_SEC = 0.02
STABILITY_DT_SEC = CAMERA_FRAME_PERIOD_SEC + VISION_PROCESSING_LATENCY_BUDGET_SEC + PX4_SETPOINT_PERIOD_SEC
LEG_CLEARANCE_M = 0.182
HOVER_PROBE_ONLY = False
INITIAL_THRUST_GAIN = 6.5

# MAVSDK/PX4
TAKEOFF_ALTITUDE_M = 5.0
EKF2_SETTLE_TIME = 5.0
MAVSDK_SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
MAVSDK_PORT_TO_FREE = 14540
MAVSDK_CONNECT_TIMEOUT_SEC = 15.0
MAVSDK_HEALTH_TIMEOUT_SEC = 30.0
MAVSDK_TAKEOFF_ALTITUDE_TIMEOUT_SEC = 130.0
ENABLE_TOUCHDOWN_MOTOR_STOP = True
ENABLE_TOUCHDOWN_KILL_FALLBACK = True

CAMERA_TOPIC = "/bee_x500/camera/image"
TRUTH_TOPIC = "/bee_land/truth"

WAIT_TAKEOFF = "mavsdk_takeoff"
PRESTREAM = "prestream_offboard"
CLOSED_LOOP = "closed_loop"
LANDED = "landed"
ABORTED = "aborted"


def _supported_kwargs(callable_obj, values):
	"""Pass only keyword arguments supported by the installed algorithm version."""
	params = inspect.signature(callable_obj).parameters
	return {k: v for k, v in values.items() if k in params}


class BeeLandNode(Node):
	def __init__(self):
		super().__init__("bee_land_node")
		self.time = TimeManager(self)
		self.bridge = CvBridge()
		self.control_law = ControlLaw()
		self._latest_target = TargetEstimate()
		self._latest_flow = None
		self._latest_setpoint = AttitudeSetpoint(thrust=self.control_law.hover_thrust)
		self._contact = ContactState()
		self._phase = WAIT_TAKEOFF
		self._phase_start_mono = self.time.monotonic_sec()
		self._last_controlled_flow_stamp = None
		self._previous_flow_stamp = None
		self._lost_target_since_mono = None
		self._vision_sequence = 0
		self._vision_dropped_frames = 0
		self._latest_frame_receipt_wall = None
		self._latest_frame_receipt_mono = None
		self._vision_metrics = {}
		self._last_publish = None
		self._motor_stop_requested = False
		self._shutdown = False

		self.mission = self._make_mission()
		self.diagnostics = DiagnosticsWriter(output_dir="logs", flush_every_row=True)

		sensor_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.VOLATILE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)
		px4_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)
		self.px4 = PX4Interface(self, px4_qos, time_manager=self.time)
		self.create_subscription(Image, CAMERA_TOPIC, self.on_camera, sensor_qos)
		self.create_subscription(Float32Array, TRUTH_TOPIC, self.on_truth, sensor_qos)

		self.mavsdk = MavsdkWorker(
			logger=self.get_logger(), on_pre_motor_stop=self._latch_zero_thrust,
			system_address=MAVSDK_SYSTEM_ADDRESS, port_to_free=MAVSDK_PORT_TO_FREE,
			takeoff_altitude_m=TAKEOFF_ALTITUDE_M,
			connect_timeout_sec=MAVSDK_CONNECT_TIMEOUT_SEC,
			health_timeout_sec=MAVSDK_HEALTH_TIMEOUT_SEC,
			takeoff_altitude_timeout_sec=MAVSDK_TAKEOFF_ALTITUDE_TIMEOUT_SEC,
			ekf2_settle_time_sec=EKF2_SETTLE_TIME,
			enable_kill_fallback=ENABLE_TOUCHDOWN_KILL_FALLBACK,
		)

		self._start_vision_worker()
		self.create_timer(CONTROL_PERIOD_SEC, self.on_control_timer)
		self.create_timer(PX4_SETPOINT_PERIOD_SEC, self.on_px4_timer)
		self.create_timer(0.1, self.on_supervisor_timer)
		self.mavsdk.start()
		self.get_logger().info(
			"BEE_LAND lean controller started: camera-only control, truth-contact-only touchdown."
		)

	def _make_mission(self):
		values = dict(
			hover_thrust=self.control_law.hover_thrust,
			control_period_sec=CONTROL_PERIOD_SEC,
			descent_divergence_setpoint=DESCENT_DIVERGENCE_SETPOINT,
			approach_divergence_setpoint=APPROACH_DIVERGENCE_SETPOINT,
			final_probe_duration_sec=FINAL_PROBE_DURATION_SEC,
			final_probe_entry_ramp_sec=FINAL_PROBE_ENTRY_RAMP_SEC,
			fov_near_area_fraction=FOV_NEAR_AREA_FRACTION,
			probe_min_duration_sec=PROBE_MIN_DURATION_SEC,
			far_probe_window_sec=FAR_PROBE_WINDOW_SEC,
			far_probe_decay_tau_sec=FAR_PROBE_DECAY_TAU_SEC,
			far_probe_highpass_tau_sec=FAR_PROBE_HIGHPASS_TAU_SEC,
			near_probe_window_sec=NEAR_PROBE_WINDOW_SEC,
			near_probe_decay_tau_sec=NEAR_PROBE_DECAY_TAU_SEC,
			near_probe_highpass_tau_sec=NEAR_PROBE_HIGHPASS_TAU_SEC,
			leg_clearance_m=LEG_CLEARANCE_M,
			ceiling_safety_factor=CEILING_SAFETY_FACTOR,
			ceiling_margin=DESCENT_CEILING_MARGIN,
			near_field_height_m=NEAR_FIELD_HEIGHT_M,
			probe_only=HOVER_PROBE_ONLY,
			initial_thrust_gain=INITIAL_THRUST_GAIN,
			d_star_ramp_in_sec=D_STAR_RAMP_IN_SEC,
			center_to_probe_lateral_ramp_sec=CENTER_TO_PROBE_LATERAL_RAMP_SEC,
			center_lateral_p_scale=CENTER_LATERAL_P_SCALE,
			center_lateral_d_scale=CENTER_LATERAL_D_SCALE,
			probe_lateral_p_scale=PROBE_LATERAL_P_SCALE,
			probe_lateral_d_scale=PROBE_LATERAL_D_SCALE,
			stability_dt_sec=STABILITY_DT_SEC,
		)
		return MissionRoutine(**_supported_kwargs(MissionRoutine, values))

	# -------------------------- vision process --------------------------
	def _start_vision_worker(self):
		ctx = mp.get_context("spawn")
		self._vision_in_q = ctx.Queue(maxsize=VISION_INPUT_QUEUE_MAX)
		self._vision_out_q = ctx.Queue()
		self._vision_worker = ctx.Process(
			target=run_vision_worker, args=(self._vision_in_q, self._vision_out_q),
			name="bee_vision_worker", daemon=True)
		self._vision_worker.start()
		self._vision_stop = threading.Event()
		self._vision_thread = threading.Thread(
			target=self._vision_drain_loop, name="bee_vision_drain", daemon=True)
		self._vision_thread.start()

	def on_camera(self, msg: Image):
		start = time.perf_counter()
		receipt = self.time.receipt_stamp()
		stamp = self.time.image_stamp_sec(msg)
		if stamp <= 0.0:
			self.get_logger().warning("Dropping camera frame without a Gazebo SIM timestamp.")
			return
		try:
			frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
			frame = cv2.rotate(frame, cv2.ROTATE_180)
		except CvBridgeError as exc:
			self.get_logger().error(f"Camera conversion failed: {exc}")
			return
		if SHOW_CAMERA:
			cv2.imshow("BEE_LAND", frame)
			cv2.waitKey(1)
		self._latest_frame_receipt_wall = receipt.wall_sec
		self._latest_frame_receipt_mono = receipt.monotonic_sec
		payload = (frame, stamp, None, receipt.wall_sec, time.perf_counter())
		try:
			self._vision_in_q.put_nowait(payload)
		except queue.Full:
			try:
				self._vision_in_q.get_nowait()
				self._vision_dropped_frames += 1
			except queue.Empty:
				pass
			try:
				self._vision_in_q.put_nowait(payload)
			except queue.Full:
				self._vision_dropped_frames += 1
		self._vision_metrics["camera_callback_ms"] = 1000.0 * (time.perf_counter() - start)
		self._vision_metrics["camera_receipt_wall_timestamp_sec"] = receipt.wall_sec
		self._vision_metrics["camera_receipt_monotonic_timestamp_sec"] = receipt.monotonic_sec

	def _vision_drain_loop(self):
		while not self._vision_stop.is_set():
			try:
				result = self._vision_out_q.get(timeout=0.2)
			except queue.Empty:
				continue
			except (OSError, ValueError):
				break
			available_perf = time.perf_counter()
			self._vision_sequence += 1
			self._latest_target = result.target
			self._latest_flow = result.flow
			self._vision_metrics.update({
				"vision_worker_target_acquisition_ms": getattr(result, "target_acquisition_ms", None),
				"vision_worker_optical_flow_ms": getattr(result, "optical_flow_ms", None),
				"frame_to_result_ms": 1000.0 * (available_perf - getattr(result, "done_perf", available_perf))
					+ float(getattr(result, "ipc_in_ms", 0.0))
					+ float(getattr(result, "target_acquisition_ms", 0.0))
					+ float(getattr(result, "optical_flow_ms", 0.0)),
				"vision_dropped_frames": self._vision_dropped_frames,
			})

	# --------------------------- truth contact --------------------------
	def on_truth(self, msg: Float32Array):
		try:
			truth = decode_truth_array(msg.data)
		except ValueError as exc:
			self.get_logger().error(f"Truth schema mismatch: {exc}")
			return
		self._contact = ContactState(
			valid=bool(truth["truth_entities_ready"] > 0.5),
			sequence=int(round(truth["truth_sequence"])),
			sim_timestamp=float(truth["truth_sim_time_sec"]),
			left_contact=bool(truth["truth_left_contact"] > 0.5),
			right_contact=bool(truth["truth_right_contact"] > 0.5),
			any_contact=bool(truth["truth_any_contact"] > 0.5),
			confirmed=bool(truth["truth_contact_confirmed"] > 0.5),
		)
		if self._contact.confirmed and self._phase not in (LANDED, ABORTED):
			self._enter_landed("Gazebo truth contact confirmed")

	# ------------------------- mission/control --------------------------
	def on_supervisor_timer(self):
		if self.mavsdk.takeoff_error:
			self._abort(f"MAVSDK takeoff failed: {self.mavsdk.takeoff_error}")
			return
		if self._phase == WAIT_TAKEOFF and self.mavsdk.takeoff_done:
			self._phase = PRESTREAM
			self._phase_start_mono = self.time.monotonic_sec()
			self._log_event("takeoff_complete")
		elif self._phase == PRESTREAM:
			if self.time.monotonic_sec() - self._phase_start_mono >= OFFBOARD_PRESTREAM_SEC:
				self.px4.engage_offboard_mode()
				stamp = float(getattr(self._latest_flow, "timestamp", 0.0) or 0.0)
				self.mission.start(stamp, TAKEOFF_ALTITUDE_M)
				self.control_law.reset_visual_integrators()
				self._phase = CLOSED_LOOP
				self._log_event("closed_loop_start")

	def on_control_timer(self):
		if self._phase != CLOSED_LOOP or self._latest_flow is None:
			return
		flow_stamp = float(getattr(self._latest_flow, "timestamp", 0.0))
		if flow_stamp <= 0.0 or flow_stamp == self._last_controlled_flow_stamp:
			return
		self._last_controlled_flow_stamp = flow_stamp
		target_ok = bool(getattr(self._latest_target, "found", False))
		flow_ok = bool(getattr(self._latest_flow, "valid", False))
		now_mono = self.time.monotonic_sec()
		if not (target_ok and flow_ok):
			self._lost_target_since_mono = self._lost_target_since_mono or now_mono
			if now_mono - self._lost_target_since_mono >= LOST_TARGET_TIMEOUT_SEC:
				self._abort("target/flow timeout")
			return
		self._lost_target_since_mono = None

		dt = self._control_dt(flow_stamp)
		start = time.perf_counter()
		previous_thrust = float(self._latest_setpoint.thrust)
		tgt = self._latest_target
		mc = self.mission.update(
			flow_stamp, dt, previous_thrust,
			offset_x=float(tgt.offset_x), offset_y=float(tgt.offset_y),
			target_found=bool(tgt.found), area_fraction=float(tgt.area_fraction),
			fov_saturated=bool(tgt.fov_saturated),
		)
		if mc.info.get("event") in ("center_done", "final_probe_start"):
			self.control_law.reset_divergence_integral()
		if mc.substate == MISSION_INFEASIBLE:
			self._abort("mission feasibility gate rejected descent")
			return

		kwargs = dict(
			divergence_setpoint=mc.divergence_setpoint,
			thrust_gain_override=mc.thrust_gain_override,
			lateral_p_scale=mc.lateral_p_scale,
			lateral_d_scale=mc.lateral_d_scale,
			lateral_gain_scale=mc.lateral_p_scale,
			enable_integral=mc.enable_integral,
		)
		self._latest_setpoint = self.control_law.compute(
			self._latest_target, self._latest_flow, dt,
			**_supported_kwargs(self.control_law.compute, kwargs))
		elapsed_ms = 1000.0 * (time.perf_counter() - start)
		self._vision_metrics["control_compute_ms"] = elapsed_ms
		self._vision_metrics["control_dt_sim_sec"] = dt
		if self._latest_frame_receipt_mono is not None:
			self._vision_metrics["frame_to_command_ms"] = 1000.0 * (
				self.time.monotonic_sec() - self._latest_frame_receipt_mono)
		self._write_row(mc)

	def _control_dt(self, stamp):
		previous = self._previous_flow_stamp
		self._previous_flow_stamp = stamp
		if previous is None:
			return CAMERA_FRAME_PERIOD_SEC
		dt = stamp - previous
		return dt if 1e-4 < dt < 0.5 else CAMERA_FRAME_PERIOD_SEC

	def on_px4_timer(self):
		if self._phase == WAIT_TAKEOFF:
			return
		if self._phase in (LANDED,):
			self._latest_setpoint = self._zero_thrust()
		elif self._phase == ABORTED:
			self._latest_setpoint = self._neutral_hold()
		self._last_publish = self.px4.publish_cycle(self._latest_setpoint)

	# -------------------------- logging/events --------------------------
	def _mission_dict(self, mc):
		gate = getattr(self.mission, "gate", None)
		probe = getattr(self.mission, "probe_result", None)
		data = {
			"substate": getattr(mc, "substate", getattr(self.mission, "substate", "")),
			"divergence_setpoint_1_s": getattr(mc, "divergence_setpoint", None),
			"thrust_gain_k": getattr(mc, "thrust_gain_override", None),
			"lateral_p_scale": getattr(mc, "lateral_p_scale", None),
			"lateral_d_scale": getattr(mc, "lateral_d_scale", None),
			"enable_integral": int(bool(getattr(mc, "enable_integral", False))),
			"peak_accel_m_s2": getattr(probe, "peak_accel", None),
			"k_min": getattr(gate, "k_min", None),
			"h_crit_m": getattr(gate, "h_crit", None),
			"feasible": int(bool(getattr(gate, "feasible", False))) if gate is not None else None,
		}
		telemetry = getattr(self.mission, "probe_telemetry", None)
		if callable(telemetry):
			raw = telemetry()
			mapping = {
				"phase": "probe_phase", "accel": "probe_accel_m_s2",
				"mean_accel": "probe_mean_accel_m_s2",
				"residual_accel": "probe_residual_accel_m_s2",
				"percentile_accel": "probe_percentile_accel_m_s2",
			}
			for old, new in mapping.items():
				if old in raw:
					data[new] = raw[old]
		return data

	def _write_row(self, mc=None, event="", detail=""):
		self.diagnostics.write(
			target=self._latest_target, flow=self._latest_flow,
			setpoint=self._latest_setpoint,
			mission=self._mission_dict(mc) if mc is not None else None,
			timing=dict(self._vision_metrics), contact=self._contact,
			publish=self._last_publish, event=event, event_detail=detail,
			divergence_integral=getattr(self.control_law, "divergence_integral", None),
			vision_sequence=self._vision_sequence,
		)

	def _log_event(self, event, detail=""):
		self._write_row(event=event, detail=detail)

	def _enter_landed(self, reason):
		if self._phase == LANDED:
			return
		self._phase = LANDED
		self._latest_setpoint = self._zero_thrust()
		stamp = float(getattr(self._latest_flow, "timestamp", 0.0) or 0.0)
		if hasattr(self.mission, "mark_landed"):
			self.mission.mark_landed(stamp)
		self._log_event("landed", reason)
		if ENABLE_TOUCHDOWN_MOTOR_STOP and not self._motor_stop_requested:
			self._motor_stop_requested = True
			self.mavsdk.request_motor_stop()

	def _abort(self, reason):
		if self._phase == ABORTED:
			return
		self._phase = ABORTED
		self.get_logger().error(f"ABORT: {reason}")
		self._log_event("aborted", reason)

	def _latch_zero_thrust(self):
		self._latest_setpoint = self._zero_thrust()

	def _zero_thrust(self):
		return AttitudeSetpoint(
			timestamp=float(getattr(self._latest_flow, "timestamp", 0.0) or 0.0),
			roll=0.0, pitch=0.0, yaw=0.0, thrust=0.0)

	def _neutral_hold(self):
		return AttitudeSetpoint(
			timestamp=float(getattr(self._latest_flow, "timestamp", 0.0) or 0.0),
			roll=0.0, pitch=0.0, yaw=0.0, thrust=self.control_law.hover_thrust)

	def close(self):
		if self._shutdown:
			return
		self._shutdown = True
		self.mavsdk.request_stop()
		self._vision_stop.set()
		try:
			self._vision_in_q.put_nowait(None)
		except Exception:
			pass
		if self._vision_thread.is_alive():
			self._vision_thread.join(timeout=1.0)
		if self._vision_worker.is_alive():
			self._vision_worker.join(timeout=2.0)
		if self._vision_worker.is_alive():
			self._vision_worker.terminate()
		self.diagnostics.close()
		if SHOW_CAMERA:
			cv2.destroyAllWindows()


def main(args=None):
	rclpy.init(args=args)
	node = BeeLandNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.close()
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
