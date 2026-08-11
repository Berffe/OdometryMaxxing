"""Lean BEE_LAND ROS 2 controller node.

Live dependencies after takeoff:
- camera images -> vision worker -> mission/control;
- Gazebo truth contact confirmation -> terminal motor stop;
- PX4 receives attitude/thrust setpoints only.

No PX4 state, platform pose, clock fitting, position reconstruction or physical
truth enters the controller. Full Gazebo truth is written to its own paired CSV
by a non-blocking diagnostics sink; only the contact subset reaches control wiring.

What this file is, and is not
-----------------------------
This node is ROS I/O, process lifecycle, and wiring.  It deliberately no longer
contains:

* tuning constants -- see ``config.py``;
* the outer takeoff/offboard/handoff state machine -- see ``flight_sequencer.py``;
* any knowledge of mission phase names or mission log columns -- the mission
  owns both (``MissionRoutine.PHASES`` and ``MissionRoutine.telemetry()``);
* the mapping from mission events to control-law side effects -- phases now
  declare ``ControlEffect``s and this node only applies them.

The practical consequence: adding a mission phase, a scheduled gain or a log
column should require no edit here at all.  If it does, the seam is in the
wrong place.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from ros_gz_interfaces.msg import Float32Array
from px4_msgs.msg import VehicleAngularVelocity, VehicleStatus

from bee_control.core.clock import TimeManager
from bee_control.core.config import BeeConfig
from bee_control.control.control_law import ControlLaw
from bee_control.core.controller_state import ControllerState, PX4Status, VisionTelemetry
from bee_control.vision.derotation import AngularRateBuffer
from bee_control.diagnostics.diagnostics_writer import DiagnosticsWriter
from bee_control.interfaces.flight_sequencer import FlightSequencer, SequencerPorts, SetpointPolicy
from bee_control.interfaces.mavsdk_worker import MavsdkWorker
from bee_control.mission.routine import MissionRoutine
from bee_control.mission.types import ActuationFeedback, ControlEffect, MissionInputs
from bee_control.interfaces.px4_interface import PX4Interface
from bee_control.core.state import AttitudeSetpoint, ContactState
from bee_control.diagnostics.truth_layout import decode_truth_array
from bee_control.vision.vision_worker import run_vision_worker


class BeeLandNode(Node):
    def __init__(self, config: BeeConfig | None = None):
        super().__init__("bee_land_node")
        self.config = config or BeeConfig.default()
        cfg = self.config

        self.time = TimeManager(self)
        self.bridge = CvBridge()

        # --------------------------------------------------------- subsystems
        self.control_law = ControlLaw(
            roll_kp=cfg.control.roll_kp,
            roll_kd=cfg.control.roll_kd,
            pitch_kp=cfg.control.pitch_kp,
            pitch_kd=cfg.control.pitch_kd,
        )
        self.mission = MissionRoutine(
            hover_thrust=self.control_law.hover_thrust,
            config=cfg.mission,
        )
        self.px4_status = PX4Status(
            nav_state_offboard=cfg.scheduling.px4_nav_state_offboard,
            arming_state_armed=cfg.scheduling.px4_arming_state_armed,
        )
        self.vision_telemetry = VisionTelemetry()
        self.state = ControllerState(
            control_law=self.control_law,
            initial_setpoint=AttitudeSetpoint(thrust=self.control_law.hover_thrust),
        )

        # Mission events map to control-law side effects HERE and only here.
        # A new effect is one entry in this table; the branching logic that used
        # to live in on_control_timer is gone.
        self._control_effects = {
            ControlEffect.RESET_DIVERGENCE_INTEGRAL:
                self.control_law.reset_divergence_integral,
            ControlEffect.RESET_VISUAL_INTEGRATORS:
                self.control_law.reset_visual_integrators,
        }

        # ------------------------------------------------------- diagnostics
        # Column order is the registration order. Each source owns its own
        # names; this node declares none of them.
        self.diagnostics = DiagnosticsWriter(
            sources=[
                self.px4_status,
                self.state,
                self.mission,
                self.vision_telemetry,
            ],
            output_dir="logs",
        )

        # ------------------------------------------------------ control state
        self._last_controlled_flow_stamp = None
        self._previous_flow_stamp = None
        self._lost_target_since_mono = None
        self._previous_camera_stamp = None
        self._vision_dropped_frames = 0
        self._latest_frame_receipt_wall = None
        self._latest_frame_receipt_mono = None
        self._angular_rates = AngularRateBuffer(maxlen=cfg.vision.angular_rate_buffer_len)
        # One atomic bundle prevents a control tick from combining a new target
        # with an old flow result while the drain thread is publishing updates.
        self._latest_vision_bundle = None
        self._last_mission_substate = None
        self._motor_stop_requested = False
        self._undeclared_timing_logged = False
        self._shutdown = False

        # --------------------------------------------------------------- ROS
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
        self.create_subscription(Image, cfg.topics.camera, self.on_camera, sensor_qos)
        self.create_subscription(
            Float32Array, cfg.topics.truth, self.on_truth, sensor_qos)
        self._angular_rate_subscriptions = [
            self.create_subscription(
                VehicleAngularVelocity, topic, self.on_angular_velocity, px4_qos)
            for topic in cfg.topics.angular_rate
        ]
        self._vehicle_status_subscriptions = [
            self.create_subscription(
                VehicleStatus, topic, self.on_vehicle_status, px4_qos)
            for topic in cfg.topics.vehicle_status
        ]

        self.mavsdk = MavsdkWorker(
            logger=self.get_logger(),
            on_pre_motor_stop=self._latch_zero_thrust,
            system_address=cfg.mavsdk.system_address,
            port_to_free=cfg.mavsdk.port_to_free,
            takeoff_altitude_m=cfg.mavsdk.takeoff_altitude_m,
            connect_timeout_sec=cfg.mavsdk.connect_timeout_sec,
            health_timeout_sec=cfg.mavsdk.health_timeout_sec,
            takeoff_altitude_timeout_sec=cfg.mavsdk.takeoff_altitude_timeout_sec,
            ekf2_settle_time_sec=cfg.mavsdk.ekf2_settle_time_sec,
            enable_kill_fallback=cfg.mavsdk.enable_kill_fallback,
        )

        self.sequencer = FlightSequencer(
            cfg, self.px4_status, self._build_sequencer_ports())

        self._start_vision_worker()
        self.create_timer(cfg.scheduling.control_period_sec, self.on_control_timer)
        self.create_timer(cfg.scheduling.px4_setpoint_period_sec, self.on_px4_timer)
        self.create_timer(cfg.scheduling.supervisor_period_sec, self.on_supervisor_timer)
        self.mavsdk.start()

        self.get_logger().info(
            "BEE_LAND lean controller started: camera-only control, "
            "truth-contact-only touchdown.")
        self.get_logger().info(f"Controller log: {self.diagnostics.filepath}")
        self.get_logger().info(f"Gazebo truth log: {self.diagnostics.truth_filepath}")
        self.get_logger().info(
            f"Controller schema {self.diagnostics.schema_version} "
            f"({len(self.diagnostics.fieldnames)} columns)")

    # ---------------------------------------------------------------- wiring
    def _build_sequencer_ports(self) -> SequencerPorts:
        return SequencerPorts(
            takeoff_done=lambda: bool(self.mavsdk.takeoff_done),
            takeoff_error=lambda: self.mavsdk.takeoff_error,
            request_offboard=self.px4.engage_offboard_mode,
            latest_vision=lambda: self._latest_vision_bundle,
            log_info=self.get_logger().info,
            log_error=self.get_logger().error,
            log_event=self._log_event,
            on_handoff=self._on_handoff,
            on_landed=self._on_landed,
            on_aborted=self._on_aborted,
            monotonic=self.time.monotonic_sec,
        )

    def _on_handoff(self, flow_stamp: float) -> None:
        """Arm the visual controller. Called once, by the sequencer."""
        self.mission.start(flow_stamp, self.config.mavsdk.takeoff_altitude_m)
        self.control_law.reset_visual_integrators()
        self._last_mission_substate = None
        self._previous_flow_stamp = None
        self._last_controlled_flow_stamp = flow_stamp

    def _on_landed(self, reason: str) -> None:
        self.state.setpoint = self.state.zero_thrust()
        self.mission.mark_landed(self.state.flow_timestamp)
        previous = self._last_mission_substate or "none"
        self._last_mission_substate = self.mission.substate
        self.get_logger().info(
            f"MISSION PHASE: LANDED ({previous} -> {self.mission.substate})")
        if self.config.mavsdk.enable_touchdown_motor_stop and not self._motor_stop_requested:
            self._motor_stop_requested = True
            self.mavsdk.request_motor_stop()

    def _on_aborted(self, reason: str) -> None:
        self.mission.mark_aborted(self.state.flow_timestamp)
        self._last_mission_substate = self.mission.substate

    def _latch_zero_thrust(self) -> None:
        self.state.setpoint = self.state.zero_thrust()

    # -------------------------------------------------------- vision process
    def _start_vision_worker(self):
        ctx = mp.get_context("spawn")
        self._vision_in_q = ctx.Queue(maxsize=self.config.vision.input_queue_max)
        self._vision_out_q = ctx.Queue()
        self._vision_worker = ctx.Process(
            target=run_vision_worker,
            args=(self._vision_in_q, self._vision_out_q,
                  self.config.vision.enable_derotation),
            name="bee_vision_worker", daemon=True)
        self._vision_worker.start()
        self._vision_stop = threading.Event()
        self._vision_thread = threading.Thread(
            target=self._vision_drain_loop, name="bee_vision_drain", daemon=True)
        self._vision_thread.start()

    def on_camera(self, msg: Image):
        callback_start_perf = time.perf_counter()
        receipt = self.time.receipt_stamp()
        frame_receipt_perf = time.perf_counter()
        stamp = self.time.image_stamp_sec(msg)
        if stamp <= 0.0:
            self.get_logger().warning(
                "Dropping camera frame without a Gazebo SIM timestamp.")
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        except CvBridgeError as exc:
            self.get_logger().error(f"Camera conversion failed: {exc}")
            return
        if self.config.vision.show_camera:
            cv2.imshow("BEE_LAND", frame)
            cv2.waitKey(1)

        self._latest_frame_receipt_wall = receipt.wall_sec
        self._latest_frame_receipt_mono = receipt.monotonic_sec

        frame_dt = (
            stamp - self._previous_camera_stamp
            if self._previous_camera_stamp is not None
            else self.config.camera.frame_period_sec
        )
        self._previous_camera_stamp = stamp
        body_rates = None
        if self.config.vision.enable_derotation:
            mean_rates, _n_rates, rates_valid = self._angular_rates.mean_recent(frame_dt)
            if rates_valid:
                body_rates = tuple(float(x) for x in mean_rates)

        def make_payload():
            ship_perf = time.perf_counter()
            camera_callback_ms = 1000.0 * (ship_perf - callback_start_perf)
            return (
                frame, stamp, body_rates, receipt.wall_sec, receipt.monotonic_sec,
                frame_receipt_perf, ship_perf, camera_callback_ms,
            )

        payload = make_payload()
        queued = False
        try:
            self._vision_in_q.put_nowait(payload)
            queued = True
        except queue.Full:
            try:
                self._vision_in_q.get_nowait()
                self._vision_dropped_frames += 1
            except queue.Empty:
                pass
            try:
                # Re-stamp the actual successful send after queue cleanup so
                # the inbound IPC duration starts at the correct put attempt.
                payload = make_payload()
                self._vision_in_q.put_nowait(payload)
                queued = True
            except queue.Full:
                self._vision_dropped_frames += 1

        self.vision_telemetry.update({
            "camera_callback_ms": payload[-1] if queued else 1000.0 * (
                time.perf_counter() - callback_start_perf),
            "camera_receipt_wall_timestamp_sec": receipt.wall_sec,
            "camera_receipt_monotonic_timestamp_sec": receipt.monotonic_sec,
        })

    def _vision_drain_loop(self):
        while not self._vision_stop.is_set():
            try:
                result = self._vision_out_q.get(timeout=0.2)
            except queue.Empty:
                continue
            except (OSError, ValueError):
                break
            available_perf = time.perf_counter()
            self.state.vision_sequence += 1

            done_perf = float(getattr(result, "done_perf", available_perf))
            receipt_perf = float(getattr(result, "frame_receipt_perf", available_perf))
            ship_perf = float(getattr(result, "ship_perf", receipt_perf))
            ipc_in_ms = float(getattr(result, "ipc_in_ms", 0.0))
            ipc_out_ms = 1000.0 * max(0.0, available_perf - done_perf)

            metrics = {
                "camera_receipt_wall_timestamp_sec": getattr(result, "frame_wall", None),
                "camera_receipt_monotonic_timestamp_sec": getattr(
                    result, "frame_monotonic", None),
                "camera_callback_ms": getattr(result, "camera_callback_ms", None),
                "camera_prequeue_ms": 1000.0 * max(0.0, ship_perf - receipt_perf),
                "vision_worker_target_acquisition_ms": getattr(
                    result, "target_acquisition_ms", None),
                "vision_worker_optical_flow_ms": getattr(
                    result, "optical_flow_ms", None),
                "vision_ipc_in_ms": ipc_in_ms,
                "vision_ipc_out_ms": ipc_out_ms,
                "vision_transport_total_ms": ipc_in_ms + ipc_out_ms,
                # Parent-process receipt and availability stamps belong to the
                # same source frame, so dropped/newer callbacks cannot bias it.
                "frame_to_result_ms": 1000.0 * max(0.0, available_perf - receipt_perf),
                "source_frame_receipt_perf_sec": receipt_perf,
                "vision_dropped_frames": self._vision_dropped_frames,
            }
            flow_internal = getattr(result, "optical_flow_timing", {}) or {}
            for key, value in flow_internal.items():
                metrics[f"optical_flow_{key}"] = value

            self._latest_vision_bundle = (result.target, result.flow, metrics)
            self.state.target = result.target
            self.state.flow = result.flow
            self.vision_telemetry.update(metrics)

    # ----------------------------------------------------------- truth/status
    def on_truth(self, msg: Float32Array):
        try:
            truth = decode_truth_array(msg.data)
        except ValueError as exc:
            self.get_logger().error(f"Truth schema mismatch: {exc}")
            return
        receipt = self.time.receipt_stamp()
        self.diagnostics.write_truth(
            truth,
            receipt_wall_sec=receipt.wall_sec,
            receipt_monotonic_sec=receipt.monotonic_sec,
        )
        self.state.contact = ContactState(
            valid=bool(truth["truth_entities_ready"] > 0.5),
            sequence=int(round(truth["truth_sequence"])),
            sim_timestamp=float(truth["truth_sim_time_sec"]),
            left_contact=bool(truth["truth_left_contact"] > 0.5),
            right_contact=bool(truth["truth_right_contact"] > 0.5),
            any_contact=bool(truth["truth_any_contact"] > 0.5),
            confirmed=bool(truth["truth_contact_confirmed"] > 0.5),
        )
        if self.state.contact.confirmed and not self.sequencer.is_terminal:
            self.sequencer.enter_landed("Gazebo truth contact confirmed")

    def on_angular_velocity(self, msg: VehicleAngularVelocity):
        """Buffer PX4 body FRD rates using the message's own PX4 timestamp."""
        stamp_sec = float(getattr(msg, "timestamp", 0)) * 1e-6
        xyz = getattr(msg, "xyz", None)
        if xyz is not None and len(xyz) >= 3:
            self._angular_rates.add(stamp_sec, xyz)

    def on_vehicle_status(self, msg: VehicleStatus):
        if self.px4_status.update(msg.nav_state, msg.arming_state, msg.failsafe):
            detail = self.px4_status.describe()
            self.get_logger().info(f"PX4 status: {detail}")
            self._log_event("px4_status", detail)

    # ------------------------------------------------------- mission/control
    def on_supervisor_timer(self):
        self.sequencer.update()

    def on_control_timer(self):
        if not self.sequencer.is_closed_loop:
            return
        bundle = self._latest_vision_bundle
        if bundle is None:
            return
        target, flow, result_metrics = bundle
        flow_stamp = float(getattr(flow, "timestamp", 0.0))
        if flow_stamp <= 0.0 or flow_stamp == self._last_controlled_flow_stamp:
            return
        self._last_controlled_flow_stamp = flow_stamp

        now_mono = self.time.monotonic_sec()
        if not (bool(getattr(target, "found", False))
                and bool(getattr(flow, "valid", False))):
            self._lost_target_since_mono = self._lost_target_since_mono or now_mono
            if (now_mono - self._lost_target_since_mono
                    >= self.config.vision.lost_target_timeout_sec):
                self.sequencer.abort("target/flow timeout")
            return
        self._lost_target_since_mono = None

        dt = self._control_dt(flow_stamp)
        start = time.perf_counter()

        mc = self.mission.update(MissionInputs(
            t=flow_stamp,
            dt=dt,
            target=target,
            flow=flow,
            actuation=ActuationFeedback.from_control_law(
                self.control_law, self.state.setpoint.thrust),
        ))
        self._announce_mission_substate(mc)

        # Apply whatever side effects the phase asked for. This node does not
        # know, and must not know, which phases want which effect.
        for effect in mc.effects:
            action = self._control_effects.get(effect)
            if action is not None:
                action()

        # An infeasible landing is not a controller abort. MissionRoutine
        # latches an active visual hover with D*=0 and a near-field-admissible
        # gain, so keep running the normal control path. This preserves vertical
        # platform tracking and lateral centering instead of freezing the last
        # command or dropping to the outer ABORTED neutral hold.
        self.state.setpoint = self.control_law.compute(
            target, flow, dt, **mc.control_kwargs())

        self.state.target = target
        self.state.flow = flow
        self.vision_telemetry.update(result_metrics)
        self.vision_telemetry.update({
            "control_compute_ms": 1000.0 * (time.perf_counter() - start),
            "control_dt_sim_sec": dt,
        })
        source_receipt_perf = result_metrics.get("source_frame_receipt_perf_sec")
        if source_receipt_perf is not None:
            self.vision_telemetry.set("frame_to_command_ms", 1000.0 * max(
                0.0, time.perf_counter() - float(source_receipt_perf)))
        self._warn_once_on_timing_drift()
        self._write_row()

    def _control_dt(self, stamp):
        previous = self._previous_flow_stamp
        self._previous_flow_stamp = stamp
        if previous is None:
            return self.config.camera.frame_period_sec
        dt = stamp - previous
        return dt if 1e-4 < dt < 0.5 else self.config.camera.frame_period_sec

    def on_px4_timer(self):
        """Publish on cadence. The sequencer decides WHO may command."""
        policy = self.sequencer.setpoint_policy
        if policy is SetpointPolicy.INHIBIT:
            return
        if policy is SetpointPolicy.ZERO_THRUST:
            self.state.setpoint = self.state.zero_thrust()
        elif policy is SetpointPolicy.NEUTRAL_HOLD:
            self.state.setpoint = self.state.neutral_hold()
        self.state.publish = self.px4.publish_cycle(self.state.setpoint)

    # ------------------------------------------------------- logging/events
    def _write_row(self, event: str = "", detail: str = ""):
        self.diagnostics.write(
            event=event,
            event_detail=detail,
            controller_phase=self.sequencer.phase,
        )

    def _log_event(self, event: str, detail: str = ""):
        self._write_row(event=event, detail=detail)

    def _warn_once_on_timing_drift(self):
        """Report timing keys the schema never declared, once per run.

        Unlike the mission source, an unexpected timing key must not raise: it
        arrives at 60 Hz from a separate process and is diagnostic only. But it
        should not vanish in silence either, which is what used to happen.
        """
        if self._undeclared_timing_logged:
            return
        undeclared = self.vision_telemetry.drain_undeclared()
        if undeclared:
            self._undeclared_timing_logged = True
            self.get_logger().warning(
                "Vision timing keys not present in the log schema (add them to "
                f"OpticalFlowEstimator.TIMING_FIELDS): {', '.join(undeclared)}")

    def _announce_mission_substate(self, mc):
        """Announce and log MissionRoutine substate transitions exactly once.

        Controller phases describe the outer PX4 / handoff lifecycle. Mission
        substates describe the visual landing sequence inside CLOSED_LOOP.
        They are intentionally tracked separately.
        """
        substate = str(mc.substate or "")
        if not substate or substate == self._last_mission_substate:
            return
        previous = self._last_mission_substate or "none"
        self._last_mission_substate = substate

        spec = self.mission.PHASES.get(substate)
        display = spec.display_name if spec is not None else substate.upper()

        detail = f"{previous} -> {substate}; {mc.summary()}"
        reason = (mc.info or {}).get("infeasible_reason")
        if reason:
            detail += f"; {reason}"
        self.get_logger().info(f"MISSION PHASE: {display} ({detail})")

        # Preserve the exact MissionControl that caused the transition in the
        # event row. This makes phase boundaries directly recoverable offline.
        self._write_row(event="mission_phase_transition", detail=detail)

    # --------------------------------------------------------------- teardown
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
        if self.config.vision.show_camera:
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
