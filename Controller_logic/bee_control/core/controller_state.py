"""Live controller snapshot, and the telemetry sources built on it.

``bee_node`` used to hold ten loose ``_latest_*`` attributes and a 100-line
``_mission_dict``/``_write_row`` pair that hand-transcribed them into CSV
columns.  Those attributes are now three small objects that each own the
columns they produce:

``PX4Status``       nav/arming/failsafe, and the derived offboard-confirmed flag
``VisionTelemetry`` the per-frame timing dictionary
``ControllerState`` latest target / flow / setpoint / contact / publish receipt

All three are plain Python and ROS-free, so the flight sequencer and the tests
can read them without a running DDS stack.

Column names are byte-identical to the previous revision, so ``analyse_log.py``
keeps working unchanged.
"""
from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional, Sequence

from bee_control.core.state import AttitudeSetpoint, ContactState, FlowResult, TargetEstimate


def _blank(value):
    """None -> empty CSV cell. A blank is a gap; 0.0 would be a fiction."""
    return "" if value is None else value


def _flag(value):
    return "" if value is None else int(bool(value))


class PX4Status:
    """PX4 vehicle-status subset the controller is allowed to see.

    This is status only -- no PX4 position, velocity or attitude estimate ever
    reaches the control law. It exists to decide whether OFFBOARD is actually
    engaged, and to make that decision auditable in the log.
    """

    TELEMETRY_PREFIX: ClassVar[str] = "px4"

    def __init__(self, *, nav_state_offboard: int, arming_state_armed: int):
        self._nav_state_offboard = int(nav_state_offboard)
        self._arming_state_armed = int(arming_state_armed)
        self.nav_state: Optional[int] = None
        self.arming_state: Optional[int] = None
        self.failsafe: Optional[bool] = None
        self.offboard_confirmed: bool = False

    @classmethod
    def telemetry_fields(cls) -> Sequence[str]:
        return ("nav_state", "arming_state", "failsafe", "offboard_confirmed")

    def telemetry(self) -> Mapping[str, Any]:
        return {
            "nav_state": _blank(self.nav_state),
            "arming_state": _blank(self.arming_state),
            "failsafe": _flag(self.failsafe),
            "offboard_confirmed": _flag(self.offboard_confirmed),
        }

    def update(self, nav_state: int, arming_state: int, failsafe: bool) -> bool:
        """Absorb a VehicleStatus message. True if anything actually changed."""
        previous = (self.nav_state, self.arming_state, self.failsafe)
        self.nav_state = int(nav_state)
        self.arming_state = int(arming_state)
        self.failsafe = bool(failsafe)
        self.offboard_confirmed = (
            self.nav_state == self._nav_state_offboard
            and self.arming_state == self._arming_state_armed
            and not self.failsafe
        )
        return (self.nav_state, self.arming_state, self.failsafe) != previous

    def describe(self) -> str:
        return (
            f"nav_state={self.nav_state}, arming_state={self.arming_state}, "
            f"failsafe={int(bool(self.failsafe))}"
        )


class VisionTelemetry:
    """The per-frame timing dictionary, and its schema.

    The optical-flow stage timings are NOT enumerated here: they come from
    ``OpticalFlowEstimator.TIMING_FIELDS``, so adding a stage timing to the
    algorithm makes it appear in the log with no edit in this file.

    Unlike the mission source, this one never raises on an unexpected key.  A
    timing dictionary arrives at 60 Hz from a separate process; an unknown key
    is a schema drift worth one warning, not a reason to take down a vehicle
    mid-descent.  ``drain_undeclared()`` hands those keys to the node once.
    """

    TELEMETRY_PREFIX: ClassVar[str] = "timing"

    #: Values measured by the node / vision transport itself.
    TRANSPORT_FIELDS: ClassVar[tuple[str, ...]] = (
        "camera_receipt_wall_timestamp_sec",
        "camera_receipt_monotonic_timestamp_sec",
        "camera_callback_ms", "camera_prequeue_ms",
        "vision_worker_target_acquisition_ms",
        "vision_worker_optical_flow_ms",
        "vision_ipc_in_ms", "vision_ipc_out_ms",
        "vision_transport_total_ms",
        "frame_to_result_ms", "frame_to_command_ms",
        "control_compute_ms", "control_dt_sim_sec",
        "vision_dropped_frames",
        # Reserved for the touchdown path; unpopulated today but kept so the
        # column set stays stable for analyse_log.py.
        "motor_stop_request_to_pickup_ms", "motor_stop_total_ms",
    )

    #: Values that are useful to the node but are not log columns.
    INTERNAL_KEYS: ClassVar[frozenset] = frozenset({"source_frame_receipt_perf_sec"})

    def __init__(self):
        self._metrics: dict[str, Any] = {}
        self._undeclared: set[str] = set()
        self._declared = frozenset(self.telemetry_fields())

    @classmethod
    def telemetry_fields(cls) -> Sequence[str]:
        # Imported HERE, not at module scope, so ``core`` keeps its rule of
        # importing nothing else in the package (see bee_control/__init__.py).
        # The dependency is real but it is a HEADER-TIME one: the estimator owns
        # its stage-timing names, and this is the only place they are needed.
        # Deferring it also keeps ``core`` free of cv2/numpy at import time.
        from bee_control.vision.optical_flow import OpticalFlowEstimator

        return tuple(cls.TRANSPORT_FIELDS) + tuple(
            f"optical_flow_{name}" for name in OpticalFlowEstimator.TIMING_FIELDS
        )

    def telemetry(self) -> Mapping[str, Any]:
        return {k: v for k, v in self._metrics.items() if k in self._declared}

    def update(self, metrics: Mapping[str, Any]) -> None:
        for key, value in metrics.items():
            if key in self._declared or key in self.INTERNAL_KEYS:
                self._metrics[key] = value
            else:
                self._undeclared.add(key)

    def set(self, key: str, value: Any) -> None:
        self.update({key: value})

    def get(self, key: str, default=None):
        return self._metrics.get(key, default)

    def drain_undeclared(self) -> Sequence[str]:
        """Return and clear keys seen but never declared, for a one-off warning."""
        found = sorted(self._undeclared)
        self._undeclared.clear()
        return found


class ControllerState:
    """Everything the visual controller currently knows and last commanded.

    Ownership note: this holds ONLY visual measurements, the command formed
    from them, and the minimal Gazebo contact latch. Simulation position,
    velocity, height and platform motion deliberately do not appear -- they
    belong to the independent truth log (see ``state.py``).
    """

    TELEMETRY_PREFIX: ClassVar[str] = ""

    def __init__(self, *, control_law, initial_setpoint: AttitudeSetpoint):
        # Read for `command_thrust_integral` only. Held rather than pushed so
        # the node does not have to remember to forward it every row.
        self._control_law = control_law
        self.target: TargetEstimate = TargetEstimate()
        self.flow: Optional[FlowResult] = None
        self.setpoint: AttitudeSetpoint = initial_setpoint
        self.contact: ContactState = ContactState()
        self.publish = None
        self.vision_sequence: int = 0

    # ---------------------------------------------------------------- schema
    @classmethod
    def telemetry_fields(cls) -> Sequence[str]:
        return (
            "vision_sequence", "vision_sim_timestamp_sec",
            "target_found", "target_offset_x", "target_offset_y",
            "target_detection_width_px", "target_detection_height_px",
            "target_confidence", "target_area_fraction", "target_fov_saturated",
            "flow_sim_timestamp_sec", "flow_valid",
            "flow_mean_x_norm_s", "flow_mean_y_norm_s",
            "flow_mean_x_px_s", "flow_mean_y_px_s",
            "flow_divergence_1_s", "flow_raw_divergence_1_s", "flow_fit_quality",
            "flow_derotated", "flow_mean_x_raw_px_s", "flow_mean_y_raw_px_s",
            "flow_divergence_prederotation_1_s",
            "flow_roi_x0", "flow_roi_y0", "flow_roi_x1", "flow_roi_y1",
            "command_source_sim_timestamp_sec",
            "command_roll_rad", "command_pitch_rad", "command_yaw_rad",
            "command_thrust", "command_thrust_integral",
            "contact_valid", "contact_truth_sequence",
            "contact_truth_sim_timestamp_sec",
            "contact_left", "contact_right", "contact_any", "contact_confirmed",
            "px4_publish_sequence", "px4_publish_wall_timestamp_sec",
            "px4_publish_monotonic_timestamp_sec",
        )

    def telemetry(self) -> Mapping[str, Any]:
        row: dict[str, Any] = {"vision_sequence": self.vision_sequence}
        target, flow = self.target, self.flow
        if target is not None:
            row.update({
                "vision_sim_timestamp_sec": _blank(target.timestamp),
                "target_found": int(bool(target.found)),
                "target_offset_x": target.offset_x,
                "target_offset_y": target.offset_y,
                "target_detection_width_px": target.detection_width,
                "target_detection_height_px": target.detection_height,
                "target_confidence": target.confidence,
                "target_area_fraction": target.area_fraction,
                "target_fov_saturated": int(bool(target.fov_saturated)),
            })
        if flow is not None:
            row.update({
                "flow_sim_timestamp_sec": _blank(flow.timestamp),
                "flow_valid": int(bool(flow.valid)),
                "flow_mean_x_norm_s": flow.mean_flow_x_norm,
                "flow_mean_y_norm_s": flow.mean_flow_y_norm,
                "flow_mean_x_px_s": flow.mean_flow_x,
                "flow_mean_y_px_s": flow.mean_flow_y,
                "flow_divergence_1_s": flow.divergence,
                "flow_raw_divergence_1_s": flow.raw_divergence,
                "flow_fit_quality": flow.fit_quality,
                "flow_derotated": int(bool(flow.derotated)),
                "flow_mean_x_raw_px_s": flow.mean_flow_x_raw,
                "flow_mean_y_raw_px_s": flow.mean_flow_y_raw,
                "flow_divergence_prederotation_1_s": flow.divergence_prederotation,
                "flow_roi_x0": flow.roi_x0, "flow_roi_y0": flow.roi_y0,
                "flow_roi_x1": flow.roi_x1, "flow_roi_y1": flow.roi_y1,
            })
        setpoint = self.setpoint
        if setpoint is not None:
            row.update({
                "command_source_sim_timestamp_sec": _blank(setpoint.timestamp),
                "command_roll_rad": setpoint.roll,
                "command_pitch_rad": setpoint.pitch,
                "command_yaw_rad": setpoint.yaw,
                "command_thrust": setpoint.thrust,
                "command_thrust_integral": _blank(
                    getattr(self._control_law, "divergence_integral", None)),
            })
        contact = self.contact
        if contact is not None:
            row.update({
                "contact_valid": int(bool(contact.valid)),
                "contact_truth_sequence": contact.sequence,
                "contact_truth_sim_timestamp_sec": _blank(contact.sim_timestamp),
                "contact_left": int(bool(contact.left_contact)),
                "contact_right": int(bool(contact.right_contact)),
                "contact_any": int(bool(contact.any_contact)),
                "contact_confirmed": int(bool(contact.confirmed)),
            })
        publish = self.publish
        if publish is not None:
            row.update({
                "px4_publish_sequence": publish.sequence,
                "px4_publish_wall_timestamp_sec": publish.wall_timestamp_sec,
                "px4_publish_monotonic_timestamp_sec": publish.monotonic_timestamp_sec,
            })
        return row

    # ----------------------------------------------------------- convenience
    @property
    def flow_timestamp(self) -> float:
        """Latest flow SIM timestamp, 0.0 if none. Never a wall clock."""
        return float(getattr(self.flow, "timestamp", 0.0) or 0.0)

    def zero_thrust(self) -> AttitudeSetpoint:
        return AttitudeSetpoint(
            timestamp=self.flow_timestamp, roll=0.0, pitch=0.0, yaw=0.0, thrust=0.0)

    def neutral_hold(self) -> AttitudeSetpoint:
        return AttitudeSetpoint(
            timestamp=self.flow_timestamp, roll=0.0, pitch=0.0, yaw=0.0,
            thrust=self._control_law.hover_thrust)
