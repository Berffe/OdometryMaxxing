"""All BEE_LAND tuning, in one place, as frozen dataclasses.

This replaces the ~100-line module-level constant block that used to sit at the
top of ``bee_node`` and the ``_supported_kwargs`` inspection hack that shipped
it into ``MissionRoutine``.

Why this matters beyond tidiness
--------------------------------
``_supported_kwargs`` filtered the mission kwargs through
``inspect.signature``.  Rename a parameter inside ``mission_routine`` and the
matching value from ``bee_node`` was silently discarded -- the vehicle then
flew the library DEFAULT with no warning, no log row, and no way to tell from
the CSV.  For a safety-gated descent that is the wrong failure direction.  A
dataclass field either exists or raises at construction.

Three groups, three owners
--------------------------
* ``SchedulingConfig`` / ``TopicsConfig`` / ``VisionConfig``: node plumbing.
* ``CameraConfig`` / ``ControlConfig`` / ``MissionConfig``: flight tuning.
* ``MavsdkConfig``: the takeoff/termination side channel.

Derived quantities (``roll_kappa``, ``stability_dt_sec``, the probe time
constants scaled off the platform period) are computed once in
``BeeConfig.default()`` so the relationship between them stays visible instead
of being re-derived at three call sites.

Overriding
----------
Every config is a frozen dataclass, so ``dataclasses.replace`` is the override
mechanism::

    cfg = BeeConfig.default()
    cfg = replace(cfg, mission=replace(cfg.mission, ceiling_margin=0.75))

``BeeConfig.from_ros_parameters(node)`` is the hook for declaring these as ROS
parameters later; it is deliberately the ONLY place that would need to know
about ROS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping


# --------------------------------------------------------------------------
# Node plumbing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TopicsConfig:
    camera: str = "/bee_x500/camera/image"
    truth: str = "/bee_land/truth"
    # PX4 renames these across releases; every candidate is subscribed and the
    # one that exists wins. Order is newest-first, purely cosmetic.
    vehicle_status: tuple[str, ...] = (
        "/fmu/out/vehicle_status_v4",
        "/fmu/out/vehicle_status_v1",
        "/fmu/out/vehicle_status",
    )
    angular_rate: tuple[str, ...] = (
        "/fmu/out/vehicle_angular_velocity_v1",
        "/fmu/out/vehicle_angular_velocity",
    )


@dataclass(frozen=True)
class SchedulingConfig:
    control_period_sec: float = 0.005
    px4_setpoint_period_sec: float = 0.01
    supervisor_period_sec: float = 0.1
    post_landing_log_period_sec: float = 0.10

    offboard_prestream_sec: float = 2.0
    px4_offboard_switch_settle_sec: float = 0.5
    px4_offboard_confirm_timeout_sec: float = 5.0
    px4_offboard_reengage_interval_sec: float = 0.5

    # Run the ROS timers on CLOCK_MONOTONIC instead of the system clock.
    #
    # A system-clock STEP (VM host time sync -- WSL2, Hyper-V) moves every
    # pending timer deadline by the size of the step. A 2 s backward step
    # stalls the control and setpoint timers for 2 s, which is four times
    # PX4's COM_OF_LOSS_T, and the vehicle drops to its offboard-loss
    # failsafe. Observed three times per flight before this was fixed.
    #
    # Leave True. False reproduces the pre-fix behaviour for A/B testing only.
    use_steady_timers: bool = True

    # Same reasoning applied to outgoing PX4 message timestamps: project the
    # Unix epoch off the monotonic clock so uORB stamps cannot jump backwards.
    use_steady_wall_clock: bool = True

    # A skew change larger than this between consecutive supervisor ticks is
    # reported as a discontinuity rather than drift.
    clock_step_threshold_sec: float = 0.05

    # uORB enum values, not tuning. Here so nothing else hardcodes them.
    px4_nav_state_offboard: int = 14
    px4_arming_state_armed: int = 2


@dataclass(frozen=True)
class VisionConfig:
    # One-line de-rotation toggle. False reproduces the exact legacy path.
    enable_derotation: bool = True
    input_queue_max: int = 2
    lost_target_timeout_sec: float = 2.0
    show_camera: bool = False
    angular_rate_buffer_len: int = 256


@dataclass(frozen=True)
class CameraConfig:
    horizontal_fov_deg: float = 80.0
    vertical_fov_deg: float = 80.0
    frame_period_sec: float = 1.0 / 60.0
    # Retained design target, not a measurement: the latency the stability
    # margin is sized against.
    processing_latency_budget_sec: float = 0.02
    # Command-shaping filter + slew group delay, lumped.
    smoothing_delay_sec: float = 0.03

    @property
    def horizontal_fov_rad(self) -> float:
        return math.radians(self.horizontal_fov_deg)

    @property
    def roll_kappa(self) -> float:
        """Normalized-flow -> angular-rate conversion for the roll axis."""
        return 1.0 / math.tan(math.radians(self.horizontal_fov_deg / 2.0))

    @property
    def pitch_kappa(self) -> float:
        return 1.0 / math.tan(math.radians(self.vertical_fov_deg / 2.0))

    def stability_dt_sec(self, scheduling: SchedulingConfig) -> float:
        """Total visual-loop delay the de Croon stability ceiling is built on.

        Frame period + vision compute budget + setpoint publication period +
        command smoothing. This is NOT the control tick: the ceiling is set by
        how stale the newest measurement can be, not by how often we act.
        """
        return (
            self.frame_period_sec
            + self.processing_latency_budget_sec
            + scheduling.px4_setpoint_period_sec
            + self.smoothing_delay_sec
        )


@dataclass(frozen=True)
class ControlConfig:
    """Constant PD gains handed to ``ControlLaw`` (acceleration domain)."""

    roll_kp: float = 5.0
    roll_kd: float = 3.5
    pitch_kp: float = 5.0
    pitch_kd: float = 3.5


@dataclass(frozen=True)
class MavsdkConfig:
    system_address: str = "udpin://0.0.0.0:14540"
    port_to_free: int = 14540
    takeoff_altitude_m: float = 5.0
    ekf2_settle_time_sec: float = 5.0
    connect_timeout_sec: float = 15.0
    health_timeout_sec: float = 30.0
    takeoff_altitude_timeout_sec: float = 130.0
    enable_touchdown_motor_stop: bool = True
    enable_kill_fallback: bool = True


# --------------------------------------------------------------------------
# Mission tuning
# --------------------------------------------------------------------------

# The platform oscillation period the probe time constants are sized against.
# Everything probe-related below is a MULTIPLE of it, so re-measuring the
# platform means changing one number.
PROBE_DESIGN_PERIOD_SEC = 6.7


@dataclass(frozen=True)
class MissionConfig:
    """Every knob ``MissionRoutine`` reads. One field, one owner, no defaults
    duplicated anywhere else.

    Adding a knob for a new phase is a single line here plus its use inside
    ``mission_routine``; nothing in ``bee_node`` changes.
    """

    # --- Vertical setpoints ---
    descent_divergence_setpoint: float = 0.50
    approach_divergence_setpoint: float = 0.08
    d_star_ramp_in_sec: float = 1.5

    # --- Phase durations / triggers ---
    final_probe_duration_sec: float = 2.0 * PROBE_DESIGN_PERIOD_SEC
    final_probe_entry_ramp_sec: float = 1.5
    fov_near_area_fraction: float = 0.8
    probe_min_duration_sec: float = 3.0 * PROBE_DESIGN_PERIOD_SEC

    # --- Geometry / feasibility ---
    leg_clearance_m: float = 0.182
    ceiling_safety_factor: float = 0.6
    # k(t) decays toward max(k_min, ceiling_margin * k_ceiling_leg) rather than
    # toward k_min: settle at 70% of the ALREADY safety-derated ceiling. This
    # is the second of two multiplicative margins, not the only one.
    ceiling_margin: float = 0.7
    # Height at which the near-field trigger actually fires. ANCHOR of the
    # whole gain schedule. A CAMERA-GEOMETRY constant (target diameter vs FOV),
    # calibratable from a log: read relative_z_m at FINAL_PROBE entry.
    near_field_height_m: float = 0.5

    # --- Gains handed to the schedule ---
    initial_thrust_gain: float = 6.50
    roll_d_gain: float = 3.5
    pitch_d_gain: float = 3.5
    roll_kappa: float = 1.0       # derived: see BeeConfig.default()
    pitch_kappa: float = 1.0      # derived: see BeeConfig.default()
    # Total visual-loop delay the de Croon ceiling is sized against. NOT the
    # control tick -- see CameraConfig.stability_dt_sec(). Derived.
    stability_dt_sec: float = 1.0 / 30.0
    # Control tick period. Used only as the stability_dt fallback; the mission
    # itself is driven by camera SIM timestamps, never by a fixed rate. Derived
    # from SchedulingConfig.
    control_period_sec: float = 0.005

    roll_flow_admissible_norm_s: float = 0.4
    pitch_flow_admissible_norm_s: float = 0.4
    max_closing_speed_m_s: float = 0.05

    # --- Probe conditioning: FAR field (APPROACH_PROBE) ---
    # Sized against the PLATFORM period, not the control rate: this phase
    # exists precisely to see the slow oscillation the short near-field hold
    # structurally cannot.
    far_probe_window_sec: float = 1.5 * PROBE_DESIGN_PERIOD_SEC
    far_probe_decay_tau_sec: float = 1.5 * PROBE_DESIGN_PERIOD_SEC
    far_probe_highpass_tau_sec: float = 4.0 * PROBE_DESIGN_PERIOD_SEC

    # --- Probe conditioning: NEAR field (FINAL_PROBE hold, after retune) ---
    # near_probe_decay_tau_sec is the handoff knob: how fast the carried
    # far-field estimate is forgotten. Long on purpose -- an under-estimated
    # peak_accel gives a too-low k_min and a too-permissive gate, so forgetting
    # slowly is the conservative direction.
    near_probe_window_sec: float = 0.6 * PROBE_DESIGN_PERIOD_SEC
    near_probe_decay_tau_sec: float = PROBE_DESIGN_PERIOD_SEC
    near_probe_highpass_tau_sec: float = 2.0 * PROBE_DESIGN_PERIOD_SEC

    # Additive m/s^2 floors: unmodeled perturbation + ground effect + cold start.
    probe_vert_accel_margin_m_s2: float = 0.05
    probe_roll_accel_margin_m_s2: float = 0.01
    probe_pitch_accel_margin_m_s2: float = 0.01
    # Multiplicative >= 1: inverts the height-dependent (few-dB) probe under-read.
    probe_attenuation_comp: float = 1.15

    # --- CENTER phase ---
    enable_center: bool = True
    enable_center_condition_gate: bool = True
    center_condition_dwell_sec: float = 0.75
    center_offset_radius_max: float = 0.12
    center_flow_radius_max_norm_s: float = 0.25
    center_timeout_sec: float = 20.0
    # Timeout stays diagnostic by default: never leave CENTER merely because
    # the clock expired while the target is still moving / off-centre.
    center_timeout_allows_handoff: bool = True
    # Legacy box-threshold gate, used only when the condition gate is off.
    center_offset_threshold: float = 0.10
    center_dwell_sec: float = 2.0

    # --- Lateral schedule ---
    center_to_probe_lateral_ramp_sec: float = 2.0
    center_lateral_p_scale: float = 0.30
    center_lateral_d_scale: float = 0.70
    probe_lateral_p_scale: float = 0.30
    probe_lateral_d_scale: float = 1.0

    # --- Visual synchronisation (tracking) gate ---
    # A one-time REJECTION test during the stationary FINAL_PROBE hold on
    #
    #     chi = Ddot - D^2 = -hddot / h   [1/s^2]
    #
    # It produces no gain correction: the authority/stability gates already
    # decide the admissible gain window. chi asks the independent bandwidth
    # question -- is the vehicle actually keeping up with the deck at k_probe?
    #
    # The gate is evaluated only before DESCENT is committed. chi continues to
    # be measured during DESCENT for diagnosis, but it cannot revoke a committed
    # landing.
    enable_tracking_gate: bool = True
    # Provisional empirical limit separating the validated low-frequency run
    # from the rejected high-frequency case. Keep this explicit until a larger
    # validation set turns it into a formal safety margin.
    tracking_chi_limit_1_s2: float = 2.0
    # Ddot is the slope of a causal least-squares fit through this much recent
    # FILTERED divergence history. The actual camera/Gazebo SIM dt values are
    # preserved in the regression, so irregular frame spacing is handled
    # correctly without a post-derivative low-pass and its extra phase lag.
    tracking_derivative_window_sec: float = 0.20
    # Minimum FINAL_PROBE-hold observation before chi is allowed to veto. The
    # robust chi envelope is restarted at hold entry while the derivative
    # history stays warm from the preceding visual samples.
    tracking_min_observation_sec: float = 1.0 * PROBE_DESIGN_PERIOD_SEC

    # --- Mode switches ---
    enable_descent: bool = True
    probe_only: bool = False

    def with_overrides(self, **overrides: Any) -> "MissionConfig":
        """Return a copy with ``overrides`` applied, rejecting unknown names.

        This is the replacement for ``_supported_kwargs``: an unknown or
        misspelled knob raises here instead of quietly reverting to a default.
        """
        if not overrides:
            return self
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(
                f"Unknown MissionConfig field(s): {sorted(unknown)}. "
                f"Known fields: {sorted(known)}"
            )
        return replace(self, **overrides)


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BeeConfig:
    topics: TopicsConfig = field(default_factory=TopicsConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    mavsdk: MavsdkConfig = field(default_factory=MavsdkConfig)

    @classmethod
    def default(cls) -> "BeeConfig":
        """The flown configuration, with derived mission values wired in.

        ``roll_kappa`` / ``pitch_kappa`` / ``stability_dt_sec`` are functions of
        the camera and scheduling configs. Computing them here keeps the single
        source of truth for FOV and loop delay in one object each, instead of
        three module constants that can drift apart.
        """
        camera = CameraConfig()
        scheduling = SchedulingConfig()
        control = ControlConfig()
        mission = MissionConfig(
            roll_kappa=camera.roll_kappa,
            pitch_kappa=camera.pitch_kappa,
            stability_dt_sec=camera.stability_dt_sec(scheduling),
            roll_d_gain=control.roll_kd,
            pitch_d_gain=control.pitch_kd,
        )
        return cls(
            scheduling=scheduling, camera=camera, control=control, mission=mission
        )

    @classmethod
    def from_ros_parameters(cls, node) -> "BeeConfig":
        """Overlay declared ROS parameters onto :meth:`default`.

        The ONLY ROS-aware entry point in this module. Parameters are declared
        flat and namespaced, e.g. ``mission.ceiling_margin``. Unknown names
        raise, by way of ``MissionConfig.with_overrides``.
        """
        base = cls.default()
        overrides: dict[str, Any] = {}
        for f in fields(base.mission):
            name = f"mission.{f.name}"
            node.declare_parameter(name, getattr(base.mission, f.name))
            value = node.get_parameter(name).value
            if value != getattr(base.mission, f.name):
                overrides[f.name] = value
        return replace(base, mission=base.mission.with_overrides(**overrides))

    def describe(self) -> Mapping[str, Any]:
        """Flat name -> value view, for a one-shot startup log row."""
        out: dict[str, Any] = {}
        for group in fields(self):
            value = getattr(self, group.name)
            for f in fields(value):
                out[f"{group.name}.{f.name}"] = getattr(value, f.name)
        return out
    