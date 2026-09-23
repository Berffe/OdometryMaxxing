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
* ``CameraConfig`` / ``ControlConfig`` / ``StabilityDelayBudget`` /
  ``MissionConfig``: flight tuning.
* ``MavsdkConfig``: the takeoff/termination side channel.

Derived quantities (``roll_kappa``, the two stability dts, the probe time
constants scaled off the platform period) are computed once in
``BeeConfig.default()`` so the relationship between them stays visible instead
of being re-derived at three call sites.

Naming conventions
------------------
Every field name carries its unit as a suffix, and the suffix is the only place
a unit is ever stated::

    _sec        seconds
    _m          metres
    _m_s        metres per second
    _m_s2       metres per second squared
    _rad        radians                 _deg    degrees
    _norm       normalized image coordinate, [-1, 1]
    _norm_s     normalized image coordinate per second (optical flow)
    _1_s        per second              _1_s2   per second squared
    (none)      dimensionless ratio, count, boolean, or enum

Prefixes name the *owner*, not the phase that happens to read the value::

    center_        CENTER phase
    approach_      APPROACH_PROBE phase
    final_probe_   FINAL_PROBE phase
    descent_       DESCENT phase
    far_ / near_   far- vs near-field conditioning of a shared estimator
    wind_trim_     the static lateral term (spans FINAL_PROBE and DESCENT)
    probe_         PlatformProbe, all three axes
    tracking_      the chi visual-synchronisation gate
    enable_        a boolean mode switch

A field whose prefix names one phase must be read by that phase only.  If two
phases share it, the prefix names the *quantity* instead -- that is why the
static lateral term is ``wind_trim_*`` and not ``descent_lateral_bias_*``.

Overriding
----------
Every config is a frozen dataclass, so ``dataclasses.replace`` is the override
mechanism::

    cfg = BeeConfig.default()
    cfg = replace(cfg, mission=replace(cfg.mission, ceiling_margin=0.75))

``BeeConfig.from_ros_parameters(node)`` is the hook for declaring these as ROS
parameters later; it is deliberately the ONLY place that would need to know
about ROS.

Renamed fields
--------------
Some ``MissionConfig`` fields were renamed to remove misleading prefixes; see
``RENAMED_MISSION_FIELDS`` at the bottom of this module.  Old names still work
in ``with_overrides()``, in ``from_ros_parameters()`` and on attribute access,
each with a ``DeprecationWarning``, so existing launch files and analysis
scripts keep running.  They will be removed once those callers are migrated.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping, Optional


# ==========================================================================
# Node plumbing
# ==========================================================================


@dataclass(frozen=True)
class TopicsConfig:
    camera: str = "/bee_x500/camera/image"
    truth: str = "/bee_land/truth"
    wind: str = "/bee_land/wind_cmd"
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
    # --- Timer periods ----------------------------------------------------
    control_period_sec: float = 0.005
    px4_setpoint_period_sec: float = 0.01
    supervisor_period_sec: float = 0.1
    post_landing_log_period_sec: float = 0.10

    # --- Terminal shutdown -------------------------------------------------
    # How long the node keeps running after a terminal outcome before it shuts
    # itself down. Every run ends by itself; nothing waits for an operator.
    #
    # LANDED gets the longer window because the truth stream is still arriving
    # and the post-touchdown rows are the contact-velocity evidence. A dead run
    # (ABORTED / INFEASIBLE) has nothing left to record, so it gets just enough
    # to flush the CSVs and let the motor stop land.
    post_landing_shutdown_sec: float = 5.0
    terminal_shutdown_grace_sec: float = 2.0

    # --- Offboard handshake -----------------------------------------------
    offboard_prestream_sec: float = 2.0
    px4_offboard_switch_settle_sec: float = 0.5
    px4_offboard_confirm_timeout_sec: float = 5.0
    px4_offboard_reengage_interval_sec: float = 0.5

    # --- Clock source -----------------------------------------------------
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

    # --- PX4 constants ----------------------------------------------------
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


@dataclass(frozen=True)
class StabilityDelayBudget:
    """The visual-loop delay the de Croon ceiling is built on, itemised.

    ``k_ceiling(h) = 2*s*h/dt`` and its lateral counterpart are set by how STALE
    the newest measurement can be by the time the vehicle acts on it -- not by
    how often the control loop runs. Every field below is one physical
    contributor to that staleness; the total is their sum.

    This group OWNS the budget: each term appears here once, under one name, and
    nothing else in the package re-declares it. Two of the terms are also real
    operational values with their own owners -- the camera's frame period and
    the PX4 setpoint timer period -- so ``BeeConfig.default()`` wires those in
    rather than letting a second literal drift away from the first.

    Adding a contributor is one field plus one line in ``total_sec`` and
    ``itemised()``. If it applies to only one channel, give it a default of 0.0
    and set it in whichever of ``BeeConfig.vertical_stability_delay`` /
    ``lateral_stability_delay`` pays it.
    """

    #: Time between camera frames: the newest measurement is on average half a
    #: frame old and at worst a whole one.
    #: <- CameraConfig.frame_period_sec, wired in BeeConfig.default().
    camera_frame_period_sec: float = 1.0 / 60.0
    #: Detection + optical flow compute. Measured across several test runs.
    vision_processing_sec: float = 0.020
    #: How long a computed setpoint waits before it is published to PX4.
    #: <- SchedulingConfig.px4_setpoint_period_sec, wired in BeeConfig.default().
    setpoint_publication_sec: float = 0.010
    #: Command-shaping filter + slew limiter group delay, lumped. Measured
    #: across several test runs.
    command_smoothing_sec: float = 0.030
    #: PX4's own attitude/rate loop: the lag between an attitude SETPOINT being
    #: accepted and the airframe actually holding that attitude. LATERAL ONLY --
    #: the vertical channel commands thrust, which PX4 applies directly with no
    #: inner loop in between, so it does not pay this term.
    #:
    #: THE ONE TERM STILL AWAITING VALIDATION. It feeds a safety ceiling, and
    #: too small a value is the OPTIMISTIC direction: it claims a higher
    #: admissible lateral gain than the airframe can actually carry. To measure
    #: it, command a small attitude step and log the delay between the setpoint
    #: timestamp and vehicle_attitude crossing ~63% of the step.
    attitude_loop_sec: float = 0.10

    @property
    def total_sec(self) -> float:
        return (
            self.camera_frame_period_sec
            + self.vision_processing_sec
            + self.setpoint_publication_sec
            + self.command_smoothing_sec
            + self.attitude_loop_sec
        )

    def itemised(self) -> Mapping[str, float]:
        """name -> seconds, in the order the delay is physically incurred."""
        return {
            "camera_frame_period_sec": self.camera_frame_period_sec,
            "vision_processing_sec": self.vision_processing_sec,
            "setpoint_publication_sec": self.setpoint_publication_sec,
            "command_smoothing_sec": self.command_smoothing_sec,
            "attitude_loop_sec": self.attitude_loop_sec,
            "total_sec": self.total_sec,
        }


@dataclass(frozen=True)
class ControlConfig:
    """Constant PD gains handed to ``ControlLaw`` (acceleration domain)."""

    # Retuned after the 2026-08-18 wind run.  The previous 5.0 / 3.5 pair
    # produced a visibly oscillatory CENTER capture.  The reduced proportional
    # authority is the main damping change; D remains high enough to dissipate
    # lateral motion without approaching the theoretical near-field ceiling.
    roll_kp: float = 4.0
    roll_kd: float = 3.5
    pitch_kp: float = 4.0
    pitch_kd: float = 3.5

    # Large-offset CENTER capture blend. P is softened more strongly to limit
    # overshoot, while D retains more authority to keep the transient damped.
    large_offset_p_gain_scale: float = 0.70
    large_offset_d_gain_scale: float = 0.75



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
    # Dead runs (ABORTED / INFEASIBLE) end in the air. Stopping the motors
    # guarantees PX4 disarms, so the next run in a campaign starts against a
    # clean SITL instead of one still hovering under a failsafe. This is a
    # TESTER policy: a real landing system would command a descent here.
    enable_terminal_motor_stop: bool = True
    enable_kill_fallback: bool = True


# ==========================================================================
# Mission tuning
# ==========================================================================

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

    Field order follows the mission timeline --

        CENTER -> APPROACH_PROBE -> FINAL_PROBE -> DESCENT

    -- with the cross-cutting sections (geometry, probe conditioning, wind
    trim, lateral schedule, tracking gate, mode switches) after it.  See the
    module docstring for the unit-suffix and prefix conventions every name
    follows.
    """

    # ======================================================================
    # 1. DERIVED -- do not hand-edit
    # ======================================================================
    # ``BeeConfig.default()`` overwrites all six from CameraConfig /
    # SchedulingConfig / ControlConfig.  The literals below are only the
    # fallbacks used when a MissionConfig is built standalone (tests).
    # ----------------------------------------------------------------------
    roll_kappa: float = 1.0               # <- CameraConfig.roll_kappa
    pitch_kappa: float = 1.0              # <- CameraConfig.pitch_kappa
    # Total visual-loop delay the de Croon ceiling is sized against.  NOT the
    # control tick: the ceiling is set by how STALE the newest measurement can
    # be, not by how often we act.  See StabilityDelayBudget for the itemised
    # breakdown and for how to add a term.
    #
    # Two of them, because the two channels do not carry the same delay.  The
    # vertical command is thrust, applied by PX4 directly.  The lateral command
    # is an ATTITUDE setpoint, so it additionally waits on PX4's inner
    # attitude/rate loop -- which means the lateral ceiling is strictly lower
    # than the vertical one at the same height, and the lateral axes have less
    # margin than a single shared dt implied.
    #
    # <- BeeConfig.vertical_stability_delay() / lateral_stability_delay()
    vertical_stability_dt_sec: float = 1.0 / 30.0
    lateral_stability_dt_sec: float = 1.0 / 30.0
    roll_d_gain: float = 3              # <- ControlConfig.roll_kd
    pitch_d_gain: float = 3             # <- ControlConfig.pitch_kd
    # Control tick period, used ONLY as the stability_dt fallback.  The mission
    # itself is driven by camera SIM timestamps, never by a fixed rate.
    # <- SchedulingConfig.control_period_sec.  Renamed from
    # ``control_period_sec``, which collided with the scheduling field of the
    # same name and read like the mission's own tick rate.
    stability_dt_fallback_sec: float = 0.005

    # ======================================================================
    # 2. CENTER -- lateral capture before the approach begins
    # ======================================================================

    # --- Handoff condition (current path) ---------------------------------
    enable_center: bool = True
    enable_center_condition_gate: bool = True
    # Require a genuinely settled lateral state before starting the approach.
    # The raw trim radius is intentionally left permissive because steady wind
    # plus camera tilt creates a legitimate non-zero visual operating point;
    # strictness is applied to residual motion, optical flow and dwell instead.
    center_condition_dwell_sec: float = 1.50
    # Final CENTER criterion in the geometric-tilt-compensated frame:
    # radius of (measured target offset - image location of the world-vertical
    # ray through the camera).  Unlike the raw image offset, this SHOULD tend to
    # zero when the adaptive visual centre has brought the platform physically
    # underneath the vehicle.
    center_offset_radius_max: float = 0.05
    center_flow_radius_max_norm_s: float = 0.10
    center_timeout_sec: float = 30.0
    # Timeout never hands off: do not leave CENTER merely because the clock
    # expired while the target is still moving / off-centre.
    center_timeout_allows_handoff: bool = False
    # ... but do not hover on it forever either. The expiry terminates the run
    # as ABORTED, because a vehicle that never centred never reaches a
    # feasibility verdict and must not be counted as a refusal.
    center_timeout_aborts: bool = True

    # --- Handoff condition (legacy box gate) ------------------------------
    # Used only when ``enable_center_condition_gate`` is False.  Renamed from
    # ``center_offset_threshold`` / ``center_dwell_sec``, which were easy to
    # confuse with the live radial gate and its dwell above.
    center_legacy_box_threshold: float = 0.25
    center_legacy_box_dwell_sec: float = 2.0

    # --- CENTER trim estimator --------------------------------------------
    # A PD lateral loop with no integrator rejects a steady wind only by
    # holding a steady image offset. The trim estimators themselves stay
    # passive; CENTER feeds the geometric physical-trim mean to the separate
    # slow VisualCenterAdaptation outer loop below.
    #
    # These were previously read out of MissionConfig with getattr() defaults,
    # which meant they could never actually be set: with_overrides() rejects
    # names that are not declared fields, so every run silently used the
    # hardcoded fallback. They are real fields now.
    #
    # Short tau, because the estimate has to settle inside the handoff dwell.
    center_trim_tau_sec: float = 1.0
    # None means DERIVE as max(center_offset_radius_max,
    # center_legacy_box_threshold).  This is the permissive RAW-image bound
    # used only while identifying the initial steady PD equilibrium; it is
    # intentionally looser than the final physical-centering bound above.
    # With the defaults above it resolves to 0.25.
    center_trim_mean_radius_max: Optional[float] = None
    center_trim_residual_radius_max: float = 0.06

    # --- Adaptive physical visual centre ----------------------------------
    # CENTER and APPROACH_PROBE continuously move the visual setpoint so the
    # non-zero P error required to reject steady wind is retained while the
    # tilt-corrected physical centring error tends to zero. The estimator is
    # frozen only when FINAL_PROBE removes lateral P authority. This outer
    # adaptation must remain slower than the lateral P/D loop.
    center_visual_adaptation_tau_sec: float = 3.0
    center_visual_adaptation_max_bias_norm: float = 0.90
    # Smooth motion gate: at this optical-flow radius adaptation runs at 50%.
    center_visual_adaptation_flow_scale_norm_s: float = 0.10
    # Independent per-axis slew protection for the moving visual reference.
    center_visual_adaptation_max_rate_norm_s: float = 0.10

    # ======================================================================
    # 3. APPROACH_PROBE -- descend to the visual-height hold
    # ======================================================================
    approach_divergence_setpoint: float = 0.08
    approach_d_star_ramp_in_sec: float = 3.0

    # APPROACH_PROBE keeps the nominal far-field divergence command until this
    # slow outer loop begins braking on visual scale.  For a planar target,
    # q = 0.5*ln(area_fraction) is a log-linear range coordinate and q_dot is
    # the same expansion-rate quantity regulated by the inner vertical PI.
    approach_hold_area_fraction: float = 0.75
    approach_visual_p_gain_1_s: float = 0.30
    approach_retreat_divergence_limit: float = 0.03
    approach_hold_log_scale_tolerance: float = 0.05
    approach_hold_divergence_tolerance_1_s: float = 0.1
    approach_hold_dwell_sec: float = 0.75
    # The APPROACH twin of center_timeout_sec. Generous relative to the nominal
    # approach: it is a stuck-run detector, not a performance bound.
    approach_timeout_sec: float = 60.0
    approach_timeout_aborts: bool = True

    # False disables only the COMMIT decision, never the measurement: the
    # probes run, k_min is computed, every gate verdict is evaluated, attributed
    # to an axis and a criterion, and logged. The descent then proceeds
    # regardless, with k_min imposed as the schedule floor and the sampled-data
    # ceiling deliberately violated. This is the ablation the campaign pairs
    # against the gated runs on the same seed; it is not a "gates off" mode in
    # the sense of an unconstrained gain.
    #
    # The near-field PROBE gain stays ceiling-limited either way. Probing above
    # the ceiling feeds self-induced oscillation into peak_accel, which would
    # corrupt the very number both arms of the ablation are compared on.
    enable_commit_gate: bool = True

    # ======================================================================
    # 4. FINAL_PROBE -- the only phase whose evidence reaches the gates
    # ======================================================================
    # FINAL_PROBE starts only after APPROACH has already established the visual
    # height hold and near-zero divergence.  Its measurements alone feed gates.
    final_probe_duration_sec: float = 3.0 * PROBE_DESIGN_PERIOD_SEC

    # ======================================================================
    # 5. DESCENT -- scheduled-gain terminal segment
    # ======================================================================
    descent_divergence_setpoint: float = 0.30
    descent_d_star_ramp_in_sec: float = 6.0

    # ======================================================================
    # 6. Geometry and feasibility bounds
    # ======================================================================
    leg_clearance_m: float = 0.182
    ceiling_safety_factor: float = 0.6
    # k(t) decays toward max(k_min, ceiling_margin * k_ceiling_leg) rather than
    # toward k_min: settle at 50% of the ALREADY safety-derated ceiling. This
    # is the second of two multiplicative margins, not the only one.
    ceiling_margin: float = 0.50
    # Height corresponding to the APPROACH visual-hold / FINAL_PROBE handoff.
    # This is a camera-geometry calibration used only by the feasibility bounds;
    # update it from truth logs whenever approach_hold_area_fraction changes.
    near_field_height_m: float = 0.40

    # Starting exploration gain k_explore for the descent schedule.
    initial_thrust_gain: float = 6.50
    roll_flow_admissible_norm_s: float = 0.4
    pitch_flow_admissible_norm_s: float = 0.4
    max_closing_speed_m_s: float = 0.05

    # ======================================================================
    # 7. Probe conditioning (PlatformProbe x3: vertical, roll, pitch)
    # ======================================================================
    # The same three estimators run throughout, retuned far -> near at the
    # FINAL_PROBE handoff.  ``far_*`` is APPROACH diagnostics; ``near_*`` is
    # gate evidence.

    # --- Far field: APPROACH diagnostics ---
    far_probe_window_sec: float = 1.5 * PROBE_DESIGN_PERIOD_SEC
    far_probe_decay_tau_sec: float = 1.5 * PROBE_DESIGN_PERIOD_SEC
    far_probe_highpass_tau_sec: float = 4.0 * PROBE_DESIGN_PERIOD_SEC

    # --- Near field: FINAL_PROBE gate evidence ---
    # FINAL_PROBE resets all three acceleration probes before applying these
    # near-field constants, so no APPROACH envelope can enter a gate.
    near_probe_window_sec: float = 0.6 * PROBE_DESIGN_PERIOD_SEC
    near_probe_decay_tau_sec: float = 8.0 * PROBE_DESIGN_PERIOD_SEC
    near_probe_highpass_tau_sec: float = 4.0 * PROBE_DESIGN_PERIOD_SEC

    # --- Envelope protection ---
    # Additive m/s^2 floors: unmodeled perturbation + ground effect + cold start.
    probe_vert_accel_margin_m_s2: float = 0.05
    probe_roll_accel_margin_m_s2: float = 0.01
    probe_pitch_accel_margin_m_s2: float = 0.01
    # Multiplicative >= 1: inverts the height-dependent (few-dB) probe under-read.
    probe_attenuation_comp: float = 1.15

    # --- Visual-trim estimators paired with the probes ---
    # They remain passive: lateral acceleration feedforward is handled
    # separately by the wind-trim section below.  Set the tau multipliers to
    # match the corresponding far/near probe conditioning windows.
    far_trim_tau_sec: float = 5.0 * PROBE_DESIGN_PERIOD_SEC
    near_trim_tau_sec: float = 4.0 * PROBE_DESIGN_PERIOD_SEC
    # False restores the previous behaviour: DESCENT freezes the single
    # instantaneous image offset from the decision tick. Kept as a one-line
    # revert for A/B comparison against earlier logs.
    descent_trim_use_probe_mean: bool = True

    # ======================================================================
    # 8. WIND REJECTION -- the static lateral acceleration term
    # ======================================================================
    # The near-field lateral law splits the commanded acceleration into a
    # static term and a dynamic term:
    #
    #     a_cmd = a_static + a_D(flow)
    #
    # ``a_static`` carries the steady wind; ``a_D`` carries the platform
    # motion.  Sequence:
    #
    #   CENTER / APPROACH   a_static == 0.  The roll/pitch PlatformProbe means
    #                       estimate the steady lateral command PASSIVELY, with
    #                       no command authority of their own.
    #   FINAL_PROBE entry   those passive means become the initial a_static, at
    #                       the same instant lateral image-position P goes to 0.
    #   FINAL_PROBE hold    a_static tracks the realized lateral command through
    #                       a causal EMA, absorbing the slow part of a_D.
    #   DESCENT             starts from the exact committed FINAL_PROBE value;
    #                       keeps adapting when the switch below is on.
    #
    # These knobs span FINAL_PROBE and DESCENT, which is why they are named for
    # the QUANTITY and not for a phase.  Renamed from ``descent_lateral_bias_*``:
    # the "descent_" prefix wrongly implied FINAL_PROBE had its own separate
    # tau and its own separate bound.
    # ----------------------------------------------------------------------

    # Adaptation time constant, shared by FINAL_PROBE and DESCENT.  It must
    # remain slower than the lateral D loop so the static estimate cannot chase
    # the oscillatory flow -- that is the loop's stability condition, not a
    # preference.
    wind_trim_tau_sec: float = 8.0 * PROBE_DESIGN_PERIOD_SEC
    # Hard bound around the passive APPROACH seed, shared across FINAL_PROBE and
    # DESCENT.  ONE neighbourhood for both phases: continuing to adapt after the
    # descent commitment does not buy a second deviation allowance.
    wind_trim_deviation_limit_m_s2: float = 0.5
    # True: DESCENT continues the same slow adaptation from the committed
    # FINAL_PROBE value, so changing wind can be tracked during the short
    # terminal segment.  False: DESCENT flies the committed value frozen.
    # Post-commit adaptation changes commands only; it can never re-open a
    # feasibility decision.
    wind_trim_adapt_in_descent: bool = False

    # ======================================================================
    # 9. Lateral gain schedule
    # ======================================================================
    # The lateral D schedule mirrors the vertical one exactly: ONE far-field
    # gain is configured, and the near-field value it decays to is DERIVED from
    # the same ceiling the descent targets.
    #
    #   vertical   k_probe    = min(initial_thrust_gain,
    #                              ceiling_margin * k_ceiling(near_field_height))
    #   lateral    Kd_probe   = min(d_gain * center_lateral_d_scale,
    #                              ceiling_margin * k_ceiling_lat(near_field_height))
    #
    # Same near_field_height_m, same ceiling_margin, same ceiling_safety_factor,
    # and the same commanded-divergence-integral driver for the decay, so the
    # lateral gain reaches its near-field value exactly when the vertical one
    # does.  The min() only ever REDUCES the gain to become admissible.
    #
    # There is deliberately no configured near-field lateral D.  Asserting one
    # makes its admissibility a coincidence between three unrelated constants:
    # the lateral ceiling is a function of kappa, lateral_stability_dt_sec and
    # near_field_height_m, so a wider lens, a slower loop or a lower handoff
    # silently moves it out from under a hand-set value.  Deriving it means a
    # platform whose lateral authority IS constrained gets a real decay instead
    # of a number that happens to fit this airframe.
    #
    # The ceiling is evaluated PER AXIS, because roll and pitch have their own
    # kappa and their own D gain whenever the lens is not square.
    #
    # FINAL_PROBE's D is deliberately NOT attenuated by the large-offset blend
    # (MissionControl.scale_lateral_d_with_offset=False), so the Kd the gates
    # are computed against is the Kd actually flown.
    #
    # P has no ceiling analogue -- it is not the flow loop the de Croon bound
    # applies to -- so its near-field floor stays configured.  During APPROACH
    # it is scheduled by the SAME commanded-divergence integral and the SAME
    # exponential floor law as D; FINAL_PROBE then holds this floor directly.
    center_lateral_p_scale: float = 1.00
    center_lateral_d_scale: float = 0.80
    probe_lateral_p_scale: float = 0.30

    # ======================================================================
    # 10. Visual synchronisation (tracking) gate
    # ======================================================================
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
    # Provisional empirical limits separating the validated low-frequency run
    # from the rejected high-frequency case. Keep these explicit until a larger
    # validation set turns them into a formal safety margin.
    #
    # The z field was ``tracking_chi_limit_1_s2``; the unqualified name read
    # like a shared limit while x and y were explicit.  The CSV column keeps
    # its legacy unqualified name -- only the config field is renamed.
    tracking_chi_z_limit_1_s2: float = 0.3
    tracking_chi_x_limit_1_s2: float = 0.5
    tracking_chi_y_limit_1_s2: float = 0.5
    # Ddot is the slope of a causal least-squares fit through this much recent
    # FILTERED divergence history. The actual camera/Gazebo SIM dt values are
    # preserved in the regression, so irregular frame spacing is handled
    # correctly without a post-derivative low-pass and its extra phase lag.
    tracking_derivative_window_sec: float = 0.20
    # Minimum FINAL_PROBE-hold observation before chi is allowed to veto. The
    # robust chi envelope is restarted at hold entry while the derivative
    # history stays warm from the preceding visual samples.
    tracking_min_observation_sec: float = 1.0 * PROBE_DESIGN_PERIOD_SEC

    # ======================================================================
    # 11. Mode switches
    # ======================================================================
    enable_descent: bool = True
    probe_only: bool = False

    # ----------------------------------------------------------------- API
    def with_overrides(self, **overrides: Any) -> "MissionConfig":
        """Return a copy with ``overrides`` applied, rejecting unknown names.

        This is the replacement for ``_supported_kwargs``: an unknown or
        misspelled knob raises here instead of quietly reverting to a default.

        Deprecated field names from before the wind-rejection rename are
        translated (with a ``DeprecationWarning``) rather than rejected, so
        existing launch files and analysis scripts keep working.
        """
        if not overrides:
            return self
        overrides = _translate_deprecated(overrides, context="with_overrides")
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(
                f"Unknown MissionConfig field(s): {sorted(unknown)}. "
                f"Known fields: {sorted(known)}"
            )
        return replace(self, **overrides)

    def __getattr__(self, name: str) -> Any:
        """Serve deprecated field names on read, with a warning.

        Only reached for attributes that do not exist, so it costs nothing on
        the normal path and cannot mask a real field.
        """
        new_name = RENAMED_MISSION_FIELDS.get(name)
        if new_name is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        warnings.warn(
            f"MissionConfig.{name} was renamed to {new_name}; "
            f"the old name will be removed in a future revision.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, new_name)


# ==========================================================================
# Aggregate
# ==========================================================================


@dataclass(frozen=True)
class BeeConfig:
    topics: TopicsConfig = field(default_factory=TopicsConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    stability_delay: StabilityDelayBudget = field(
        default_factory=StabilityDelayBudget
    )
    mission: MissionConfig = field(default_factory=MissionConfig)
    mavsdk: MavsdkConfig = field(default_factory=MavsdkConfig)

    @classmethod
    def default(cls) -> "BeeConfig":
        """The flown configuration, with derived mission values wired in.

        ``roll_kappa`` / ``pitch_kappa`` / the two stability dts are functions of
        the camera and scheduling configs. Computing them here keeps the single
        source of truth for FOV and loop delay in one object each, instead of
        three module constants that can drift apart.

        Every field in ``MissionConfig`` section 1 is set here. If you add a
        derived field, add it here too -- ``test_derived_values_are_wired`` is
        what catches the omission.
        """
        camera = CameraConfig()
        scheduling = SchedulingConfig()
        control = ControlConfig()
        # The two terms that are also real operational values keep ONE owner
        # each; the literals in StabilityDelayBudget are only the standalone
        # fallback, exactly like MissionConfig section 1.
        stability_delay = replace(
            StabilityDelayBudget(),
            camera_frame_period_sec=camera.frame_period_sec,
            setpoint_publication_sec=scheduling.px4_setpoint_period_sec,
        )
        mission = MissionConfig(
            roll_kappa=camera.roll_kappa,
            pitch_kappa=camera.pitch_kappa,
            vertical_stability_dt_sec=replace(
                stability_delay, attitude_loop_sec=0.0
            ).total_sec,
            lateral_stability_dt_sec=stability_delay.total_sec,
            roll_d_gain=control.roll_kd,
            pitch_d_gain=control.pitch_kd,
            # Previously omitted: the mission kept its own literal 0.005 while
            # the comment claimed the value was derived, so changing the
            # scheduling tick left the mission's fallback behind.
            stability_dt_fallback_sec=scheduling.control_period_sec,
        )
        return cls(
            scheduling=scheduling,
            camera=camera,
            control=control,
            stability_delay=stability_delay,
            mission=mission,
        )

    def vertical_stability_delay(self) -> StabilityDelayBudget:
        """Delay on the thrust channel: no inner loop between us and the motors.

        The whole budget minus the attitude-loop term, which the vertical
        channel does not pay.
        """
        return replace(self.stability_delay, attitude_loop_sec=0.0)

    def lateral_stability_delay(self) -> StabilityDelayBudget:
        """The full budget: a lateral command is an ATTITUDE setpoint, so no
        lateral acceleration exists until PX4 has driven the airframe there."""
        return self.stability_delay

    @classmethod
    def from_ros_parameters(cls, node) -> "BeeConfig":
        """Overlay declared ROS parameters onto :meth:`default`.

        The ONLY ROS-aware entry point in this module. Parameters are declared
        flat and namespaced, e.g. ``mission.ceiling_margin``. Unknown names
        raise, by way of ``MissionConfig.with_overrides``.

        Deprecated names are declared alongside the current ones so an existing
        launch file keeps working; setting one emits a ``DeprecationWarning``
        and is translated to the current field. Setting both raises.
        """
        base = cls.default()
        overrides: dict[str, Any] = {}
        for f in fields(base.mission):
            name = f"mission.{f.name}"
            node.declare_parameter(name, getattr(base.mission, f.name))
            value = node.get_parameter(name).value
            if value != getattr(base.mission, f.name):
                overrides[f.name] = value

        # Deprecated aliases are declared with the CURRENT default, so leaving
        # one alone is a no-op and only a genuine override is picked up.
        for old_name, new_name in RENAMED_MISSION_FIELDS.items():
            current = getattr(base.mission, new_name)
            param = f"mission.{old_name}"
            node.declare_parameter(param, current)
            value = node.get_parameter(param).value
            if value != current:
                if new_name in overrides:
                    raise ValueError(
                        f"ROS parameters set both {param} and "
                        f"mission.{new_name}; use only mission.{new_name}."
                    )
                overrides[old_name] = value

        return replace(base, mission=base.mission.with_overrides(**overrides))

    def describe(self) -> Mapping[str, Any]:
        """Flat name -> value view, for a one-shot startup log row.

        The per-channel totals are added on top of the ``stability_delay``
        group: they are DERIVED (the group holds the terms, not the two sums),
        and they are the numbers a review most often wants spelled out.
        """
        out: dict[str, Any] = {}
        for group in fields(self):
            value = getattr(self, group.name)
            for f in fields(value):
                out[f"{group.name}.{f.name}"] = getattr(value, f.name)
        out["stability_delay.vertical_total_sec"] = (
            self.vertical_stability_delay().total_sec
        )
        out["stability_delay.lateral_total_sec"] = (
            self.lateral_stability_delay().total_sec
        )
        return out


# ==========================================================================
# Deprecated names
# ==========================================================================

#: Old ``MissionConfig`` field name -> current name.
#:
#: Accepted by ``with_overrides()``, ``from_ros_parameters()`` and attribute
#: access, each with a ``DeprecationWarning``.  Delete an entry once no launch
#: file, notebook or analysis script still uses it.
#:
#: Only true RENAMES belong here. ``stability_dt_sec`` was deliberately NOT
#: added when it became vertical_/lateral_: aliasing a SPLIT onto one of its two
#: halves would quietly leave the other half at its default, which is the exact
#: class of silent mis-tuning this module exists to prevent. An old launch file
#: setting it now fails loudly instead.
RENAMED_MISSION_FIELDS: Mapping[str, str] = {
    # The static lateral term spans FINAL_PROBE and DESCENT; the old prefix
    # implied FINAL_PROBE had a separate tau and a separate bound.
    "descent_lateral_bias_tau_sec": "wind_trim_tau_sec",
    "descent_lateral_bias_deviation_limit_m_s2": "wind_trim_deviation_limit_m_s2",
    "descent_lateral_bias_adaptive": "wind_trim_adapt_in_descent",
    # Unqualified z limit sat next to explicit x/y limits.
    "tracking_chi_limit_1_s2": "tracking_chi_z_limit_1_s2",
    # Legacy CENTER box gate, easily confused with the live radial gate.
    "center_offset_threshold": "center_legacy_box_threshold",
    "center_dwell_sec": "center_legacy_box_dwell_sec",
    # Collided with SchedulingConfig.control_period_sec.
    "control_period_sec": "stability_dt_fallback_sec",
}


def _translate_deprecated(
    overrides: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    """Map deprecated override names onto current ones, warning on each."""
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        new_key = RENAMED_MISSION_FIELDS.get(key)
        if new_key is None:
            out[key] = value
            continue
        if new_key in overrides:
            raise TypeError(
                f"{context} received both {key!r} and its replacement "
                f"{new_key!r}; pass only {new_key!r}."
            )
        warnings.warn(
            f"MissionConfig field {key!r} was renamed to {new_key!r}; "
            f"the old name will be removed in a future revision.",
            DeprecationWarning,
            stacklevel=3,
        )
        out[new_key] = value
    return out
