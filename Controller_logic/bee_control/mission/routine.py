"""Bio-inspired near-field mission routine -- assembly and shared state.

Sequence:
    CENTER -> APPROACH_PROBE -> FINAL_PROBE -> DESCEND

The per-phase logic lives in ``mission/phases/``; this module owns what the
phases share: configuration, the three acceleration probes, the feasibility
gates, phase dispatch, and the log schema.

Three acceleration probes run in parallel on the vertical, roll and pitch
command accelerations. APPROACH_PROBE uses their far-field envelopes for
diagnostics only. At the visual-height handoff, FINAL_PROBE freezes those
diagnostic peaks, resets all three probes, applies the near-field conditioning,
and collects the only acceleration evidence allowed into the feasibility gates.
The static lateral feedforward is seeded at that handoff, adapted causally after
lateral P is removed, and may remain adaptive through DESCENT; post-commit
adaptation changes commands only and never re-opens a feasibility decision.

Vertical feasibility compares the Herisse disturbance-rejection floor with the
safety-scaled de Croon ceiling at landing-gear height. Roll and pitch feasibility
use the acceleration-domain lateral bound

    K_min = c_max / kappa + peak_accel / omega_adm

and verify that the planned D gain at touchdown lies below the corresponding
discrete lateral ceiling. No online height estimate is used.

All timers use the camera/image timestamp supplied as ``inputs.t``, which is
Gazebo SIM time. Never a wall clock.

Where to edit what
------------------
``phases/<name>.py``   one phase's behaviour and transitions
``probe.py``           the command-acceleration probe
``trim.py``            the slow visual-trim (wind equilibrium) estimator
``gates.py``           feasibility maths
``schedule.py``        the k(t) descent trajectory
``types.py``           the contract with the caller
this file              config plumbing, shared state, dispatch, telemetry
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
import math
from typing import Any, ClassVar, Mapping, Optional, Sequence

from bee_control.core.config import CameraConfig, MissionConfig
from bee_control.core.state import FlowResult, TargetEstimate

from . import phases
from .gates import (
    GateResult,
    LateralGateResult,
    TrackingGateResult,
    compute_tracking_gate,
    ceiling_gain_at_height,
    lateral_ceiling_gain_at_height,
    compute_lateral_gate,
)
from .math_utils import blank, clamp
from .phases import PHASES, TERMINAL_SUBSTATES
from .probe import PlatformProbe, ProbeResult, ThrustModel
from .trim import VisualTrim
from .visual_mismatch import VisualMismatchProbe
from .visual_center_adaptation import VisualCenterAdaptation
from .types import (
    ABORTED,
    APPROACH_PROBE,
    CENTER,
    DESCEND,
    FINAL_PROBE,
    INFEASIBLE,
    LANDED,
    PROBE_HOLD,
    ActuationFeedback,
    MissionControl,
    MissionInputs,
)


def _trim_row(snapshot: tuple[float, float, float, float]):
    """``VisualTrim.snapshot()`` -> CSV-safe values, ``inf`` becoming ``None``.

    ``inf`` is the correct in-code sentinel for "no estimate yet" because it can
    never satisfy a ``<= max`` gate by accident, but it is the wrong thing to
    write to a log: it plots as a number. A blank cell is a gap, which is true.
    """
    mean_x, mean_y, mean_radius, residual_radius = snapshot
    if not math.isfinite(mean_radius):
        return None, None, None, None
    return (
        mean_x,
        mean_y,
        mean_radius,
        residual_radius if math.isfinite(residual_radius) else None,
    )


class MissionRoutine:
    PHASES: ClassVar[Mapping[str, Any]] = PHASES

    """Shared state and dispatch for the visual landing sequence.

    The phases in ``mission/phases/`` receive this object as ``routine`` and
    read its configuration, probes and gates directly. That is deliberate: they
    are behaviours OF this object, split across files for readability, not
    independent components with their own state.
    """

    def __init__(
        self,
        hover_thrust: float,
        config: Optional[MissionConfig] = None,
        camera: Optional[CameraConfig] = None,
        **overrides,
    ):
        """Build the routine from a :class:`~config.MissionConfig`.

        Previously this took ~50 keyword arguments whose defaults duplicated the
        constants in ``bee_node``, and the caller filtered them through
        ``inspect.signature`` -- so a renamed parameter silently reverted to the
        library default with the vehicle in the air. ``overrides`` goes through
        ``MissionConfig.with_overrides``, which raises on an unknown name.

        ``hover_thrust`` stays a positional argument: it is not tuning, it is a
        measured property of the airframe that ``ControlLaw`` owns.
        """
        cfg = (config or MissionConfig()).with_overrides(**overrides)
        self.config = cfg
        camera_cfg = camera or CameraConfig()

        # Camera geometry is not a mission tuning knob.  The far-field lateral
        # controller uses the command it produced on the previous visual tick
        # as a visual-apparatus correction: a tilted nadir camera does not see
        # the point directly below the vehicle at image centre.  Keep the
        # geometry here, next to the phase decision that decides WHEN that
        # correction is meaningful; config.py can expose sign/tuning knobs only
        # after the behaviour is validated in simulation.
        self._tan_half_hfov = math.tan(0.5 * camera_cfg.horizontal_fov_rad)
        self._tan_half_vfov = math.tan(
            0.5 * math.radians(camera_cfg.vertical_fov_deg)
        )

        self._dt = float(cfg.stability_dt_fallback_sec)
        # Two delays, because the two channels do not carry the same one: the
        # lateral command is an attitude setpoint and additionally waits on
        # PX4's inner attitude loop. See StabilityDelayBudget in config.py.
        self._vertical_stability_dt = float(cfg.vertical_stability_dt_sec)
        self._lateral_stability_dt = float(cfg.lateral_stability_dt_sec)

        self._d_star = max(0.0, float(cfg.descent_divergence_setpoint))
        self._approach_d_star = max(0.0, float(cfg.approach_divergence_setpoint))
        self._approach_hold_area_fraction = clamp(
            cfg.approach_hold_area_fraction, 1e-6, 1.0
        )
        self._approach_visual_p_gain = max(0.0, float(cfg.approach_visual_p_gain_1_s))
        self._approach_retreat_d_star_limit = max(
            0.0, float(cfg.approach_retreat_divergence_limit)
        )
        self._approach_hold_log_scale_tol = max(
            0.0, float(cfg.approach_hold_log_scale_tolerance)
        )
        self._approach_hold_divergence_tol = max(
            0.0, float(cfg.approach_hold_divergence_tolerance_1_s)
        )
        self._approach_hold_dwell = max(0.0, float(cfg.approach_hold_dwell_sec))
        self._final_probe_duration = max(0.0, float(cfg.final_probe_duration_sec))
        self._leg_clearance = float(cfg.leg_clearance_m)
        self._enable_descent = bool(cfg.enable_descent)
        self._probe_only = bool(cfg.probe_only)

        self._safety = max(1e-3, min(1.0, float(cfg.ceiling_safety_factor)))
        self._initial_thrust_gain = max(0.0, float(cfg.initial_thrust_gain))
        self._roll_d_gain = max(0.0, float(cfg.roll_d_gain))
        self._pitch_d_gain = max(0.0, float(cfg.pitch_d_gain))
        self._roll_kappa = max(1e-6, float(cfg.roll_kappa))
        self._pitch_kappa = max(1e-6, float(cfg.pitch_kappa))
        self._roll_flow_admissible = max(1e-6, float(cfg.roll_flow_admissible_norm_s))
        self._pitch_flow_admissible = max(1e-6, float(cfg.pitch_flow_admissible_norm_s))
        self._max_closing_speed = max(0.0, float(cfg.max_closing_speed_m_s))
        self._ceiling_margin = max(0.0, float(cfg.ceiling_margin))
        self._near_field_height = max(1e-3, float(cfg.near_field_height_m))

        self._enable_center = bool(cfg.enable_center)
        self._center_offset_thr = max(0.0, float(cfg.center_legacy_box_threshold))
        self._center_dwell = max(0.0, float(cfg.center_legacy_box_dwell_sec))
        self._center_timeout = max(0.0, float(cfg.center_timeout_sec))

        self._enable_center_condition_gate = bool(cfg.enable_center_condition_gate)
        self._center_condition_dwell = max(0.0, float(cfg.center_condition_dwell_sec))
        self._center_offset_radius_max = max(0.0, float(cfg.center_offset_radius_max))
        self._center_flow_radius_max = max(0.0, float(cfg.center_flow_radius_max_norm_s))
        self._center_timeout_allows_handoff = bool(cfg.center_timeout_allows_handoff)

        # Trim-aware CENTER gate. These are declared MissionConfig fields; they
        # used to be read with getattr() defaults, which silently made them
        # unsettable because with_overrides() rejects undeclared names.
        self._center_trim_tau = max(1e-3, float(cfg.center_trim_tau_sec))
        # None -> derive, preserving the original coupling: the bound inherits
        # the looser of the modern radial gate and the legacy box threshold, so
        # tuning either of those still moves the handoff bound with them.
        self._center_trim_mean_radius_max = max(0.0, float(
            max(self._center_offset_radius_max, self._center_offset_thr)
            if cfg.center_trim_mean_radius_max is None
            else cfg.center_trim_mean_radius_max
        ))
        self._center_trim_residual_radius_max = max(
            0.0, float(cfg.center_trim_residual_radius_max)
        )

        # Slow outer loop that moves the visual reference until the
        # geometric-tilt-compensated physical centring error vanishes. It stays
        # active through CENTER and APPROACH_PROBE, then stops when FINAL_PROBE
        # removes lateral image-position P authority.
        self._center_visual_adaptation_tau = max(
            1e-3, float(cfg.center_visual_adaptation_tau_sec)
        )
        self._center_visual_adaptation_max_bias = abs(
            float(cfg.center_visual_adaptation_max_bias_norm)
        )
        self._center_visual_adaptation_flow_scale = max(
            1e-6, float(cfg.center_visual_adaptation_flow_scale_norm_s)
        )
        self._center_visual_adaptation_max_rate = abs(
            float(cfg.center_visual_adaptation_max_rate_norm_s)
        )

        # DESCENT handoff trim: averaged over the same window as the probes.
        self._far_trim_tau = max(1e-3, float(cfg.far_trim_tau_sec))
        self._near_trim_tau = max(1e-3, float(cfg.near_trim_tau_sec))
        self._descent_trim_use_probe_mean = bool(cfg.descent_trim_use_probe_mean)
        # Static lateral (wind) trim.  ONE tau and ONE deviation bound are
        # shared by FINAL_PROBE and DESCENT; only the *continuation* into
        # DESCENT is switchable.  See config.py section 8.
        self._wind_trim_adapt_in_descent = bool(cfg.wind_trim_adapt_in_descent)
        self._wind_trim_tau = max(1e-3, float(cfg.wind_trim_tau_sec))
        self._wind_trim_deviation_limit = abs(
            float(cfg.wind_trim_deviation_limit_m_s2)
        )


        self._approach_d_star_ramp_in = max(0.0, float(cfg.approach_d_star_ramp_in_sec))
        self._descent_d_star_ramp_in = max(0.0, float(cfg.descent_d_star_ramp_in_sec))

        self._center_lateral_p_scale = max(0.0, float(cfg.center_lateral_p_scale))
        self._center_lateral_d_scale = max(0.0, float(cfg.center_lateral_d_scale))
        self._probe_lateral_p_scale = max(0.0, float(cfg.probe_lateral_p_scale))
        # Residual image-position P retained through FINAL_PROBE.  Small by
        # design: the static acceleration trim owns the steady wind force, so
        # this term only closes the centring error the flow branch cannot see.
        self._final_probe_lateral_p_scale = max(
            0.0, float(cfg.final_probe_lateral_p_scale)
        )

        self._tm = ThrustModel(hover_thrust)

        # Three parallel probes use far-field conditioning for APPROACH
        # diagnostics and are reset into near-field conditioning at FINAL_PROBE.
        self._probe = PlatformProbe(
            self._tm,
            highpass_tau_sec=float(cfg.far_probe_highpass_tau_sec),
            percentile_window_sec=float(cfg.far_probe_window_sec),
            peak_decay_tau_sec=float(cfg.far_probe_decay_tau_sec),
            accel_margin=float(cfg.probe_vert_accel_margin_m_s2),
            probe_attenuation_comp=float(cfg.probe_attenuation_comp),
        )
        self._roll_probe = PlatformProbe(
            self._tm,
            highpass_tau_sec=float(cfg.far_probe_highpass_tau_sec),
            percentile_window_sec=float(cfg.far_probe_window_sec),
            peak_decay_tau_sec=float(cfg.far_probe_decay_tau_sec),
            accel_margin=float(cfg.probe_roll_accel_margin_m_s2),
            probe_attenuation_comp=float(cfg.probe_attenuation_comp),
        )
        self._pitch_probe = PlatformProbe(
            self._tm,
            highpass_tau_sec=float(cfg.far_probe_highpass_tau_sec),
            percentile_window_sec=float(cfg.far_probe_window_sec),
            peak_decay_tau_sec=float(cfg.far_probe_decay_tau_sec),
            accel_margin=float(cfg.probe_pitch_accel_margin_m_s2),
            probe_attenuation_comp=float(cfg.probe_attenuation_comp),
        )
        self._far_probe_window = float(cfg.far_probe_window_sec)
        self._far_probe_decay_tau = float(cfg.far_probe_decay_tau_sec)
        self._far_probe_highpass_tau = float(cfg.far_probe_highpass_tau_sec)

        self._near_probe_window = float(cfg.near_probe_window_sec)
        self._near_probe_decay_tau = float(cfg.near_probe_decay_tau_sec)
        self._near_probe_highpass_tau = float(cfg.near_probe_highpass_tau_sec)

        # Three visual synchronisation probes. The vertical channel keeps the
        # original chi_z = omega_z_dot - omega_z^2 definition. The lateral
        # channels use chi_i = omega_i_dot - omega_i*omega_z, so the range-rate
        # coupling that infiltrates lateral optical flow is removed explicitly.
        # All derivatives are causal least-squares slopes over the real
        # camera/Gazebo-SIM sample spacings.
        self._chi_probe = VisualMismatchProbe(
            percentile_window_sec=cfg.far_probe_window_sec,
            peak_decay_tau_sec=cfg.far_probe_decay_tau_sec,
            derivative_window_sec=cfg.tracking_derivative_window_sec,
        )
        self._chi_x_probe = VisualMismatchProbe(
            percentile_window_sec=cfg.far_probe_window_sec,
            peak_decay_tau_sec=cfg.far_probe_decay_tau_sec,
            derivative_window_sec=cfg.tracking_derivative_window_sec,
        )
        self._chi_y_probe = VisualMismatchProbe(
            percentile_window_sec=cfg.far_probe_window_sec,
            peak_decay_tau_sec=cfg.far_probe_decay_tau_sec,
            derivative_window_sec=cfg.tracking_derivative_window_sec,
        )
        self._enable_tracking_gate = bool(cfg.enable_tracking_gate)
        # Separate rejection limits per visual axis.  The config fields are
        # explicit (``tracking_chi_z_limit_1_s2``); the CSV column for the
        # vertical channel keeps its legacy unqualified name ``chi_limit_1_s2``.
        self._tracking_z_chi_limit = max(
            0.0, float(cfg.tracking_chi_z_limit_1_s2)
        )
        self._tracking_x_chi_limit = max(
            0.0, float(cfg.tracking_chi_x_limit_1_s2)
        )
        self._tracking_y_chi_limit = max(
            0.0, float(cfg.tracking_chi_y_limit_1_s2)
        )
        self._tracking_min_observation_sec = max(
            0.0, float(cfg.tracking_min_observation_sec)
        )
        # Only the stationary FINAL_PROBE hold contributes to the gate clock and
        # decision envelope. chi itself is still updated in every active visual
        # phase for diagnosis.
        self._chi_gate_window_active = False
        self._chi_observed_sec = 0.0
        self._chi_x_observed_sec = 0.0
        self._chi_y_observed_sec = 0.0

        self.gate = GateResult()
        self.roll_gate = LateralGateResult()
        self.pitch_gate = LateralGateResult()
        self.tracking_gate = TrackingGateResult()
        self.tracking_x_gate = TrackingGateResult()
        self.tracking_y_gate = TrackingGateResult()
        self.probe_result = ProbeResult()
        self.roll_probe_result = ProbeResult()
        self.pitch_probe_result = ProbeResult()
        # APPROACH diagnostic envelopes frozen immediately before FINAL_PROBE
        # resets the three gate probes.
        self.peak_accel_at_handoff: Optional[float] = None
        self.roll_peak_accel_at_handoff: Optional[float] = None
        self.pitch_peak_accel_at_handoff: Optional[float] = None

        self._substate = CENTER if self._enable_center else APPROACH_PROBE
        self._t0: Optional[float] = None
        self._h0: Optional[float] = None
        self._k_explore = self._initial_thrust_gain

        self._centered_since: Optional[float] = None
        self._center_start_t: Optional[float] = None

        # Two VISUAL trim estimators, updated every visual tick by update()
        # rather than by one phase. CENTER's is fast because it gates a dwell;
        # the handoff estimator is slow and is reset/retuned with the probes so
        # the near-field offset equilibrium remains available for diagnostics.
        # It no longer commands DESCENT: image-position P authority is zero from
        # FINAL_PROBE onward.
        self._center_trim = VisualTrim(self._center_trim_tau)
        # Same time constant, but in the geometric-tilt-compensated frame.
        # Its mean is the physical lateral mis-centering signal: zero means the
        # deck centre lies on the world-vertical ray through the camera, even
        # though a tilted camera sees that point away from image centre.
        self._center_physical_trim = VisualTrim(self._center_trim_tau)
        self._handoff_trim = VisualTrim(self._far_trim_tau)
        self._center_trim_snapshot = self._center_trim.snapshot()
        self._center_physical_trim_snapshot = self._center_physical_trim.snapshot()
        self._handoff_trim_snapshot = self._handoff_trim.snapshot()

        # Adaptive physical visual centre.  Unlike the previous two-dwell
        # calibration, this state evolves continuously in CENTER: the visual
        # reference moves slowly while the lateral P/D loop remains the fast
        # controller.  At physical centring the adaptation naturally stops,
        # leaving exactly the P bias required to reject the steady wind.
        self._visual_center_adaptation = VisualCenterAdaptation(
            tau_sec=self._center_visual_adaptation_tau,
            max_bias_norm=self._center_visual_adaptation_max_bias,
            flow_scale_norm_s=self._center_visual_adaptation_flow_scale,
            max_rate_norm_s=self._center_visual_adaptation_max_rate,
        )
        self._visual_center_adaptation_snapshot = (
            self._visual_center_adaptation.snapshot()
        )
        self._center_geometric_offset_x = 0.0
        self._center_geometric_offset_y = 0.0

        # Lateral static-acceleration handoff.
        #
        # CENTER / APPROACH_PROBE only MEASURE the steady command through the
        # existing PlatformProbes; they do not inject a feedforward term.
        # FINAL_PROBE activates that passive estimate and adapts it while P is
        # disabled.  DESCENT starts bumplessly from the committed FINAL_PROBE
        # value and, when configured, keeps the same slow estimator online so
        # low-frequency wind changes can be absorbed during the short descent.
        self._descent_lateral_trim_frozen = False
        self._final_probe_roll_accel_bias = 0.0
        self._final_probe_pitch_accel_bias = 0.0
        self._final_probe_roll_bias_initial = 0.0
        self._final_probe_pitch_bias_initial = 0.0
        self._descent_roll_accel_bias = 0.0
        self._descent_pitch_accel_bias = 0.0
        # DESCENT-entry anchors retained for telemetry.  The live descent bias
        # may move away from these values when online adaptation is enabled; the
        # deviation columns then show exactly how much correction was learned
        # after commitment.
        self._descent_roll_bias_frozen = 0.0
        self._descent_pitch_bias_frozen = 0.0
        self._descent_trim_offset_x = 0.0
        self._descent_trim_offset_y = 0.0
        self._t_approach_entry: Optional[float] = None
        self._approach_hold_since: Optional[float] = None
        self._approach_divergence_integral = 0.0
        self._t_final_probe_entry: Optional[float] = None
        self._t_final_probe_hold_start: Optional[float] = None
        self._t_descend_start: Optional[float] = None
        self._t_landed: Optional[float] = None
        self._t_aborted: Optional[float] = None
        self.last_control = MissionControl(substate=self._substate)

    def mark_landed(self, t: float) -> None:
        """Latch the terminal LANDED substate. Called by bee_node when its own
        touchdown detector fires (_enter_landed_phase).

        The mission routine cannot detect touchdown itself -- it is visual-only and
        has no height -- so bee_node owns the detection and simply TELLS us. Without
        this, mission_substate stayed "descend" through the entire post-touchdown
        zero-thrust hold, and every statistic computed per-phase from the log mixed
        flying rows with sitting-on-the-platform rows.

        Idempotent: the first call wins, so a repeated latch cannot move t_landed.
        """
        if self._substate == LANDED:
            return
        self._substate = LANDED
        self._t_landed = float(t)

    @staticmethod
    def _terminal_inputs(t: float) -> MissionInputs:
        """Minimal MissionInputs for a terminal phase latched from outside.

        LANDED and ABORTED are latched by the node, which has no obligation to
        supply a visual measurement at that instant -- touchdown is detected
        from Gazebo truth contact, and an abort from a vision timeout. Both
        phases read only ``t``.
        """
        return MissionInputs(
            t=float(t), dt=0.0,
            target=TargetEstimate(timestamp=float(t)),
            flow=FlowResult(timestamp=float(t)),
            actuation=ActuationFeedback(),
        )

    def reset(self) -> None:
        for probe in (self._probe, self._roll_probe, self._pitch_probe):
            probe.reset()
            probe.retune(
                highpass_tau_sec=self._far_probe_highpass_tau,
                percentile_window_sec=self._far_probe_window,
                peak_decay_tau_sec=self._far_probe_decay_tau,
            )
        for probe in (self._chi_probe, self._chi_x_probe, self._chi_y_probe):
            probe.reset()
        self._chi_gate_window_active = False
        self._chi_observed_sec = 0.0
        self._chi_x_observed_sec = 0.0
        self._chi_y_observed_sec = 0.0
        self.gate = GateResult()
        self.roll_gate = LateralGateResult()
        self.pitch_gate = LateralGateResult()
        self.tracking_gate = TrackingGateResult()
        self.tracking_x_gate = TrackingGateResult()
        self.tracking_y_gate = TrackingGateResult()
        self.probe_result = ProbeResult()
        self.roll_probe_result = ProbeResult()
        self.pitch_probe_result = ProbeResult()
        self.peak_accel_at_handoff = None
        self.roll_peak_accel_at_handoff = None
        self.pitch_peak_accel_at_handoff = None

        self._substate = CENTER if self._enable_center else APPROACH_PROBE
        self._t0 = None
        self._h0 = None
        self._k_explore = self._initial_thrust_gain

        self._centered_since = None
        self._center_start_t = None
        self._center_trim.reset()
        self._center_trim.retune(self._center_trim_tau)
        self._center_physical_trim.reset()
        self._center_physical_trim.retune(self._center_trim_tau)
        self._handoff_trim.reset()
        self._handoff_trim.retune(self._far_trim_tau)
        self._center_trim_snapshot = self._center_trim.snapshot()
        self._center_physical_trim_snapshot = self._center_physical_trim.snapshot()
        self._handoff_trim_snapshot = self._handoff_trim.snapshot()
        self._visual_center_adaptation.reset()
        self._visual_center_adaptation_snapshot = (
            self._visual_center_adaptation.snapshot()
        )
        self._center_geometric_offset_x = 0.0
        self._center_geometric_offset_y = 0.0
        self._descent_lateral_trim_frozen = False
        self._final_probe_roll_accel_bias = 0.0
        self._final_probe_pitch_accel_bias = 0.0
        self._final_probe_roll_bias_initial = 0.0
        self._final_probe_pitch_bias_initial = 0.0
        self._descent_roll_accel_bias = 0.0
        self._descent_pitch_accel_bias = 0.0
        self._descent_roll_bias_frozen = 0.0
        self._descent_pitch_bias_frozen = 0.0
        self._descent_trim_offset_x = 0.0
        self._descent_trim_offset_y = 0.0
        self._t_approach_entry = None
        self._approach_hold_since = None
        self._approach_divergence_integral = 0.0
        self._t_final_probe_entry = None
        self._t_final_probe_hold_start = None
        self._t_descend_start = None
        self._t_landed = None
        self._t_aborted = None
        self.last_control = MissionControl(substate=self._substate)

    def start(self, t: float, start_height_m: float) -> None:
        self.reset()
        self._t0 = float(t)
        self._h0 = max(1e-3, float(start_height_m))
        self._k_explore = self._initial_thrust_gain

    @property
    def substate(self) -> str:
        return self._substate

    def mark_aborted(self, t: float) -> None:
        """Latch the terminal ABORTED substate.

        The mission cannot detect an abort itself -- loss of target, an offboard
        dropout or a MAVSDK failure are all outside its visual world -- so the
        node tells it, exactly as with :meth:`mark_landed`.

        This exists so that ``mission_substate`` stays a real, self-describing
        column. Previously the node overwrote the logged substate with the string
        "aborted" at write time, which meant the CSV disagreed with what the
        mission object actually believed.

        Idempotent: the first call wins.
        """
        # Derived from the registry (spec.terminal), so a future terminal phase
        # is covered by adding it to phases/, not by editing this tuple.
        if self._substate in TERMINAL_SUBSTATES:
            return
        self._substate = ABORTED
        self._t_aborted = float(t)
        self.last_control = phases.aborted.run(
            self, self._terminal_inputs(float(t)), just_entered=True)

    TELEMETRY_PREFIX: ClassVar[str] = "mission"

    TELEMETRY_FIELDS: ClassVar[tuple] = (
        "substate",
        "divergence_setpoint_1_s", "thrust_gain_k",
        "lateral_p_scale", "lateral_d_scale",
        # 1 when the large-offset blend is allowed to attenuate the D branch.
        # FINAL_PROBE logs 0: its damping must equal the commanded d_scale,
        # because that is the number the lateral feasibility gates rest on.
        "lateral_d_offset_attenuated",
        "roll_p_scale", "roll_d_scale", "pitch_p_scale", "pitch_d_scale",
        "roll_offset_setpoint", "pitch_offset_setpoint",
        "roll_accel_feedforward_m_s2", "pitch_accel_feedforward_m_s2",
        "center_trim_mean_x", "center_trim_mean_y",
        "center_trim_mean_radius", "center_trim_residual_radius",
        "center_trim_tau_sec", "center_trim_mean_radius_max",
        "center_trim_residual_radius_max",
        # Physical-centering diagnostics and the slow adaptive visual reference.
        "center_geometric_offset_x", "center_geometric_offset_y",
        "center_physical_mean_x", "center_physical_mean_y",
        "center_physical_mean_radius", "center_physical_residual_radius",
        "center_visual_bias_x", "center_visual_bias_y",
        "center_visual_bias_radius",
        "center_visual_adaptation_rate_x_norm_s",
        "center_visual_adaptation_rate_y_norm_s",
        "center_visual_adaptation_weight", "center_visual_adaptation_active",
        "center_visual_adaptation_tau_sec",
        # Slow VISUAL handoff trim, conditioned and reset with the acceleration
        # probes.  DESCENT retains this image-space value for diagnostics only;
        # lateral P has zero command authority there.  The center_* columns above
        # stay the fast CENTER-gate estimator so existing analyses keep meaning.
        "handoff_trim_mean_x", "handoff_trim_mean_y",
        "handoff_trim_mean_radius", "handoff_trim_residual_radius",
        "handoff_trim_tau_sec",
        "descent_trim_offset_x", "descent_trim_offset_y",
        "descent_lateral_trim_frozen",
        # Frozen reference vs live estimate: the difference is how far the wind
        # has moved since FINAL_PROBE committed.
        "descent_roll_bias_frozen_m_s2", "descent_pitch_bias_frozen_m_s2",
        "descent_bias_adaptive", "descent_bias_tau_sec",
        "descent_bias_deviation_limit_m_s2",
        "descent_roll_bias_deviation_m_s2", "descent_pitch_bias_deviation_m_s2",
        "enable_integral",
        "approach_hold_area_fraction", "approach_log_scale_error",
        "approach_measured_divergence_1_s", "approach_hold_condition",
        "approach_hold_dwell_sec",
        # Feasibility gate inputs and outputs.
        "peak_accel_m_s2", "roll_peak_accel_m_s2", "pitch_peak_accel_m_s2",
        "k_min", "k_explore", "k_probe", "k_floor", "k_ceiling_leg",
        "k_ceiling_probe", "h_crit_m", "h_pred_m",
        "vertical_feasible", "roll_feasible", "pitch_feasible", "feasible",
        # Explicit three-question verdict per controlled axis:
        # upper bound -> stable?, lower bound -> enough authority?,
        # visual mismatch -> enough tracking bandwidth?
        "vertical_stability_ok", "vertical_authority_ok",
        "vertical_gain_margin_exists", "vertical_tracking_ok",
        "vertical_landing_feasible",
        "roll_stability_ok", "roll_authority_ok",
        "roll_gain_margin_exists", "roll_tracking_ok",
        "roll_landing_feasible",
        "pitch_stability_ok", "pitch_authority_ok",
        "pitch_gain_margin_exists", "pitch_tracking_ok",
        "pitch_landing_feasible",
        "roll_k_min", "pitch_k_min", "roll_k_probe", "pitch_k_probe",
        "roll_k_applied", "pitch_k_applied",
        "roll_k_target", "pitch_k_target",
        "roll_k_floor", "pitch_k_floor",
        "roll_k_touchdown", "pitch_k_touchdown",
        "roll_k_ceiling_leg", "pitch_k_ceiling_leg",
        "roll_k_ceiling_probe", "pitch_k_ceiling_probe",
        "vertical_accel_capacity_floor_m_s2",
        "vertical_accel_capacity_ceiling_m_s2",
        "roll_accel_capacity_floor_m_s2",
        "roll_accel_capacity_ceiling_m_s2",
        "pitch_accel_capacity_floor_m_s2",
        "pitch_accel_capacity_ceiling_m_s2",
        "vertical_window_exists", "roll_window_exists", "pitch_window_exists",
        "vertical_probe_within_ceiling", "roll_probe_within_ceiling",
        "pitch_probe_within_ceiling",
        "vertical_floor_within_ceiling", "roll_floor_within_ceiling",
        "pitch_floor_within_ceiling",
        "infeasible_axes", "infeasible_criteria", "infeasible_reason",
        # Live probe state.
        "probe_phase", "probe_active",
        "probe_accel_m_s2", "probe_mean_accel_m_s2",
        "probe_residual_accel_m_s2", "probe_percentile_accel_m_s2",
        "probe_peak_accel_m_s2", "probe_peak_accel_at_handoff_m_s2",
        "probe_elapsed_sec", "probe_total_elapsed_sec",
        "roll_probe_accel_m_s2", "roll_probe_mean_accel_m_s2",
        "roll_probe_residual_accel_m_s2", "roll_probe_percentile_accel_m_s2",
        "roll_probe_peak_accel_m_s2", "roll_probe_peak_accel_at_handoff_m_s2",
        "pitch_probe_accel_m_s2", "pitch_probe_mean_accel_m_s2",
        "pitch_probe_residual_accel_m_s2", "pitch_probe_percentile_accel_m_s2",
        "pitch_probe_peak_accel_m_s2", "pitch_probe_peak_accel_at_handoff_m_s2",
        "near_field_height_m",
        # Visual synchronisation gates. Legacy unqualified ``chi`` fields remain
        # the vertical (z) channel; x/y are added explicitly.
        "chi", "chi_abs_1_s2", "chi_divergence_rate_1_s2",
        "chi_percentile_1_s2", "chi_peak_1_s2", "chi_limit_1_s2",
        "chi_observed_sec", "chi_derivative_ready",
        "tracking_decision_chi_peak_1_s2",
        # Explicit z aliases make the tracking schema parallel to x/y while
        # preserving every legacy unqualified vertical column above.
        "tracking_z_decision_chi_peak_1_s2",
        "tracking_z_ready", "tracking_z_synchronized", "tracking_z_feasible",
        "chi_x", "chi_x_abs_1_s2", "chi_x_rate_1_s2",
        "chi_x_percentile_1_s2", "chi_x_peak_1_s2", "chi_x_limit_1_s2",
        "chi_x_observed_sec", "chi_x_derivative_ready",
        "tracking_x_decision_chi_peak_1_s2",
        "tracking_x_ready", "tracking_x_synchronized", "tracking_x_feasible",
        "chi_y", "chi_y_abs_1_s2", "chi_y_rate_1_s2",
        "chi_y_percentile_1_s2", "chi_y_peak_1_s2", "chi_y_limit_1_s2",
        "chi_y_observed_sec", "chi_y_derivative_ready",
        "tracking_y_decision_chi_peak_1_s2",
        "tracking_y_ready", "tracking_y_synchronized", "tracking_y_feasible",
        "tracking_ready", "tracking_synchronized",
        "tracking_enabled", "tracking_feasible",
    )

    _PROBE_COLUMNS: ClassVar[dict] = {
        "probe_phase": "probe_phase",
        "probe_active": "probe_active",
        "probe_accel": "probe_accel_m_s2",
        "probe_mean_accel": "probe_mean_accel_m_s2",
        "probe_residual_accel": "probe_residual_accel_m_s2",
        "probe_percentile_accel": "probe_percentile_accel_m_s2",
        "probe_peak_accel": "probe_peak_accel_m_s2",
        "probe_peak_accel_at_handoff": "probe_peak_accel_at_handoff_m_s2",
        "probe_elapsed_sec": "probe_elapsed_sec",
        "probe_total_elapsed_sec": "probe_total_elapsed_sec",
        "roll_probe_accel": "roll_probe_accel_m_s2",
        "roll_probe_mean_accel": "roll_probe_mean_accel_m_s2",
        "roll_probe_residual_accel": "roll_probe_residual_accel_m_s2",
        "roll_probe_percentile_accel": "roll_probe_percentile_accel_m_s2",
        "roll_probe_peak_accel": "roll_probe_peak_accel_m_s2",
        "roll_probe_peak_accel_at_handoff": "roll_probe_peak_accel_at_handoff_m_s2",
        "pitch_probe_accel": "pitch_probe_accel_m_s2",
        "pitch_probe_mean_accel": "pitch_probe_mean_accel_m_s2",
        "pitch_probe_residual_accel": "pitch_probe_residual_accel_m_s2",
        "pitch_probe_percentile_accel": "pitch_probe_percentile_accel_m_s2",
        "pitch_probe_peak_accel": "pitch_probe_peak_accel_m_s2",
        "pitch_probe_peak_accel_at_handoff": "pitch_probe_peak_accel_at_handoff_m_s2",
        "near_field_height_m": "near_field_height_m",
    }

    @classmethod
    def telemetry_fields(cls) -> Sequence[str]:
        return cls.TELEMETRY_FIELDS

    def telemetry(self) -> Mapping[str, Any]:
        """One row's worth of mission diagnostics, in the mission's own words."""
        mc = self.last_control
        info = mc.info or {}
        gate, roll_gate, pitch_gate = self.gate, self.roll_gate, self.pitch_gate

        # An unseeded estimator reports inf, which would litter the CSV with a
        # sentinel that reads like a number. Map "no estimate yet" to None here
        # and let blank() turn it into an empty cell.
        center_trim_x, center_trim_y, center_trim_radius, center_trim_residual = (
            _trim_row(self._center_trim_snapshot)
        )
        (
            center_phys_x, center_phys_y, center_phys_radius, center_phys_residual
        ) = _trim_row(self._center_physical_trim_snapshot)
        handoff_trim_x, handoff_trim_y, handoff_trim_radius, handoff_trim_residual = (
            _trim_row(self._handoff_trim_snapshot)
        )

        row: dict = {
            "substate": self._substate,
            "divergence_setpoint_1_s": mc.divergence_setpoint,
            "thrust_gain_k": blank(mc.thrust_gain_override),
            "lateral_p_scale": mc.lateral_p_scale,
            "lateral_d_scale": mc.lateral_d_scale,
            "lateral_d_offset_attenuated": int(bool(mc.scale_lateral_d_with_offset)),
            "roll_p_scale": blank(mc.roll_p_scale),
            "roll_d_scale": blank(mc.roll_d_scale),
            "pitch_p_scale": blank(mc.pitch_p_scale),
            "pitch_d_scale": blank(mc.pitch_d_scale),
            "roll_offset_setpoint": mc.roll_offset_setpoint,
            "pitch_offset_setpoint": mc.pitch_offset_setpoint,
            "roll_accel_feedforward_m_s2": mc.roll_accel_feedforward_m_s2,
            "pitch_accel_feedforward_m_s2": mc.pitch_accel_feedforward_m_s2,
            # Read from the estimator, NOT from info: these are properties of
            # the routine, so every phase logs them without having to remember
            # to copy four keys into its info dict. Blank only while the mean
            # is unseeded, which is a real gap rather than a measured zero.
            "center_trim_mean_x": blank(center_trim_x),
            "center_trim_mean_y": blank(center_trim_y),
            "center_trim_mean_radius": blank(center_trim_radius),
            "center_trim_residual_radius": blank(center_trim_residual),
            "center_trim_tau_sec": self._center_trim_tau,
            "center_trim_mean_radius_max": self._center_trim_mean_radius_max,
            "center_trim_residual_radius_max": self._center_trim_residual_radius_max,
            "center_geometric_offset_x": self._center_geometric_offset_x,
            "center_geometric_offset_y": self._center_geometric_offset_y,
            "center_physical_mean_x": blank(center_phys_x),
            "center_physical_mean_y": blank(center_phys_y),
            "center_physical_mean_radius": blank(center_phys_radius),
            "center_physical_residual_radius": blank(center_phys_residual),
            "center_visual_bias_x": self._visual_center_adaptation_snapshot.bias_x,
            "center_visual_bias_y": self._visual_center_adaptation_snapshot.bias_y,
            "center_visual_bias_radius": (
                self._visual_center_adaptation_snapshot.bias_radius
            ),
            "center_visual_adaptation_rate_x_norm_s": (
                self._visual_center_adaptation_snapshot.rate_x_norm_s
            ),
            "center_visual_adaptation_rate_y_norm_s": (
                self._visual_center_adaptation_snapshot.rate_y_norm_s
            ),
            "center_visual_adaptation_weight": (
                self._visual_center_adaptation_snapshot.adaptation_weight
            ),
            "center_visual_adaptation_active": int(
                self._visual_center_adaptation_snapshot.active
            ),
            "center_visual_adaptation_tau_sec": self._center_visual_adaptation_tau,
            "handoff_trim_mean_x": blank(handoff_trim_x),
            "handoff_trim_mean_y": blank(handoff_trim_y),
            "handoff_trim_mean_radius": blank(handoff_trim_radius),
            "handoff_trim_residual_radius": blank(handoff_trim_residual),
            "handoff_trim_tau_sec": self._handoff_trim.tau_sec,
            # The frozen values are meaningless before commitment, so they stay
            # blank until DESCENT actually owns them.
            "descent_trim_offset_x": blank(
                self._descent_trim_offset_x
                if self._descent_lateral_trim_frozen else None
            ),
            "descent_trim_offset_y": blank(
                self._descent_trim_offset_y
                if self._descent_lateral_trim_frozen else None
            ),
            "descent_lateral_trim_frozen": int(bool(self._descent_lateral_trim_frozen)),
            "descent_roll_bias_frozen_m_s2": blank(
                self._descent_roll_bias_frozen
                if self._descent_lateral_trim_frozen else None
            ),
            "descent_pitch_bias_frozen_m_s2": blank(
                self._descent_pitch_bias_frozen
                if self._descent_lateral_trim_frozen else None
            ),
            "descent_bias_adaptive": int(
                bool(self._wind_trim_adapt_in_descent and self._substate == DESCEND)
            ),
            "descent_bias_tau_sec": self._wind_trim_tau,
            "descent_bias_deviation_limit_m_s2": self._wind_trim_deviation_limit,
            # Deviation from the exact FINAL_PROBE value committed at DESCENT
            # entry.  Zero means no post-commit correction was required.
            "descent_roll_bias_deviation_m_s2": blank(
                self._descent_roll_accel_bias - self._descent_roll_bias_frozen
                if self._descent_lateral_trim_frozen else None
            ),
            "descent_pitch_bias_deviation_m_s2": blank(
                self._descent_pitch_accel_bias - self._descent_pitch_bias_frozen
                if self._descent_lateral_trim_frozen else None
            ),
            "enable_integral": int(bool(mc.enable_integral)),
            "approach_hold_area_fraction": self._approach_hold_area_fraction,
            "approach_log_scale_error": blank(info.get("approach_log_scale_error")),
            "approach_measured_divergence_1_s": blank(
                info.get("approach_measured_divergence")
            ),
            "approach_hold_condition": blank(info.get("approach_hold_condition")),
            "approach_hold_dwell_sec": blank(info.get("approach_hold_dwell_sec")),

            "peak_accel_m_s2": self.probe_result.peak_accel,
            "roll_peak_accel_m_s2": self.roll_probe_result.peak_accel,
            "pitch_peak_accel_m_s2": self.pitch_probe_result.peak_accel,

            "k_min": gate.k_min,
            "k_explore": gate.k_explore,
            "k_probe": self.probe_gain,
            "k_floor": gate.k_floor,
            "k_ceiling_leg": gate.k_ceiling_leg,
            "k_ceiling_probe": gate.k_ceiling_probe,
            "h_crit_m": gate.h_crit,
            "h_pred_m": blank(info.get("h_pred")),

            # Legacy gain-gate verdicts. These deliberately remain gain-only.
            "vertical_feasible": int(bool(gate.feasible)),
            "roll_feasible": int(bool(roll_gate.feasible)),
            "pitch_feasible": int(bool(pitch_gate.feasible)),
            "feasible": int(bool(self.feasible)),

            # Parallel per-axis interpretation used by the INFEASIBLE report.
            "vertical_stability_ok": int(bool(self.vertical_stability_ok)),
            "vertical_authority_ok": int(bool(self.vertical_authority_ok)),
            "vertical_gain_margin_exists": int(bool(self.vertical_gain_margin_exists)),
            "vertical_tracking_ok": int(bool(self.vertical_tracking_ok)),
            "vertical_landing_feasible": int(bool(self.vertical_landing_feasible)),
            "roll_stability_ok": int(bool(self.roll_stability_ok)),
            "roll_authority_ok": int(bool(self.roll_authority_ok)),
            "roll_gain_margin_exists": int(bool(self.roll_gain_margin_exists)),
            "roll_tracking_ok": int(bool(self.roll_tracking_ok)),
            "roll_landing_feasible": int(bool(self.roll_landing_feasible)),
            "pitch_stability_ok": int(bool(self.pitch_stability_ok)),
            "pitch_authority_ok": int(bool(self.pitch_authority_ok)),
            "pitch_gain_margin_exists": int(bool(self.pitch_gain_margin_exists)),
            "pitch_tracking_ok": int(bool(self.pitch_tracking_ok)),
            "pitch_landing_feasible": int(bool(self.pitch_landing_feasible)),

            "roll_k_min": roll_gate.k_min,
            "pitch_k_min": pitch_gate.k_min,
            "roll_k_probe": roll_gate.k_probe,
            "pitch_k_probe": pitch_gate.k_probe,
            "roll_k_applied": blank(info.get("roll_k")),
            "pitch_k_applied": blank(info.get("pitch_k")),
            "roll_k_target": roll_gate.k_target,
            "pitch_k_target": pitch_gate.k_target,
            "roll_k_floor": roll_gate.k_floor,
            "pitch_k_floor": pitch_gate.k_floor,
            "roll_k_touchdown": roll_gate.k_touchdown,
            "pitch_k_touchdown": pitch_gate.k_touchdown,
            "roll_k_ceiling_leg": roll_gate.k_ceiling_leg,
            "pitch_k_ceiling_leg": pitch_gate.k_ceiling_leg,
            "roll_k_ceiling_probe": roll_gate.k_ceiling_probe,
            "pitch_k_ceiling_probe": pitch_gate.k_ceiling_probe,

            "vertical_accel_capacity_floor_m_s2": gate.accel_capacity_floor,
            "vertical_accel_capacity_ceiling_m_s2": gate.accel_capacity_ceiling,
            "roll_accel_capacity_floor_m_s2": roll_gate.accel_capacity_floor,
            "roll_accel_capacity_ceiling_m_s2": roll_gate.accel_capacity_ceiling,
            "pitch_accel_capacity_floor_m_s2": pitch_gate.accel_capacity_floor,
            "pitch_accel_capacity_ceiling_m_s2": pitch_gate.accel_capacity_ceiling,

            "vertical_window_exists": int(bool(gate.window_exists)),
            "roll_window_exists": int(bool(roll_gate.window_exists)),
            "pitch_window_exists": int(bool(pitch_gate.window_exists)),
            "vertical_probe_within_ceiling": int(bool(gate.probe_within_ceiling)),
            "roll_probe_within_ceiling": int(bool(roll_gate.probe_within_ceiling)),
            "pitch_probe_within_ceiling": int(bool(pitch_gate.probe_within_ceiling)),
            "vertical_floor_within_ceiling": int(bool(gate.floor_within_ceiling)),
            "roll_floor_within_ceiling": int(bool(roll_gate.floor_within_ceiling)),
            "pitch_floor_within_ceiling": int(bool(pitch_gate.floor_within_ceiling)),
            "infeasible_axes": blank(info.get("infeasible_axes")),
            "infeasible_criteria": blank(info.get("infeasible_criteria")),
            "infeasible_reason": blank(info.get("infeasible_reason")),
        }

        tracking = self.tracking_gate
        tracking_x = self.tracking_x_gate
        tracking_y = self.tracking_y_gate
        row.update({
            "chi": self._chi_probe.chi,
            "chi_abs_1_s2": self._chi_probe.abs_chi,
            "chi_divergence_rate_1_s2": self._chi_probe.signal_rate,
            "chi_percentile_1_s2": self._chi_probe.percentile_chi,
            "chi_peak_1_s2": self._chi_probe.peak_chi,
            "chi_limit_1_s2": self._tracking_z_chi_limit,
            "chi_observed_sec": self._chi_observed_sec,
            "chi_derivative_ready": int(bool(self._chi_probe.derivative_ready)),
            # Frozen value used by the one-time FINAL_PROBE decision. The live
            # chi_peak above is free to keep evolving during DESCENT diagnostics.
            "tracking_decision_chi_peak_1_s2": tracking.chi_peak,
            "tracking_z_decision_chi_peak_1_s2": tracking.chi_peak,
            "tracking_z_ready": int(bool(tracking.ready)),
            "tracking_z_synchronized": int(bool(tracking.synchronized)),
            "tracking_z_feasible": int(bool(tracking.feasible)),
            "chi_x": self._chi_x_probe.chi,
            "chi_x_abs_1_s2": self._chi_x_probe.abs_chi,
            "chi_x_rate_1_s2": self._chi_x_probe.signal_rate,
            "chi_x_percentile_1_s2": self._chi_x_probe.percentile_chi,
            "chi_x_peak_1_s2": self._chi_x_probe.peak_chi,
            "chi_x_limit_1_s2": self._tracking_x_chi_limit,
            "chi_x_observed_sec": self._chi_x_observed_sec,
            "chi_x_derivative_ready": int(bool(self._chi_x_probe.derivative_ready)),
            "tracking_x_decision_chi_peak_1_s2": tracking_x.chi_peak,
            "tracking_x_ready": int(bool(tracking_x.ready)),
            "tracking_x_synchronized": int(bool(tracking_x.synchronized)),
            "tracking_x_feasible": int(bool(tracking_x.feasible)),
            "chi_y": self._chi_y_probe.chi,
            "chi_y_abs_1_s2": self._chi_y_probe.abs_chi,
            "chi_y_rate_1_s2": self._chi_y_probe.signal_rate,
            "chi_y_percentile_1_s2": self._chi_y_probe.percentile_chi,
            "chi_y_peak_1_s2": self._chi_y_probe.peak_chi,
            "chi_y_limit_1_s2": self._tracking_y_chi_limit,
            "chi_y_observed_sec": self._chi_y_observed_sec,
            "chi_y_derivative_ready": int(bool(self._chi_y_probe.derivative_ready)),
            "tracking_y_decision_chi_peak_1_s2": tracking_y.chi_peak,
            "tracking_y_ready": int(bool(tracking_y.ready)),
            "tracking_y_synchronized": int(bool(tracking_y.synchronized)),
            "tracking_y_feasible": int(bool(tracking_y.feasible)),
            # Aggregate status across z, x and y. The legacy per-axis decision
            # peak above remains vertical so existing analyses do not change
            # meaning silently.
            "tracking_ready": int(bool(
                tracking.ready and tracking_x.ready and tracking_y.ready
            )),
            "tracking_synchronized": int(bool(
                tracking.synchronized
                and tracking_x.synchronized
                and tracking_y.synchronized
            )),
            "tracking_enabled": int(bool(tracking.enabled)),
            "tracking_feasible": int(bool(self.tracking_feasible)),
        })

        probe = self.probe_telemetry()
        for key, column in self._PROBE_COLUMNS.items():
            if key in probe:
                value = probe[key]
                if isinstance(value, bool):
                    value = int(value)
                row[column] = blank(value)
        return row

    def probe_telemetry(self) -> dict:
        """Return per-step vertical, roll and pitch probe diagnostics."""
        active = self._substate in (APPROACH_PROBE, FINAL_PROBE)
        near = self._t_final_probe_hold_start is not None

        # The PER-STEP fields are None whenever the probe is not running (CENTER,
        # DESCEND, INFEASIBLE) -- the probe simply stops being updated there, so its
        # last values would otherwise persist and draw a flat line that looks like a
        # measurement but is just a frozen register. None -> blank CSV cell -> a gap
        # in the plot, which is the truth.
        #
        # FINAL_PROBE peak_accel stays available after the probe because it is a
        # gate input.  peak_accel_at_handoff is the frozen APPROACH diagnostic,
        # kept separately so the two phases never share evidence.
        return {
            "probe_active": active,
            "probe_phase": ("near" if near else "far") if active else "",
            "probe_accel": self._probe.accel if active else None,
            "probe_mean_accel": self._probe.mean_accel if active else None,
            "probe_residual_accel": self._probe.residual_accel if active else None,
            "probe_percentile_accel": self._probe.percentile_accel if active else None,
            "probe_peak_accel": self._probe.peak_accel,
            "probe_peak_decay_tau_sec": self._probe.peak_decay_tau_sec,
            "probe_elapsed_sec": self.probe_result.duration_sec,
            "probe_total_elapsed_sec": self.probe_result.total_duration_sec,
            "probe_peak_accel_at_handoff": self.peak_accel_at_handoff,
            "roll_probe_accel": self._roll_probe.accel if active else None,
            "roll_probe_mean_accel": self._roll_probe.mean_accel if active else None,
            "roll_probe_residual_accel": self._roll_probe.residual_accel if active else None,
            "roll_probe_percentile_accel": self._roll_probe.percentile_accel if active else None,
            "roll_probe_peak_accel": self._roll_probe.peak_accel,
            "roll_probe_peak_accel_at_handoff": self.roll_peak_accel_at_handoff,
            "pitch_probe_accel": self._pitch_probe.accel if active else None,
            "pitch_probe_mean_accel": self._pitch_probe.mean_accel if active else None,
            "pitch_probe_residual_accel": self._pitch_probe.residual_accel if active else None,
            "pitch_probe_percentile_accel": self._pitch_probe.percentile_accel if active else None,
            "pitch_probe_peak_accel": self._pitch_probe.peak_accel,
            "pitch_probe_peak_accel_at_handoff": self.pitch_peak_accel_at_handoff,
            "vertical_probe_feasible": self.gate.feasible,
            "roll_probe_feasible": self.roll_gate.feasible,
            "pitch_probe_feasible": self.pitch_gate.feasible,
            "landing_feasible": self.feasible,
            "k_probe": self._compute_probe_gain(),
            "near_field_height_m": self._near_field_height,
        }

    def _compute_probe_gain(self) -> float:
        """The gain flown through FINAL_PROBE -- and the anchor of the schedule.

        k_explore (6.5) is a FAR-FIELD gain. The de Croon ceiling shrinks with
        height, k_ceiling(h) = 2*s*h/dt, so k_explore is only admissible above
        h = k_explore*dt/(2*s). At the flown constants that is ~0.46 m -- but
        FINAL_PROBE fires on FOV saturation, i.e. WELL BELOW it. Holding k_explore
        into the near field therefore probes ABOVE the stability ceiling, and since
        the probe measures the THRUST-COMMAND RESIDUAL, any resulting self-induced
        oscillation is counted as platform acceleration. That does not merely risk
        instability: it corrupts the one number the feasibility gate rests on.

        So the near-field probe is flown at the same fraction of the ceiling that
        the descent targets, evaluated at the height the probe actually happens at:

            k_probe = min(k_explore, ceiling_margin * k_ceiling(near_field_height))

        The min() means that if the near-field trigger fires while still high enough
        that k_explore is already under the ceiling, nothing is dropped -- the gain
        is only ever reduced to become admissible, never raised.
        """
        ceiling = ceiling_gain_at_height(
            self._near_field_height, self._vertical_stability_dt, self._safety
        )
        return min(self._initial_thrust_gain, self._ceiling_margin * ceiling)

    @property
    def probe_gain(self) -> float:
        """Gain held flat through FINAL_PROBE; the descent schedule starts here."""
        return self._compute_probe_gain()

    def _compute_lateral_probe_gain(self, kappa: float, d_gain: float) -> float:
        """The lateral D gain flown through FINAL_PROBE, derived not asserted.

        Exactly the vertical construction in _compute_probe_gain(), transposed
        onto the lateral ceiling::

            Kd_probe = min(Kd_far, ceiling_margin * k_ceiling_lat(near_field_height))

        with the same near_field_height_m, the same ceiling_margin and the same
        ceiling_safety_factor.  ``Kd_far = d_gain * center_lateral_d_scale`` is
        the lateral analogue of k_explore: the far-field gain, admissible while
        high and progressively less so as the ceiling shrinks with height.

        The lateral ceiling carries two terms the vertical one does not --

            k_ceiling_lat(h) = 2*s*h/(kappa*dt) - c_max/kappa

        -- so it depends on the FOV through kappa and on the admitted closing
        speed.  That is why this takes the axis's own kappa and D gain: on a
        non-square lens the roll and pitch ceilings genuinely differ, and a
        single shared scale would silently fly the wider axis closer to its
        bound than the narrower one.

        As with the vertical, the min() only ever reduces: if the far-field gain
        is already admissible at the handoff height nothing is dropped, and the
        schedule below is flat.  A platform with less lateral authority -- wider
        lens, slower visual loop, lower handoff -- gets a real decay from the
        same code and the same constants.
        """
        ceiling = lateral_ceiling_gain_at_height(
            self._near_field_height,
            self._lateral_stability_dt,
            kappa,
            self._max_closing_speed,
            self._safety,
        )
        far_field_gain = max(0.0, float(d_gain)) * self._center_lateral_d_scale
        return min(far_field_gain, self._ceiling_margin * ceiling)

    @property
    def roll_probe_lateral_gain(self) -> float:
        return self._compute_lateral_probe_gain(self._roll_kappa, self._roll_d_gain)

    @property
    def pitch_probe_lateral_gain(self) -> float:
        return self._compute_lateral_probe_gain(self._pitch_kappa, self._pitch_d_gain)

    @property
    def roll_probe_lateral_d_scale(self) -> float:
        """``roll_probe_lateral_gain`` expressed as the scale ControlLaw wants."""
        return (
            self.roll_probe_lateral_gain / self._roll_d_gain
            if self._roll_d_gain > 1e-9
            else 0.0
        )

    @property
    def pitch_probe_lateral_d_scale(self) -> float:
        return (
            self.pitch_probe_lateral_gain / self._pitch_d_gain
            if self._pitch_d_gain > 1e-9
            else 0.0
        )

    @property
    def feasible(self) -> bool:
        """All authority gates plus all three visual tracking checks.

        The z/x/y tracking verdicts are frozen at the FINAL_PROBE decision and
        become diagnostic-only after DESCENT begins.
        """
        return bool(
            self.vertical_landing_feasible
            and self.roll_landing_feasible
            and self.pitch_landing_feasible
        )

    @property
    def tracking_feasible(self) -> bool:
        return bool(
            self.tracking_gate.feasible
            and self.tracking_x_gate.feasible
            and self.tracking_y_gate.feasible
        )

    @property
    def tracking_z_feasible(self) -> bool:
        return bool(self.tracking_gate.feasible)

    @property
    def tracking_x_feasible(self) -> bool:
        return bool(self.tracking_x_gate.feasible)

    @property
    def tracking_y_feasible(self) -> bool:
        return bool(self.tracking_y_gate.feasible)

    @property
    def vertical_feasible(self) -> bool:
        return bool(self.gate.feasible)

    @property
    def roll_feasible(self) -> bool:
        return bool(self.roll_gate.feasible)

    @property
    def pitch_feasible(self) -> bool:
        return bool(self.pitch_gate.feasible)

    # ------------------------------------------------------------------
    # Explicit per-axis feasibility interpretation.  The legacy *_feasible
    # properties above remain gain-window verdicts; these properties expose the
    # three questions independently and then combine them with visual tracking.
    @property
    def vertical_stability_ok(self) -> bool:
        return bool(
            self.gate.probe_within_ceiling
            and self.gate.floor_within_ceiling
        )

    @property
    def vertical_authority_ok(self) -> bool:
        return bool(self.gate.start_above_floor)

    @property
    def vertical_gain_margin_exists(self) -> bool:
        return bool(self.gate.window_exists)

    @property
    def vertical_tracking_ok(self) -> bool:
        return bool(self.tracking_z_feasible)

    @property
    def vertical_landing_feasible(self) -> bool:
        return bool(self.gate.feasible and self.vertical_tracking_ok)

    @property
    def roll_stability_ok(self) -> bool:
        return bool(
            self.roll_gate.probe_within_ceiling
            and self.roll_gate.floor_within_ceiling
        )

    @property
    def roll_authority_ok(self) -> bool:
        return bool(self.roll_gate.start_above_floor)

    @property
    def roll_gain_margin_exists(self) -> bool:
        return bool(self.roll_gate.window_exists)

    @property
    def roll_tracking_ok(self) -> bool:
        # image-x optical flow is the roll-controlled lateral channel
        return bool(self.tracking_x_feasible)

    @property
    def roll_landing_feasible(self) -> bool:
        return bool(self.roll_gate.feasible and self.roll_tracking_ok)

    @property
    def pitch_stability_ok(self) -> bool:
        return bool(
            self.pitch_gate.probe_within_ceiling
            and self.pitch_gate.floor_within_ceiling
        )

    @property
    def pitch_authority_ok(self) -> bool:
        return bool(self.pitch_gate.start_above_floor)

    @property
    def pitch_gain_margin_exists(self) -> bool:
        return bool(self.pitch_gate.window_exists)

    @property
    def pitch_tracking_ok(self) -> bool:
        # image-y optical flow is the pitch-controlled lateral channel
        return bool(self.tracking_y_feasible)

    @property
    def pitch_landing_feasible(self) -> bool:
        return bool(self.pitch_gate.feasible and self.pitch_tracking_ok)

    def update(self, inputs: MissionInputs) -> MissionControl:
        """Advance the mission by one controlled frame.

        Takes a single :class:`~mission_types.MissionInputs` rather than the
        previous eleven keyword arguments. The caller no longer unpacks
        ``target``/``flow`` on the mission's behalf, so a phase that needs one
        more visual signal is a change in this file only.

        Dispatch is table-driven via :attr:`PHASES`. Adding a phase means writing
        ``_do_<name>`` and adding one ``PhaseSpec``; there is no if-chain to
        extend and no display name to register anywhere else.
        """
        t = float(inputs.t)
        dt = max(1e-3, float(inputs.dt))
        inputs = dc_replace(inputs, t=t, dt=dt)

        if self._t0 is None:
            self._t0 = t
        if self._h0 is None:
            self._h0 = 5.0

        # Static lateral acceleration is absent in CENTER/APPROACH.  It adapts
        # from the FINAL_PROBE handoff onward -- the tick where the steady wind
        # force moves off the P error -- and may keep adapting through DESCENT
        # to follow slow wind changes.  The update is causal: it uses the
        # allocator-realized command from the previous control tick, so the
        # residual FINAL_PROBE P contribution is folded in automatically.
        self._update_near_field_lateral_bias(inputs)

        # Shared per-tick trim estimators run BEFORE dispatch, so every phase
        # reads the same value and none of them owns that measurement update.
        # The adaptive visual-centre state itself is advanced by CENTER and
        # APPROACH_PROBE only, where lateral image-position P remains meaningful.
        self._update_lateral_trim(inputs)
        # An unregistered substate falls through to DESCEND, preserving the
        # previous if-chain's final `return self._do_descend(t)`.
        spec = PHASES.get(self._substate) or PHASES[DESCEND]
        control = spec.handler(self, inputs)
        # Retained so telemetry() can describe the tick without the caller
        # handing the MissionControl back to the logger.
        self.last_control = control
        return control

    def _update_near_field_lateral_bias(self, inputs: MissionInputs) -> None:
        """Adapt the static lateral acceleration term after lateral P is removed.

        APPROACH_PROBE passively estimates the steady command through the
        existing roll/pitch PlatformProbe means.  At the FINAL_PROBE handoff that
        estimate is activated as feedforward, and lateral image-position P drops
        from its APPROACH value to a small residual
        (``final_probe_lateral_p_scale``).  The near-field law is then

            a_cmd = a_static + a_P(offset - e_geom) + a_D(flow)

        and the causal EMA

            a_static <- a_static + beta * (a_realized - a_static)

        absorbs the slow part of BOTH corrections.  That is what makes the
        descent boundary bumpless even though P is dropped there: by the time
        the gates pass, whatever mean force the residual P was contributing has
        already migrated into ``a_static``, and DESCENT inherits it in the
        feedforward.  The wind rejection moves from the P error to the static
        term gradually rather than at a single tick.

        FINAL_PROBE uses ``_final_probe_*_accel_bias``; DESCENT starts from that
        exact value and, when ``wind_trim_adapt_in_descent`` is enabled, keeps
        the same estimator running on ``_descent_*_accel_bias``.

        Both phases share ONE safety neighbourhood centred on the passive
        APPROACH seed.  Continuing adaptation in DESCENT therefore does not buy
        a second independent deviation allowance after the FINAL_PROBE update.
        """
        if self._substate not in (FINAL_PROBE, DESCEND):
            return
        if self._substate == DESCEND and not self._wind_trim_adapt_in_descent:
            return
        if not (inputs.target_found and inputs.flow_valid):
            return

        beta = 1.0 - math.exp(
            -max(1e-3, float(inputs.dt)) / max(1e-3, self._wind_trim_tau)
        )
        limit = self._wind_trim_deviation_limit

        if self._substate == FINAL_PROBE:
            roll_current = self._final_probe_roll_accel_bias
            pitch_current = self._final_probe_pitch_accel_bias
        else:
            # DESCENT begins only after _commit_descent_lateral_trim(), so these
            # are the exact FINAL_PROBE values flown on the handoff tick.
            roll_current = self._descent_roll_accel_bias
            pitch_current = self._descent_pitch_accel_bias

        roll_bias = roll_current + beta * (
            float(inputs.actuation.last_roll_accel_cmd) - roll_current
        )
        pitch_bias = pitch_current + beta * (
            float(inputs.actuation.last_pitch_accel_cmd) - pitch_current
        )

        roll_bias = clamp(
            roll_bias,
            self._final_probe_roll_bias_initial - limit,
            self._final_probe_roll_bias_initial + limit,
        )
        pitch_bias = clamp(
            pitch_bias,
            self._final_probe_pitch_bias_initial - limit,
            self._final_probe_pitch_bias_initial + limit,
        )

        if self._substate == FINAL_PROBE:
            self._final_probe_roll_accel_bias = roll_bias
            self._final_probe_pitch_accel_bias = pitch_bias
        else:
            self._descent_roll_accel_bias = roll_bias
            self._descent_pitch_accel_bias = pitch_bias

    @staticmethod
    def _tilt_to_normalized_offset(angle_rad: float, tan_half_fov: float) -> float:
        """Exact pinhole projection into the target's [-1, 1] image coordinate."""
        denom = max(1e-9, abs(float(tan_half_fov)))
        return math.tan(float(angle_rad)) / denom

    def _geometric_tilt_offset(
        self, inputs: MissionInputs
    ) -> tuple[float, float]:
        """Image location of the point directly below the camera.

        This is geometry only. It intentionally contains no wind/disturbance
        compensation. The previous shaped command keeps it causal and matches
        the attitude actually sent to PX4 closely in the validated wind logs.

        Used by CENTER/APPROACH (as one half of the adaptive visual setpoint)
        AND by FINAL_PROBE (on its own).  It carries no far-field assumption:
        it is derived from the attitude command, so it stays exact regardless
        of how much of the frame the target fills.
        """
        roll_geom = -self._tilt_to_normalized_offset(
            inputs.actuation.last_roll_cmd_rad, self._tan_half_hfov
        )
        pitch_geom = -self._tilt_to_normalized_offset(
            inputs.actuation.last_pitch_cmd_rad, self._tan_half_vfov
        )
        roll_geom = clamp(roll_geom, -1.0, 1.0)
        pitch_geom = clamp(pitch_geom, -1.0, 1.0)
        self._center_geometric_offset_x = roll_geom
        self._center_geometric_offset_y = pitch_geom
        return roll_geom, pitch_geom

    def _far_field_lateral_control_terms(
        self, inputs: MissionInputs
    ) -> tuple[float, float, float, float]:
        """Return the adaptive far-field visual setpoint.

        Let ``e_geom`` be the image location of the world-vertical ray through
        the tilted camera and ``b`` the slow CENTER adaptation.  The setpoint is

            e_sp = e_geom - b.

        The physical centring error is ``e_phys = e_meas - e_geom``.  CENTER
        evolves ``b`` slowly from that error; when e_phys -> 0 the bias freezes
        at the value needed to keep the steady P counter-force.

        CENTER and APPROACH_PROBE both evolve the learned bias online.  The
        same direct visual-reference law is therefore used in both phases:
        changing the scheduled P multiplier may temporarily expose a physical
        centring error, and the slow adaptation is allowed to learn the new
        equilibrium naturally instead of algebraically rescaling the bias.
        """
        roll_geom, pitch_geom = self._geometric_tilt_offset(inputs)

        snap = self._visual_center_adaptation_snapshot
        roll_sp = roll_geom - float(snap.bias_x)
        pitch_sp = pitch_geom - float(snap.bias_y)
        roll_sp = clamp(roll_sp, -1.0, 1.0)
        pitch_sp = clamp(pitch_sp, -1.0, 1.0)

        # No acceleration feedforward before FINAL_PROBE.  The adaptive centre
        # is a moving visual reference, not a second force-command path.
        return roll_sp, pitch_sp, 0.0, 0.0

    def _near_field_lateral_setpoint(
        self, inputs: MissionInputs
    ) -> tuple[float, float]:
        """FINAL_PROBE visual setpoint: geometric tilt compensation, no bias.

        The far-field setpoint is ``e_geom - b``.  The learned bias ``b`` exists
        so that the steady P error generates the counter-wind force, which is
        precisely the job the static acceleration trim takes over at this
        handoff.  Carrying ``b`` here as well would ask the loop to supply that
        force twice: the feedforward once, and P again by holding an error of
        about ``b``.  On the 2026-08-19 09:42 run ``b`` had frozen at -0.519
        normalized, so the double count would have parked the vehicle roughly
        26 cm off centre -- worse than the drift this whole change removes.

        ``e_geom`` alone is kept because it is the difference between "the
        target is at image centre" and "the target is physically underneath
        us".  Dropping it too would settle at the tilt offset instead: about
        0.09 normalized, ~5 cm at the FINAL_PROBE height, which is a third of
        ``leg_clearance_m``.
        """
        roll_geom, pitch_geom = self._geometric_tilt_offset(inputs)
        return (
            clamp(roll_geom, -1.0, 1.0),
            clamp(pitch_geom, -1.0, 1.0),
        )

    def _update_visual_center_adaptation(self, inputs: MissionInputs) -> None:
        """Advance the slow visual-centre finder in CENTER and APPROACH_PROBE.

        The physical trim is already based on ``e_meas - e_geom`` and is a
        one-second EMA, so the adaptive state sees a mean/static displacement
        rather than frame noise.  Optical flow only modulates adaptation speed;
        the fast lateral P/D controller remains solely responsible for motion.

        Adaptation deliberately stops at FINAL_PROBE.  The trigger is no longer
        "P authority is removed" -- FINAL_PROBE keeps a small residual P -- but
        that the bias's PURPOSE is transferred there: the static acceleration
        trim takes over the steady wind force, so a bias that exists to hold a
        counter-wind P error has nothing left to do and would double-count it.
        """
        px, py, _radius, _residual = self._center_physical_trim_snapshot
        valid = (
            self._substate in (CENTER, APPROACH_PROBE)
            and bool(inputs.target_found)
            and bool(inputs.flow_valid)
            and not bool(inputs.fov_saturated)
            and math.isfinite(px)
            and math.isfinite(py)
        )
        self._visual_center_adaptation.update(
            physical_error_x=float(px) if math.isfinite(px) else 0.0,
            physical_error_y=float(py) if math.isfinite(py) else 0.0,
            flow_x_norm_s=float(inputs.flow_x_norm_s),
            flow_y_norm_s=float(inputs.flow_y_norm_s),
            valid=valid,
            dt=inputs.dt,
        )
        self._visual_center_adaptation_snapshot = (
            self._visual_center_adaptation.snapshot()
        )

    def _update_visual_mismatch(self, inputs: MissionInputs) -> None:
        """Fold one visual sample into the height-free bandwidth diagnostic.

        chi_z is updated from vertical divergence as before. chi_x and chi_y use

            chi_i = domega_i/dt - omega_i*omega_z,

        where omega_z is the simultaneous vertical divergence. The mission's
        ``inputs.dt`` already comes from consecutive fresh camera/Gazebo-SIM
        timestamps, so all three regressions preserve the same true spacing.

        Only the stationary FINAL_PROBE hold activates the gate clock/envelope;
        DESCENT continues updating all chi channels for diagnosis but cannot
        revoke commitment.
        """
        if not inputs.flow_valid:
            return
        omega_z = float(getattr(inputs.flow, "divergence", 0.0))
        omega_x = inputs.flow_x_norm_s
        omega_y = inputs.flow_y_norm_s

        self._chi_probe.update(omega_z, inputs.dt)
        self._chi_x_probe.update(omega_x, inputs.dt, coupling=omega_z)
        self._chi_y_probe.update(omega_y, inputs.dt, coupling=omega_z)

        if self._chi_gate_window_active and self._chi_probe.derivative_ready:
            self._chi_observed_sec += inputs.dt
        if self._chi_gate_window_active and self._chi_x_probe.derivative_ready:
            self._chi_x_observed_sec += inputs.dt
        if self._chi_gate_window_active and self._chi_y_probe.derivative_ready:
            self._chi_y_observed_sec += inputs.dt

    def _begin_tracking_gate_window(self) -> None:
        """Start FINAL_PROBE envelopes without cooling any derivative history."""
        for probe in (self._chi_probe, self._chi_x_probe, self._chi_y_probe):
            probe.retune(
                percentile_window_sec=self._near_probe_window,
                peak_decay_tau_sec=self._near_probe_decay_tau,
            )
            probe.reset_envelope()
        self._chi_observed_sec = 0.0
        self._chi_x_observed_sec = 0.0
        self._chi_y_observed_sec = 0.0
        self._chi_gate_window_active = True
        self.tracking_gate = TrackingGateResult()
        self.tracking_x_gate = TrackingGateResult()
        self.tracking_y_gate = TrackingGateResult()

    def _refresh_tracking_gate(self) -> TrackingGateResult:
        z_ready = self._chi_probe.result(
            min_duration_sec=self._tracking_min_observation_sec
        ).ready and self._chi_observed_sec >= self._tracking_min_observation_sec
        x_ready = self._chi_x_probe.result(
            min_duration_sec=self._tracking_min_observation_sec
        ).ready and self._chi_x_observed_sec >= self._tracking_min_observation_sec
        y_ready = self._chi_y_probe.result(
            min_duration_sec=self._tracking_min_observation_sec
        ).ready and self._chi_y_observed_sec >= self._tracking_min_observation_sec

        self.tracking_gate = compute_tracking_gate(
            chi_peak=self._chi_probe.peak_chi,
            chi_limit=self._tracking_z_chi_limit,
            ready=z_ready,
            enabled=self._enable_tracking_gate,
        )
        self.tracking_x_gate = compute_tracking_gate(
            chi_peak=self._chi_x_probe.peak_chi,
            chi_limit=self._tracking_x_chi_limit,
            ready=x_ready,
            enabled=self._enable_tracking_gate,
        )
        self.tracking_y_gate = compute_tracking_gate(
            chi_peak=self._chi_y_probe.peak_chi,
            chi_limit=self._tracking_y_chi_limit,
            ready=y_ready,
            enabled=self._enable_tracking_gate,
        )
        return self.tracking_gate

    def _update_probes(
        self,
        last_thrust_cmd: float,
        last_vertical_accel_cmd: Optional[float],
        last_roll_accel_cmd: float,
        last_pitch_accel_cmd: float,
        dt: float,
    ) -> None:
        if last_vertical_accel_cmd is None:
            self._probe.update(last_thrust_cmd, dt)
        else:
            self._probe.update_accel(last_vertical_accel_cmd, dt)
        self._roll_probe.update_accel(last_roll_accel_cmd, dt)
        self._pitch_probe.update_accel(last_pitch_accel_cmd, dt)

    def _retune_probes(self) -> None:
        for probe in (self._probe, self._roll_probe, self._pitch_probe):
            probe.retune(
                highpass_tau_sec=self._near_probe_highpass_tau,
                percentile_window_sec=self._near_probe_window,
                peak_decay_tau_sec=self._near_probe_decay_tau,
            )
        # The handoff trim pairs with PlatformProbe.mean_accel, so it follows
        # the same far -> near conditioning change at the same instant.
        self._handoff_trim.retune(self._near_trim_tau)

    def _refresh_probe_results(self, min_duration_sec: float) -> None:
        self.probe_result = self._probe.result(min_duration_sec)
        self.roll_probe_result = self._roll_probe.result(min_duration_sec)
        self.pitch_probe_result = self._pitch_probe.result(min_duration_sec)

    def _begin_final_probe_measurement(
        self, t: float, inputs: Optional[MissionInputs] = None
    ) -> None:
        """Activate the passive APPROACH static estimate, then reset gate evidence.

        The acceleration feedforward has had ZERO authority up to this point.
        The roll/pitch PlatformProbe means therefore provide a passive estimate
        of the steady lateral command produced by compensated P+D control.  We
        capture those means before resetting the probes and use them as the
        initial FINAL_PROBE static term.

        A direct FINAL_PROBE dispatch (tests/state restoration) may have no
        APPROACH samples.  In that exceptional case the last realized command is
        the best causal seed available; otherwise the seed is zero.
        """
        roll_samples = self._roll_probe.result(0.0).n_samples
        pitch_samples = self._pitch_probe.result(0.0).n_samples
        roll_seed = (
            float(self._roll_probe.mean_accel)
            if roll_samples > 0
            else float(inputs.actuation.last_roll_accel_cmd) if inputs is not None else 0.0
        )
        pitch_seed = (
            float(self._pitch_probe.mean_accel)
            if pitch_samples > 0
            else float(inputs.actuation.last_pitch_accel_cmd) if inputs is not None else 0.0
        )
        self._final_probe_roll_accel_bias = roll_seed
        self._final_probe_pitch_accel_bias = pitch_seed
        self._final_probe_roll_bias_initial = roll_seed
        self._final_probe_pitch_bias_initial = pitch_seed

        self.peak_accel_at_handoff = self._probe.peak_accel
        self.roll_peak_accel_at_handoff = self._roll_probe.peak_accel
        self.pitch_peak_accel_at_handoff = self._pitch_probe.peak_accel

        for probe in (self._probe, self._roll_probe, self._pitch_probe):
            probe.reset()
        # Same provenance rule the probes follow: no APPROACH-era sample may
        # enter the operating point FINAL_PROBE hands to DESCENT.
        self._handoff_trim.reset()
        self._retune_probes()
        self.probe_result = ProbeResult()
        self.roll_probe_result = ProbeResult()
        self.pitch_probe_result = ProbeResult()

        self._t_final_probe_entry = float(t)
        self._t_final_probe_hold_start = float(t)
        self._begin_tracking_gate_window()

    def _compute_lateral_gates(self) -> None:
        # Same derivation the schedule flies, not a second copy of it: a gate
        # that judged a different gain than the vehicle uses would be measuring
        # a vehicle that does not exist.
        roll_probe_gain = self.roll_probe_lateral_gain
        pitch_probe_gain = self.pitch_probe_lateral_gain
        self.roll_gate = compute_lateral_gate(
            peak_accel=self.roll_probe_result.peak_accel,
            flow_admissible_norm_s=self._roll_flow_admissible,
            max_closing_speed_m_s=self._max_closing_speed,
            kappa=self._roll_kappa,
            probe_gain=roll_probe_gain,
            near_field_height_m=self._near_field_height,
            leg_clearance_m=self._leg_clearance,
            stability_dt_sec=self._lateral_stability_dt,
            ceiling_safety_factor=self._safety,
            ceiling_margin=self._ceiling_margin,
        )
        self.pitch_gate = compute_lateral_gate(
            peak_accel=self.pitch_probe_result.peak_accel,
            flow_admissible_norm_s=self._pitch_flow_admissible,
            max_closing_speed_m_s=self._max_closing_speed,
            kappa=self._pitch_kappa,
            probe_gain=pitch_probe_gain,
            near_field_height_m=self._near_field_height,
            leg_clearance_m=self._leg_clearance,
            stability_dt_sec=self._lateral_stability_dt,
            ceiling_safety_factor=self._safety,
            ceiling_margin=self._ceiling_margin,
        )

    def _feasibility_failures(self) -> list[tuple[str, str, str]]:
        """Structured FINAL_PROBE rejection reasons.

        Each entry is ``(axis, criterion, detail)``.  The axis names follow the
        controller channels (VERTICAL/ROLL/PITCH); the visual observables are
        mapped as z->VERTICAL, x->ROLL and y->PITCH.  Keeping the criterion
        explicit makes the eventual report read exactly like the theory:

        * STABILITY_UPPER_BOUND -- is the commanded gain below the ceiling?
        * GAIN_MARGIN -- do the lower and upper bounds overlap?
        * AUTHORITY_LOWER_BOUND -- does the flown gain reach the required floor?
        * VISUAL_MISMATCH -- is the closed loop actually fast enough to track?
        """
        failures: list[tuple[str, str, str]] = []

        def gain_failures(axis: str, gate) -> None:
            if not gate.probe_within_ceiling:
                failures.append((
                    axis, "STABILITY_UPPER_BOUND",
                    f"FINAL_PROBE gain above near-field ceiling: "
                    f"K_probe={gate.k_descend_start:.3f} > "
                    f"K_ceiling_probe={gate.k_ceiling_probe:.3f}",
                ))
            if not gate.window_exists:
                failures.append((
                    axis, "GAIN_MARGIN",
                    f"no admissible gain window: K_min={gate.k_min:.3f} > "
                    f"K_ceiling_leg={gate.k_ceiling_leg:.3f}",
                ))
            elif not gate.start_above_floor:
                failures.append((
                    axis, "AUTHORITY_LOWER_BOUND",
                    f"FINAL_PROBE gain below disturbance-rejection floor: "
                    f"K_probe={gate.k_descend_start:.3f} < K_min={gate.k_min:.3f}",
                ))
            if gate.window_exists and not gate.floor_within_ceiling:
                failures.append((
                    axis, "STABILITY_UPPER_BOUND",
                    f"scheduled touchdown floor above ceiling: "
                    f"K_floor={gate.k_floor:.3f} > "
                    f"K_ceiling_leg={gate.k_ceiling_leg:.3f}",
                ))

        gain_failures("VERTICAL", self.gate)
        gain_failures("ROLL", self.roll_gate)
        gain_failures("PITCH", self.pitch_gate)

        for axis, tracking in (
            ("VERTICAL", self.tracking_gate),
            ("ROLL", self.tracking_x_gate),   # image x -> roll channel
            ("PITCH", self.tracking_y_gate),  # image y -> pitch channel
        ):
            if not tracking.feasible and tracking.reason:
                failures.append((axis, "VISUAL_MISMATCH", tracking.reason))

        if not self._enable_descent:
            failures.append((
                "MISSION", "DESCENT_DISABLED",
                "DESCENT disabled by configuration",
            ))

        return failures

    def _gate_failure_reasons(self) -> list[str]:
        """Human-readable axis + criterion strings for console/CSV logging."""
        return [
            f"{axis} [{criterion}] {detail}"
            for axis, criterion, detail in self._feasibility_failures()
        ]

    def _failed_axes(self) -> str:
        axes: list[str] = []
        for axis, _criterion, _detail in self._feasibility_failures():
            if axis not in axes:
                axes.append(axis)
        return "|".join(axes)

    def _failed_criteria(self) -> str:
        criteria: list[str] = []
        for _axis, criterion, _detail in self._feasibility_failures():
            if criterion not in criteria:
                criteria.append(criterion)
        return "|".join(criteria)

    def _update_lateral_trim(self, inputs: MissionInputs) -> None:
        """Advance both visual-trim estimators for one tick.

        Three visual estimates are maintained: raw image trim, the
        geometric-tilt-compensated physical trim, and the slow handoff trim.
        These estimators remain passive; CENTER passes the physical-trim mean to
        the separate VisualCenterAdaptation outer loop, while FINAL_PROBE/DESCENT
        retain their existing acceleration-trim handoff semantics.

        Called from :meth:`update` rather than from a phase, so the estimate is
        continuous across the whole visual sequence and its CSV columns are
        populated in every phase that has a live measurement.
        """
        self._center_trim.update(
            inputs.offset_x, inputs.offset_y, inputs.target_found, inputs.dt
        )
        geom_x, geom_y = self._geometric_tilt_offset(inputs)
        self._center_physical_trim.update(
            float(inputs.offset_x) - geom_x,
            float(inputs.offset_y) - geom_y,
            inputs.target_found,
            inputs.dt,
        )
        self._handoff_trim.update(
            inputs.offset_x, inputs.offset_y, inputs.target_found, inputs.dt
        )
        self._center_trim_snapshot = self._center_trim.snapshot()
        self._center_physical_trim_snapshot = self._center_physical_trim.snapshot()
        self._handoff_trim_snapshot = self._handoff_trim.snapshot()

    def _commit_descent_lateral_trim(self, inputs: MissionInputs) -> None:
        """Commit the FINAL_PROBE static acceleration trim into DESCENT.

        FINAL_PROBE flies a small residual image-position P on top of the static
        trim, and DESCENT drops it.  No re-parameterisation is needed anyway:
        the wind-trim EMA has been integrating the REALIZED command all through
        the hold, so the mean of that P contribution is already inside the
        static term.  Starting DESCENT from the exact feedforward FINAL_PROBE
        was flying therefore remains the bumpless choice -- what is lost at the
        boundary is P's response to the residual centring error, not the steady
        force, which has already migrated.

        DESCENT drops P because image offset stops being a position measurement
        near contact.  On the 2026-08-19 09:42 run the centroid tracked truth to
        within 7% for the whole of FINAL_PROBE, degraded past area_fraction
        ~0.74 about 2.6 s before touchdown, and read +0.000 while the vehicle
        was still 10.6 cm off.  ``fov_saturated`` fired 1.5 s AFTER that, so it
        is a lagging indicator and cannot serve as the trigger; the phase
        boundary is used instead.

        The committed value is also retained as a telemetry anchor.  If descent
        adaptation is enabled, subsequent ticks may refine the live bias while
        remaining inside the same APPROACH-seeded safety neighbourhood.
        """
        self._descent_roll_accel_bias = float(self._final_probe_roll_accel_bias)
        self._descent_pitch_accel_bias = float(self._final_probe_pitch_accel_bias)
        self._descent_roll_bias_frozen = self._descent_roll_accel_bias
        self._descent_pitch_bias_frozen = self._descent_pitch_accel_bias
        # Keep the near-field visual trim for diagnostics only.  The lateral P
        # scales are zero from FINAL_PROBE onward, so these values no longer
        # enter ControlLaw as setpoints.
        if self._descent_trim_use_probe_mean and self._handoff_trim.has_mean:
            self._descent_trim_offset_x = float(self._handoff_trim.mean_x)
            self._descent_trim_offset_y = float(self._handoff_trim.mean_y)
        else:
            # Explicit legacy path, and the fallback if FINAL_PROBE never saw a
            # single valid target sample to seed the mean.
            self._descent_trim_offset_x = float(inputs.offset_x)
            self._descent_trim_offset_y = float(inputs.offset_y)
        self._descent_lateral_trim_frozen = True

    def _is_centered(self, offset_x: float, offset_y: float, target_found: bool) -> bool:
        return (
            bool(target_found)
            and abs(float(offset_x)) <= self._center_offset_thr
            and abs(float(offset_y)) <= self._center_offset_thr
        )
