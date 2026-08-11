"""Bio-inspired near-field mission routine -- assembly and shared state.

Sequence:
    CENTER -> APPROACH_PROBE -> FINAL_PROBE -> DESCEND

The per-phase logic lives in ``mission/phases/``; this module owns what the
phases share: configuration, the three acceleration probes, the feasibility
gates, phase dispatch, and the log schema.

Three acceleration probes run in parallel from APPROACH_PROBE through
FINAL_PROBE:
    - vertical command acceleration;
    - roll-channel command acceleration;
    - pitch-channel command acceleration.

Each probe uses the same de-biasing, rolling-percentile and leaky-peak logic. The
far-field estimates carry into FINAL_PROBE, where all three probes are retuned to
the near-field time constants without resetting their accumulated envelopes.
The final transition requires all three probes to be ready and all three gain
windows to be feasible.

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
``gates.py``           feasibility maths
``schedule.py``        the k(t) descent trajectory
``types.py``           the contract with the caller
this file              config plumbing, shared state, dispatch, telemetry
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any, ClassVar, Mapping, Optional, Sequence

from bee_control.core.config import MissionConfig
from bee_control.core.state import FlowResult, TargetEstimate

from . import phases
from .gates import (
    GateResult,
    LateralGateResult,
    ceiling_gain_at_height,
    compute_lateral_gate,
)
from .math_utils import blank as _blank, clamp
from .phases import PHASES
from .probe import PlatformProbe, ProbeResult, ThrustModel
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


class MissionRoutine:
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

        self._dt = float(cfg.control_period_sec)
        self._stability_dt = (
            float(cfg.stability_dt_sec) if cfg.stability_dt_sec is not None else self._dt
        )

        self._d_star = max(0.0, float(cfg.descent_divergence_setpoint))
        self._approach_d_star = max(0.0, float(cfg.approach_divergence_setpoint))
        self._final_probe_duration = max(0.0, float(cfg.final_probe_duration_sec))
        self._final_probe_entry_ramp = max(0.0, float(cfg.final_probe_entry_ramp_sec))
        self._fov_near_area_fraction = clamp(cfg.fov_near_area_fraction, 0.0, 1.0)

        self._probe_min = max(0.0, float(cfg.probe_min_duration_sec))
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
        self._center_offset_thr = max(0.0, float(cfg.center_offset_threshold))
        self._center_dwell = max(0.0, float(cfg.center_dwell_sec))
        self._center_timeout = max(0.0, float(cfg.center_timeout_sec))

        self._enable_center_condition_gate = bool(cfg.enable_center_condition_gate)
        self._center_condition_dwell = max(0.0, float(cfg.center_condition_dwell_sec))
        self._center_offset_radius_max = max(0.0, float(cfg.center_offset_radius_max))
        self._center_flow_radius_max = max(0.0, float(cfg.center_flow_radius_max_norm_s))
        self._center_timeout_allows_handoff = bool(cfg.center_timeout_allows_handoff)

        self._d_star_ramp_in = max(0.0, float(cfg.d_star_ramp_in_sec))
        self._lateral_ramp = max(0.0, float(cfg.center_to_probe_lateral_ramp_sec))

        self._center_lateral_p_scale = max(0.0, float(cfg.center_lateral_p_scale))
        self._center_lateral_d_scale = max(0.0, float(cfg.center_lateral_d_scale))
        self._probe_lateral_p_scale = max(0.0, float(cfg.probe_lateral_p_scale))
        self._probe_lateral_d_scale = max(0.0, float(cfg.probe_lateral_d_scale))

        self._tm = ThrustModel(hover_thrust)

        # Three parallel probes share the same far/near timing design.
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

        self.gate = GateResult()
        self.roll_gate = LateralGateResult()
        self.pitch_gate = LateralGateResult()
        self.probe_result = ProbeResult()
        self.roll_probe_result = ProbeResult()
        self.pitch_probe_result = ProbeResult()
        # peak_accel at the instant of the far->near handoff, frozen for diagnostics:
        # comparing it against the final peak_accel shows how much the near field
        # actually revised the far-field estimate (and in which direction).
        self.peak_accel_at_handoff: Optional[float] = None
        self.roll_peak_accel_at_handoff: Optional[float] = None
        self.pitch_peak_accel_at_handoff: Optional[float] = None

        self._substate = CENTER if self._enable_center else APPROACH_PROBE
        self._t0: Optional[float] = None
        self._h0: Optional[float] = None
        self._k_explore = self._initial_thrust_gain

        self._centered_since: Optional[float] = None
        self._center_start_t: Optional[float] = None
        self._t_approach_entry: Optional[float] = None
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
        self.gate = GateResult()
        self.roll_gate = LateralGateResult()
        self.pitch_gate = LateralGateResult()
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
        self._t_approach_entry = None
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
        if self._substate in (LANDED, ABORTED):
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
        "roll_p_scale", "roll_d_scale", "pitch_p_scale", "pitch_d_scale",
        "enable_integral",
        # Feasibility gate inputs and outputs.
        "peak_accel_m_s2", "roll_peak_accel_m_s2", "pitch_peak_accel_m_s2",
        "k_min", "k_explore", "k_probe", "k_floor", "k_ceiling_leg",
        "h_crit_m", "h_pred_m",
        "vertical_feasible", "roll_feasible", "pitch_feasible", "feasible",
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
        "roll_probe_within_ceiling", "pitch_probe_within_ceiling",
        "infeasible_reason",
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

        row: dict = {
            "substate": self._substate,
            "divergence_setpoint_1_s": mc.divergence_setpoint,
            "thrust_gain_k": _blank(mc.thrust_gain_override),
            "lateral_p_scale": mc.lateral_p_scale,
            "lateral_d_scale": mc.lateral_d_scale,
            "roll_p_scale": _blank(mc.roll_p_scale),
            "roll_d_scale": _blank(mc.roll_d_scale),
            "pitch_p_scale": _blank(mc.pitch_p_scale),
            "pitch_d_scale": _blank(mc.pitch_d_scale),
            "enable_integral": int(bool(mc.enable_integral)),

            "peak_accel_m_s2": self.probe_result.peak_accel,
            "roll_peak_accel_m_s2": self.roll_probe_result.peak_accel,
            "pitch_peak_accel_m_s2": self.pitch_probe_result.peak_accel,

            "k_min": gate.k_min,
            "k_explore": gate.k_explore,
            "k_probe": self.probe_gain,
            "k_floor": gate.k_floor,
            "k_ceiling_leg": gate.k_ceiling_leg,
            "h_crit_m": gate.h_crit,
            "h_pred_m": _blank(info.get("h_pred")),

            "vertical_feasible": int(bool(gate.feasible)),
            "roll_feasible": int(bool(roll_gate.feasible)),
            "pitch_feasible": int(bool(pitch_gate.feasible)),
            "feasible": int(bool(self.feasible)),

            "roll_k_min": roll_gate.k_min,
            "pitch_k_min": pitch_gate.k_min,
            "roll_k_probe": roll_gate.k_probe,
            "pitch_k_probe": pitch_gate.k_probe,
            "roll_k_applied": _blank(info.get("roll_k")),
            "pitch_k_applied": _blank(info.get("pitch_k")),
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
            "roll_probe_within_ceiling": int(bool(roll_gate.probe_within_ceiling)),
            "pitch_probe_within_ceiling": int(bool(pitch_gate.probe_within_ceiling)),
            "infeasible_reason": _blank(info.get("infeasible_reason")),
        }

        probe = self.probe_telemetry()
        for key, column in self._PROBE_COLUMNS.items():
            if key in probe:
                value = probe[key]
                if isinstance(value, bool):
                    value = int(value)
                row[column] = _blank(value)
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
        # The ENVELOPE (probe_peak_accel) and the handoff value are NOT blanked:
        # they are the gate's inputs and stay meaningful for the whole descent
        # (k_min = peak/D* is what the schedule's floor was built from).
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
            self._near_field_height, self._stability_dt, self._safety
        )
        return min(self._initial_thrust_gain, self._ceiling_margin * ceiling)

    @property
    def probe_gain(self) -> float:
        """Gain held flat through FINAL_PROBE; the descent schedule starts here."""
        return self._compute_probe_gain()

    @property
    def feasible(self) -> bool:
        return self.gate.feasible and self.roll_gate.feasible and self.pitch_gate.feasible

    @property
    def vertical_feasible(self) -> bool:
        return bool(self.gate.feasible)

    @property
    def roll_feasible(self) -> bool:
        return bool(self.roll_gate.feasible)

    @property
    def pitch_feasible(self) -> bool:
        return bool(self.pitch_gate.feasible)

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

        # An unregistered substate falls through to DESCEND, preserving the
        # previous if-chain's final `return self._do_descend(t)`.
        spec = PHASES.get(self._substate) or PHASES[DESCEND]
        control = spec.handler(self, inputs)
        # Retained so telemetry() can describe the tick without the caller
        # handing the MissionControl back to the logger.
        self.last_control = control
        return control

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

    def _refresh_probe_results(
        self,
        min_duration_sec: float,
        min_total_duration_sec: float,
    ) -> None:
        kwargs = dict(
            min_duration_sec=min_duration_sec,
            min_total_duration_sec=min_total_duration_sec,
        )
        self.probe_result = self._probe.result(**kwargs)
        self.roll_probe_result = self._roll_probe.result(**kwargs)
        self.pitch_probe_result = self._pitch_probe.result(**kwargs)

    def _compute_lateral_gates(self) -> None:
        roll_probe_gain = self._roll_d_gain * self._probe_lateral_d_scale
        pitch_probe_gain = self._pitch_d_gain * self._probe_lateral_d_scale
        self.roll_gate = compute_lateral_gate(
            peak_accel=self.roll_probe_result.peak_accel,
            flow_admissible_norm_s=self._roll_flow_admissible,
            max_closing_speed_m_s=self._max_closing_speed,
            kappa=self._roll_kappa,
            probe_gain=roll_probe_gain,
            near_field_height_m=self._near_field_height,
            leg_clearance_m=self._leg_clearance,
            control_period_sec=self._stability_dt,
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
            control_period_sec=self._stability_dt,
            ceiling_safety_factor=self._safety,
            ceiling_margin=self._ceiling_margin,
        )

    def _gate_failure_reasons(self) -> list[str]:
        """Return concise, axis-specific reasons for an INFEASIBLE decision."""
        reasons: list[str] = []

        if not self.gate.window_exists:
            reasons.append(
                "VERTICAL no gain window: "
                f"K_min={self.gate.k_min:.3f} > "
                f"K_ceiling_leg={self.gate.k_ceiling_leg:.3f}"
            )
        elif not self.gate.start_above_floor:
            reasons.append(
                "VERTICAL FINAL_PROBE gain below floor: "
                f"K_probe={self.gate.k_descend_start:.3f} < "
                f"K_min={self.gate.k_min:.3f}"
            )

        for label, gate in (("ROLL", self.roll_gate), ("PITCH", self.pitch_gate)):
            if not gate.probe_within_ceiling:
                reasons.append(
                    f"{label} probe gain above near-field ceiling: "
                    f"K_probe={gate.k_probe:.3f} > "
                    f"K_ceiling_probe={gate.k_ceiling_probe:.3f}"
                )
            if not gate.window_exists:
                reasons.append(
                    f"{label} no gain window: "
                    f"K_min={gate.k_min:.3f} > "
                    f"K_ceiling_leg={gate.k_ceiling_leg:.3f}"
                )
            elif not gate.start_above_floor:
                reasons.append(
                    f"{label} FINAL_PROBE gain below floor: "
                    f"K_probe={gate.k_probe:.3f} < K_min={gate.k_min:.3f}"
                )
            elif not gate.floor_within_ceiling:
                reasons.append(
                    f"{label} scheduled floor above touchdown ceiling: "
                    f"K_floor={gate.k_floor:.3f} > "
                    f"K_ceiling_leg={gate.k_ceiling_leg:.3f}"
                )

        if not self._enable_descent:
            reasons.append("DESCENT disabled by configuration")

        return reasons

    def _is_centered(self, offset_x: float, offset_y: float, target_found: bool) -> bool:
        return (
            bool(target_found)
            and abs(float(offset_x)) <= self._center_offset_thr
            and abs(float(offset_y)) <= self._center_offset_thr
        )

    def _near_field_reached(
        self,
        offset_x: float,
        offset_y: float,
        target_found: bool,
        area_fraction: float,
        fov_saturated: bool,
    ) -> bool:
        centered = self._is_centered(offset_x, offset_y, target_found)

        # Area fraction is the monotone near-field trigger. fov_saturated remains a
        # diagnostic but does not drive the phase transition.
        visually_close = float(area_fraction) >= self._fov_near_area_fraction
        return centered and visually_close

    def status_line(self) -> str:
        if self._substate == LANDED:
            since = (
                f"{(self._t_landed):.1f}s"
                if self._t_landed is not None else "--"
            )
            return f"[landed] touchdown latched at t={since}; mission complete"

        if self._substate == CENTER:
            return (
                f"[center] waiting for target within +/-{self._center_offset_thr:.2f} "
                f"for {self._center_dwell:.1f}s (timeout {self._center_timeout:.0f}s)"
            )

        if self._substate == APPROACH_PROBE:
            return (
                f"[approach_probe] D*_approach={self._approach_d_star:.2f} "
                f"probe={self.probe_result.total_duration_sec:.1f}/{self._probe_min:.1f}s "
                f"peaks z/r/p={self.probe_result.peak_accel:.3f}/"
                f"{self.roll_probe_result.peak_accel:.3f}/"
                f"{self.pitch_probe_result.peak_accel:.3f} m/s^2 "
                f"near_area_thr={self._fov_near_area_fraction:.2f}"
            )

        if self._substate == FINAL_PROBE:
            handoff = (
                f"{self.peak_accel_at_handoff:.3f}"
                if self.peak_accel_at_handoff is not None else "--"
            )
            return (
                f"[final_probe] hold={self.probe_result.duration_sec:.1f}/"
                f"{self._final_probe_duration:.1f}s "
                f"total={self.probe_result.total_duration_sec:.1f}/{self._probe_min:.1f}s "
                f"peaks z/r/p={self.probe_result.peak_accel:.3f}/"
                f"{self.roll_probe_result.peak_accel:.3f}/"
                f"{self.pitch_probe_result.peak_accel:.3f} m/s^2 "
                f"(at handoff {handoff})"
            )

        if self._substate == PROBE_HOLD:
            verdict = "WOULD-LAND" if self.feasible else "WOULD-ABORT"
            handoff = (
                f"{self.peak_accel_at_handoff:.3f}"
                if self.peak_accel_at_handoff is not None else "--"
            )
            return (
                f"[probe_hold] peak_accel={self.probe_result.peak_accel:.3f} m/s^2 "
                f"(at handoff {handoff}) "
                f"k_min={self.gate.k_min:.2f} k_floor={self.gate.k_floor:.2f} "
                f"k_ceiling_leg={self.gate.k_ceiling_leg:.2f} "
                f"h_crit={self.gate.h_crit:.2f}m "
                f"flags z/r/p={int(self.vertical_feasible)}/"
                f"{int(self.roll_feasible)}/{int(self.pitch_feasible)} "
                f"vs leg={self._leg_clearance:.2f}m -> {verdict} (hovering, no descent)"
            )

        if self._substate == INFEASIBLE:
            reasons = "; ".join(self._gate_failure_reasons())
            return (
                f"[infeasible] flags z/r/p={int(self.vertical_feasible)}/"
                f"{int(self.roll_feasible)}/{int(self.pitch_feasible)}: {reasons}"
            )

        return (
            f"[descend] h_crit={self.gate.h_crit:.2f}m "
            f"k: {self.gate.k_descend_start:.2f} (probe) -> {self.gate.k_floor:.2f} "
            f"(floor = max(k_min {self.gate.k_min:.2f}, "
            f"{self.gate.ceiling_margin:.2f} x k_ceiling_leg "
            f"{self.gate.k_ceiling_leg:.2f}))"
        )

    #: Phase table, imported from ``mission/phases/``. Kept as a class
    #: attribute so existing callers (``bee_node``, tests) can still reach it
    #: as ``MissionRoutine.PHASES``.
    PHASES: ClassVar[dict] = PHASES

