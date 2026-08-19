"""ROS-free regression tests for the module contracts.

These exist because the refactor's whole claim is that a subsystem can now be
changed without touching ``bee_node``.  That claim is only true if the seams
are actually checked, so each test below pins one of them:

* a source cannot emit a column it never declared (the failure mode that used
  to be a silent drop at CSV-write time);
* two sources cannot claim the same column;
* an unknown mission knob raises instead of reverting to a default;
* every registered phase is dispatchable and reachable by name;
* the outer lifecycle reaches CLOSED_LOOP and both terminal states.

Run with ``python3 -m pytest test_contracts.py`` or directly:
``python3 test_contracts.py``.  Nothing here imports rclpy, cv_bridge or
px4_msgs, so it runs on a laptop with no ROS install.
"""
from __future__ import annotations

import math
import time
import tempfile
import warnings
from dataclasses import fields

from bee_control.core.clock import SteadyWallClock, TimeManager
from bee_control.core.config import (
    RENAMED_MISSION_FIELDS,
    BeeConfig,
    MissionConfig,
)
from bee_control.core.controller_state import ControllerState, PX4Status, VisionTelemetry
from bee_control.diagnostics.diagnostics_writer import DiagnosticsWriter
from bee_control.interfaces.flight_sequencer import (
    CLOSED_LOOP,
    FlightSequencer,
    SequencerPorts,
    SetpointPolicy,
)
from bee_control.mission.gates import compute_tracking_gate
from bee_control.mission.routine import MissionRoutine
from bee_control.mission.visual_center_adaptation import VisualCenterAdaptation
from bee_control.mission.trim import VisualTrim
from bee_control.mission.types import ActuationFeedback, ControlEffect, MissionInputs
from bee_control.mission.visual_mismatch import VisualMismatchProbe
from bee_control.vision.optical_flow import OpticalFlowEstimator
from bee_control.core.state import AttitudeSetpoint, FlowResult, TargetEstimate
from bee_control.diagnostics.telemetry import TelemetrySchemaError, collect_fields
from bee_control.control.control_law import ControlLaw


class _FakeControlLaw:
    hover_thrust = 0.73
    divergence_integral = 0.0
    last_vertical_accel_cmd = 0.0
    last_roll_accel_cmd = 0.0
    last_pitch_accel_cmd = 0.0


def _sources():
    control_law = _FakeControlLaw()
    return [
        PX4Status(nav_state_offboard=14, arming_state_armed=2),
        ControllerState(control_law=control_law,
                        initial_setpoint=AttitudeSetpoint(thrust=0.73)),
        MissionRoutine(hover_thrust=0.73, config=BeeConfig.default().mission),
        VisionTelemetry(),
    ]


# --------------------------------------------------------------------------
# Schema contracts
# --------------------------------------------------------------------------
def test_every_source_declares_what_it_emits():
    for source in _sources():
        declared = set(source.telemetry_fields())
        emitted = set(source.telemetry())
        assert not emitted - declared, (
            f"{type(source).__name__} emits undeclared "
            f"{sorted(emitted - declared)}")


def test_no_column_collisions_between_sources():
    fields = collect_fields(_sources())
    assert len(fields) == len(set(fields))


def test_writer_rejects_undeclared_keys():
    class Drifting:
        TELEMETRY_PREFIX = "x"

        @classmethod
        def telemetry_fields(cls):
            return ("declared",)

        def telemetry(self):
            return {"declared": 1, "undeclared": 2}

    writer = DiagnosticsWriter(sources=[Drifting()],
                               output_dir=tempfile.mkdtemp(), strict=True)
    try:
        writer.write()
    except TelemetrySchemaError:
        pass
    else:
        raise AssertionError("strict mode did not catch the undeclared key")
    finally:
        writer.close()


def test_optical_flow_owns_its_timing_schema():
    """The estimator's declared timings must reach the log with a prefix."""
    vision = VisionTelemetry()
    columns = set(vision.telemetry_fields())
    for name in OpticalFlowEstimator.TIMING_FIELDS:
        assert f"optical_flow_{name}" in columns


def test_unknown_timing_key_is_reported_not_raised():
    vision = VisionTelemetry()
    vision.update({"control_dt_sim_sec": 0.016, "invented_key": 1})
    assert "invented_key" in vision.drain_undeclared()
    assert vision.drain_undeclared() == []       # drained, warn-once


# --------------------------------------------------------------------------
# Config contracts
# --------------------------------------------------------------------------
def test_unknown_mission_knob_raises():
    try:
        MissionConfig().with_overrides(ceilling_margin=0.7)   # typo on purpose
    except TypeError as exc:
        assert "ceilling_margin" in str(exc)
    else:
        raise AssertionError(
            "a misspelled knob was accepted -- this is the failure mode that "
            "used to fly the library default silently")


def test_derived_values_are_wired():
    cfg = BeeConfig.default()
    assert math.isclose(cfg.mission.roll_kappa, cfg.camera.roll_kappa)
    assert math.isclose(
        cfg.mission.stability_dt_sec,
        cfg.camera.stability_dt_sec(cfg.scheduling))

    # Previously omitted, so a change to the scheduling tick silently left the
    # mission's stability_dt fallback behind at its own literal.
    assert math.isclose(
        cfg.mission.stability_dt_fallback_sec,
        cfg.scheduling.control_period_sec)
    assert math.isclose(cfg.mission.roll_d_gain, cfg.control.roll_kd)
    assert math.isclose(cfg.mission.pitch_d_gain, cfg.control.pitch_kd)


def test_renamed_fields_map_onto_real_fields():
    """Every alias must point at a field that actually exists today."""
    known = {f.name for f in fields(MissionConfig())}
    for old, new in RENAMED_MISSION_FIELDS.items():
        assert new in known, f"alias {old!r} points at missing field {new!r}"
        assert old not in known, f"{old!r} is both a live field and an alias"


def test_deprecated_names_still_work_and_warn():
    """An existing launch file or notebook must not break on the rename."""
    base = MissionConfig()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = base.with_overrides(descent_lateral_bias_tau_sec=3.25)
        value = cfg.descent_lateral_bias_tau_sec
    assert math.isclose(cfg.wind_trim_tau_sec, 3.25)
    assert math.isclose(value, 3.25)
    assert len(caught) == 2
    assert all(issubclass(w.category, DeprecationWarning) for w in caught)


def test_setting_both_old_and_new_name_raises():
    """Silently picking one of two conflicting values is the failure to avoid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            MissionConfig().with_overrides(
                descent_lateral_bias_tau_sec=1.0, wind_trim_tau_sec=2.0)
        except TypeError as exc:
            assert "wind_trim_tau_sec" in str(exc)
        else:
            raise AssertionError("a conflicting override pair was accepted")


def test_unknown_attribute_still_raises_attribute_error():
    """The alias __getattr__ must not swallow genuine typos."""
    try:
        MissionConfig().wind_trimm_tau_sec   # typo on purpose
    except AttributeError as exc:
        assert "wind_trimm_tau_sec" in str(exc)
    else:
        raise AssertionError("a misspelled attribute returned a value")


def test_every_mission_field_declares_its_unit():
    """Config naming convention: a float knob names its unit in its suffix.

    Dimensionless ratios, counts and booleans are the documented exception.
    """
    unit_suffixes = (
        "_sec", "_m", "_m_s", "_m_s2", "_rad", "_deg",
        "_norm", "_norm_s", "_1_s", "_1_s2",
    )
    dimensionless = {
        "roll_kappa", "pitch_kappa", "roll_d_gain", "pitch_d_gain",
        "ceiling_safety_factor", "ceiling_margin", "initial_thrust_gain",
        "approach_hold_area_fraction", "approach_retreat_divergence_limit",
        "approach_hold_log_scale_tolerance", "approach_divergence_setpoint",
        "descent_divergence_setpoint", "probe_attenuation_comp",
        "center_offset_radius_max", "center_trim_mean_radius_max",
        "center_trim_residual_radius_max", "center_legacy_box_threshold",
        "center_visual_adaptation_max_bias_norm",
        "center_lateral_p_scale", "center_lateral_d_scale",
        "probe_lateral_p_scale", "probe_lateral_d_scale",
    }
    offenders = [
        f.name for f in fields(MissionConfig())
        if not f.name.startswith(("enable_", "probe_only", "descent_trim_use",
                                  "center_timeout_allows", "wind_trim_adapt"))
        and f.name not in dimensionless
        and not f.name.endswith(unit_suffixes)
    ]
    assert not offenders, f"fields with no unit suffix: {sorted(offenders)}"


# --------------------------------------------------------------------------
# Mission contracts
# --------------------------------------------------------------------------
def test_every_registered_phase_dispatches():
    cfg = BeeConfig.default().mission
    inputs = MissionInputs(
        t=1.0, dt=1 / 60, target=TargetEstimate(found=True),
        flow=FlowResult(valid=True), actuation=ActuationFeedback())
    for name, spec in MissionRoutine.PHASES.items():
        routine = MissionRoutine(hover_thrust=0.73, config=cfg)
        routine.start(0.0, 5.0)
        control = spec.handler(routine, inputs)
        assert control.substate, f"{name} returned no substate"
        assert spec.display_name


def test_transitions_declare_their_control_effects():
    """The three integral resets must be declared, not inferred from strings."""
    cfg = BeeConfig.default().mission
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.start(0.0, 3.0)
    seen, t, dt = set(), 0.0, 0.05
    while t < 60.0:
        a = 0.3 * math.cos(2 * math.pi * 0.15 * t)
        control = routine.update(MissionInputs(
            t=t, dt=dt,
            target=TargetEstimate(timestamp=t, found=True,
                                  area_fraction=0.9 if t > 2 else 0.2),
            flow=FlowResult(timestamp=t, valid=True),
            actuation=ActuationFeedback(
                last_thrust_cmd=0.73, last_vertical_accel_cmd=a,
                last_roll_accel_cmd=0.6 * a, last_pitch_accel_cmd=0.4 * a)))
        seen.update(control.effects)
        t += dt
    assert ControlEffect.RESET_DIVERGENCE_INTEGRAL in seen


def test_approach_visual_height_loop_brakes_and_can_retreat():
    cfg = MissionConfig(enable_center=False)
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.start(0.0, 5.0)

    def update(t, area):
        return routine.update(MissionInputs(
            t=t, dt=0.05,
            target=TargetEstimate(timestamp=t, found=True, area_fraction=area),
            flow=FlowResult(timestamp=t, valid=True, divergence=0.05),
            actuation=ActuationFeedback(last_thrust_cmd=0.73)))

    update(0.0, 0.2)
    far = update(4.0, 0.2)
    at_hold = update(4.05, cfg.approach_hold_area_fraction)
    too_close = update(4.10, min(0.95, cfg.approach_hold_area_fraction + 0.15))

    assert math.isclose(far.divergence_setpoint, cfg.approach_divergence_setpoint)
    assert abs(at_hold.divergence_setpoint) < 1e-12
    assert too_close.divergence_setpoint < 0.0
    assert abs(too_close.divergence_setpoint) <= cfg.approach_retreat_divergence_limit


def test_final_probe_starts_from_fresh_gate_evidence_without_integral_reset():
    cfg = MissionConfig(enable_center=False, approach_hold_dwell_sec=0.10)
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.start(0.0, 5.0)

    def update(t, area, div, accel):
        return routine.update(MissionInputs(
            t=t, dt=0.05,
            target=TargetEstimate(
                timestamp=t, found=True, area_fraction=area,
                offset_x=0.0, offset_y=0.0),
            flow=FlowResult(timestamp=t, valid=True, divergence=div),
            actuation=ActuationFeedback(
                last_thrust_cmd=0.73, last_vertical_accel_cmd=accel,
                last_roll_accel_cmd=0.5 * accel,
                last_pitch_accel_cmd=0.25 * accel)))

    # Build a large APPROACH-only diagnostic envelope.
    for i in range(20):
        update(0.05 * i, 0.4, 0.03, 2.0 if i % 2 else -2.0)
    assert routine.probe_result.peak_accel > 1.0

    control = None
    for i in range(6):
        control = update(1.1 + 0.05 * i, cfg.approach_hold_area_fraction, 0.0, 0.1)
        if control.substate == "final_probe":
            break

    assert control is not None and control.substate == "final_probe"
    assert control.lateral_p_scale == 0.0
    assert routine.peak_accel_at_handoff > 1.0
    assert routine.probe_result.peak_accel == 0.0
    assert ControlEffect.RESET_DIVERGENCE_INTEGRAL not in control.effects

    # The next sample starts a new FINAL_PROBE envelope; APPROACH cannot leak in.
    update(1.5, cfg.approach_hold_area_fraction, 0.0, 0.1)
    assert routine.probe_result.peak_accel < 0.5


def _fly_to_descend(routine, *, trim_x=0.10, trim_y=-0.05, osc=0.05,
                    freq_hz=0.15, dt=1 / 30.0, max_sec=140.0):
    """Synthetic whole-sequence flight with adaptive physical centring.

    ``trim_x/y`` represent the steady P error required to reject wind.  The
    adaptive visual bias ``b`` moves the physical equilibrium according to

        e_phys = e_wind - b,

    while the controller error remains

        e_phys - e_sp = (e_wind - b) - (-b) = e_wind.

    Thus the synthetic plant recentres without losing its steady counter-force.
    """
    kp_eff = 0.22 * 9.80665 * 0.75
    rows, t = {}, 0.0
    while t < max_sec:
        t += dt
        adapt = routine._visual_center_adaptation_snapshot
        decay = math.exp(-t / 3.0)
        offset_x = trim_x - adapt.bias_x + osc * decay * math.sin(
            2 * math.pi * freq_hz * t
        )
        offset_y = trim_y - adapt.bias_y
        prev_sp_x = float(getattr(routine.last_control, "roll_offset_setpoint", 0.0))
        prev_sp_y = float(getattr(routine.last_control, "pitch_offset_setpoint", 0.0))
        routine.update(MissionInputs(
            t=t, dt=dt,
            target=TargetEstimate(timestamp=t, found=True, offset_x=offset_x,
                                  offset_y=offset_y, area_fraction=0.70),
            flow=FlowResult(timestamp=t, valid=True, divergence=0.0),
            actuation=ActuationFeedback(
                last_thrust_cmd=0.73, last_vertical_accel_cmd=0.0,
                last_roll_accel_cmd=-kp_eff * (offset_x - prev_sp_x),
                last_pitch_accel_cmd=-kp_eff * (offset_y - prev_sp_y))))
        rows.setdefault(routine.substate, routine.telemetry())
    return rows

def test_lateral_trim_is_logged_in_every_visual_phase():
    """The trim columns must not go blank after CENTER hands off.

    The estimator used to be advanced inside ``center.py`` alone and the CSV
    read it out of that phase's ``info`` dict, so both the estimate and its four
    columns froze at the CENTER handoff -- blank for APPROACH_PROBE and
    FINAL_PROBE, which is exactly where a wind equilibrium is worth measuring.
    """
    routine = MissionRoutine(hover_thrust=0.73, config=MissionConfig())
    routine.start(0.0, 5.0)
    rows = _fly_to_descend(routine)

    for phase in ("center", "approach_probe", "final_probe", "descend"):
        assert phase in rows, f"never reached {phase}"
        for column in ("center_trim_mean_x", "center_trim_mean_y",
                       "center_trim_mean_radius", "center_trim_residual_radius",
                       "handoff_trim_mean_x", "handoff_trim_mean_radius"):
            assert rows[phase][column] != "", f"{column} blank during {phase}"


def test_descent_handoff_keeps_the_last_final_probe_static_feedforward():
    """DESCENT starts from the LAST, not first, FINAL_PROBE static term.

    FINAL_PROBE is intentionally adaptive, so its first row may differ from the
    committed value.  The descent entry itself must remain bumpless even though
    subsequent DESCENT ticks are now allowed to refine that term.
    """
    routine = MissionRoutine(hover_thrust=0.73, config=MissionConfig())
    routine.start(0.0, 5.0)
    rows = _fly_to_descend(routine)

    assert "descend" in rows
    assert routine._descent_lateral_trim_frozen
    assert math.isclose(
        routine._final_probe_roll_accel_bias,
        rows["descend"]["roll_accel_feedforward_m_s2"],
        abs_tol=1e-12,
    )
    assert not math.isclose(
        rows["final_probe"]["roll_accel_feedforward_m_s2"],
        rows["descend"]["roll_accel_feedforward_m_s2"],
        abs_tol=1e-6,
    ), "the FINAL_PROBE static term never adapted"
    assert rows["final_probe"]["lateral_p_scale"] == 0.0
    assert rows["descend"]["lateral_p_scale"] == 0.0
    assert rows["descend"]["roll_p_scale"] == 0.0
    assert rows["descend"]["pitch_p_scale"] == 0.0


def test_far_field_tilt_compensation_uses_previous_shaped_command():
    routine = MissionRoutine(hover_thrust=0.73, config=MissionConfig())
    routine.start(0.0, 5.0)
    roll = math.radians(4.0)
    pitch = math.radians(-3.0)
    denom = math.tan(math.radians(40.0))
    roll_geom = -math.tan(roll) / denom
    pitch_geom = -math.tan(pitch) / denom
    control = routine.update(MissionInputs(
        t=0.1,
        dt=0.1,
        # A pure-tilt case must present the geometric image displacement.
        # offset=0 with a tilted camera would correctly look like physical
        # mis-centring to the adaptive outer loop.
        target=TargetEstimate(timestamp=0.1, found=True,
                              offset_x=roll_geom, offset_y=pitch_geom),
        flow=FlowResult(timestamp=0.1, valid=True),
        actuation=ActuationFeedback(
            last_thrust_cmd=0.73,
            last_roll_cmd_rad=roll,
            last_pitch_cmd_rad=pitch,
        ),
    ))
    assert math.isclose(control.roll_offset_setpoint, -math.tan(roll) / denom)
    assert math.isclose(control.pitch_offset_setpoint, -math.tan(pitch) / denom)
    assert math.isclose(abs(control.roll_offset_setpoint), 0.083336, rel_tol=1e-4)




def test_visual_center_adaptation_recenters_while_preserving_p_error():
    """CENTER learns the moving reference continuously, without subphases."""
    cfg = MissionConfig(
        center_condition_dwell_sec=0.20,
        center_flow_radius_max_norm_s=0.05,
        center_trim_residual_radius_max=0.02,
        center_offset_radius_max=0.02,
        center_visual_adaptation_tau_sec=0.40,
        center_visual_adaptation_max_rate_norm_s=1.0,
        center_visual_adaptation_flow_scale_norm_s=0.10,
    )
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.start(0.0, 5.0)

    required_p_error_y = -0.18
    t = 0.0
    observed_controller_errors = []
    for _ in range(300):
        t += 1 / 30.0
        bias_y = routine._visual_center_adaptation_snapshot.bias_y
        # Synthetic steady-wind equilibrium: as b learns -0.18, the physical
        # offset approaches zero but the P error remains -0.18.
        offset_y = required_p_error_y - bias_y
        control = routine.update(MissionInputs(
            t=t, dt=1 / 30.0,
            target=TargetEstimate(timestamp=t, found=True, offset_x=0.0,
                                  offset_y=offset_y, area_fraction=0.70),
            flow=FlowResult(timestamp=t, valid=True, divergence=0.0),
            actuation=ActuationFeedback(last_thrust_cmd=0.73),
        ))
        observed_controller_errors.append(offset_y - control.pitch_offset_setpoint)
        if routine.substate == "approach_probe":
            break

    assert routine.substate == "approach_probe"
    snap = routine._visual_center_adaptation_snapshot
    assert snap.bias_y < -0.15
    row = routine.telemetry()
    assert row["center_physical_mean_radius"] <= cfg.center_offset_radius_max
    # Once the short estimator transient is gone, the moving reference preserves
    # the wind-rejection P error while physical displacement vanishes.
    assert math.isclose(observed_controller_errors[-1], required_p_error_y, abs_tol=0.02)


def test_visual_center_adaptation_slows_during_large_flow():
    fast = VisualCenterAdaptation(
        tau_sec=1.0, max_bias_norm=0.5, flow_scale_norm_s=0.1,
        max_rate_norm_s=1.0,
    )
    slow = VisualCenterAdaptation(
        tau_sec=1.0, max_bias_norm=0.5, flow_scale_norm_s=0.1,
        max_rate_norm_s=1.0,
    )
    fast.update(physical_error_x=0.2, physical_error_y=0.0,
                flow_x_norm_s=0.0, flow_y_norm_s=0.0, valid=True, dt=0.1)
    slow.update(physical_error_x=0.2, physical_error_y=0.0,
                flow_x_norm_s=0.3, flow_y_norm_s=0.0, valid=True, dt=0.1)
    assert fast.bias_x > slow.bias_x > 0.0
    assert math.isclose(fast.snapshot().adaptation_weight, 1.0)
    assert slow.snapshot().adaptation_weight < 0.2


def test_visual_center_bias_keeps_adapting_during_approach():
    cfg = MissionConfig(
        center_condition_dwell_sec=0.10,
        center_offset_radius_max=0.02,
        center_visual_adaptation_tau_sec=0.25,
        center_visual_adaptation_max_rate_norm_s=1.0,
    )
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.start(0.0, 5.0)
    t = 0.0
    for _ in range(300):
        t += 1/30
        by = routine._visual_center_adaptation_snapshot.bias_y
        routine.update(MissionInputs(
            t=t, dt=1/30,
            target=TargetEstimate(timestamp=t, found=True, offset_y=-0.12 - by,
                                  area_fraction=0.70),
            flow=FlowResult(timestamp=t, valid=True, divergence=0.0),
            actuation=ActuationFeedback(last_thrust_cmd=0.73),
        ))
        if routine.substate == "approach_probe":
            break
    assert routine.substate == "approach_probe"
    bias_at_handoff = routine._visual_center_adaptation_snapshot.bias_y
    for _ in range(30):
        t += 1/30
        routine.update(MissionInputs(
            t=t, dt=1/30,
            target=TargetEstimate(timestamp=t, found=True, offset_y=0.30,
                                  area_fraction=0.50),
            flow=FlowResult(timestamp=t, valid=True, divergence=0.0),
            actuation=ActuationFeedback(last_thrust_cmd=0.73),
        ))
    assert routine.substate == "approach_probe"
    assert routine._visual_center_adaptation_snapshot.bias_y > bias_at_handoff
    assert routine._visual_center_adaptation_snapshot.active


def test_far_field_visual_bias_is_not_inverse_p_scaled():
    """The same learned b must map to the same visual setpoint at any P scale."""
    cfg = MissionConfig()
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.start(0.0, 5.0)
    routine._visual_center_adaptation.update(
        physical_error_x=0.10, physical_error_y=-0.20,
        flow_x_norm_s=0.0, flow_y_norm_s=0.0, valid=True, dt=0.5,
    )
    routine._visual_center_adaptation_snapshot = routine._visual_center_adaptation.snapshot()
    inputs = MissionInputs(
        t=1.0, dt=1/60,
        target=TargetEstimate(found=True),
        flow=FlowResult(valid=True),
        actuation=ActuationFeedback(),
    )
    center_terms = routine._far_field_lateral_control_terms(inputs)
    # Changing the mission P schedule must not algebraically change e_sp.
    routine._substate = "approach_probe"
    approach_terms = routine._far_field_lateral_control_terms(inputs)
    assert math.isclose(center_terms[0], approach_terms[0], abs_tol=1e-12)
    assert math.isclose(center_terms[1], approach_terms[1], abs_tol=1e-12)


def test_far_field_adaptive_center_keeps_accel_feedforward_zero():
    cfg = MissionConfig(center_visual_adaptation_tau_sec=0.1,
                        center_visual_adaptation_max_rate_norm_s=1.0)
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.start(0.0, 5.0)
    routine._visual_center_adaptation.update(
        physical_error_x=0.12, physical_error_y=-0.18,
        flow_x_norm_s=0.0, flow_y_norm_s=0.0, valid=True, dt=0.1,
    )
    routine._visual_center_adaptation_snapshot = routine._visual_center_adaptation.snapshot()
    control = routine._far_field_lateral_control_terms(
        MissionInputs(
            t=1.0, dt=1/60,
            target=TargetEstimate(found=True),
            flow=FlowResult(valid=True),
            actuation=ActuationFeedback(),
        )
    )
    assert control[2] == 0.0
    assert control[3] == 0.0

def test_d_only_near_field_does_not_inherit_offset_gain_attenuation():
    """With P disabled, a saturated image offset must not weaken flow D."""
    def command_for(offset):
        law = ControlLaw(command_filter_alpha=0.0, enable_slew_rate_limits=False)
        law.compute(
            TargetEstimate(found=True, offset_x=offset, offset_y=0.0),
            FlowResult(valid=True, mean_flow_x_norm=0.08, mean_flow_y_norm=0.0),
            1 / 60.0,
            lateral_p_scale=0.0,
            lateral_d_scale=0.6,
            enable_integral=False,
        )
        return law.last_roll_accel_cmd

    centered = command_for(0.0)
    saturated = command_for(0.9)
    assert math.isclose(centered, saturated, rel_tol=1e-9, abs_tol=1e-12)


def test_handoff_trim_admits_no_approach_era_sample():
    """Same provenance rule the probes follow, for the same reason.

    The offset setpoint DESCENT freezes is evidence about the near field. If an
    APPROACH-era sample can survive into it, the static operating point no
    longer matches the FINAL_PROBE ``mean_accel`` it is paired with.
    """
    cfg = MissionConfig(enable_center=False, approach_hold_dwell_sec=0.10)
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.start(0.0, 5.0)

    def update(t, area, offset_x):
        return routine.update(MissionInputs(
            t=t, dt=0.05,
            target=TargetEstimate(timestamp=t, found=True, area_fraction=area,
                                  offset_x=offset_x, offset_y=0.0),
            flow=FlowResult(timestamp=t, valid=True, divergence=0.0),
            actuation=ActuationFeedback(last_thrust_cmd=0.73,
                                        last_vertical_accel_cmd=0.1)))

    for i in range(40):                       # APPROACH at a far-field offset
        update(0.05 * i, 0.4, 0.40)
    assert routine._handoff_trim.mean_x > 0.30

    control = None
    for i in range(6):
        control = update(2.1 + 0.05 * i, cfg.approach_hold_area_fraction, 0.02)
        if control.substate == "final_probe":
            break
    assert control is not None and control.substate == "final_probe"

    update(2.5, cfg.approach_hold_area_fraction, 0.02)
    assert routine._handoff_trim.mean_x < 0.05, (
        "an APPROACH-era offset survived into the FINAL_PROBE trim")


def test_trim_knobs_are_reachable_config_fields():
    """These were read with ``getattr(cfg, ..., default)`` but never declared.

    ``with_overrides`` raises on undeclared names, so the knobs could not be set
    at all: every run silently flew the hardcoded fallback.
    """
    cfg = MissionConfig().with_overrides(
        center_trim_tau_sec=4.5,
        center_trim_mean_radius_max=0.33,
        center_trim_residual_radius_max=0.07,
        descent_trim_use_probe_mean=False,
    )
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    assert routine._center_trim.tau_sec == 4.5
    assert routine._center_trim_mean_radius_max == 0.33
    assert routine._center_trim_residual_radius_max == 0.07
    assert routine._descent_trim_use_probe_mean is False


def test_unseeded_trim_cannot_satisfy_a_gate_and_survives_dropout():
    """``inf`` before the first sample; a lost target holds, never decays.

    A dropout is missing evidence, not evidence of being centred -- folding
    zeros in during a dropout would walk the estimate toward image centre and
    report a settled trim that was never measured.
    """
    trim = VisualTrim(1.0)
    assert not (trim.mean_radius <= 0.20)
    assert not (trim.residual_radius <= 0.12)

    for _ in range(200):
        trim.update(0.10, 0.0, True, 1 / 30.0)
    settled = trim.mean_x
    for _ in range(200):
        trim.update(0.0, 0.0, False, 1 / 30.0)
    assert math.isclose(trim.mean_x, settled)

    trim.retune(13.4)                    # conditioning change keeps the mean
    assert trim.has_mean and math.isclose(trim.mean_x, settled)
    trim.reset()                         # fresh evidence discards it
    assert not trim.has_mean and not math.isfinite(trim.mean_radius)


def _final_probe_ready(**cfg_kw):
    """A routine in FINAL_PROBE with a known static-bias operating point."""
    routine = MissionRoutine(hover_thrust=0.73, config=MissionConfig(**cfg_kw))
    routine.start(0.0, 5.0)
    routine._substate = "final_probe"
    routine._final_probe_roll_accel_bias = 0.20
    routine._final_probe_pitch_accel_bias = 0.20
    routine._final_probe_roll_bias_initial = 0.20
    routine._final_probe_pitch_bias_initial = 0.20
    return routine


def _bias_inputs(t, realized, *, found=True, flow_valid=True, dt=1 / 30.0):
    return MissionInputs(
        t=t, dt=dt,
        target=TargetEstimate(timestamp=t, found=found, offset_x=0.1, offset_y=0.0),
        flow=FlowResult(timestamp=t, valid=flow_valid),
        actuation=ActuationFeedback(last_thrust_cmd=0.73,
                                    last_roll_accel_cmd=realized,
                                    last_pitch_accel_cmd=realized))


def test_far_field_accel_feedforward_is_zero_until_final_probe():
    """The new static term must not recreate hidden integral action far away."""
    routine = MissionRoutine(hover_thrust=0.73, config=MissionConfig())
    routine.start(0.0, 5.0)
    rows = _fly_to_descend(routine)
    assert rows["center"]["roll_accel_feedforward_m_s2"] == 0.0
    assert rows["center"]["pitch_accel_feedforward_m_s2"] == 0.0
    assert rows["approach_probe"]["roll_accel_feedforward_m_s2"] == 0.0
    assert rows["approach_probe"]["pitch_accel_feedforward_m_s2"] == 0.0


def test_final_probe_seed_comes_from_passive_approach_probe_mean():
    """Capture the APPROACH mean before the near-field probe reset."""
    routine = MissionRoutine(hover_thrust=0.73, config=MissionConfig())
    routine.start(0.0, 5.0)
    for _ in range(120):
        routine._roll_probe.update_accel(0.42, 1 / 30.0)
        routine._pitch_probe.update_accel(-0.31, 1 / 30.0)
    routine._begin_final_probe_measurement(4.0)
    assert math.isclose(routine._final_probe_roll_accel_bias, 0.42, abs_tol=1e-12)
    assert math.isclose(routine._final_probe_pitch_accel_bias, -0.31, abs_tol=1e-12)
    assert routine._roll_probe.result(0.0).n_samples == 0
    assert routine._pitch_probe.result(0.0).n_samples == 0


def test_final_probe_bias_tracks_the_realized_lateral_command():
    """During FINAL_PROBE it integrates the slow part of the D-only command."""
    tau = 2.0
    routine = _final_probe_ready(wind_trim_tau_sec=tau,
                                 wind_trim_deviation_limit_m_s2=10.0)
    target, dt, t = 0.70, 1 / 30.0, 0.0
    start = routine._final_probe_roll_accel_bias
    for _ in range(int(round(tau / dt))):
        t += dt
        routine._update_near_field_lateral_bias(_bias_inputs(t, target, dt=dt))
    reached = (routine._final_probe_roll_accel_bias - start) / (target - start)
    assert 0.60 < reached < 0.68, f"one tau moved {reached:.3f} of the way, want ~0.63"


def test_final_probe_bias_holds_outside_phase_or_without_measurement():
    """No far-field adaptation and no update on missing visual evidence."""
    far = _final_probe_ready()
    far._substate = "approach_probe"
    far._update_near_field_lateral_bias(_bias_inputs(1.0, 0.9))
    assert far._final_probe_roll_accel_bias == 0.20

    lost = _final_probe_ready()
    for i in range(200):
        lost._update_near_field_lateral_bias(
            _bias_inputs(i / 30.0, 0.0, found=False)
        )
    assert lost._final_probe_roll_accel_bias == 0.20

    no_flow = _final_probe_ready()
    for i in range(200):
        no_flow._update_near_field_lateral_bias(
            _bias_inputs(i / 30.0, 0.9, flow_valid=False)
        )
    assert no_flow._final_probe_roll_accel_bias == 0.20


def test_final_probe_bias_cannot_leave_the_approach_neighbourhood():
    """Near-field adaptation stays bounded around the passively measured seed."""
    limit = 0.35
    routine = _final_probe_ready(wind_trim_deviation_limit_m_s2=limit,
                                 wind_trim_tau_sec=0.2)
    for i in range(600):
        routine._update_near_field_lateral_bias(_bias_inputs(i / 30.0, 9.0))
    assert math.isclose(routine._final_probe_roll_accel_bias, 0.20 + limit)

    for i in range(600):
        routine._update_near_field_lateral_bias(_bias_inputs(i / 30.0, -9.0))
    assert math.isclose(routine._final_probe_roll_accel_bias, 0.20 - limit)


def _descent_bias_ready(*, adaptive=True, tau=2.0, limit=10.0):
    """A DESCEND routine with a known committed FINAL_PROBE operating point."""
    cfg = MissionConfig(
        wind_trim_adapt_in_descent=adaptive,
        wind_trim_tau_sec=tau,
        wind_trim_deviation_limit_m_s2=limit,
    )
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.start(0.0, 5.0)
    routine._substate = "descend"
    routine._t_descend_start = 0.0
    routine._descent_lateral_trim_frozen = True
    routine._final_probe_roll_bias_initial = 0.20
    routine._final_probe_pitch_bias_initial = -0.15
    routine._descent_roll_accel_bias = 0.20
    routine._descent_pitch_accel_bias = -0.15
    routine._descent_roll_bias_frozen = 0.20
    routine._descent_pitch_bias_frozen = -0.15
    return routine


def test_descent_bias_tracks_the_realized_lateral_command():
    """DESCENT continues the same slow static-trim estimator after commitment."""
    tau = 2.0
    routine = _descent_bias_ready(adaptive=True, tau=tau, limit=10.0)
    target, dt, t = 0.80, 1 / 30.0, 0.0
    start = routine._descent_roll_accel_bias
    for _ in range(int(round(tau / dt))):
        t += dt
        routine._update_near_field_lateral_bias(_bias_inputs(t, target, dt=dt))
    reached = (routine._descent_roll_accel_bias - start) / (target - start)
    assert 0.60 < reached < 0.68, f"one tau moved {reached:.3f} of the way, want ~0.63"
    row = routine.telemetry()
    assert row["descent_bias_adaptive"] == 1
    assert row["descent_roll_bias_deviation_m_s2"] > 0.0


def test_descent_bias_can_be_frozen_by_config():
    """The existing config switch remains a one-line A/B revert to frozen DESCENT."""
    routine = _descent_bias_ready(adaptive=False)
    for i in range(1, 300):
        routine.update(_bias_inputs(i / 30.0, 1.20))

    assert routine._descent_roll_accel_bias == 0.20
    assert routine._descent_pitch_accel_bias == -0.15
    row = routine.telemetry()
    assert row["descent_bias_adaptive"] == 0
    assert row["descent_roll_bias_deviation_m_s2"] == 0.0
    assert row["descent_pitch_bias_deviation_m_s2"] == 0.0


def test_descent_bias_shares_the_approach_seed_safety_neighbourhood():
    """FINAL_PROBE + DESCENT together get only one configured deviation allowance."""
    limit = 0.35
    routine = _descent_bias_ready(adaptive=True, tau=0.2, limit=limit)
    # Pretend FINAL_PROBE already used most of the positive allowance before
    # DESCENT was committed. DESCENT must still clamp around the original seed,
    # not around this later entry anchor.
    routine._descent_roll_accel_bias = 0.20 + 0.30
    routine._descent_roll_bias_frozen = routine._descent_roll_accel_bias
    for i in range(600):
        routine._update_near_field_lateral_bias(_bias_inputs(i / 30.0, 9.0))
    assert math.isclose(routine._descent_roll_accel_bias, 0.20 + limit)


def test_descent_bias_holds_without_visual_evidence():
    """A missing target or flow sample cannot move terminal static feedforward."""
    routine = _descent_bias_ready(adaptive=True)
    routine._update_near_field_lateral_bias(_bias_inputs(1.0, 1.0, found=False))
    routine._update_near_field_lateral_bias(_bias_inputs(2.0, 1.0, flow_valid=False))
    assert routine._descent_roll_accel_bias == 0.20
    assert routine._descent_pitch_accel_bias == -0.15


def test_center_trim_bound_still_derives_from_the_offset_gates():
    """This bound decides whether CENTER can hand off at all.

    It was originally a ``getattr`` fallback that DERIVED the value as
    ``max(center_offset_radius_max, center_legacy_box_threshold)`` -- the looser of
    the modern radial gate and the legacy box threshold. Replacing that with a
    hardcoded number silently tightened the handoff gate from 0.25 to 0.20 and
    stranded the mission in CENTER. Keep it derived so tuning either input still
    moves it.
    """
    cfg = MissionConfig()
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    assert routine._center_trim_mean_radius_max == max(
        cfg.center_offset_radius_max, cfg.center_legacy_box_threshold)

    wider = MissionRoutine(hover_thrust=0.73, config=MissionConfig(
        center_offset_radius_max=0.30, center_legacy_box_threshold=0.25))
    assert wider._center_trim_mean_radius_max == 0.30

    explicit = MissionRoutine(hover_thrust=0.73, config=MissionConfig(
        center_trim_mean_radius_max=0.42))
    assert explicit._center_trim_mean_radius_max == 0.42


def test_making_trim_knobs_explicit_preserved_their_values():
    """Every knob that was a ``getattr`` default must still resolve to the same
    number now that it is a declared field. A refactor that makes a constant
    settable must not also change it."""
    routine = MissionRoutine(hover_thrust=0.73, config=MissionConfig())
    assert routine._center_trim_tau == 1.0
    assert routine._center_trim_residual_radius_max == 0.06
    assert routine._center_trim_mean_radius_max == 0.25


def test_terminal_substates_are_latched_not_overwritten():
    cfg = BeeConfig.default().mission
    landed = MissionRoutine(hover_thrust=0.73, config=cfg)
    landed.mark_landed(1.0)
    assert landed.telemetry()["substate"] == "landed"

    aborted = MissionRoutine(hover_thrust=0.73, config=cfg)
    aborted.mark_aborted(1.0)
    assert aborted.telemetry()["substate"] == "aborted"
    aborted.mark_aborted(9.0)                     # idempotent
    assert aborted._t_aborted == 1.0


# --------------------------------------------------------------------------
# Sequencer contract
# --------------------------------------------------------------------------
def test_sequencer_reaches_closed_loop_and_terminals():
    clock = {"t": 0.0}
    events = []
    took_off = {"done": False}
    bundle = {"value": None}

    ports = SequencerPorts(
        takeoff_done=lambda: took_off["done"],
        takeoff_error=lambda: None,
        request_offboard=lambda: events.append("offboard_requested"),
        latest_vision=lambda: bundle["value"],
        log_info=lambda m: None,
        log_error=lambda m: None,
        log_event=lambda e, d: events.append(e),
        on_handoff=lambda stamp: events.append(f"handoff@{stamp}"),
        on_landed=lambda r: events.append("on_landed"),
        on_aborted=lambda r: events.append("on_aborted"),
        monotonic=lambda: clock["t"],
    )
    cfg = BeeConfig.default()
    status = PX4Status(nav_state_offboard=14, arming_state_armed=2)
    seq = FlightSequencer(cfg, status, ports)

    assert seq.setpoint_policy is SetpointPolicy.INHIBIT
    took_off["done"] = True
    seq.update()
    assert seq.setpoint_policy is SetpointPolicy.NEUTRAL_HOLD

    clock["t"] += cfg.scheduling.offboard_prestream_sec + 0.1
    seq.update()
    status.update(14, 2, False)
    seq.update()
    clock["t"] += cfg.scheduling.px4_offboard_switch_settle_sec + 0.1
    bundle["value"] = (TargetEstimate(timestamp=2.0, found=True),
                       FlowResult(timestamp=2.0, valid=True), {})
    seq.update()

    assert seq.phase == CLOSED_LOOP
    assert seq.setpoint_policy is SetpointPolicy.CONTROL
    assert "handoff@2.0" in events

    seq.enter_landed("truth contact")
    assert seq.setpoint_policy is SetpointPolicy.ZERO_THRUST
    seq.abort("too late")                          # must not override touchdown
    assert seq.phase == "landed"


def test_offboard_timeout_aborts():
    clock = {"t": 0.0}
    aborted = []
    ports = SequencerPorts(
        takeoff_done=lambda: True, takeoff_error=lambda: None,
        request_offboard=lambda: None, latest_vision=lambda: None,
        log_info=lambda m: None, log_error=lambda m: None,
        log_event=lambda e, d: None, on_handoff=lambda s: None,
        on_landed=lambda r: None, on_aborted=aborted.append,
        monotonic=lambda: clock["t"])
    cfg = BeeConfig.default()
    seq = FlightSequencer(cfg, PX4Status(nav_state_offboard=14,
                                         arming_state_armed=2), ports)
    seq.update()
    clock["t"] += cfg.scheduling.offboard_prestream_sec + 0.1
    seq.update()
    clock["t"] += cfg.scheduling.px4_offboard_confirm_timeout_sec + 0.1
    seq.update()
    assert seq.phase == "aborted" and aborted


# --------------------------------------------------------------------------
# Visual synchronisation gate
# --------------------------------------------------------------------------
def test_chi_regression_uses_old_values_and_real_dt_spacing():
    """The causal derivative must use the whole timestamped window, not only
    the previous sample. A linear D(t) has an exact least-squares slope even
    with deliberately irregular camera intervals."""
    probe = VisualMismatchProbe(
        derivative_window_sec=0.20,
        percentile_window_sec=1.0,
        peak_decay_tau_sec=2.0,
    )
    slope, intercept = 1.75, 0.12
    t = 0.0
    dts = [0.010, 0.022, 0.014, 0.019, 0.011, 0.025, 0.016]
    for i in range(80):
        dt = dts[i % len(dts)]
        t += dt
        d = intercept + slope * t
        probe.update(d, dt)

    assert probe.derivative_ready
    assert abs(probe.signal_rate - slope) < 1e-10
    expected_chi = slope - (intercept + slope * t) ** 2
    assert abs(probe.chi - expected_chi) < 1e-10


def test_constant_divergence_keeps_physical_chi_in_envelope():
    """No D* subtraction and no high-pass: a persistent non-zero chi is real
    mismatch evidence and must not be learned away as a bias."""
    probe = VisualMismatchProbe(
        derivative_window_sec=0.20,
        percentile_window_sec=1.0,
        peak_decay_tau_sec=2.0,
    )
    d, dt = 0.5, 1 / 60
    for _ in range(600):
        probe.update(d, dt)

    expected = d * d
    assert abs(probe.signal_rate) < 1e-12
    assert abs(probe.chi + expected) < 1e-12
    assert abs(probe.abs_chi - expected) < 1e-12
    assert abs(probe.percentile_chi - expected) < 1e-12
    assert abs(probe.peak_chi - expected) < 1e-12


def test_final_probe_envelope_reset_keeps_derivative_warm():
    """FINAL_PROBE restarts only the gate evidence, not the Ddot history."""
    probe = VisualMismatchProbe(
        derivative_window_sec=0.20,
        percentile_window_sec=1.0,
        peak_decay_tau_sec=2.0,
    )
    dt = 1 / 60
    for i in range(60):
        probe.update(0.2 + 0.4 * i * dt, dt)
    assert probe.derivative_ready
    rate_before = probe.signal_rate
    assert probe.peak_chi > 0.0

    probe.reset_envelope()
    assert probe.derivative_ready
    assert probe.peak_chi == 0.0
    probe.update(0.2 + 0.4 * 60 * dt, dt)
    assert probe.derivative_ready
    assert abs(probe.signal_rate - rate_before) < 1e-10
    assert probe.peak_chi > 0.0


def test_gate_is_advisory_until_enabled():
    """Disabled, the gate must never veto -- it only logs."""
    verdict = compute_tracking_gate(
        chi_peak=99.0, chi_limit=2.0, ready=True, enabled=False)
    assert not verdict.synchronized and verdict.feasible


def test_unready_probe_never_rejects():
    """No evidence must not read as evidence of desynchronisation."""
    verdict = compute_tracking_gate(
        chi_peak=99.0, chi_limit=2.0, ready=False, enabled=True)
    assert verdict.feasible


def test_absolute_chi_gate_separates_inside_and_outside_limit():
    good = compute_tracking_gate(
        chi_peak=0.8, chi_limit=2.0, ready=True, enabled=True)
    bad = compute_tracking_gate(
        chi_peak=3.5, chi_limit=2.0, ready=True, enabled=True)
    assert good.synchronized and good.feasible
    assert not bad.synchronized and not bad.feasible


def test_tracking_gate_joins_overall_feasibility():
    cfg = BeeConfig.default().mission
    routine = MissionRoutine(hover_thrust=0.73, config=cfg)
    routine.gate.feasible = True
    routine.roll_gate.feasible = True
    routine.pitch_gate.feasible = True
    assert routine.feasible
    routine.tracking_gate = compute_tracking_gate(
        chi_peak=99.0, chi_limit=2.0, ready=True, enabled=True)
    assert not routine.feasible, "a desynchronised vehicle was still feasible"
    assert any("mismatch" in r for r in routine._gate_failure_reasons())


# --------------------------------------------------------------------------
# Host clock steps
# --------------------------------------------------------------------------
def test_steady_wall_clock_survives_a_backward_step():
    """A backward system-clock step must not move the outgoing timebase.

    This is the failure that produced three PX4 offboard failsafes per flight:
    a VM host time sync stepped the guest clock back ~2.2 s, which moved every
    pending timer deadline forward by the same amount and starved the setpoint
    stream well past COM_OF_LOSS_T.
    """
    import bee_control.core.clock as clock_module
    real_time, offset = time.time, {"value": 0.0}
    clock_module.time.time = lambda: real_time() + offset["value"]
    try:
        clock = SteadyWallClock(step_threshold_sec=0.05)
        before = clock.wall_sec()
        offset["value"] = -2.2
        step = clock.check_step()
        after = clock.wall_sec()
        assert step is not None and step.backward
        assert after >= before, "the steady clock moved backwards"
        assert abs(step.step_sec + 2.2) < 0.01
    finally:
        clock_module.time.time = real_time


def test_px4_stamps_are_monotonic_across_a_step():
    """uORB timestamps must never go backwards; PX4 may treat them as stale."""
    import bee_control.core.clock as clock_module
    real_time, offset = time.time, {"value": 0.0}
    clock_module.time.time = lambda: real_time() + offset["value"]
    try:
        manager = TimeManager(steady_wall=True)
        stamps = []
        for i in range(5):
            if i == 2:
                offset["value"] -= 2.2
            stamps.append(manager.px4_timestamp_us())
            time.sleep(0.002)
        assert all(b > a for a, b in zip(stamps, stamps[1:])), stamps
    finally:
        clock_module.time.time = real_time


def test_clock_steps_reach_the_log():
    """A step is no longer fatal, but it is still a host fault worth seeing."""
    manager = TimeManager(steady_wall=True)
    assert set(manager.telemetry()) <= set(manager.telemetry_fields())
    assert "backward_steps_detected" in manager.telemetry_fields()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
            passed += 1
    print(f"\n{passed} contract tests passed")

def test_mission_package_is_lazy_facade_not_phase_registry():
    """Keep the mission package boundary acyclic.

    ``bee_control.mission`` is the public lazy facade.  The concrete phase
    registry belongs to ``bee_control.mission.phases``.  Copying registry
    imports into the package ``__init__`` recreates the circular-import failure
    that prevents ``bee_node`` from starting.
    """
    import bee_control.mission as mission
    from bee_control.mission.phases import PHASES
    from bee_control.mission.routine import MissionRoutine as Routine

    assert mission.MissionRoutine is Routine
    assert mission.display_name("center") == "CENTER"
    assert "display_name" in mission.__all__
    assert "PHASES" not in mission.__all__
    assert "center" in PHASES
