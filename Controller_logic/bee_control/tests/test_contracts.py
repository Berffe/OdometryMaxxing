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

from bee_control.core.clock import SteadyWallClock, TimeManager
from bee_control.core.config import BeeConfig, MissionConfig
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
from bee_control.mission.types import ActuationFeedback, ControlEffect, MissionInputs
from bee_control.mission.visual_mismatch import VisualMismatchProbe
from bee_control.vision.optical_flow import OpticalFlowEstimator
from bee_control.core.state import AttitudeSetpoint, FlowResult, TargetEstimate
from bee_control.diagnostics.telemetry import TelemetrySchemaError, collect_fields


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
    assert abs(probe.divergence_rate - slope) < 1e-10
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
    assert abs(probe.divergence_rate) < 1e-12
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
    rate_before = probe.divergence_rate
    assert probe.peak_chi > 0.0

    probe.reset_envelope()
    assert probe.derivative_ready
    assert probe.peak_chi == 0.0
    probe.update(0.2 + 0.4 * 60 * dt, dt)
    assert probe.derivative_ready
    assert abs(probe.divergence_rate - rate_before) < 1e-10
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