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
import tempfile

from bee_control.core.config import BeeConfig, MissionConfig
from bee_control.core.controller_state import ControllerState, PX4Status, VisionTelemetry
from bee_control.diagnostics.diagnostics_writer import DiagnosticsWriter
from bee_control.interfaces.flight_sequencer import (
    CLOSED_LOOP,
    FlightSequencer,
    SequencerPorts,
    SetpointPolicy,
)
from bee_control.mission.routine import MissionRoutine
from bee_control.mission.types import ActuationFeedback, ControlEffect, MissionInputs
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


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
            passed += 1
    print(f"\n{passed} contract tests passed")
