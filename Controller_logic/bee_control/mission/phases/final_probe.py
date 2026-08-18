"""FINAL_PROBE phase.

Stationary near-field probe at D*=0 and k_probe.  APPROACH_PROBE has already
established the visual-height hold; every acceleration and mismatch sample used
by the feasibility gates is collected fresh in this phase.
"""
from __future__ import annotations

from ..gates import compute_gate
from ..types import DESCEND, FINAL_PROBE, INFEASIBLE, MissionControl, PROBE_HOLD, PhaseSpec
from . import descend, infeasible, probe_hold


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    t = inputs.t
    dt = inputs.dt
    last_thrust_cmd = inputs.actuation.last_thrust_cmd
    last_vertical_accel_cmd = inputs.actuation.last_vertical_accel_cmd
    last_roll_accel_cmd = inputs.actuation.last_roll_accel_cmd
    last_pitch_accel_cmd = inputs.actuation.last_pitch_accel_cmd

    # Direct dispatch in tests / external state restoration still starts a clean
    # FINAL_PROBE window.  Normal flight already does this at APPROACH handoff.
    if routine._t_final_probe_hold_start is None:
        routine._begin_final_probe_measurement(t, inputs)

    routine._update_visual_mismatch(inputs)
    routine._update_probes(
        last_thrust_cmd, last_vertical_accel_cmd,
        last_roll_accel_cmd, last_pitch_accel_cmd, dt,
    )
    routine._refresh_probe_results(routine._final_probe_duration)

    if (
        routine.probe_result.ready
        and routine.roll_probe_result.ready
        and routine.pitch_probe_result.ready
    ):
        routine.gate = compute_gate(
            peak_accel=routine.probe_result.peak_accel,
            descent_divergence_setpoint=routine._d_star,
            initial_thrust_gain=routine._initial_thrust_gain,
            control_period_sec=routine._stability_dt,
            leg_clearance_m=routine._leg_clearance,
            ceiling_safety_factor=routine._safety,
            ceiling_margin=routine._ceiling_margin,
            descend_start_gain=routine._compute_probe_gain(),
            near_field_height_m=routine._near_field_height,
        )
        routine._compute_lateral_gates()
        routine._refresh_tracking_gate()

        vertical_ok = routine.vertical_landing_feasible
        roll_ok = routine.roll_landing_feasible
        pitch_ok = routine.pitch_landing_feasible

        if routine._probe_only:
            routine._substate = PROBE_HOLD
            return probe_hold.run(routine, inputs, just_entered=True)
        if vertical_ok and roll_ok and pitch_ok and routine._enable_descent:
            # FINAL_PROBE has kept the static term live. Commit its CURRENT
            # value only after every feasibility gate has passed; DESCENT starts
            # from that exact point and may keep adapting it slowly online.
            routine._commit_descent_lateral_trim(inputs)
            routine._substate = DESCEND
            routine._t_descend_start = t
            return descend.run(routine, inputs, just_entered=True)

        routine._substate = INFEASIBLE
        return infeasible.run(routine, inputs, just_entered=True)

    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=routine._compute_probe_gain(),
        lateral_p_scale=0.0,
        lateral_d_scale=routine._probe_lateral_d_scale,
        roll_accel_feedforward_m_s2=routine._final_probe_roll_accel_bias,
        pitch_accel_feedforward_m_s2=routine._final_probe_pitch_accel_bias,
        enable_integral=True,
        substate=FINAL_PROBE,
        info={
            "event": "final_probe_hold",
            "k": routine._compute_probe_gain(),
            "peak_accel": routine.probe_result.peak_accel,
            "roll_peak_accel": routine.roll_probe_result.peak_accel,
            "pitch_peak_accel": routine.pitch_probe_result.peak_accel,
            "peak_accel_at_handoff": routine.peak_accel_at_handoff,
            "roll_peak_accel_at_handoff": routine.roll_peak_accel_at_handoff,
            "pitch_peak_accel_at_handoff": routine.pitch_peak_accel_at_handoff,
            "hold_elapsed_sec": routine.probe_result.duration_sec,
            "hold_min_sec": routine._final_probe_duration,
            "k_explore": routine._k_explore,
        },
    )


SPEC = PhaseSpec(
    name=FINAL_PROBE,
    display_name="FINAL_PROBE",
    description="Fresh near-field feasibility probe at visual hover.",
    terminal=False,
    handler=run,
)
