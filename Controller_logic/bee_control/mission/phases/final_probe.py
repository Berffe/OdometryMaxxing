"""FINAL_PROBE phase.

Stationary near-field probe at D*=0 and k_probe.  APPROACH_PROBE has already
established the visual-height hold; every acceleration and mismatch sample used
by the feasibility gates is collected fresh in this phase.

Lateral law here is three terms:

    a_cmd = a_static + a_P(offset - e_geom) + a_D(flow)

``a_static`` carries the steady wind (see routine's wind-trim section), ``a_P``
holds the configured near-field ``probe_lateral_p_scale`` reached by the
APPROACH schedule about the GEOMETRIC tilt setpoint only, and ``a_D`` is the optical-flow damping the
feasibility gates are measured against -- which is why this phase asks the
control law NOT to attenuate D with the large-offset blend.
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
            stability_dt_sec=routine._vertical_stability_dt,
            leg_clearance_m=routine._leg_clearance,
            ceiling_safety_factor=routine._safety,
            ceiling_margin=routine._ceiling_margin,
            descend_start_gain=routine._compute_probe_gain(),
            near_field_height_m=routine._near_field_height,
            impose_ceiling_floor=routine._enable_commit_gate,
        )
        routine._compute_lateral_gates()
        routine._refresh_tracking_gate()
        routine._verdict_reached = True

        if routine._probe_only:
            routine._substate = PROBE_HOLD
            return probe_hold.run(routine, inputs, just_entered=True)
        if routine._enable_descent and routine._commit_gate_permits_descent():
            # FINAL_PROBE has kept the static term live. Commit its CURRENT
            # value only after every feasibility gate has passed; DESCENT starts
            # from that exact point and may keep adapting it slowly online.
            routine._commit_descent_lateral_trim(inputs)
            routine._substate = DESCEND
            routine._t_descend_start = t
            return descend.run(routine, inputs, just_entered=True)

        routine._substate = INFEASIBLE
        return infeasible.run(routine, inputs, just_entered=True)

    # Geometric tilt compensation ONLY -- deliberately not the adaptive
    # far-field bias, which would double-count the wind the static feedforward
    # is already supplying.  See MissionRoutine._near_field_lateral_setpoint.
    roll_offset_setpoint, pitch_offset_setpoint = (
        routine._near_field_lateral_setpoint(inputs)
    )

    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=routine._compute_probe_gain(),
        lateral_p_scale=routine._probe_lateral_p_scale,
        lateral_d_scale=max(
            routine.roll_probe_lateral_d_scale,
            routine.pitch_probe_lateral_d_scale,
        ),
        roll_d_scale=routine.roll_probe_lateral_d_scale,
        pitch_d_scale=routine.pitch_probe_lateral_d_scale,
        roll_offset_setpoint=roll_offset_setpoint,
        pitch_offset_setpoint=pitch_offset_setpoint,
        roll_accel_feedforward_m_s2=routine._final_probe_roll_accel_bias,
        pitch_accel_feedforward_m_s2=routine._final_probe_pitch_accel_bias,
        # No large-offset blend: it is CENTER-only.  This matters most here,
        # because the lateral feasibility gates are computed against this
        # phase's commanded gains -- an offset-dependent multiplier on either
        # branch would make the gates claim authority the vehicle does not
        # have, which is optimistic in the unsafe direction.
        apply_offset_gain_blend=False,
        # Subsumed by the line above while the blend stays CENTER-only. Kept
        # because it is the field the log schema and the gate contract name,
        # and because it states the D exemption independently of who else
        # happens to be running the blend.
        scale_lateral_d_with_offset=False,
        enable_integral=True,
        substate=FINAL_PROBE,
        info={
            "event": "final_probe_hold",
            "lateral_p_scale": routine._probe_lateral_p_scale,
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