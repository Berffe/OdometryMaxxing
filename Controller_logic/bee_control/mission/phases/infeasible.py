"""INFEASIBLE phase.

Active visual hover at probe gains after any feasibility rejection: stability
ceiling, disturbance-authority floor/gain margin, or visual mismatch bandwidth.

Phase contract
--------------
``run(routine, inputs, just_entered=False)`` returns the ``MissionControl`` for
one tick, and may latch ``routine._substate`` when it hands off. ``SPEC`` is
what the registry in ``mission/routine.py`` picks up.

Adding a phase means writing a file shaped like this one and listing it in
``phases/__init__.py``. Nothing outside ``mission/`` changes -- not the node,
not the diagnostics writer, not the log schema.

The body is unchanged from the single-file revision apart from the mechanical
``self`` -> ``routine`` rename that comes with being a free function.
"""
from __future__ import annotations

from ..types import INFEASIBLE, MissionControl, PhaseSpec


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    # The verdict is already latched, but keep chi alive for diagnosis while
    # the vehicle visually hovers at the probe gain.
    routine._update_visual_mismatch(inputs)

    reasons = routine._gate_failure_reasons()
    reason = "; ".join(reasons) if reasons else "unknown feasibility failure"
    failed_axes = routine._failed_axes()
    failed_criteria = routine._failed_criteria()
    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=routine._compute_probe_gain(),
        lateral_p_scale=0.0,
        lateral_d_scale=max(
            routine.roll_probe_lateral_d_scale,
            routine.pitch_probe_lateral_d_scale,
        ),
        roll_d_scale=routine.roll_probe_lateral_d_scale,
        pitch_d_scale=routine.pitch_probe_lateral_d_scale,
        roll_accel_feedforward_m_s2=routine._final_probe_roll_accel_bias,
        pitch_accel_feedforward_m_s2=routine._final_probe_pitch_accel_bias,
        enable_integral=True,
        substate=INFEASIBLE,
        info={
            "just_entered": just_entered,
            "reason": reason,
            "infeasible_reason": reason,
            "infeasible_axes": failed_axes,
            "infeasible_criteria": failed_criteria,
            "h_crit": routine.gate.h_crit,
            "leg_clearance_m": routine._leg_clearance,
            "k_min": routine.gate.k_min,
            "peak_accel": routine.probe_result.peak_accel,
            "vertical_accel_capacity_floor": routine.gate.accel_capacity_floor,
            "vertical_accel_capacity_ceiling": routine.gate.accel_capacity_ceiling,
            "roll_k_min": routine.roll_gate.k_min,
            "roll_k_floor": routine.roll_gate.k_floor,
            "roll_k_ceiling_leg": routine.roll_gate.k_ceiling_leg,
            "roll_accel_capacity_floor": routine.roll_gate.accel_capacity_floor,
            "roll_accel_capacity_ceiling": routine.roll_gate.accel_capacity_ceiling,
            "pitch_k_min": routine.pitch_gate.k_min,
            "pitch_k_floor": routine.pitch_gate.k_floor,
            "pitch_k_ceiling_leg": routine.pitch_gate.k_ceiling_leg,
            "pitch_accel_capacity_floor": routine.pitch_gate.accel_capacity_floor,
            "pitch_accel_capacity_ceiling": routine.pitch_gate.accel_capacity_ceiling,
            "vertical_feasible": routine.vertical_feasible,
            "roll_feasible": routine.roll_feasible,
            "pitch_feasible": routine.pitch_feasible,
            "vertical_landing_feasible": routine.vertical_landing_feasible,
            "roll_landing_feasible": routine.roll_landing_feasible,
            "pitch_landing_feasible": routine.pitch_landing_feasible,
            "vertical_tracking_ok": routine.vertical_tracking_ok,
            "roll_tracking_ok": routine.roll_tracking_ok,
            "pitch_tracking_ok": routine.pitch_tracking_ok,
        },
    )


SPEC = PhaseSpec(
    name=INFEASIBLE,
    display_name="INFEASIBLE",
    description=(
        "Active visual hover after stability, authority/margin, or visual "
        "tracking-bandwidth rejection."
    ),
    terminal=False,
    handler=run,
)
