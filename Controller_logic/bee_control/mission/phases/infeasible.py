"""INFEASIBLE phase.

Active visual hover at probe gains; no gain window exists.

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
    # This phase is time-invariant: it holds the same command indefinitely,
    # so it reads nothing off `inputs`. The signature stays uniform with every
    # other phase so the registry can dispatch them identically.
    del inputs

    reasons = routine._gate_failure_reasons()
    reason = "; ".join(reasons) if reasons else "unknown feasibility failure"
    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=routine._compute_probe_gain(),
        lateral_p_scale=routine._probe_lateral_p_scale,
        lateral_d_scale=routine._probe_lateral_d_scale,
        enable_integral=True,
        substate=INFEASIBLE,
        info={
            "just_entered": just_entered,
            "reason": reason,
            "infeasible_reason": reason,
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
        },
    )


SPEC = PhaseSpec(
    name=INFEASIBLE,
    display_name="INFEASIBLE",
    description="Active visual hover at probe gains; no gain window exists.",
    terminal=False,
    handler=run,
)
