"""PROBE_HOLD phase.

Probe-only mode: hold indefinitely, never descend.

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

from ..types import MissionControl, PROBE_HOLD, PhaseSpec


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    # Probe-only mode keeps the vehicle hovering after the one-time verdict,
    # but chi remains useful diagnostic evidence. It cannot change the frozen
    # FINAL_PROBE decision.
    routine._update_visual_mismatch(inputs)

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
        substate=PROBE_HOLD,
        info={
            "just_entered": just_entered,
            "probe_only": True,
            "peak_accel": routine.probe_result.peak_accel,
            "roll_peak_accel": routine.roll_probe_result.peak_accel,
            "pitch_peak_accel": routine.pitch_probe_result.peak_accel,
            "k_min": routine.gate.k_min,
            "h_crit": routine.gate.h_crit,
            "k_explore": routine.gate.k_explore,
            "vertical_feasible": routine.vertical_feasible,
            "roll_feasible": routine.roll_feasible,
            "pitch_feasible": routine.pitch_feasible,
            "feasible_if_descended": routine.feasible,
            "leg_clearance_m": routine._leg_clearance,
        },
    )


SPEC = PhaseSpec(
    name=PROBE_HOLD,
    display_name="PROBE_HOLD",
    description="Probe-only mode: hold indefinitely, never descend.",
    terminal=False,
    handler=run,
)
