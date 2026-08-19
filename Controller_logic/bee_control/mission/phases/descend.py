"""DESCENT phase.

Scheduled-gain descent riding the safety-derated ceiling.

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

from ..math_utils import raised_cosine01
from ..schedule import critical_time, predicted_height, scheduled_gain_at_time
from ..types import ControlEffect, DESCEND, MissionControl, PhaseSpec


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    t = inputs.t

    elapsed = t - (routine._t_descend_start if routine._t_descend_start is not None else t)

    # Commitment has already been made in FINAL_PROBE. Keep measuring chi all
    # the way down because it is valuable bandwidth diagnostic data, but never
    # turn a post-commit mismatch into an INFEASIBLE transition.
    routine._update_visual_mismatch(inputs)

    # Each translational axis now follows the same exponential trajectory but
    # keeps its own independently probed disturbance floor.
    k = scheduled_gain_at_time(
        elapsed_sec=elapsed,
        descent_divergence_setpoint=routine._d_star,
        k_floor=routine.gate.k_floor,
        k_explore=routine.gate.k_descend_start,
        d_star_ramp_in_sec=routine._descent_d_star_ramp_in,
    )
    roll_k = scheduled_gain_at_time(
        elapsed_sec=elapsed,
        descent_divergence_setpoint=routine._d_star,
        k_floor=routine.roll_gate.k_floor,
        k_explore=routine.roll_gate.k_descend_start,
        d_star_ramp_in_sec=routine._descent_d_star_ramp_in,
    )
    pitch_k = scheduled_gain_at_time(
        elapsed_sec=elapsed,
        descent_divergence_setpoint=routine._d_star,
        k_floor=routine.pitch_gate.k_floor,
        k_explore=routine.pitch_gate.k_descend_start,
        d_star_ramp_in_sec=routine._descent_d_star_ramp_in,
    )

    roll_ratio = (
        roll_k / routine.roll_gate.k_descend_start
        if routine.roll_gate.k_descend_start > 1e-9 else 1.0
    )
    pitch_ratio = (
        pitch_k / routine.pitch_gate.k_descend_start
        if routine.pitch_gate.k_descend_start > 1e-9 else 1.0
    )
    roll_p_scale = 0.0
    roll_d_scale = routine.roll_probe_lateral_d_scale * roll_ratio
    pitch_p_scale = 0.0
    pitch_d_scale = routine.pitch_probe_lateral_d_scale * pitch_ratio

    # Legacy shared fields remain populated for compatibility and represent the
    # most demanding active lateral axis. ControlLaw receives the specific
    # per-axis fields below.
    lateral_p_scale = 0.0
    lateral_d_scale = max(roll_d_scale, pitch_d_scale)
    h_pred = predicted_height(routine._h0, routine._d_star, elapsed, routine._descent_d_star_ramp_in)

    if routine._descent_d_star_ramp_in <= 1e-9:
        linear_frac = 1.0
    else:
        linear_frac = min(1.0, elapsed / routine._descent_d_star_ramp_in)

    ramp_frac = raised_cosine01(linear_frac)
    d_star_cmd = routine._d_star * ramp_frac

    return MissionControl(
        divergence_setpoint=d_star_cmd,
        thrust_gain_override=k,
        lateral_p_scale=lateral_p_scale,
        lateral_d_scale=lateral_d_scale,
        roll_p_scale=roll_p_scale,
        roll_d_scale=roll_d_scale,
        pitch_p_scale=pitch_p_scale,
        pitch_d_scale=pitch_d_scale,
        # Near field is deliberately flow-only in feedback: the static wind
        # trim was adapted throughout FINAL_PROBE, carried bumplessly across the
        # descent commitment, and may continue adapting slowly as wind changes.
        # Saturated image position has zero command authority from FINAL_PROBE
        # onward.
        roll_offset_setpoint=0.0,
        pitch_offset_setpoint=0.0,
        roll_accel_feedforward_m_s2=routine._descent_roll_accel_bias,
        pitch_accel_feedforward_m_s2=routine._descent_pitch_accel_bias,
        enable_integral=False,
        substate=DESCEND,
        # DESCEND starts with no inherited vertical bias, and further integral
        # accumulation is disabled above, so the contribution stays exactly zero
        # thereafter. Only the entry tick carries the effect.
        effects=(
            (ControlEffect.RESET_DIVERGENCE_INTEGRAL,) if just_entered else ()
        ),
        info={
            "just_entered": just_entered,
            "event": "descent_start" if just_entered else "",
            "h_pred": h_pred,
            "k": k,
            "k_min": routine.gate.k_min,
            "k_floor": routine.gate.k_floor,
            "k_target": routine.gate.k_target,
            "k_ceiling_leg": routine.gate.k_ceiling_leg,
            "k_explore": routine.gate.k_explore,
            "k_descend_start": routine.gate.k_descend_start,
            "h_crit": routine.gate.h_crit,
            "vertical_accel_capacity_floor": routine.gate.accel_capacity_floor,
            "vertical_accel_capacity_ceiling": routine.gate.accel_capacity_ceiling,
            "roll_k": roll_k,
            "roll_k_floor": routine.roll_gate.k_floor,
            "roll_k_target": routine.roll_gate.k_target,
            "roll_k_ceiling_leg": routine.roll_gate.k_ceiling_leg,
            "roll_accel_capacity_floor": routine.roll_gate.accel_capacity_floor,
            "roll_accel_capacity_ceiling": routine.roll_gate.accel_capacity_ceiling,
            "pitch_k": pitch_k,
            "pitch_k_floor": routine.pitch_gate.k_floor,
            "pitch_k_target": routine.pitch_gate.k_target,
            "pitch_k_ceiling_leg": routine.pitch_gate.k_ceiling_leg,
            "pitch_accel_capacity_floor": routine.pitch_gate.accel_capacity_floor,
            "pitch_accel_capacity_ceiling": routine.pitch_gate.accel_capacity_ceiling,
            "roll_p_scale": roll_p_scale,
            "roll_d_scale": roll_d_scale,
            "pitch_p_scale": pitch_p_scale,
            "pitch_d_scale": pitch_d_scale,
            "descent_trim_offset_x": routine._descent_trim_offset_x,
            "descent_trim_offset_y": routine._descent_trim_offset_y,
            "descent_roll_accel_bias_m_s2": routine._descent_roll_accel_bias,
            "descent_pitch_accel_bias_m_s2": routine._descent_pitch_accel_bias,
            "elapsed_sec": elapsed,
            "t_crit_sec": critical_time(
                routine._h0,
                routine._d_star,
                routine.gate.h_crit,
                routine._descent_d_star_ramp_in,
            ),
            "d_star_ramp_frac": ramp_frac,
            "d_star_ramp_linear_frac": linear_frac,
            "d_star_target": routine._d_star,
        },
    )


SPEC = PhaseSpec(
    name=DESCEND,
    display_name="DESCENT",
    description="Scheduled-gain descent riding the safety-derated ceiling.",
    terminal=False,
    handler=run,
)
