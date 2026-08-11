"""APPROACH_PROBE phase.

Far-field descent while the three probes build their envelopes.

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
from ..schedule import scheduled_gain_at_time
from ..types import APPROACH_PROBE, ControlEffect, FINAL_PROBE, MissionControl, PhaseSpec


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    t = inputs.t
    dt = inputs.dt
    last_thrust_cmd = inputs.actuation.last_thrust_cmd
    last_vertical_accel_cmd = inputs.actuation.last_vertical_accel_cmd
    last_roll_accel_cmd = inputs.actuation.last_roll_accel_cmd
    last_pitch_accel_cmd = inputs.actuation.last_pitch_accel_cmd
    offset_x = inputs.offset_x
    offset_y = inputs.offset_y
    target_found = inputs.target_found
    area_fraction = inputs.area_fraction
    fov_saturated = inputs.fov_saturated

    if routine._t_approach_entry is None:
        routine._t_approach_entry = t

    elapsed = t - routine._t_approach_entry

    lateral_frac = (
        1.0 if routine._lateral_ramp <= 1e-9 else min(1.0, elapsed / routine._lateral_ramp)
    )
    lateral_blend = raised_cosine01(lateral_frac)

    lateral_p = routine._center_lateral_p_scale + (
        routine._probe_lateral_p_scale - routine._center_lateral_p_scale
    ) * lateral_blend

    lateral_d = routine._center_lateral_d_scale + (
        routine._probe_lateral_d_scale - routine._center_lateral_d_scale
    ) * lateral_blend

    approach_frac = (
        1.0 if routine._d_star_ramp_in <= 1e-9 else min(1.0, elapsed / routine._d_star_ramp_in)
    )
    approach_blend = raised_cosine01(approach_frac)
    d_approach_cmd = routine._approach_d_star * approach_blend

    # GAIN DROP, far -> near. k(t) decays from k_explore toward k_probe on the
    # integral of the COMMANDED approach D*. Because h(t) also decays as
    # exp(-integral D* dt), and k_ceiling is proportional to h, this decay is
    # PARALLEL to the shrinking stability ceiling -- the gain comes down at the
    # same rate the ceiling does, instead of holding at a far-field value while
    # the ceiling collapses beneath it. It bottoms out at k_probe, the gain the
    # near-field probe is flown at (see _compute_probe_gain).
    k_approach = scheduled_gain_at_time(
        elapsed_sec=elapsed,
        descent_divergence_setpoint=routine._approach_d_star,
        k_floor=routine._compute_probe_gain(),
        k_explore=routine._k_explore,
        d_star_ramp_in_sec=routine._d_star_ramp_in,
    )

    # FAR-field probing supplies the long observation interval needed to resolve
    # the platform motion. The three envelopes carry into FINAL_PROBE.
    #
    # CAVEAT, deliberate: k is RAMPING during this phase, so the loop gain that
    # shapes the thrust-command residual is itself moving, and the far-field
    # peak_accel is contaminated by that. This is accepted rather than fixed,
    # because (a) the far estimate exists to be SUPERSEDED by the near-field one
    # (which is flown at a constant k_probe -- see _do_final_probe), and (b) the
    # alternative, dropping the gain before probing starts, would fly the whole
    # approach far below the ceiling and throw away the bandwidth this design is
    # built to exploit.
    # Visual synchronisation evidence, gathered from the same frames.
    routine._update_visual_mismatch(inputs)
    routine._update_probes(
        last_thrust_cmd, last_vertical_accel_cmd,
        last_roll_accel_cmd, last_pitch_accel_cmd, dt,
    )
    routine._refresh_probe_results(0.0, routine._probe_min)

    near_field = routine._near_field_reached(
        offset_x=offset_x,
        offset_y=offset_y,
        target_found=target_found,
        area_fraction=area_fraction,
        fov_saturated=fov_saturated,
    )

    if near_field:
        routine._substate = FINAL_PROBE
        routine._t_final_probe_entry = t
        routine._t_final_probe_hold_start = None
        # All three accumulated envelopes carry into the near-field hold.

        return MissionControl(
            divergence_setpoint=d_approach_cmd,
            thrust_gain_override=k_approach,
            lateral_p_scale=lateral_p,
            lateral_d_scale=lateral_d,
            enable_integral=True,
            substate=FINAL_PROBE,
            effects=(ControlEffect.RESET_DIVERGENCE_INTEGRAL,),
            info={
                "event": "final_probe_start",
                "k": k_approach,
                "area_fraction": float(area_fraction),
                "fov_saturated": bool(fov_saturated),
                "near_field": True,
                "approach_elapsed_sec": elapsed,
                "approach_d_star_cmd": d_approach_cmd,
            },
        )

    return MissionControl(
        divergence_setpoint=d_approach_cmd,
        thrust_gain_override=k_approach,
        lateral_p_scale=lateral_p,
        lateral_d_scale=lateral_d,
        enable_integral=True,
        substate=APPROACH_PROBE,
        info={
            "k": k_approach,
            "k_probe": routine._compute_probe_gain(),
            "area_fraction": float(area_fraction),
            "fov_saturated": bool(fov_saturated),
            "near_field": False,
            "approach_elapsed_sec": elapsed,
            "approach_d_star_cmd": d_approach_cmd,
            "approach_ramp_frac": approach_blend,
            "lateral_ramp_frac": lateral_blend,
            "peak_accel": routine.probe_result.peak_accel,
            "roll_peak_accel": routine.roll_probe_result.peak_accel,
            "pitch_peak_accel": routine.pitch_probe_result.peak_accel,
            "probe_elapsed_sec": routine.probe_result.total_duration_sec,
        },
    )


SPEC = PhaseSpec(
    name=APPROACH_PROBE,
    display_name="APPROACH_PROBE",
    description="Far-field descent while the three probes build their envelopes.",
    terminal=False,
    handler=run,
)
