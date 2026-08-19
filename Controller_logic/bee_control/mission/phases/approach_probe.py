"""APPROACH_PROBE phase.

Far-field approach with diagnostic probing and a visual-height outer P loop.
The outer loop regulates log visual scale and hands FINAL_PROBE an already
settled operating point: target at the requested visual fraction and vertical
divergence near zero.
"""
from __future__ import annotations

import math

from ..math_utils import clamp, raised_cosine01
from ..schedule import scheduled_gain_from_integral
from ..types import APPROACH_PROBE, FINAL_PROBE, MissionControl, PhaseSpec


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    t = inputs.t
    dt = inputs.dt
    last_thrust_cmd = inputs.actuation.last_thrust_cmd
    last_vertical_accel_cmd = inputs.actuation.last_vertical_accel_cmd
    last_roll_accel_cmd = inputs.actuation.last_roll_accel_cmd
    last_pitch_accel_cmd = inputs.actuation.last_pitch_accel_cmd
    area_fraction = inputs.area_fraction
    if routine._t_approach_entry is None:
        routine._t_approach_entry = t

    elapsed = t - routine._t_approach_entry

    # Keep the adaptive visual-centre finder online throughout APPROACH.
    # The lateral P schedule is allowed to change the equilibrium naturally;
    # the outer adaptation then follows the resulting physical centring error
    # instead of using the former inverse-P-scale algebraic correction.
    routine._update_visual_center_adaptation(inputs)
    (
        roll_offset_setpoint,
        pitch_offset_setpoint,
        roll_accel_feedforward,
        pitch_accel_feedforward,
    ) = routine._far_field_lateral_control_terms(inputs)

    # Ramp only the maximum allowed closing divergence.  The slow visual-scale
    # P loop then backs D* away from that limit as the requested handoff height
    # is approached, and may command a small retreat after overshoot.
    approach_frac = (
        1.0 if routine._approach_d_star_ramp_in <= 1e-9
        else min(1.0, elapsed / routine._approach_d_star_ramp_in)
    )
    approach_blend = raised_cosine01(approach_frac)
    d_closing_limit = routine._approach_d_star * approach_blend

    log_scale_error = float("inf")
    if inputs.target_found and not inputs.fov_saturated and area_fraction > 1e-9:
        log_scale_error = 0.5 * math.log(
            routine._approach_hold_area_fraction / max(1e-9, float(area_fraction))
        )
        d_visual = routine._approach_visual_p_gain * log_scale_error
        d_approach_cmd = clamp(
            d_visual,
            -routine._approach_retreat_d_star_limit,
            d_closing_limit,
        )
    elif inputs.target_found and inputs.fov_saturated:
        # Saturated box geometry is not a range measurement.  Retreat until a
        # valid visual scale returns instead of trying to regulate a clipped box.
        d_approach_cmd = -routine._approach_retreat_d_star_limit
    else:
        d_approach_cmd = 0.0

    # Keep the approach gain schedule tied to what was actually commanded.  It
    # remains monotone: retreat never raises the gain, which is conservative as
    # the vehicle enters the near-field stability region.
    routine._approach_divergence_integral += max(0.0, d_approach_cmd) * dt
    k_approach = scheduled_gain_from_integral(
        commanded_divergence_integral=routine._approach_divergence_integral,
        k_floor=routine._compute_probe_gain(),
        k_explore=routine._k_explore,
    )

    # Lateral D rides the SAME accumulated integral, per axis, so it reaches its
    # near-field value exactly when the vertical gain reaches k_probe.  The
    # endpoint is derived from the lateral ceiling rather than configured; when
    # the far-field gain is already admissible at the handoff height the two
    # endpoints coincide and this is flat, which is the correct answer for a
    # platform whose lateral authority is not the binding constraint.
    roll_kd_far = routine._roll_d_gain * routine._center_lateral_d_scale
    pitch_kd_far = routine._pitch_d_gain * routine._center_lateral_d_scale
    roll_kd = scheduled_gain_from_integral(
        commanded_divergence_integral=routine._approach_divergence_integral,
        k_floor=routine.roll_probe_lateral_gain,
        k_explore=roll_kd_far,
    )
    pitch_kd = scheduled_gain_from_integral(
        commanded_divergence_integral=routine._approach_divergence_integral,
        k_floor=routine.pitch_probe_lateral_gain,
        k_explore=pitch_kd_far,
    )
    roll_d = roll_kd / routine._roll_d_gain if routine._roll_d_gain > 1e-9 else 0.0
    pitch_d = pitch_kd / routine._pitch_d_gain if routine._pitch_d_gain > 1e-9 else 0.0
    lateral_d = max(roll_d, pitch_d)

    # P is not the flow loop the de Croon bound constrains, so it has no derived
    # endpoint.  It rides the same decay curve as D -- ``1 - exp(-integral)`` --
    # rather than the D endpoints themselves, because those coincide whenever
    # the ceiling is not binding and would leave P with no progress signal.
    lateral_blend = clamp(
        1.0 - math.exp(-max(0.0, routine._approach_divergence_integral)), 0.0, 1.0
    )
    lateral_p = routine._center_lateral_p_scale + (
        routine._probe_lateral_p_scale - routine._center_lateral_p_scale
    ) * lateral_blend

    # APPROACH probes are diagnostics only.  FINAL_PROBE resets these envelopes
    # before collecting any evidence used by a feasibility gate.
    routine._update_visual_mismatch(inputs)
    routine._update_probes(
        last_thrust_cmd, last_vertical_accel_cmd,
        last_roll_accel_cmd, last_pitch_accel_cmd, dt,
    )
    routine._refresh_probe_results(0.0)

    centered = routine._is_centered(
        inputs.offset_x, inputs.offset_y, inputs.target_found
    )
    hold_condition = (
        centered
        and inputs.flow_valid
        and not inputs.fov_saturated
        and math.isfinite(log_scale_error)
        and abs(log_scale_error) <= routine._approach_hold_log_scale_tol
        and abs(inputs.divergence_1_s) <= routine._approach_hold_divergence_tol
    )

    if hold_condition:
        if routine._approach_hold_since is None:
            routine._approach_hold_since = t
        hold_dwell = t - routine._approach_hold_since
    else:
        routine._approach_hold_since = None
        hold_dwell = 0.0

    if hold_condition and hold_dwell >= routine._approach_hold_dwell:
        routine._substate = FINAL_PROBE
        routine._begin_final_probe_measurement(t, inputs)

        # This is the FIRST FINAL_PROBE command, so it must already BE a
        # FINAL_PROBE command. It duplicates the lateral contract in
        # phases/final_probe.py rather than delegating, because delegating
        # would re-run the probe update this tick has already performed --
        # keep the two blocks in step.
        handoff_roll_sp, handoff_pitch_sp = routine._near_field_lateral_setpoint(
            inputs
        )

        return MissionControl(
            divergence_setpoint=0.0,
            thrust_gain_override=routine._compute_probe_gain(),
            # Near-field handoff, three simultaneous changes on ONE tick:
            # image-position P collapses from the APPROACH value to the small
            # residual, the visual setpoint drops the learned wind bias and
            # keeps only geometric tilt, and the passive APPROACH acceleration
            # mean is activated as the initial static term. The steady wind
            # force therefore moves from the P error to the feedforward
            # without ever passing through zero.
            lateral_p_scale=routine._final_probe_lateral_p_scale,
            lateral_d_scale=max(
                routine.roll_probe_lateral_d_scale,
                routine.pitch_probe_lateral_d_scale,
            ),
            roll_d_scale=routine.roll_probe_lateral_d_scale,
            pitch_d_scale=routine.pitch_probe_lateral_d_scale,
            roll_offset_setpoint=handoff_roll_sp,
            pitch_offset_setpoint=handoff_pitch_sp,
            roll_accel_feedforward_m_s2=routine._final_probe_roll_accel_bias,
            pitch_accel_feedforward_m_s2=routine._final_probe_pitch_accel_bias,
            scale_lateral_d_with_offset=False,
            enable_integral=True,
            substate=FINAL_PROBE,
            info={
                "event": "final_probe_start",
                "k": routine._compute_probe_gain(),
                "area_fraction": float(area_fraction),
                "approach_hold_area_fraction": routine._approach_hold_area_fraction,
                "approach_log_scale_error": log_scale_error,
                "approach_measured_divergence": inputs.divergence_1_s,
                "approach_hold_condition": True,
                "approach_hold_dwell_sec": hold_dwell,
                "lateral_p_scale": routine._final_probe_lateral_p_scale,
            },
        )

    return MissionControl(
        divergence_setpoint=d_approach_cmd,
        thrust_gain_override=k_approach,
        lateral_p_scale=lateral_p,
        lateral_d_scale=lateral_d,
        # Per-axis, because the two ceilings differ on a non-square lens.
        # lateral_d_scale above stays populated as the most demanding axis.
        roll_d_scale=roll_d,
        pitch_d_scale=pitch_d,
        roll_offset_setpoint=roll_offset_setpoint,
        pitch_offset_setpoint=pitch_offset_setpoint,
        roll_accel_feedforward_m_s2=roll_accel_feedforward,
        pitch_accel_feedforward_m_s2=pitch_accel_feedforward,
        enable_integral=True,
        substate=APPROACH_PROBE,
        info={
            "k": k_approach,
            "k_probe": routine._compute_probe_gain(),
            "area_fraction": float(area_fraction),
            "fov_saturated": bool(inputs.fov_saturated),
            "approach_elapsed_sec": elapsed,
            "approach_d_star_cmd": d_approach_cmd,
            "approach_d_star_limit": d_closing_limit,
            "approach_hold_area_fraction": routine._approach_hold_area_fraction,
            "approach_log_scale_error": log_scale_error,
            "approach_measured_divergence": inputs.divergence_1_s,
            "approach_hold_condition": hold_condition,
            "approach_hold_dwell_sec": hold_dwell,
            "approach_ramp_frac": approach_blend,
            "lateral_ramp_frac": lateral_blend,
            "peak_accel": routine.probe_result.peak_accel,
            "roll_peak_accel": routine.roll_probe_result.peak_accel,
            "pitch_peak_accel": routine.pitch_probe_result.peak_accel,
            "probe_elapsed_sec": routine.probe_result.duration_sec,
        },
    )


SPEC = PhaseSpec(
    name=APPROACH_PROBE,
    display_name="APPROACH_PROBE",
    description="Visual-height approach hold with diagnostic-only probing.",
    terminal=False,
    handler=run,
)
