"""CENTER phase.

Adaptive physical centring under steady wind.

CENTER keeps the ordinary lateral P/D controller intact but moves its visual
reference slowly.  Geometric camera-tilt compensation identifies where the
world-vertical ray appears in the image; ``VisualCenterAdaptation`` then shifts
that reference so the non-zero P error required to reject wind is preserved
while the platform moves onto that vertical ray.
"""
from __future__ import annotations

import math

from ..probe import ProbeResult
from ..types import APPROACH_PROBE, CENTER, ControlEffect, MissionControl, PhaseSpec


def _diagnostics(routine, *, offset_radius: float, flow_radius: float, dwell: float,
                 required_dwell: float, elapsed: float) -> dict:
    trim_mean_x, trim_mean_y, trim_mean_radius, trim_residual_radius = (
        routine._center_trim_snapshot
    )
    physical_mean_x, physical_mean_y, physical_mean_radius, physical_residual_radius = (
        routine._center_physical_trim_snapshot
    )
    adapt = routine._visual_center_adaptation_snapshot
    return {
        "center_dwell_sec": dwell,
        "center_required_dwell_sec": required_dwell,
        "center_elapsed_sec": elapsed,
        "center_offset_radius": offset_radius,
        "center_flow_radius_norm_s": flow_radius,
        "center_trim_mean_x": trim_mean_x,
        "center_trim_mean_y": trim_mean_y,
        "center_trim_mean_radius": trim_mean_radius,
        "center_trim_residual_radius": trim_residual_radius,
        "center_geometric_offset_x": routine._center_geometric_offset_x,
        "center_geometric_offset_y": routine._center_geometric_offset_y,
        "center_physical_mean_x": physical_mean_x,
        "center_physical_mean_y": physical_mean_y,
        "center_physical_mean_radius": physical_mean_radius,
        "center_physical_residual_radius": physical_residual_radius,
        "center_visual_bias_x": adapt.bias_x,
        "center_visual_bias_y": adapt.bias_y,
        "center_visual_bias_radius": adapt.bias_radius,
        "center_visual_adaptation_rate_x_norm_s": adapt.rate_x_norm_s,
        "center_visual_adaptation_rate_y_norm_s": adapt.rate_y_norm_s,
        "center_visual_adaptation_weight": adapt.adaptation_weight,
        "center_visual_adaptation_active": adapt.active,
    }


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    """Continuously adapt the visual centre, then hand off once physically settled.

    There is no calibration/recentring subphase anymore.  Every valid CENTER
    sample contributes to the slow outer adaptation:

        e_phys = e_meas - e_geom
        e_sp   = e_geom - b
        b_dot  ~ e_phys

    Large optical flow smoothly slows ``b_dot`` inside VisualCenterAdaptation,
    so the moving visual reference cannot chase the fast lateral transient.
    CENTER finishes only when the normal trim/flow conditions AND the physical
    tilt-corrected centring condition remain satisfied for the configured dwell.
    """
    t = inputs.t
    offset_x = inputs.offset_x
    offset_y = inputs.offset_y
    target_found = inputs.target_found
    flow_x_norm_s = inputs.flow_x_norm_s
    flow_y_norm_s = inputs.flow_y_norm_s
    flow_valid = inputs.flow_valid

    # Shared trim estimators were already advanced by MissionRoutine.update().
    # Fold their current physical-error mean into the slow moving reference
    # before producing this tick's setpoint.
    routine._update_visual_center_adaptation(inputs)
    (
        roll_offset_setpoint,
        pitch_offset_setpoint,
        roll_accel_feedforward,
        pitch_accel_feedforward,
    ) = routine._far_field_lateral_control_terms(inputs)

    # Height-free visual mismatch remains diagnostic in CENTER.
    routine._update_visual_mismatch(inputs)

    if routine._center_start_t is None:
        routine._center_start_t = t

    offset_radius = math.hypot(float(offset_x), float(offset_y))
    flow_radius = (
        math.hypot(float(flow_x_norm_s), float(flow_y_norm_s))
        if flow_valid else float("inf")
    )

    (
        _trim_mean_x,
        _trim_mean_y,
        trim_mean_radius,
        trim_residual_radius,
    ) = routine._center_trim_snapshot
    (
        _physical_mean_x,
        _physical_mean_y,
        physical_mean_radius,
        _physical_residual_radius,
    ) = routine._center_physical_trim_snapshot

    if routine._enable_center_condition_gate:
        settled_and_centered = (
            bool(target_found)
            and bool(flow_valid)
            and trim_mean_radius <= routine._center_trim_mean_radius_max
            and trim_residual_radius <= routine._center_trim_residual_radius_max
            and flow_radius <= routine._center_flow_radius_max
            and math.isfinite(physical_mean_radius)
            and physical_mean_radius <= routine._center_offset_radius_max
        )
        required_dwell = routine._center_condition_dwell
    else:
        # Legacy gate remains available for A/B work.  Even here, require the
        # geometric physical-centre criterion so enabling the adaptive finder
        # cannot hand off at a merely wind-biased raw image equilibrium.
        settled_and_centered = (
            routine._is_centered(offset_x, offset_y, target_found)
            and math.isfinite(physical_mean_radius)
            and physical_mean_radius <= routine._center_offset_radius_max
        )
        required_dwell = routine._center_dwell

    if settled_and_centered:
        if routine._centered_since is None:
            routine._centered_since = t
        dwell = t - routine._centered_since
    else:
        routine._centered_since = None
        dwell = 0.0

    elapsed = t - routine._center_start_t
    timed_out = routine._center_timeout > 0.0 and elapsed >= routine._center_timeout
    timeout_handoff = timed_out and routine._center_timeout_allows_handoff
    handoff_ready = settled_and_centered and dwell >= required_dwell

    # Operational gate. A vehicle that has not centred within the timeout will
    # not produce a feasibility verdict, so the run is over: ABORTED, not a
    # refusal. Without this the phase hovers on a valid target indefinitely,
    # which is what an unattended campaign cannot afford.
    if timed_out and not timeout_handoff and routine._center_timeout_aborts:
        return routine.abort(
            inputs,
            f"CENTER did not converge within {routine._center_timeout:.1f} s "
            f"(physical centring radius {physical_mean_radius:.3f} > "
            f"{routine._center_offset_radius_max:.3f})",
        )

    if handoff_ready or timeout_handoff:
        for probe in (routine._probe, routine._roll_probe, routine._pitch_probe):
            probe.reset()
            probe.retune(
                highpass_tau_sec=routine._far_probe_highpass_tau,
                percentile_window_sec=routine._far_probe_window,
                peak_decay_tau_sec=routine._far_probe_decay_tau,
            )
        routine.probe_result = ProbeResult()
        routine.roll_probe_result = ProbeResult()
        routine.pitch_probe_result = ProbeResult()
        routine.peak_accel_at_handoff = None
        routine.roll_peak_accel_at_handoff = None
        routine.pitch_peak_accel_at_handoff = None
        routine._t_approach_entry = t
        routine._approach_hold_since = None
        routine._approach_divergence_integral = 0.0
        routine._substate = APPROACH_PROBE

        info = _diagnostics(
            routine,
            offset_radius=offset_radius,
            flow_radius=flow_radius,
            dwell=dwell,
            required_dwell=required_dwell,
            elapsed=elapsed,
        )
        info.update({
            "event": "center_done",
            "centered_ok": bool(handoff_ready),
            "center_timed_out": bool(timed_out),
            "center_timeout_handoff": bool(timeout_handoff),
        })
        return MissionControl(
            divergence_setpoint=0.0,
            thrust_gain_override=routine._k_explore,
            lateral_p_scale=routine._center_lateral_p_scale,
            lateral_d_scale=routine._center_lateral_d_scale,
            roll_offset_setpoint=roll_offset_setpoint,
            pitch_offset_setpoint=pitch_offset_setpoint,
            # This control already belongs to APPROACH_PROBE -- it is the first
            # tick of that phase, merely constructed here -- so it drops the
            # large-offset blend along with the rest of APPROACH. Setting it
            # here rather than in approach_probe.py is what keeps the blend
            # from surviving one extra tick past the handoff.
            apply_offset_gain_blend=False,
            roll_accel_feedforward_m_s2=roll_accel_feedforward,
            pitch_accel_feedforward_m_s2=pitch_accel_feedforward,
            enable_integral=True,
            substate=APPROACH_PROBE,
            effects=(ControlEffect.RESET_DIVERGENCE_INTEGRAL,),
            info=info,
        )

    info = _diagnostics(
        routine,
        offset_radius=offset_radius,
        flow_radius=flow_radius,
        dwell=dwell,
        required_dwell=required_dwell,
        elapsed=elapsed,
    )
    info.update({
        "offset_x": offset_x,
        "offset_y": offset_y,
        "target_found": target_found,
        "flow_valid": flow_valid,
        "centered": bool(settled_and_centered),
        "center_timed_out": bool(timed_out),
        "center_timeout_allows_handoff": bool(routine._center_timeout_allows_handoff),
        "center_trim_mean_radius_max": routine._center_trim_mean_radius_max,
        "center_trim_residual_radius_max": routine._center_trim_residual_radius_max,
        "center_offset_radius_max": routine._center_offset_radius_max,
        "center_flow_radius_max_norm_s": routine._center_flow_radius_max,
    })
    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=routine._k_explore,
        lateral_p_scale=routine._center_lateral_p_scale,
        lateral_d_scale=routine._center_lateral_d_scale,
        roll_offset_setpoint=roll_offset_setpoint,
        pitch_offset_setpoint=pitch_offset_setpoint,
        # CENTER is the ONLY phase that runs the large-offset gain blend. The
        # capture transient it guards -- a compound P+D request large enough to
        # reach the collapsing-gain part of the angle soft limit -- happens
        # here and nowhere else, because every later phase entered through the
        # CENTER gate and therefore starts already centred.
        #
        # Blend on the PHYSICAL centring error: geometric tilt only, with the
        # learned wind bias excluded. See MissionControl.
        apply_offset_gain_blend=True,
        roll_gain_blend_setpoint=routine._center_geometric_offset_x,
        pitch_gain_blend_setpoint=routine._center_geometric_offset_y,
        roll_accel_feedforward_m_s2=roll_accel_feedforward,
        pitch_accel_feedforward_m_s2=pitch_accel_feedforward,
        enable_integral=True,
        substate=CENTER,
        info=info,
    )


SPEC = PhaseSpec(
    name=CENTER,
    display_name="CENTER",
    description="Adaptive physical centring under steady wind.",
    terminal=False,
    handler=run,
)