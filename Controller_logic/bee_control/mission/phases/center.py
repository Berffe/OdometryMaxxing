"""CENTER phase.

Visual hover until the target is centred and laterally settled.

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

import math

from ..probe import ProbeResult
from ..types import APPROACH_PROBE, CENTER, ControlEffect, MissionControl, PhaseSpec


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    """Hold visual hover until the target is centred AND laterally settled.

        This is mission-phase logic, not command formation. ControlLaw continues
        receiving the real target/flow throughout CENTER; this method only decides
        when the mission is allowed to enter APPROACH_PROBE.
    """
    t = inputs.t
    offset_x = inputs.offset_x
    offset_y = inputs.offset_y
    target_found = inputs.target_found
    flow_x_norm_s = inputs.flow_x_norm_s
    flow_y_norm_s = inputs.flow_y_norm_s
    flow_valid = inputs.flow_valid

    # Height-free visual mismatch is diagnostic in CENTER as well. The robust
    # FINAL_PROBE decision envelope is not active yet.
    routine._update_visual_mismatch(inputs)

    if routine._center_start_t is None:
        routine._center_start_t = t

    offset_radius = math.hypot(float(offset_x), float(offset_y))
    flow_radius = (
        math.hypot(float(flow_x_norm_s), float(flow_y_norm_s))
        if flow_valid else float("inf")
    )

    if routine._enable_center_condition_gate:
        centered = (
            bool(target_found)
            and bool(flow_valid)
            and offset_radius <= routine._center_offset_radius_max
            and flow_radius <= routine._center_flow_radius_max
        )
        required_dwell = routine._center_condition_dwell
    else:
        centered = routine._is_centered(offset_x, offset_y, target_found)
        required_dwell = routine._center_dwell

    if centered:
        if routine._centered_since is None:
            routine._centered_since = t
        dwell = t - routine._centered_since
    else:
        routine._centered_since = None
        dwell = 0.0

    elapsed = t - routine._center_start_t
    settled = centered and dwell >= required_dwell
    timed_out = (
        routine._center_timeout > 0.0
        and elapsed >= routine._center_timeout
    )
    timeout_handoff = (
        timed_out and routine._center_timeout_allows_handoff
    )

    if settled or timeout_handoff:
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
        routine._substate = APPROACH_PROBE

        return MissionControl(
            divergence_setpoint=0.0,
            thrust_gain_override=routine._k_explore,
            lateral_p_scale=routine._center_lateral_p_scale,
            lateral_d_scale=routine._center_lateral_d_scale,
            enable_integral=True,
            substate=APPROACH_PROBE,
            # APPROACH_PROBE must not inherit vertical bias accumulated while
            # CENTER was fighting a moving target. Declared here rather than
            # recognised from the event string by bee_node.
            effects=(ControlEffect.RESET_DIVERGENCE_INTEGRAL,),
            info={
                "event": "center_done",
                "centered_ok": bool(settled),
                "center_timed_out": bool(timed_out),
                "center_timeout_handoff": bool(timeout_handoff),
                "center_elapsed_sec": elapsed,
                "center_dwell_sec": dwell,
                "center_offset_radius": offset_radius,
                "center_flow_radius_norm_s": flow_radius,
            },
        )

    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=routine._k_explore,
        lateral_p_scale=routine._center_lateral_p_scale,
        lateral_d_scale=routine._center_lateral_d_scale,
        enable_integral=True,
        substate=CENTER,
        info={
            "offset_x": offset_x,
            "offset_y": offset_y,
            "target_found": target_found,
            "flow_valid": flow_valid,
            "centered": centered,
            "center_dwell_sec": dwell,
            "center_required_dwell_sec": required_dwell,
            "center_elapsed_sec": elapsed,
            "center_timed_out": bool(timed_out),
            "center_timeout_allows_handoff": bool(
                routine._center_timeout_allows_handoff
            ),
            "center_offset_radius": offset_radius,
            "center_flow_radius_norm_s": flow_radius,
            "center_offset_radius_max": routine._center_offset_radius_max,
            "center_flow_radius_max_norm_s": routine._center_flow_radius_max,
        },
    )


SPEC = PhaseSpec(
    name=CENTER,
    display_name="CENTER",
    description="Visual hover until the target is centred and laterally settled.",
    terminal=False,
    handler=run,
)
