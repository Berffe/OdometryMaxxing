"""FINAL_PROBE phase.

Near-field hold at k_probe; probes retuned, envelopes carried over.

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

    if routine._t_final_probe_entry is None:
        routine._t_final_probe_entry = t

    entry_elapsed = t - routine._t_final_probe_entry

    # Smooth D* from approach descent back to zero before the near-field probe.
    if routine._final_probe_entry_ramp > 1e-9 and entry_elapsed < routine._final_probe_entry_ramp:
        frac = raised_cosine01(entry_elapsed / routine._final_probe_entry_ramp)
        d_cmd = routine._approach_d_star * (1.0 - frac)
        routine._update_visual_mismatch(inputs)
        routine._update_probes(
            last_thrust_cmd, last_vertical_accel_cmd,
            last_roll_accel_cmd, last_pitch_accel_cmd, dt,
        )
        routine._refresh_probe_results(0.0, routine._probe_min)

        return MissionControl(
            divergence_setpoint=d_cmd,
            thrust_gain_override=routine._compute_probe_gain(),
            lateral_p_scale=routine._probe_lateral_p_scale,
            lateral_d_scale=routine._probe_lateral_d_scale,
            enable_integral=True,
            substate=FINAL_PROBE,
            info={
                "event": "final_probe_entry_ramp",
                "k": routine._compute_probe_gain(),
                "final_probe_entry_elapsed_sec": entry_elapsed,
                "final_probe_entry_ramp_frac": frac,
                "probe_elapsed_sec": routine.probe_result.total_duration_sec,
            },
        )

    if routine._t_final_probe_hold_start is None:
        routine._t_final_probe_hold_start = t
        # Retune all probes together while preserving their far-field envelopes.
        routine.peak_accel_at_handoff = routine._probe.peak_accel
        routine.roll_peak_accel_at_handoff = routine._roll_probe.peak_accel
        routine.pitch_peak_accel_at_handoff = routine._pitch_probe.peak_accel
        routine._retune_probes()
        # FINAL_PROBE is the one operating point that decides bandwidth. Keep
        # the causal Ddot history warm, but start a fresh robust |chi| envelope
        # and observation clock so APPROACH/ramp transients cannot veto landing.
        routine._begin_tracking_gate_window()

    routine._update_visual_mismatch(inputs)
    routine._update_probes(
        last_thrust_cmd, last_vertical_accel_cmd,
        last_roll_accel_cmd, last_pitch_accel_cmd, dt,
    )
    # ready requires BOTH: a full near-field hold, AND enough TOTAL probing
    # across both phases (~ one platform period) -- the hold alone is far too
    # short to supply the latter, so this is what keeps a fast FOV saturation
    # from gating on a fraction of one platform cycle.
    routine._refresh_probe_results(routine._final_probe_duration, routine._probe_min)

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
        )
        routine._compute_lateral_gates()

        vertical_probe_ok = routine.vertical_feasible
        roll_probe_ok = routine.roll_feasible
        pitch_probe_ok = routine.pitch_feasible
        # Fourth, independent question: authority is not bandwidth. FINAL_PROBE
        # is stationary (D*=0, k=k_probe), so its robust height-free |chi|
        # envelope is the actual pre-commit synchronisation decision.
        routine._refresh_tracking_gate()
        tracking_ok = routine.tracking_feasible

        if routine._probe_only:
            routine._substate = PROBE_HOLD
            return probe_hold.run(routine, inputs, just_entered=True)
        if (
            vertical_probe_ok
            and roll_probe_ok
            and pitch_probe_ok
            and tracking_ok
            and routine._enable_descent
        ):
            routine._substate = DESCEND
            routine._t_descend_start = t
            return descend.run(routine, inputs, just_entered=True)

        routine._substate = INFEASIBLE
        return infeasible.run(routine, inputs, just_entered=True)

    # THE PROBE ITSELF. k is held FLAT at k_probe here -- deliberately, twice
    # over: (1) it is the admissible gain at this height (k_explore is above the
    # ceiling in the near field, and probing there would feed routine-induced
    # oscillation straight into peak_accel), and (2) a gain that MOVES during
    # the probe moves the closed-loop transfer function with it, so the residual
    # would mix platform motion with our own gain sweep. D*=0 in this phase, so
    # the exp(-integral D*) schedule holds k constant here of its own accord --
    # the flatness is structural, not a special case.
    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=routine._compute_probe_gain(),
        lateral_p_scale=routine._probe_lateral_p_scale,
        lateral_d_scale=routine._probe_lateral_d_scale,
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
            "probe_total_elapsed_sec": routine.probe_result.total_duration_sec,
            "hold_min_sec": routine._final_probe_duration,
            "probe_total_min_sec": routine._probe_min,
            "k_explore": routine._k_explore,
        },
    )


SPEC = PhaseSpec(
    name=FINAL_PROBE,
    display_name="FINAL_PROBE",
    description="Near-field hold at k_probe; probes retuned, envelopes carried over.",
    terminal=False,
    handler=run,
)