"""ABORTED phase.

Terminal. An operational failure stopped the flight before the feasibility
gates could return a verdict: loss of the target, a CENTER or APPROACH gate
that never converged, an offboard dropout, a MAVSDK failure.

This is deliberately NOT the phase for "the gates ran and refused". That is
INFEASIBLE, and keeping the two apart is what stops a run the gate never
evaluated from being counted as a rejection.

Phase contract
--------------
``run(routine, inputs, just_entered=False)`` returns the ``MissionControl`` for
one tick, and may latch ``routine._substate`` when it hands off. ``SPEC`` is
what the registry in ``mission/routine.py`` picks up.

Ending the run
--------------
This phase emits a ``TerminalRequest``; the node applies it. The phase owns the
decision and the reason, the node owns what ending a run means, and neither has
to know the other's vocabulary.
"""
from __future__ import annotations

from ..types import ABORTED, MissionControl, PhaseSpec, TerminalRequest


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    """Terminal hold after an operational failure.

    Emits no gain and no divergence setpoint: ``thrust_gain_override`` is None
    ("no opinion"), because by this point the node is publishing its own neutral
    hold and any number here would be a fiction that only ever shows up in the
    log.
    """
    t = inputs.t
    reason = routine.terminal_reason or "unspecified operational failure"

    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=None,
        lateral_p_scale=0.0,
        lateral_d_scale=0.0,
        enable_integral=False,
        substate=ABORTED,
        terminal_request=TerminalRequest(outcome=ABORTED, reason=reason),
        info={
            "event": "aborted" if just_entered else "",
            "reason": reason,
            "terminal_reason": reason,
            "abort_phase": routine.terminal_origin_substate,
            "aborted_since_sec": (
                (t - routine._t_aborted) if routine._t_aborted is not None else 0.0
            ),
            "peak_accel": routine.probe_result.peak_accel,
            "feasible": routine.feasible,
        },
    )


SPEC = PhaseSpec(
    name=ABORTED,
    display_name="ABORTED",
    description="Terminal. Operational failure before the gates returned a verdict.",
    terminal=True,
    handler=run,
)
