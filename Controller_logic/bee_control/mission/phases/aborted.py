"""ABORTED phase.

Terminal. Latched by the node on an outer-loop failure.

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

from ..types import ABORTED, MissionControl, PhaseSpec


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    """Terminal hold after an outer-loop abort.

        Like LANDED, this emits no gain and no divergence setpoint:
        ``thrust_gain_override`` is None ("no opinion"), because by this point the
        node is publishing its own neutral hold and any number here would be a
        fiction that only ever shows up in the log.

        This method is the worked example of what adding a phase now costs: one
        handler, one ``PhaseSpec`` in :attr:`PHASES`, and -- because the substate
        string already flows through ``telemetry()`` -- nothing at all in
        ``bee_node`` or ``diagnostics_writer``.
    """
    t = inputs.t

    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=None,
        lateral_p_scale=0.0,
        lateral_d_scale=0.0,
        enable_integral=False,
        substate=ABORTED,
        info={
            "event": "aborted" if just_entered else "",
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
    description="Terminal. Latched by the node on an outer-loop failure.",
    terminal=True,
    handler=run,
)
