"""LANDED phase.

Terminal. Latched by the node on confirmed truth contact.

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

from ..types import LANDED, MissionControl, PhaseSpec


def run(routine, inputs, *, just_entered: bool = False) -> MissionControl:
    """Terminal hold after touchdown. Emits no gain and no divergence setpoint.

        thrust_gain_override is None (not k_probe, not 0.0) on purpose: "no opinion".
        bee_node is publishing its own zero-thrust landed setpoint by now, so any
        number we put here would be a fiction that only shows up in the log and in
        the gain-schedule plot -- exactly the kind of phantom that made the last
        run's descent statistics unreadable.
    """
    t = inputs.t

    return MissionControl(
        divergence_setpoint=0.0,
        thrust_gain_override=None,
        lateral_p_scale=0.0,
        lateral_d_scale=0.0,
        enable_integral=False,
        substate=LANDED,
        info={
            "event": "landed",
            "landed_since_sec": (t - routine._t_landed) if routine._t_landed is not None else 0.0,
            "peak_accel": routine.probe_result.peak_accel,
            "roll_peak_accel": routine.roll_probe_result.peak_accel,
            "pitch_peak_accel": routine.pitch_probe_result.peak_accel,
            "k_min": routine.gate.k_min,
            "k_floor": routine.gate.k_floor,
            "feasible": routine.feasible,
        },
    )


SPEC = PhaseSpec(
    name=LANDED,
    display_name="LANDED",
    description="Terminal. Latched by the node on confirmed truth contact.",
    terminal=True,
    handler=run,
)
