"""The visual landing sequence.

    CENTER -> APPROACH_PROBE -> FINAL_PROBE -> DESCEND

``routine``    MissionRoutine: config, shared state, dispatch, telemetry
``phases/``    one file per phase, plus the registry
``probe``      the command-acceleration probe (three instances run in parallel)
``gates``      feasibility maths
``schedule``   the k(t) descent trajectory and its look-ahead predicates
``math_utils`` clamp, raised cosine, G
``types``      the contract with the caller: MissionInputs, MissionControl, effects

Only ``routine`` and ``types`` are meant to be imported from outside this
package. ``bee_node`` uses exactly two names: ``MissionRoutine`` and the types
it needs to build one tick's input.
"""
from .routine import MissionRoutine
from .types import ActuationFeedback, ControlEffect, MissionControl, MissionInputs

__all__ = [
    "MissionRoutine",
    "MissionInputs",
    "MissionControl",
    "ActuationFeedback",
    "ControlEffect",
]
