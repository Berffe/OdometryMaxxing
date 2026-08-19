"""The visual landing sequence.

    CENTER -> APPROACH_PROBE -> FINAL_PROBE -> DESCEND

``routine``                   MissionRoutine: config, shared state, dispatch, telemetry
``phases/``                   one file per phase, plus the registry
``probe``                     the command-acceleration probe (three run in parallel)
``gates``                     feasibility maths
``schedule``                  the k(t) descent trajectory and its look-ahead predicates
``trim``                      the slow visual-trim (wind equilibrium) estimator
``visual_center_adaptation``  the far-field adaptive visual centre (wind rejection)
``visual_mismatch``           the chi bandwidth diagnostic behind the tracking gate
``math_utils``                clamp, raised cosine, G
``types``                     the contract with the caller: MissionInputs, MissionControl, effects

Wind rejection spans three of these -- ``visual_center_adaptation`` in the far
field, the roll/pitch ``probe`` means as the passive seed, and the static term
that ``routine`` adapts from FINAL_PROBE onward.  See ``docs/WIND_REJECTION.md``.

Only ``routine`` and ``types`` are meant to be imported from outside this
package. ``bee_node`` uses exactly two names: ``MissionRoutine`` and the types
it needs to build one tick's input.

Why these re-exports are LAZY
-----------------------------
``routine`` imports ``.phases``, and every phase imports ``..types``. Both are
submodule imports, which Python resolves against ``sys.modules`` even while
this package is still initialising -- so an EAGER ``from .routine import
MissionRoutine`` here happens to work today.

It works by luck, not by construction. The moment any submodule imports a NAME
re-exported here rather than a submodule -- ``from .. import MissionControl``
inside a phase, say -- that same cycle raises::

    ImportError: cannot import name 'MissionControl' from partially
    initialized module 'bee_control.mission' (most likely due to a
    circular import)

and it raises no matter which module the process imports first, so it cannot be
worked around by import ordering at the call site.

Resolving the re-exports on FIRST ATTRIBUTE ACCESS (PEP 562) removes the cycle
outright: this module now executes no imports at all, so there is no partially
initialised window for a submodule to observe. ``from bee_control.mission import
MissionRoutine`` behaves exactly as before.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import-time only for type checkers; never at runtime.
    from .phases import display_name
    from .routine import MissionRoutine
    from .types import (
        ActuationFeedback,
        ControlEffect,
        MissionControl,
        MissionInputs,
    )

__all__ = [
    "MissionRoutine",
    "MissionInputs",
    "MissionControl",
    "ActuationFeedback",
    "ControlEffect",
    "display_name",
]

#: Re-exported name -> submodule that defines it.
_EXPORTS = {
    "MissionRoutine": "routine",
    "MissionInputs": "types",
    "MissionControl": "types",
    "ActuationFeedback": "types",
    "ControlEffect": "types",
    "display_name": "phases",
}


def __getattr__(name: str):
    """Resolve a re-export on first access (PEP 562). See the module docstring."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # cache: __getattr__ is not called again.
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
