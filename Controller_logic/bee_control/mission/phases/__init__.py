"""Mission phase registry.

This package is the single place that knows the mission's phase set.  It
replaced both the ``if``-chain that used to open ``MissionRoutine.update()``
and the ``display`` dict that lived inside ``bee_node``.

Adding a phase
--------------
1. Write ``phases/<name>.py`` with a ``run(routine, inputs, *, just_entered)``
   function and a module-level ``SPEC = PhaseSpec(...)``.
2. Add the module to ``_PHASE_MODULES`` below.

That is the whole change.  ``MissionRoutine`` picks the new phase up through
``PHASES``, the console picks its ``display_name`` up through the same table,
and the CSV picks its substate string up through ``MissionRoutine.telemetry()``
-- so neither ``bee_node`` nor ``diagnostics_writer`` is touched.

Import order note
-----------------
``final_probe`` imports its three successor phases directly, because it hands
off to them mid-tick with ``just_entered=True``. Those successors import no
phases in turn, so the graph stays acyclic. If a future phase ever needs a
mutual handoff, resolve it through ``PHASES[...]`` at call time rather than at
import time.
"""
from __future__ import annotations

from . import (
    aborted,
    approach_probe,
    center,
    descend,
    final_probe,
    infeasible,
    landed,
    probe_hold,
)

#: Registration order. Roughly the flight order, then the terminal states --
#: it has no semantic effect, but keeping it readable makes the table itself
#: serve as the sequence documentation.
_PHASE_MODULES = (
    center,
    approach_probe,
    final_probe,
    probe_hold,
    descend,
    infeasible,
    landed,
    aborted,
)

PHASES = {module.SPEC.name: module.SPEC for module in _PHASE_MODULES}

#: Substates from which the mission never advances on its own.
#:
#: Named SUBSTATES, not PHASES, to keep it distinct from
#: ``flight_sequencer.TERMINAL_PHASES``, which is the same idea in the OUTER
#: controller vocabulary. The two state machines are deliberately separate and
#: their terminal sets are not interchangeable.
TERMINAL_SUBSTATES = tuple(
    spec.name for spec in PHASES.values() if spec.terminal
)


def display_name(substate: str) -> str:
    """Human-readable name for a substate; the registry is the only source."""
    spec = PHASES.get(substate)
    return spec.display_name if spec is not None else str(substate).upper()


__all__ = ["PHASES", "TERMINAL_SUBSTATES", "display_name"]
