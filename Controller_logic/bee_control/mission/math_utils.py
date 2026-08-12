"""Small numeric helpers shared by the probe, the gates and the phases.

Dependency-free on purpose: everything else in ``mission/`` may import this,
and it may import nothing back.
"""
from __future__ import annotations

import math

#: Standard gravity. Converts normalized collective thrust to world-vertical
#: acceleration for the Herisse / de Croon accel-domain thrust law.
G_ACCEL = 9.80665


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def raised_cosine01(x: float) -> float:
    """Smooth 0->1 blend with zero slope at both ends."""
    x = clamp(x, 0.0, 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * x))


def blank(value):
    """None -> empty CSV cell.

    A blank is a gap in the plot, which is the truth when a quantity is not
    being measured. 0.0 would draw a flat line that looks like a measurement.
    """
    return "" if value is None else value
