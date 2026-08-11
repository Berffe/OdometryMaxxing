"""The descent gain schedule and its safety predicates.

``scheduled_gain_at_time`` is the k(t) trajectory flown through DESCEND;
``critical_time`` and ``predicted_height`` are the look-ahead used to decide
whether that trajectory stays inside the window the gates opened.
"""
from __future__ import annotations

import math




def commanded_divergence_integral(
    elapsed_sec: float,
    divergence_setpoint: float,
    ramp_in_sec: float = 0.0,
) -> float:
    """Integral of a raised-cosine ramp from 0 to D* followed by constant D*."""
    t = max(0.0, float(elapsed_sec))
    d = max(0.0, float(divergence_setpoint))
    T = max(0.0, float(ramp_in_sec))

    if d <= 1e-12 or t <= 0.0:
        return 0.0

    if T <= 1e-9:
        return d * t

    if t < T:
        return d * (0.5 * t - 0.5 * T / math.pi * math.sin(math.pi * t / T))

    return d * (t - 0.5 * T)


def scheduled_gain_at_time(
    elapsed_sec: float,
    descent_divergence_setpoint: float,
    k_floor: float,
    k_explore: float,
    d_star_ramp_in_sec: float = 0.0,
) -> float:
    """K(t) = clamp(k_explore * exp(-integral(D*_cmd dt)), k_floor, k_explore).

    The TRAJECTORY is unchanged from before (a conservative exponential that, by
    design, decays faster than the height does under the same commanded D*). What
    changed is the ASYMPTOTE: k_floor is now GateResult.k_floor = max(k_min,
    ceiling_margin * k_ceiling_leg), so the descent settles just under the de
    Croon ceiling at leg height instead of sinking all the way to the Herisse
    floor k_min. See the module docstring.

    Depends only on elapsed time and the commanded D* -- NOT on any height
    estimate. h_pred/h0 stay diagnostic-only.

    The ceiling margin enters through k_floor, which compute_gate() derives from
    the known landing-gear height and the stability period.
    """
    exponent = commanded_divergence_integral(
        elapsed_sec, descent_divergence_setpoint, d_star_ramp_in_sec
    )
    decay = math.exp(-exponent)
    return max(float(k_floor), min(float(k_explore), float(k_explore) * decay))


def critical_time(
    h0: float,
    descent_divergence_setpoint: float,
    h_crit: float,
    d_star_ramp_in_sec: float = 0.0,
) -> float:
    """Predicted time for h(t)=h_crit under the same ramp-aware D* schedule."""
    d = float(descent_divergence_setpoint)
    h0 = float(h0)
    h_crit = float(h_crit)

    if h_crit <= 0.0 or h_crit >= h0 or d <= 1e-9:
        return float("inf")

    target_integral = math.log(h0 / h_crit)
    T = max(0.0, float(d_star_ramp_in_sec))

    if T <= 1e-9:
        return target_integral / d

    ramp_integral = commanded_divergence_integral(T, d, T)
    if target_integral >= ramp_integral:
        return target_integral / d + 0.5 * T

    lo, hi = 0.0, T
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if commanded_divergence_integral(mid, d, T) < target_integral:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def predicted_height(
    h0: float,
    descent_divergence_setpoint: float,
    elapsed_sec: float,
    d_star_ramp_in_sec: float = 0.0,
) -> float:
    exponent = commanded_divergence_integral(
        elapsed_sec, descent_divergence_setpoint, d_star_ramp_in_sec
    )
    return max(0.0, float(h0)) * math.exp(-exponent)

