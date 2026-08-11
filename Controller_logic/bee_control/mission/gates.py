"""Feasibility gates: does a usable gain window exist at landing-gear height?

Vertical feasibility compares the Herisse disturbance-rejection floor with the
safety-scaled de Croon ceiling. Roll and pitch use the acceleration-domain
lateral bound

    K_min = c_max / kappa + peak_accel / omega_adm

No online height estimate is used anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional




@dataclass
class GateResult:
    k_min: float = 0.0          # Herisse floor: peak_accel / D*
    h_crit: float = 0.0         # height at which the safety-scaled ceiling == k_min
    k_explore: float = 0.0      # hand-tuned exploration gain (schedule's start value)
    feasible: bool = False      # landing window exists and FINAL_PROBE gain reaches k_min

    # --- Ceiling-riding descent target (see module docstring). ---
    k_ceiling_leg: float = 0.0    # de Croon safety-scaled ceiling AT LEG HEIGHT
    k_target: float = 0.0         # ceiling_margin * k_ceiling_leg
    k_floor: float = 0.0          # max(k_min, k_target) -- what the schedule decays TO
    ceiling_margin: float = 0.0
    k_descend_start: float = 0.0  # gain at DESCEND entry = k_probe, NOT k_explore.
                                  # The descent continues down from where FINAL_PROBE
                                  # left the gain; it never jumps back up to the
                                  # far-field value.
    accel_capacity_floor: float = 0.0
    accel_capacity_ceiling: float = 0.0
    window_exists: bool = False
    start_above_floor: bool = False


@dataclass
class LateralGateResult:
    peak_accel: float = 0.0
    k_min: float = 0.0
    k_probe: float = 0.0
    k_touchdown: float = 0.0
    k_target: float = 0.0
    k_floor: float = 0.0
    k_descend_start: float = 0.0
    k_ceiling_probe: float = 0.0
    k_ceiling_leg: float = 0.0
    ceiling_margin: float = 0.0
    accel_capacity_floor: float = 0.0
    accel_capacity_ceiling: float = 0.0
    probe_within_ceiling: bool = False
    window_exists: bool = False
    start_above_floor: bool = False
    floor_within_ceiling: bool = False
    feasible: bool = False


def lateral_ceiling_gain_at_height(
    height_m: float,
    control_period_sec: float,
    kappa: float,
    max_closing_speed_m_s: float,
    safety: float = 1.0,
) -> float:
    """Safety-scaled discrete lateral ceiling with bounded closing speed."""
    s = max(1e-3, float(safety))
    dt = max(1e-6, float(control_period_sec))
    kappa = max(1e-6, float(kappa))
    c_max = max(0.0, float(max_closing_speed_m_s))
    return max(0.0, 2.0 * s * max(0.0, float(height_m)) / (kappa * dt) - c_max / kappa)


def lateral_accel_capacity(
    gain: float,
    flow_admissible_norm_s: float,
    max_closing_speed_m_s: float,
    kappa: float,
) -> float:
    """Maximum lateral platform acceleration rejected by ``gain``."""
    omega_adm = max(1e-6, float(flow_admissible_norm_s))
    kappa = max(1e-6, float(kappa))
    closing_term = max(0.0, float(max_closing_speed_m_s)) / kappa
    return omega_adm * max(0.0, float(gain) - closing_term)


def compute_lateral_gate(
    peak_accel: float,
    flow_admissible_norm_s: float,
    max_closing_speed_m_s: float,
    kappa: float,
    probe_gain: float,
    near_field_height_m: float,
    leg_clearance_m: float,
    control_period_sec: float,
    ceiling_safety_factor: float = 0.5,
    ceiling_margin: float = 0.8,
) -> LateralGateResult:
    """Build an independent lateral gain floor and verify its gain window."""
    kappa = max(1e-6, float(kappa))
    omega_adm = max(1e-6, float(flow_admissible_norm_s))
    c_max = max(0.0, float(max_closing_speed_m_s))
    margin = max(0.0, float(ceiling_margin))

    k_min = c_max / kappa + max(0.0, float(peak_accel)) / omega_adm
    k_probe = max(0.0, float(probe_gain))
    k_ceiling_probe = lateral_ceiling_gain_at_height(
        near_field_height_m, control_period_sec, kappa, c_max,
        ceiling_safety_factor,
    )
    k_ceiling_leg = lateral_ceiling_gain_at_height(
        leg_clearance_m, control_period_sec, kappa, c_max,
        ceiling_safety_factor,
    )

    k_target = margin * k_ceiling_leg
    k_floor = max(k_min, k_target)
    # A scheduled floor may never exceed the gain at descent entry: the
    # trajectory is a monotone decay from the FINAL_PROBE gain.
    k_floor = min(k_floor, k_probe) if k_probe > 0.0 else k_floor
    k_touchdown = k_floor

    probe_within_ceiling = k_probe <= k_ceiling_probe
    window_exists = k_min <= k_ceiling_leg
    start_above_floor = k_probe >= k_min
    floor_within_ceiling = k_floor <= k_ceiling_leg
    feasible = (
        probe_within_ceiling
        and window_exists
        and start_above_floor
        and floor_within_ceiling
    )

    return LateralGateResult(
        peak_accel=max(0.0, float(peak_accel)),
        k_min=float(k_min),
        k_probe=float(k_probe),
        k_touchdown=float(k_touchdown),
        k_target=float(k_target),
        k_floor=float(k_floor),
        k_descend_start=float(k_probe),
        k_ceiling_probe=float(k_ceiling_probe),
        k_ceiling_leg=float(k_ceiling_leg),
        ceiling_margin=float(margin),
        accel_capacity_floor=lateral_accel_capacity(
            k_floor, omega_adm, c_max, kappa
        ),
        accel_capacity_ceiling=lateral_accel_capacity(
            k_ceiling_leg, omega_adm, c_max, kappa
        ),
        probe_within_ceiling=bool(probe_within_ceiling),
        window_exists=bool(window_exists),
        start_above_floor=bool(start_above_floor),
        floor_within_ceiling=bool(floor_within_ceiling),
        feasible=bool(feasible),
    )


def critical_height(k_min: float, control_period_sec: float, safety: float = 1.0) -> float:
    """Height where the safety-scaled de Croon ceiling reaches k_min."""
    s = max(1e-3, float(safety))
    return float(k_min) * float(control_period_sec) / (2.0 * s)


def ceiling_gain_at_height(
    height_m: float, control_period_sec: float, safety: float = 1.0
) -> float:
    """Safety-scaled de Croon stability ceiling on k, at a given height.

    k_ceiling(h) = 2*s*h/dt -- the exact inverse of critical_height(), which
    solves k_ceiling(h_crit) = k_min. Evaluated at leg_clearance_m this is the
    largest gain that is still admissible at touchdown, and (scaled by
    ceiling_margin) the value the descent schedule now decays toward instead of
    k_min.
    """
    s = max(1e-3, float(safety))
    dt = max(1e-6, float(control_period_sec))
    return 2.0 * s * max(0.0, float(height_m)) / dt


def compute_gate(
    peak_accel: float,
    descent_divergence_setpoint: float,
    initial_thrust_gain: float,
    control_period_sec: float,
    leg_clearance_m: float,
    ceiling_safety_factor: float = 0.5,
    min_divergence_setpoint: float = 0.01,
    ceiling_margin: float = 0.8,
    descend_start_gain: Optional[float] = None,
) -> GateResult:
    """Turn a probed peak_accel into the descent gain window.

    ceiling_margin: how close to the safety-scaled ceiling AT LEG HEIGHT the
        descent should settle (the Bode/bandwidth argument -- higher gain at
        touchdown means better synchronization with the platform). This is a
        SECOND multiplicative margin on top of ceiling_safety_factor; see the
        module docstring before reading 0.8 as "80% of the stability limit".

    The returned k_floor -- not k_min -- is what scheduled_gain_at_time() decays
    toward. k_min survives as a hard floor UNDER k_target so that a marginally-
    feasible mission (k_min close to k_ceiling_leg, where margin<1 would place
    k_target below the disturbance-rejection floor) safely falls back to the old
    conservative behavior instead of under-gaining.
    """
    d_star = max(float(min_divergence_setpoint), float(descent_divergence_setpoint))
    s = max(1e-3, float(ceiling_safety_factor))
    margin = max(0.0, float(ceiling_margin))

    k_min = max(0.0, float(peak_accel)) / d_star
    h_crit = critical_height(k_min, control_period_sec, s)
    k_explore = max(0.0, float(initial_thrust_gain))

    k_ceiling_leg = ceiling_gain_at_height(leg_clearance_m, control_period_sec, s)
    k_target = margin * k_ceiling_leg

    # The descent starts from wherever FINAL_PROBE left the gain (k_probe), not
    # from the far-field k_explore -- the gain has already been walked down the
    # ceiling during the approach and must not step back up.
    k_start = float(k_explore if descend_start_gain is None else descend_start_gain)
    window_exists = h_crit <= float(leg_clearance_m)
    start_above_floor = k_start >= k_min
    feasible = window_exists and start_above_floor

    # Never below the Herisse floor, and never above the gain the schedule starts
    # from (it only ever decays -- a k_floor above the start would turn the
    # "decay" into a step up, which is not what the trajectory means).
    k_floor = max(float(k_min), float(k_target))
    k_floor = min(k_floor, k_start) if k_start > 0.0 else k_floor

    return GateResult(
        k_min=float(k_min),
        h_crit=float(h_crit),
        k_explore=float(k_explore),
        feasible=bool(feasible),
        k_ceiling_leg=float(k_ceiling_leg),
        k_target=float(k_target),
        k_floor=float(k_floor),
        ceiling_margin=float(margin),
        k_descend_start=k_start,
        accel_capacity_floor=float(d_star * k_floor),
        accel_capacity_ceiling=float(d_star * k_ceiling_leg),
        window_exists=bool(window_exists),
        start_above_floor=bool(start_above_floor),
    )

