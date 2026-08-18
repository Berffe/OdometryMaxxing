"""Slow adaptive visual-centre finder used during CENTER.

Why this state exists
---------------------
A pure lateral PD loop rejects a steady wind by holding a non-zero visual
position error.  Geometric tilt compensation alone therefore cannot make
physical centring an equilibrium: if the platform is directly below the camera,
the geometric error is zero and the proportional counter-wind force disappears.

The adaptive centre finder moves the *visual reference* slowly enough that the
steady P error is retained while the physical centring error tends to zero.
For one lateral axis,

    e_phys = e_meas - e_geom
    e_sp   = e_geom - b
    b_dot  = w(flow) * e_phys / tau

so at physical centring (e_phys -> 0) the bias ``b`` freezes at the value needed
to preserve the counter-wind P error.  This is intentionally a slow outer loop;
the normal P/D controller remains the fast lateral controller.

The estimator is active only when its caller asks it to update. MissionRoutine
keeps it online through CENTER and APPROACH_PROBE, then stops adapting exactly
when FINAL_PROBE removes lateral image-position P authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .math_utils import clamp


@dataclass(frozen=True)
class VisualCenterAdaptationSnapshot:
    """Immutable diagnostic view of the current adaptive-centre state."""

    bias_x: float
    bias_y: float
    bias_radius: float
    rate_x_norm_s: float
    rate_y_norm_s: float
    adaptation_weight: float
    active: bool


class VisualCenterAdaptation:
    """Bounded, motion-aware integrator of physical visual centring error.

    Parameters
    ----------
    tau_sec:
        Nominal adaptation time constant.  It must be slower than the lateral
        P/D dynamics so the moving reference cannot chase normal oscillations.
    max_bias_norm:
        Per-axis hard bound on the learned visual-reference bias.
    flow_scale_norm_s:
        Optical-flow radius at which adaptation weight falls to 1/2.  The
        smooth weight ``1 / (1 + (flow/scale)^2)`` avoids a hard on/off switch.
    max_rate_norm_s:
        Per-axis slew bound on the learned bias.  This protects against a
        temporary detection error even when the physical-error mean is large.
    """

    def __init__(
        self,
        *,
        tau_sec: float,
        max_bias_norm: float,
        flow_scale_norm_s: float,
        max_rate_norm_s: float,
    ) -> None:
        self._tau = max(1e-3, float(tau_sec))
        self._max_bias = abs(float(max_bias_norm))
        self._flow_scale = max(1e-6, float(flow_scale_norm_s))
        self._max_rate = abs(float(max_rate_norm_s))
        self.reset()

    def reset(self) -> None:
        self._bias_x = 0.0
        self._bias_y = 0.0
        self._rate_x = 0.0
        self._rate_y = 0.0
        self._weight = 0.0
        self._active = False

    def update(
        self,
        *,
        physical_error_x: float,
        physical_error_y: float,
        flow_x_norm_s: float,
        flow_y_norm_s: float,
        valid: bool,
        dt: float,
    ) -> None:
        """Advance the adaptive visual centre by one far-field tick.

        ``physical_error`` is already geometric-tilt compensated.  Invalid
        target/flow data freezes the state.  Large flow smoothly slows the
        adaptation instead of changing the lateral reference aggressively while
        the vehicle is in a transient.
        """
        self._rate_x = 0.0
        self._rate_y = 0.0
        self._weight = 0.0
        self._active = False
        if not bool(valid):
            return

        ex = float(physical_error_x)
        ey = float(physical_error_y)
        fx = float(flow_x_norm_s)
        fy = float(flow_y_norm_s)
        if not all(math.isfinite(v) for v in (ex, ey, fx, fy)):
            return

        flow_radius = math.hypot(fx, fy)
        ratio = flow_radius / self._flow_scale
        weight = 1.0 / (1.0 + ratio * ratio)

        rate_x = weight * ex / self._tau
        rate_y = weight * ey / self._tau
        if self._max_rate > 0.0:
            rate_x = clamp(rate_x, -self._max_rate, self._max_rate)
            rate_y = clamp(rate_y, -self._max_rate, self._max_rate)

        step = max(1e-3, float(dt))
        self._bias_x = clamp(
            self._bias_x + rate_x * step, -self._max_bias, self._max_bias
        )
        self._bias_y = clamp(
            self._bias_y + rate_y * step, -self._max_bias, self._max_bias
        )
        self._rate_x = rate_x
        self._rate_y = rate_y
        self._weight = weight
        self._active = True

    @property
    def bias_x(self) -> float:
        return float(self._bias_x)

    @property
    def bias_y(self) -> float:
        return float(self._bias_y)

    @property
    def bias_radius(self) -> float:
        return math.hypot(self._bias_x, self._bias_y)

    def snapshot(self) -> VisualCenterAdaptationSnapshot:
        return VisualCenterAdaptationSnapshot(
            bias_x=self.bias_x,
            bias_y=self.bias_y,
            bias_radius=self.bias_radius,
            rate_x_norm_s=float(self._rate_x),
            rate_y_norm_s=float(self._rate_y),
            adaptation_weight=float(self._weight),
            active=bool(self._active),
        )
