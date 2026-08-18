"""Command-acceleration probe: de-biasing, rolling percentile, leaky peak.

Three instances run in parallel (vertical, roll, pitch). APPROACH_PROBE uses
them for diagnostics; FINAL_PROBE resets them and is the only phase whose
envelopes feed the feasibility gates. They measure the THRUST-COMMAND RESIDUAL,
never a physical acceleration.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .math_utils import G_ACCEL, clamp


class ThrustModel:
    """a_drone = g * (u / u_hover - 1), world-up, tilt ignored."""

    def __init__(self, hover_thrust: float, g: float = G_ACCEL):
        self._u_hover = max(1e-3, float(hover_thrust))
        self._g = float(g)

    def accel_from_thrust(self, u: float) -> float:
        return self._g * (float(u) / self._u_hover - 1.0)


@dataclass
class ProbeResult:
    peak_accel: float = 0.0
    n_samples: int = 0
    duration_sec: float = 0.0
    total_duration_sec: float = 0.0  # diagnostic elapsed time since the latest reset
    ready: bool = False


class PlatformProbe:
    """Estimate a robust acceleration envelope from a command history.

    The EMA removes the slow command bias, the rolling percentile rejects isolated
    outliers, and the leaky maximum retains the largest recent excursion. The same
    object can consume normalized thrust through update() or an acceleration
    directly through update_accel().
    """

    def __init__(
        self,
        thrust_model: ThrustModel,
        highpass_tau_sec: float = 7.0,
        percentile_window_sec: float = 2.0,
        peak_decay_tau_sec: float = 3.0,
        peak_percentile: float = 0.95,
        accel_margin: float = 0.05,
        probe_attenuation_comp: float = 1.2,
    ):
        self._tm = thrust_model
        self._tau = max(1e-3, float(highpass_tau_sec))

        self._mean = 0.0
        self._has_mean = False
        self._peak = 0.0
        self._n = 0
        self._elapsed = 0.0        # since the last retune() (the CURRENT phase)
        self._total_elapsed = 0.0  # since reset() (across the retune)

        self._percentile_window_sec = max(1e-3, float(percentile_window_sec))
        self._peak_percentile = clamp(peak_percentile, 0.0, 1.0)
        self._accel_margin = max(0.0, float(accel_margin))
        self._probe_attenuation_comp = max(1.0, float(probe_attenuation_comp))
        self._peak_decay_tau = max(1e-3, float(peak_decay_tau_sec))
        self._residual_window: Deque[tuple[float, float]] = deque()

        # Per-step values from the LAST update(), exposed for logging. They are what
        # the peak envelope is actually built from, so plotting them against
        # peak_accel shows whether the envelope is tracking real excursions or
        # coasting on a stale peak.
        self._last_accel = 0.0       # a = accel_from_thrust(thrust_cmd)
        self._last_residual = 0.0    # |a - mean|, the de-biased signal
        self._last_percentile = 0.0  # the window percentile the peak chases

    def reset(self) -> None:
        self._mean = 0.0
        self._has_mean = False
        self._peak = 0.0
        self._n = 0
        self._elapsed = 0.0
        self._total_elapsed = 0.0
        self._residual_window.clear()
        self._last_accel = 0.0
        self._last_residual = 0.0
        self._last_percentile = 0.0

    def retune(
        self,
        highpass_tau_sec: Optional[float] = None,
        percentile_window_sec: Optional[float] = None,
        peak_decay_tau_sec: Optional[float] = None,
    ) -> None:
        """Apply new time constants while preserving the mean and peak.

        The phase clock and rolling window restart, while the accumulated envelope
        carries into the near-field probe.
        """
        if highpass_tau_sec is not None:
            self._tau = max(1e-3, float(highpass_tau_sec))
        if percentile_window_sec is not None:
            self._percentile_window_sec = max(1e-3, float(percentile_window_sec))
        if peak_decay_tau_sec is not None:
            self._peak_decay_tau = max(1e-3, float(peak_decay_tau_sec))

        # Phase clock restarts (so `ready` can require a minimum NEAR-field hold),
        # but the total clock and the peak/mean state carry through.
        self._elapsed = 0.0
        self._residual_window.clear()

    @property
    def peak_accel(self) -> float:
        """The leaky-max envelope -- the number the gate consumes."""
        return float(self._peak)

    @property
    def accel(self) -> float:
        """Instantaneous commanded acceleration used by the last probe update."""
        return float(self._last_accel)

    @property
    def mean_accel(self) -> float:
        """The EMA bias being subtracted (hover trim + any slow descent term)."""
        return float(self._mean)

    @property
    def residual_accel(self) -> float:
        """|a - mean|: the de-biased per-step signal the envelope is built from."""
        return float(self._last_residual)

    @property
    def percentile_accel(self) -> float:
        """The rolling-window percentile the peak chases each step."""
        return float(self._last_percentile)

    @property
    def peak_decay_tau_sec(self) -> float:
        return float(self._peak_decay_tau)

    def update(self, thrust_cmd: float, dt: float) -> None:
        """Update from normalized collective thrust (vertical probe compatibility)."""
        self.update_accel(self._tm.accel_from_thrust(thrust_cmd), dt)

    def update_accel(self, accel_cmd: float, dt: float) -> None:
        """Update directly from a commanded acceleration in m/s^2."""
        dt = max(1e-3, float(dt))
        a = float(accel_cmd)

        if not self._has_mean:
            self._mean = a
            self._has_mean = True
        else:
            alpha = math.exp(-dt / self._tau)
            self._mean = alpha * self._mean + (1.0 - alpha) * a

        residual = abs(a - self._mean)
        self._last_accel = a
        self._last_residual = residual

        self._n += 1
        self._elapsed += dt
        self._total_elapsed += dt
        self._residual_window.append((self._elapsed, residual))

        window_sec = max(1e-3, self._percentile_window_sec)
        while (
            self._residual_window
            and self._elapsed - self._residual_window[0][0] > window_sec
        ):
            self._residual_window.popleft()

        values = sorted(v for _, v in self._residual_window)
        if not values:
            target_peak = residual
        elif len(values) == 1:
            target_peak = values[0]
        else:
            q = clamp(self._peak_percentile, 0.0, 1.0)
            idx = q * (len(values) - 1)
            lo = int(math.floor(idx))
            hi = int(math.ceil(idx))
            if lo == hi:
                target_peak = values[lo]
            else:
                w = idx - lo
                target_peak = (1.0 - w) * values[lo] + w * values[hi]

        # Raw percentile stays the diagnostic ("what the probe actually saw").
        self._last_percentile = float(target_peak)

        # Two margins, two shapes: factor inverts the dB attenuation (scales with the
        # value); additive covers unmodeled/ground-effect terms AND guarantees a
        # nonzero floor when the percentile is small, where a bare factor collapses.
        protected_peak = target_peak * self._probe_attenuation_comp + self._accel_margin

        # Leaky max, unchanged. Note the additive term now also acts as a hard lower
        # bound: _peak can never leak below accel_margin.
        decay = math.exp(-dt / max(1e-3, self._peak_decay_tau))
        self._peak = max(protected_peak, decay * self._peak)

    def result(self, min_duration_sec: float) -> ProbeResult:
        """Return the current phase-local envelope and readiness."""
        return ProbeResult(
            peak_accel=float(self._peak),
            n_samples=int(self._n),
            duration_sec=float(self._elapsed),
            total_duration_sec=float(self._total_elapsed),
            ready=self._elapsed >= float(min_duration_sec),
        )

