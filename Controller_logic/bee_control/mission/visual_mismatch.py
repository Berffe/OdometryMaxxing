"""Visual tracking mismatch: chi(t) = Ddot(t) - D(t)^2.

What chi measures
-----------------
Let ``h`` be the camera-to-deck distance along the optical axis and ``D`` the
observed divergence, with the usual sign convention ``D = -hdot / h`` (positive
while closing). Differentiating,

    Ddot = -hddot/h + (hdot/h)^2 = -hddot/h + D^2

so

    chi = Ddot - D^2 = -hddot / h

which is the RELATIVE vertical acceleration between vehicle and deck, scaled by
range. No height, platform state, or divergence setpoint is needed: chi is built
entirely from the visual divergence history.

Why this is a bandwidth test
----------------------------
The acceleration probes ask whether the commanded controller has enough
AUTHORITY to reject the disturbance. chi asks a different question: after the
real vision/filter/control delays have acted, how much relative acceleration is
still left between the vehicle and the deck? A disturbance can fit inside the
authority envelope and still vary too quickly for the closed loop to follow.

Derivative estimator
--------------------
A two-sample backward difference is intentionally NOT used. Differentiation is
the noisiest part of the metric, so Ddot is the slope of a causal least-squares
line fitted to the recent filtered-divergence history. The sample abscissa is
built from the actual per-frame Gazebo-SIM ``dt`` values supplied by the mission;
irregular camera intervals therefore receive their true temporal spacing.

The regression window performs the smoothing itself, avoiding the extra phase
lag of differentiating first and then low-pass filtering the result.

Robust envelope
---------------
Unlike the command-acceleration probe, chi is NOT high-pass/de-biased. A
persistent non-zero chi is itself evidence of poor synchronisation and must not
be learned away as a bias. The gate therefore uses

    |chi| -> rolling percentile -> leaky maximum.

The derivative history and the robust envelope are separate. FINAL_PROBE can
restart only the decision envelope while keeping the derivative warm from the
preceding visual samples.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque

from .math_utils import clamp


@dataclass
class VisualMismatchResult:
    peak_chi: float = 0.0
    n_samples: int = 0
    duration_sec: float = 0.0
    ready: bool = False


class VisualMismatchProbe:
    """Causal robust envelope of ``chi = Ddot - D^2`` [1/s^2]."""

    def __init__(
        self,
        *,
        derivative_window_sec: float = 0.20,
        percentile_window_sec: float,
        peak_decay_tau_sec: float,
        peak_percentile: float = 0.95,
        margin: float = 0.0,
        attenuation_comp: float = 1.0,
        min_derivative_samples: int = 5,
    ):
        self._derivative_window_sec = max(1e-3, float(derivative_window_sec))
        self._percentile_window_sec = max(1e-3, float(percentile_window_sec))
        self._peak_decay_tau = max(1e-3, float(peak_decay_tau_sec))
        self._peak_percentile = clamp(peak_percentile, 0.0, 1.0)
        self._margin = max(0.0, float(margin))
        self._attenuation_comp = max(1.0, float(attenuation_comp))
        self._min_derivative_samples = max(3, int(min_derivative_samples))

        self._time = 0.0
        self._history: Deque[tuple[float, float]] = deque()
        self._magnitude_window: Deque[tuple[float, float]] = deque()

        self._divergence_rate = 0.0
        self._last_chi = 0.0
        self._last_abs_chi = 0.0
        self._last_percentile = 0.0
        self._peak = 0.0
        self._derivative_ready = False

        # Envelope clocks. reset_envelope() restarts these without touching the
        # derivative history, which is exactly what FINAL_PROBE needs.
        self._envelope_elapsed = 0.0
        self._envelope_samples = 0

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Reset derivative history and envelope."""
        self._time = 0.0
        self._history.clear()
        self._divergence_rate = 0.0
        self._last_chi = 0.0
        self._last_abs_chi = 0.0
        self._derivative_ready = False
        self.reset_envelope()

    def reset_envelope(self) -> None:
        """Restart only the robust decision envelope; keep Ddot warm."""
        self._magnitude_window.clear()
        self._last_percentile = 0.0
        self._peak = 0.0
        self._envelope_elapsed = 0.0
        self._envelope_samples = 0

    def retune(
        self,
        *,
        percentile_window_sec: float | None = None,
        peak_decay_tau_sec: float | None = None,
        derivative_window_sec: float | None = None,
    ) -> None:
        """Apply new time constants without resetting derivative or peak state."""
        if percentile_window_sec is not None:
            self._percentile_window_sec = max(1e-3, float(percentile_window_sec))
        if peak_decay_tau_sec is not None:
            self._peak_decay_tau = max(1e-3, float(peak_decay_tau_sec))
        if derivative_window_sec is not None:
            self._derivative_window_sec = max(1e-3, float(derivative_window_sec))
            self._trim_derivative_history()

    # ------------------------------------------------------------- properties
    @property
    def divergence_rate(self) -> float:
        """Causal regression estimate of dD/dt [1/s^2]."""
        return float(self._divergence_rate)

    @property
    def derivative_ready(self) -> bool:
        return bool(self._derivative_ready)

    @property
    def chi(self) -> float:
        return float(self._last_chi)

    @property
    def abs_chi(self) -> float:
        return float(self._last_abs_chi)

    @property
    def percentile_chi(self) -> float:
        return float(self._last_percentile)

    @property
    def peak_chi(self) -> float:
        return float(self._peak)

    @property
    def envelope_elapsed_sec(self) -> float:
        return float(self._envelope_elapsed)

    @property
    def peak_decay_tau_sec(self) -> float:
        return float(self._peak_decay_tau)

    # ---------------------------------------------------------------- helpers
    def _trim_derivative_history(self) -> None:
        window = self._derivative_window_sec
        while self._history and self._time - self._history[0][0] > window:
            self._history.popleft()

    def _fit_divergence_rate(self) -> tuple[float, bool]:
        """Least-squares slope over the causal (time, divergence) history."""
        if len(self._history) < self._min_derivative_samples:
            return self._divergence_rate, False

        t0 = self._history[0][0]
        times = [t - t0 for t, _ in self._history]
        values = [d for _, d in self._history]
        span = times[-1] - times[0]
        if span <= 1e-6:
            return self._divergence_rate, False

        mean_t = sum(times) / len(times)
        mean_d = sum(values) / len(values)
        denom = sum((t - mean_t) ** 2 for t in times)
        if denom <= 1e-12:
            return self._divergence_rate, False

        numer = sum(
            (t - mean_t) * (d - mean_d)
            for t, d in zip(times, values)
        )
        return numer / denom, True

    def _rolling_percentile(self, value: float) -> float:
        self._magnitude_window.append((self._envelope_elapsed, value))
        window = self._percentile_window_sec
        while (
            self._magnitude_window
            and self._envelope_elapsed - self._magnitude_window[0][0] > window
        ):
            self._magnitude_window.popleft()

        values = sorted(v for _, v in self._magnitude_window)
        if not values:
            return value
        if len(values) == 1:
            return values[0]

        idx = self._peak_percentile * (len(values) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return values[lo]
        weight = idx - lo
        return (1.0 - weight) * values[lo] + weight * values[hi]

    # ------------------------------------------------------------------ update
    def update(self, divergence: float, dt: float) -> None:
        """Fold one FILTERED visual-divergence sample into the estimator.

        ``dt`` is the elapsed Gazebo-SIM time since the previous fresh controlled
        visual result. The mission/node already derive it from consecutive camera
        source timestamps; this estimator simply preserves those true spacings in
        its causal regression history.
        """
        dt = max(1e-3, float(dt))
        divergence = float(divergence)
        self._time += dt
        self._history.append((self._time, divergence))
        self._trim_derivative_history()

        rate, ready = self._fit_divergence_rate()
        self._derivative_ready = ready
        if not ready:
            return

        self._divergence_rate = float(rate)
        self._last_chi = self._divergence_rate - divergence * divergence
        self._last_abs_chi = abs(self._last_chi)

        self._envelope_elapsed += dt
        self._envelope_samples += 1
        target = self._rolling_percentile(self._last_abs_chi)
        self._last_percentile = float(target)

        protected = target * self._attenuation_comp + self._margin
        decay = math.exp(-dt / self._peak_decay_tau)
        self._peak = max(protected, decay * self._peak)

    def result(self, min_duration_sec: float = 0.0) -> VisualMismatchResult:
        return VisualMismatchResult(
            peak_chi=float(self._peak),
            n_samples=int(self._envelope_samples),
            duration_sec=float(self._envelope_elapsed),
            ready=(
                self._derivative_ready
                and self._envelope_elapsed >= float(min_duration_sec)
            ),
        )
