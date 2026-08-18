"""Slow visual-trim estimator for mission gates and diagnostics.

The raw image centroid can have a non-zero slow mean for several reasons:
disturbance-rejection tilt, camera geometry, and target-relative position.  This
estimator separates that slow image equilibrium (the ``mean``) from the fast
motion about it (the ``residual``) without assigning either one direct control
authority.

Two instances run in parallel in :class:`~bee_control.mission.routine.MissionRoutine`,
with different time constants because they answer different questions:

``center trim``   fast (~1 s). Feeds the CENTER handoff gate: "has the vehicle
                  settled around a safe equilibrium yet?" It must react within
                  the dwell window, so it is deliberately short.

``handoff trim``  slow, retuned and reset in lockstep with
                  :class:`~bee_control.mission.probe.PlatformProbe`. It records
                  the near-field visual operating point for diagnostics.  Since
                  lateral image-position P authority is intentionally removed at
                  FINAL_PROBE, this mean is no longer a DESCENT control setpoint.

Like the probe, this estimator observes only what the mission already sees. It
adds no control authority of its own and cannot alter a commanded acceleration
by itself -- the phase decides what, if anything, to do with the numbers.
"""
from __future__ import annotations

import math


class VisualTrim:
    """Exponential mean of the target image offset, plus its live residual.

    ``tau_sec`` is the EMA time constant. Samples are folded in only while the
    target is found, so a dropout holds the estimate rather than dragging it
    toward zero -- a lost target is missing evidence, not evidence of being
    centred.
    """

    def __init__(self, tau_sec: float = 1.0):
        self._tau = max(1e-3, float(tau_sec))
        self._mean_x = 0.0
        self._mean_y = 0.0
        self._has_mean = False
        self._last_x = 0.0
        self._last_y = 0.0
        self._has_sample = False

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        """Discard the mean entirely. The next sample re-seeds it."""
        self._mean_x = 0.0
        self._mean_y = 0.0
        self._has_mean = False
        self._last_x = 0.0
        self._last_y = 0.0
        self._has_sample = False

    def retune(self, tau_sec: float) -> None:
        """Change the time constant WITHOUT discarding the current mean.

        Separate from :meth:`reset` for the same reason the probe separates the
        two: a phase transition may want new conditioning, fresh evidence, or
        both, and conflating them silently carries an old envelope into a new
        decision.
        """
        self._tau = max(1e-3, float(tau_sec))

    def update(
        self, offset_x: float, offset_y: float, target_found: bool, dt: float
    ) -> None:
        if not bool(target_found):
            return
        x = float(offset_x)
        y = float(offset_y)
        self._last_x = x
        self._last_y = y
        self._has_sample = True
        if not self._has_mean:
            self._mean_x = x
            self._mean_y = y
            self._has_mean = True
            return
        alpha = math.exp(-max(1e-3, float(dt)) / self._tau)
        self._mean_x = alpha * self._mean_x + (1.0 - alpha) * x
        self._mean_y = alpha * self._mean_y + (1.0 - alpha) * y

    # ------------------------------------------------------------ accessors
    @property
    def tau_sec(self) -> float:
        return float(self._tau)

    @property
    def has_mean(self) -> bool:
        return bool(self._has_mean)

    @property
    def mean_x(self) -> float:
        return float(self._mean_x)

    @property
    def mean_y(self) -> float:
        return float(self._mean_y)

    @property
    def mean_radius(self) -> float:
        """Distance of the equilibrium from image centre. ``inf`` until seeded.

        ``inf`` rather than 0.0 so that an unseeded estimator can never satisfy
        a ``<= max`` gate by accident.
        """
        if not self._has_mean:
            return float("inf")
        return math.hypot(self._mean_x, self._mean_y)

    @property
    def residual_radius(self) -> float:
        """Distance of the LATEST sample from the equilibrium. ``inf`` if none."""
        if not (self._has_mean and self._has_sample):
            return float("inf")
        return math.hypot(self._last_x - self._mean_x, self._last_y - self._mean_y)

    def snapshot(self) -> tuple[float, float, float, float]:
        """``(mean_x, mean_y, mean_radius, residual_radius)`` for logging/gates."""
        return (self.mean_x, self.mean_y, self.mean_radius, self.residual_radius)
