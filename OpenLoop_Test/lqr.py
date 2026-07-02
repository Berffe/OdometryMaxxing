"""
Discrete-time LQR building blocks (ROS-free), shared by control_law.py.

Plant:        x[k+1] = A x[k] + B u[k]
Cost:         J = sum_k ( x[k]^T Q x[k] + u[k]^T R u[k] )
DARE:         P = A^T P A - A^T P B (R + B^T P B)^-1 B^T P A + Q
Optimal gain: K = (R + B^T P B)^-1 B^T P A,   with   u[k] = -K x[k]
Closed loop:  x[k+1] = (A - B K) x[k],  stable iff all |eig(A - B K)| < 1

ScheduledLQR carries a small bank of local linear models indexed by a scalar
scheduling variable (here area_fraction) and blends their pre-solved gains by
linear interpolation (the standard LPV / gain-scheduling pattern).

It also supports an element-wise gain_scale s, applied as K_eff = s (.) K. This
is the manual-tuning hook used by control_law.py: the LQR gives the baseline
gain shape, and s lets each gain component (e.g. the damping term) be trimmed
experimentally without re-deriving Q/R. baseline_gain_at() returns the
un-scaled optimal gain so the starting point can still be inspected/logged.
"""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def solve_discrete_lqr(A, B, Q, R, iterations: int = 20000, tol: float = 1e-12) -> np.ndarray:
	"""
	Return the LQR gain K (u = -K x) for x[k+1] = A x[k] + B u[k].

	A/B/Q/R may be scalars, nested lists, or arrays; they are coerced to 2D, so
	a 1-state/1-input axis can pass A=[[a]], B=[[b]], Q=[[q]], R=[[r]].

	Uses scipy.linalg.solve_discrete_are (Schur-based, robust) when available;
	this runs only a handful of times at construction, never per control tick.
	Falls back to fixed-point Riccati iteration otherwise. The fallback verifies
	the closed-loop eigenvalues and raises rather than silently returning a
	non-stabilizing K -- a model with |eig(A)| near/above 1 and small B can need
	P to grow for many thousands of steps before the control term bites.
	"""
	A = np.atleast_2d(np.asarray(A, dtype=float))
	B = np.atleast_2d(np.asarray(B, dtype=float))
	Q = np.atleast_2d(np.asarray(Q, dtype=float))
	R = np.atleast_2d(np.asarray(R, dtype=float))

	_validate_cost_weights(Q, R)

	try:
		from scipy.linalg import solve_discrete_are

		P = solve_discrete_are(A, B, Q, R)
		return _gain_from_p(A, B, R, P)
	except ImportError:
		pass

	P = Q.copy()
	converged = False
	for _ in range(iterations):
		K = _gain_from_p(A, B, R, P)
		p_next = Q + A.T @ P @ A - A.T @ P @ B @ K
		if np.max(np.abs(p_next - P)) < tol:
			P = p_next
			converged = True
			break
		P = p_next

	K = _gain_from_p(A, B, R, P)
	eigvals = np.linalg.eigvals(A - B @ K)
	if not converged or np.any(np.abs(eigvals) >= 1.0 - 1e-9):
		raise RuntimeError(
			"solve_discrete_lqr: pure-numpy fallback did not reach a verified "
			f"stabilizing solution in {iterations} iterations (closed-loop "
			f"eigenvalues {eigvals}, converged={converged}). Likely near the "
			"stability boundary with weak control authority -- install scipy "
			"(pip install scipy) or raise `iterations`."
		)
	return K


def _gain_from_p(A: np.ndarray, B: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
	"""K = (R + B^T P B)^-1 B^T P A."""
	bt_p = B.T @ P
	return np.linalg.solve(R + bt_p @ B, bt_p @ A)


def _validate_cost_weights(Q: np.ndarray, R: np.ndarray):
	"""
	Q must be PSD and R strictly PD, else the minimization is ill-posed. Guards
	the easy mistake of passing a fitted plant parameter into a Q/R slot: the
	number looks reasonable, the solver does not crash, it just returns a gain
	that barely reacts (or, for negative Q, solves the wrong problem).
	"""
	q_eigs = np.linalg.eigvalsh((Q + Q.T) / 2.0)
	r_eigs = np.linalg.eigvalsh((R + R.T) / 2.0)
	if np.any(q_eigs < -1e-9):
		raise ValueError(f"Q must be positive semi-definite, got eigenvalues {q_eigs}.")
	if np.any(r_eigs <= 1e-12):
		raise ValueError(f"R must be strictly positive definite, got eigenvalues {r_eigs}.")


class ScalarSchedule:
	"""
	A single manual-tuning knob (e.g. roll_prop_scale) that is either a
	constant, or a piecewise-linear function of a scheduling variable
	(area_fraction) -- same clamp-to-endpoint convention as ScheduledLQR,
	just for a plain scalar instead of a gain matrix.

	Exists for closed-loop, per-operating-point gain tuning (hover at a
	fixed altitude, tune by hand, repeat at the next altitude) as an
	alternative to deriving the whole schedule from open-loop calibration:
	pass a single number for the old "one value everywhere" behavior, or a
	list of (area_fraction, value) pairs collected one altitude at a time.

	value(0.1) -> 0.1                                  (constant)
	value([(0.07, 0.4), (0.13, 0.5), (0.21, 0.6)])     (tuned per altitude)
	"""

	def __init__(self, value):
		if isinstance(value, (int, float)):
			self._constant = float(value)
			self._afs = None
			self._values = None
			return

		points = sorted(value, key=lambda p: p[0])
		if not points:
			raise ValueError("ScalarSchedule needs at least one point if not a constant")
		self._constant = None
		self._afs = [float(p[0]) for p in points]
		self._values = [float(p[1]) for p in points]

	def value_at(self, area_fraction: float) -> float:
		if self._constant is not None:
			return self._constant

		afs, values = self._afs, self._values
		if area_fraction <= afs[0]:
			return values[0]
		if area_fraction >= afs[-1]:
			return values[-1]
		for i in range(len(afs) - 1):
			lo, hi = afs[i], afs[i + 1]
			if lo <= area_fraction <= hi:
				t = (area_fraction - lo) / max(hi - lo, 1e-9)
				return (1.0 - t) * values[i] + t * values[i + 1]
		return values[-1]


class ScheduledLQR:
	"""
	Bank of LQR gains along a scalar scheduling variable, blended by linear
	interpolation. Each entry is (scheduling_value, A, B, Q, R). Values outside
	the provided range are clamped to the nearest end (no extrapolation).

	gain_scale (optional) is an element-wise multiplier applied to every gain:
	K_eff = gain_scale (.) K. Use it to trim individual gain components (e.g.
	scale up the damping/velocity term) on top of the optimal baseline.
	"""

	def __init__(
		self,
		schedule: Iterable[Tuple[float, object, object, object, object]],
		gain_scale: Optional[object] = None,
	):
		points = sorted(schedule, key=lambda item: item[0])
		if not points:
			raise ValueError("ScheduledLQR needs at least one schedule point")

		self._values = [float(p[0]) for p in points]
		self._baseline_gains = [solve_discrete_lqr(p[1], p[2], p[3], p[4]) for p in points]

		self._gain_scale = None if gain_scale is None else np.atleast_2d(np.asarray(gain_scale, float))
		self._gains = [self._apply_scale(K) for K in self._baseline_gains]

	def _apply_scale(self, K: np.ndarray) -> np.ndarray:
		return K if self._gain_scale is None else self._gain_scale * K

	def _interpolate(self, gains, scheduling_value: float) -> np.ndarray:
		v = self._values
		if scheduling_value <= v[0]:
			return gains[0]
		if scheduling_value >= v[-1]:
			return gains[-1]
		for i in range(len(v) - 1):
			lo, hi = v[i], v[i + 1]
			if lo <= scheduling_value <= hi:
				t = (scheduling_value - lo) / max(hi - lo, 1e-9)
				return (1.0 - t) * gains[i] + t * gains[i + 1]
		return gains[-1]

	def gain_at(self, scheduling_value: float) -> np.ndarray:
		"""Interpolated gain K_eff (= baseline (.) gain_scale) for scheduling_value."""
		return self._interpolate(self._gains, scheduling_value)

	def baseline_gain_at(self, scheduling_value: float) -> np.ndarray:
		"""Interpolated optimal gain before gain_scale -- the LQR starting point."""
		return self._interpolate(self._baseline_gains, scheduling_value)

	def schedule_values(self) -> Sequence[float]:
		return tuple(self._values)