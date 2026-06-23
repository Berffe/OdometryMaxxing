"""
Small, ROS-free discrete-time LQR building blocks, shared by control_law.py.

solve_discrete_lqr(A, B, Q, R) solves the infinite-horizon discrete-time
LQR problem for

	x_{k+1} = A x_k + B u_k

minimizing sum_k (x_k^T Q x_k + u_k^T R u_k), and returns the gain K such
that u_k = -K x_k is optimal. It works for any state/input dimension —
a 1-state/1-input axis (today) or a multi-state axis (e.g. once you add
an optical-flow-derived velocity state for damping) use the same code
path, A/B/Q/R just become bigger matrices.

ScheduledLQR wraps a handful of (operating_point, A, B, Q, R) tuples,
solves each once at construction time, and linearly interpolates between
the two nearest gains at runtime. This is the standard gain-scheduling
("LPV") pattern: rather than one linearization valid only locally, you
carry a small bank of local linear models spanning the operating range
and blend between them using a scheduling variable (here, area_fraction).
"""

from typing import Iterable, Sequence, Tuple

import numpy as np


def solve_discrete_lqr(
	A,
	B,
	Q,
	R,
	iterations: int = 20000,
	tol: float = 1e-12,
) -> np.ndarray:
	"""
	Solve the discrete-time algebraic Riccati equation and return the
	corresponding LQR gain K (u = -K x).

	A, B, Q, R may be plain numbers/nested lists or numpy arrays; they
	are coerced to 2D arrays internally, so a scalar 1-state/1-input axis
	can simply pass e.g. A=[[1.0]], B=[[b]], Q=[[q]], R=[[r]].

	Uses scipy.linalg.solve_discrete_are when scipy is importable — it's
	the robust, standard approach (Schur-based, no iteration count to
	tune) and this function is only ever called a handful of times at
	ControlLaw construction, not per control tick, so the dependency
	costs nothing at runtime. Falls back to fixed-point Riccati iteration
	if scipy isn't available.

	The fallback is NOT just "more iterations and hope": a system with
	a near or above 1 (open-loop marginal/unstable, exactly what real
	calibration data can produce — see fit_axis_models.py) combined
	with a small b needs the Riccati iterate P to grow to a large value
	before the control term meaningfully bites, and naive iteration can
	need many thousands of steps to get there. Returning a gain before
	that happens silently hands back an unstabilizing K with no warning.
	So the fallback explicitly checks the resulting closed-loop
	eigenvalues and raises rather than returning a gain it hasn't
	verified is actually stabilizing.
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
		k = _gain_from_p(A, B, R, P)
		p_next = Q + A.T @ P @ A - A.T @ P @ B @ k

		if np.max(np.abs(p_next - P)) < tol:
			P = p_next
			converged = True
			break

		P = p_next

	K = _gain_from_p(A, B, R, P)
	closed_loop_eigvals = np.linalg.eigvals(A - B @ K)

	if not converged or np.any(np.abs(closed_loop_eigvals) >= 1.0 - 1e-9):
		raise RuntimeError(
			"solve_discrete_lqr: the pure-numpy fallback did not reach a "
			f"verified stabilizing solution after {iterations} iterations "
			f"(closed-loop eigenvalues: {closed_loop_eigvals}, "
			f"converged={converged}). This combination of A/B/Q/R is likely "
			"close to the stability boundary with weak control authority — "
			"install scipy for a robust solve (pip install scipy), or pass "
			"a larger `iterations`."
		)

	return K


def _gain_from_p(A: np.ndarray, B: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
	bt_p = B.T @ P
	s = R + bt_p @ B
	return np.linalg.solve(s, bt_p @ A)


def _validate_cost_weights(Q: np.ndarray, R: np.ndarray):
	"""
	Q must be positive semi-definite (state error is never allowed to
	reduce cost) and R strictly positive definite (control effort must
	have a real, invertible cost). This exists because of a real
	mistake: it's easy to accidentally pass a fitted model parameter
	(e.g. the intercept `c` from fit_axis_models.py) into a Q/R slot
	instead of an actual cost weight — the numbers look perfectly
	reasonable (small, finite) and the solver won't crash on them, it
	just quietly returns a gain that barely reacts to anything, or
	(if Q ends up negative) one that doesn't even correspond to a
	well-posed minimization problem. Catch that here instead.
	"""
	q_eigs = np.linalg.eigvalsh((Q + Q.T) / 2.0)
	r_eigs = np.linalg.eigvalsh((R + R.T) / 2.0)

	if np.any(q_eigs < -1e-9):
		raise ValueError(
			f"solve_discrete_lqr: Q must be positive semi-definite, got "
			f"eigenvalues {q_eigs}. If this Q came from a constructor "
			f"argument named *_state_cost, double check you passed a real "
			f"cost weight (e.g. 1.0) and not a fitted model parameter "
			f"like the intercept `c` from fit_axis_models.py — that's a "
			f"property of the plant, not something you choose, and it has "
			f"no slot in this 1-state model."
		)

	if np.any(r_eigs <= 1e-12):
		raise ValueError(
			f"solve_discrete_lqr: R must be strictly positive definite, "
			f"got eigenvalues {r_eigs}. Check the *_control_cost argument."
		)


class ScheduledLQR:
	"""
	A bank of discrete-time LQR gains, one per operating point along a
	scalar scheduling variable, blended by linear interpolation.

	Each schedule entry is (scheduling_value, A, B, Q, R). The gain for
	an arbitrary scheduling_value is obtained by linearly interpolating
	between the two nearest entries; values outside the provided range
	are clamped to the nearest end rather than extrapolated.
	"""

	def __init__(
		self,
		schedule: Iterable[Tuple[float, object, object, object, object]],
	):
		points = sorted(schedule, key=lambda item: item[0])

		if not points:
			raise ValueError("ScheduledLQR needs at least one schedule point")

		self._schedule_values = [float(point[0]) for point in points]
		self._gains = [
			solve_discrete_lqr(point[1], point[2], point[3], point[4])
			for point in points
		]

	def gain_at(self, scheduling_value: float) -> np.ndarray:
		"""Return the interpolated gain matrix K for scheduling_value."""
		values = self._schedule_values
		gains = self._gains

		if scheduling_value <= values[0]:
			return gains[0]

		if scheduling_value >= values[-1]:
			return gains[-1]

		for i in range(len(values) - 1):
			low, high = values[i], values[i + 1]

			if low <= scheduling_value <= high:
				span = max(high - low, 1e-9)
				t = (scheduling_value - low) / span

				return (1.0 - t) * gains[i] + t * gains[i + 1]

		return gains[-1]

	def schedule_values(self) -> Sequence[float]:
		return tuple(self._schedule_values)