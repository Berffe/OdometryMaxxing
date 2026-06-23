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
	iterations: int = 200,
	tol: float = 1e-10,
) -> np.ndarray:
	"""
	Solve the discrete-time algebraic Riccati equation by fixed-point
	iteration and return the corresponding LQR gain K (u = -K x).

	A, B, Q, R may be plain numbers/nested lists or numpy arrays; they
	are coerced to 2D arrays internally, so a scalar 1-state/1-input axis
	can simply pass e.g. A=[[1.0]], B=[[b]], Q=[[q]], R=[[r]].

	This converges to the steady-state solution for any stabilizable,
	detectable (A, B, Q, R) — in particular, any scalar system with
	B != 0 — well within `iterations`, since the Riccati recursion is a
	contraction for such systems.
	"""
	A = np.atleast_2d(np.asarray(A, dtype=float))
	B = np.atleast_2d(np.asarray(B, dtype=float))
	Q = np.atleast_2d(np.asarray(Q, dtype=float))
	R = np.atleast_2d(np.asarray(R, dtype=float))

	P = Q.copy()

	for _ in range(iterations):
		bt_p = B.T @ P
		s = R + bt_p @ B
		k = np.linalg.solve(s, bt_p @ A)
		p_next = Q + A.T @ P @ A - A.T @ P @ B @ k

		if np.max(np.abs(p_next - P)) < tol:
			P = p_next
			break

		P = p_next

	bt_p = B.T @ P
	s = R + bt_p @ B
	k = np.linalg.solve(s, bt_p @ A)

	return k


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