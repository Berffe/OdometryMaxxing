"""
Low-frequency vision-only control law.

This module turns a TargetEstimate + FlowResult into a desired
attitude/thrust setpoint, using a discrete-time LQR regulator on each
axis instead of a hand-tuned proportional gain:

	roll_cmd   = -Kroll(area_fraction)   * offset_x
	pitch_cmd  = -Kpitch(area_fraction)  * offset_y
	thrust_cmd = hover_thrust - Kthrust(area_fraction) * (divergence - divergence_setpoint)
	yaw_cmd    = yaw_setpoint

Kroll, Kpitch, and Kthrust are not single fixed gains: each comes from a
ScheduledLQR (see lqr.py) that solves the discrete-time LQR problem at a
handful of operating points spread across area_fraction and interpolates
between them. This replaces the earlier hand-rolled `_attitude_gain_scale`
taper — the same "gain must shrink as the target fills more of the
frame" behavior now falls out of the LQR solve itself, rather than
being imposed after the fact.

Why area_fraction is the scheduling variable, and why it changes the
gain at all:

	Each axis is modeled, locally, as a 1-state discrete-time system

		e[k+1] = a * e[k] + b(area_fraction) * u[k]

	where e is the regulation error (offset_x, offset_y, or divergence
	minus its setpoint) and u is the corresponding command (roll, pitch,
	or thrust minus hover_thrust). The control-effectiveness term b is
	not constant: a fixed roll/pitch command produces a bigger swing in
	offset_x/offset_y at low altitude than at high altitude, because
	offset_x, offset_y, and divergence are themselves ratios that scale
	with 1/altitude. Since area_fraction scales with 1/altitude^2 for a
	target of roughly fixed real size, b(area_fraction) is modeled here
	as

		b(area_fraction) = b_ref * sqrt(area_fraction / area_fraction_ref)

	i.e. proportional to 1/altitude, up to the reference calibration
	constant b_ref. b_ref, the state/control cost weights, and `a`
	(the state's natural persistence absent control input) are placeholders
	for your own system identification / tuning — there is no physical
	parameter (mass, FOV, thrust curve) baked in here, by design: this
	keeps the model in the same scale-free offset/divergence space the
	rest of the pipeline already works in, rather than requiring a
	metric calibration the project deliberately avoids (see
	target_acquisition.py — no assumed target size).

Each axis currently uses a 1-state model (the regulation error alone,
no separate velocity/damping state). That is intentional for now and
is exactly the extension point for adding optical-flow-based damping
later: extending `_build_axis_schedule` (or adding a sibling builder)
to a 2-state model — e.g. state = [offset_x, mean_flow_x] for roll,
state = [offset_y, mean_flow_y] for pitch — only means supplying 2x2
A/B/Q/R instead of 1x1, and building the corresponding 2-element state
vector in compute(). Nothing else in this module, in ScheduledLQR, or
in solve_discrete_lqr assumes a 1-state axis; the scalar case here is
the simplest instance of the same machinery, not a special case of it.

The three axes (roll, pitch, thrust) are treated as independent
single-input/single-output loops, each with its own ScheduledLQR. That
is a deliberate simplification, not an oversight: see the project
discussion on why a tilt-dependent loss of vertical thrust is a
second-order effect that doesn't enter this linearization, and is
better handled as a thrust-mixing correction than as cross-axis terms
in these per-axis models, if/when it's added.

No PX4 position or velocity feedback is used here.
"""

import math

import numpy as np

try:
	from .lqr import ScheduledLQR
	from .state import AttitudeSetpoint, FlowResult, TargetEstimate
except ImportError:
	from lqr import ScheduledLQR
	from state import AttitudeSetpoint, FlowResult, TargetEstimate


class ControlLaw:
	def __init__(
		self,
		hover_thrust=0.45,
		yaw_setpoint=0.0,

		divergence_setpoint=0.0,

		roll_limit=0.10,
		pitch_limit=0.10,

		thrust_min=0.35,
		thrust_max=0.65,

		require_target_for_descent=True,

		# Local 1-state model + LQR cost weights, one set per axis. See
		# the module docstring for what `a`, `b_ref`, and
		# area_fraction_ref mean. state_cost/control_cost are the
		# diagonal LQR weights (Q, R) for that axis' single state/input.
		roll_a=1.0,
		roll_b_ref=0.6,
		roll_state_cost=1.0,
		roll_control_cost=60.0,

		pitch_a=1.0,
		pitch_b_ref=0.6,
		pitch_state_cost=1.0,
		pitch_control_cost=60.0,

		thrust_a=1.0,
		thrust_b_ref=0.6,
		thrust_state_cost=1.0,
		thrust_control_cost=60.0,

		# Operating points (in area_fraction) at which the LQR gain is
		# actually solved; gain_at() interpolates between these at
		# runtime. The values mirror the breakpoints already used in
		# target_acquisition.py's large-area handling, so "far",
		# "approaching", and "very close" mean the same thing project-wide.
		schedule_points=(0.05, 0.30, 0.60, 0.80, 0.95, 1.0),
		area_fraction_ref=0.05,
	):
		self._hover_thrust = hover_thrust
		self._yaw_setpoint = yaw_setpoint
		self._divergence_setpoint = divergence_setpoint

		self._roll_limit = abs(roll_limit)
		self._pitch_limit = abs(pitch_limit)

		self._thrust_min = thrust_min
		self._thrust_max = thrust_max

		self._require_target_for_descent = require_target_for_descent

		self._roll_lqr = self._build_axis_schedule(
			a=roll_a,
			b_ref=roll_b_ref,
			state_cost=roll_state_cost,
			control_cost=roll_control_cost,
			schedule_points=schedule_points,
			area_fraction_ref=area_fraction_ref,
		)

		self._pitch_lqr = self._build_axis_schedule(
			a=pitch_a,
			b_ref=pitch_b_ref,
			state_cost=pitch_state_cost,
			control_cost=pitch_control_cost,
			schedule_points=schedule_points,
			area_fraction_ref=area_fraction_ref,
		)

		self._thrust_lqr = self._build_axis_schedule(
			a=thrust_a,
			b_ref=thrust_b_ref,
			state_cost=thrust_state_cost,
			control_cost=thrust_control_cost,
			schedule_points=schedule_points,
			area_fraction_ref=area_fraction_ref,
		)

	def compute(
		self,
		target: TargetEstimate,
		flow: FlowResult,
		dt: float,
	) -> AttitudeSetpoint:
		# dt is unused for now: all three axis models are local
		# discrete-time models implicitly tied to the (fixed) control
		# period they were tuned against, rather than re-derived from
		# continuous dynamics on every call. Kept in the signature for a
		# future variable-dt or explicit derivative term.
		roll_cmd = 0.0
		pitch_cmd = 0.0
		yaw_cmd = self._yaw_setpoint
		thrust_cmd = self._hover_thrust

		area_fraction = float(getattr(target, "area_fraction", 0.0))

		if target.found:
			roll_state = np.array([[target.offset_x]])
			roll_gain = self._roll_lqr.gain_at(area_fraction)

			roll_cmd = float(-(roll_gain @ roll_state)[0, 0])

			roll_cmd = self._clamp(
				roll_cmd,
				-self._roll_limit,
				self._roll_limit,
			)

			pitch_state = np.array([[target.offset_y]])
			pitch_gain = self._pitch_lqr.gain_at(area_fraction)

			pitch_cmd = float(-(pitch_gain @ pitch_state)[0, 0])

			pitch_cmd = self._clamp(
				pitch_cmd,
				-self._pitch_limit,
				self._pitch_limit,
			)

		can_use_divergence = flow is not None and flow.valid

		if self._require_target_for_descent:
			can_use_divergence = can_use_divergence and target.found

		if can_use_divergence:
			thrust_state = np.array(
				[[flow.divergence - self._divergence_setpoint]]
			)
			thrust_gain = self._thrust_lqr.gain_at(area_fraction)

			thrust_correction = float(-(thrust_gain @ thrust_state)[0, 0])

			thrust_cmd = self._hover_thrust + thrust_correction

		thrust_cmd = self._clamp(
			thrust_cmd,
			self._thrust_min,
			self._thrust_max,
		)

		return AttitudeSetpoint(
			timestamp=target.timestamp,
			roll=roll_cmd,
			pitch=pitch_cmd,
			yaw=yaw_cmd,
			thrust=thrust_cmd,
		)

	@staticmethod
	def _clamp(value: float, lower: float, upper: float) -> float:
		return max(lower, min(upper, float(value)))

	@staticmethod
	def _build_axis_schedule(
		a: float,
		b_ref: float,
		state_cost: float,
		control_cost: float,
		schedule_points,
		area_fraction_ref: float,
	) -> ScheduledLQR:
		"""
		Build a ScheduledLQR for one axis: a 1-state local model at each
		point in `schedule_points`, with control effectiveness
		`b_ref * sqrt(area_fraction / area_fraction_ref)` (see the module
		docstring for the 1/altitude reasoning behind the square root).
		"""
		schedule = []

		for area_fraction in schedule_points:
			b = b_ref * math.sqrt(
				max(area_fraction, 1e-6) / max(area_fraction_ref, 1e-6)
			)

			schedule.append(
				(area_fraction, [[a]], [[b]], [[state_cost]], [[control_cost]])
			)

		return ScheduledLQR(schedule)