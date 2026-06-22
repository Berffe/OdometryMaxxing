"""
Low-frequency vision-only control law.

This module turns a TargetEstimate + FlowResult into a desired
attitude/thrust setpoint.

PX4 keeps the fast inner stabilization loop. This controller only
produces slow attitude/thrust references from vision-derived quantities:

	roll_cmd   = -Kx * offset_x
	pitch_cmd  = -Ky * offset_y
	thrust_cmd = hover_thrust + Kdiv * (divergence - divergence_setpoint)
	yaw_cmd    = yaw_setpoint

No PX4 position or velocity feedback is used here.
"""

from .state import AttitudeSetpoint, FlowResult, TargetEstimate


class ControlLaw:
	def __init__(
		self,
		hover_thrust=0.45, # 0.45
		yaw_setpoint=0.0,

		roll_gain=0.05,
		pitch_gain=0,

		divergence_gain=0.03,
		divergence_setpoint=0.15,

		roll_limit=0.10,
		pitch_limit=0.10,

		thrust_min=0.35,
		thrust_max=0.65,

		require_target_for_descent=True,
	):
		self._hover_thrust = hover_thrust
		self._yaw_setpoint = yaw_setpoint

		self._roll_gain = roll_gain
		self._pitch_gain = pitch_gain

		self._divergence_gain = divergence_gain
		self._divergence_setpoint = divergence_setpoint

		self._roll_limit = abs(roll_limit)
		self._pitch_limit = abs(pitch_limit)

		self._thrust_min = thrust_min
		self._thrust_max = thrust_max

		self._require_target_for_descent = require_target_for_descent

	def compute(
		self,
		target: TargetEstimate,
		flow: FlowResult,
		dt: float,
	) -> AttitudeSetpoint:
		# dt is unused for now, but kept for future I/D terms.
		roll_cmd = 0.0
		pitch_cmd = 0.0
		yaw_cmd = self._yaw_setpoint
		thrust_cmd = self._hover_thrust

		if target.found:
			roll_cmd = -self._roll_gain * target.offset_x
			pitch_cmd = -self._pitch_gain * target.offset_y

			roll_cmd = self._clamp(
				roll_cmd,
				-self._roll_limit,
				self._roll_limit,
			)

			pitch_cmd = self._clamp(
				pitch_cmd,
				-self._pitch_limit,
				self._pitch_limit,
			)

		can_use_divergence = flow is not None and flow.valid

		if self._require_target_for_descent:
			can_use_divergence = can_use_divergence and target.found

		if can_use_divergence:
			divergence_error = flow.divergence - self._divergence_setpoint

			thrust_cmd = (
				self._hover_thrust
				+ self._divergence_gain * divergence_error
			)

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