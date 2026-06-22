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
	"""
	Simple first implementation of the target-centering + divergence
	landing law.

	Target centering:

		roll_cmd  = -roll_gain  * target.offset_x
		pitch_cmd = -pitch_gain * target.offset_y

	Optical-flow divergence landing:

		thrust = hover_thrust + divergence_gain * (
			flow.divergence - divergence_setpoint
		)

	Interpretation with positive divergence_gain:

		flow.divergence < divergence_setpoint
			=> thrust decreases
			=> drone descends faster

		flow.divergence > divergence_setpoint
			=> thrust increases
			=> drone slows down its descent

	The signs may need to be flipped after the first Gazebo test depending
	on camera convention, PX4 attitude convention, and the sign of the
	divergence estimator.
	"""

	def __init__(
		self,
		hover_thrust: float = 0.5,
		yaw_setpoint: float = 0.0,

		# Lateral target-centering gains.
		roll_gain: float = 0.10,
		pitch_gain: float = 0.10,

		# Divergence-based vertical control.
		divergence_gain: float = 0.05,
		divergence_setpoint: float = 0.15,

		# Safety limits.
		roll_limit: float = 0.15,
		pitch_limit: float = 0.15,
		thrust_min: float = 0.20,
		thrust_max: float = 0.80,

		# Conservative default:
		# only use divergence landing when the target is found.
		require_target_for_descent: bool = True,
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
		"""
		Compute the latest attitude/thrust setpoint.

		dt is accepted so the public API is already compatible with future
		integral/derivative terms, but this first version does not use it.
		"""

		# ------------------------------------------------------------
		# 1. Default command: neutral attitude, hover thrust.
		# ------------------------------------------------------------
		roll_cmd = 0.0
		pitch_cmd = 0.0
		yaw_cmd = self._yaw_setpoint
		thrust_cmd = self._hover_thrust

		# ------------------------------------------------------------
		# 2. Lateral target-centering control.
		# ------------------------------------------------------------
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

		# ------------------------------------------------------------
		# 3. Divergence-based thrust control.
		# ------------------------------------------------------------
		can_use_divergence = flow is not None and flow.valid

		if self._require_target_for_descent:
			can_use_divergence = can_use_divergence and target.found

		if can_use_divergence:
			divergence_error = flow.divergence - self._divergence_setpoint

			thrust_cmd = (
				self._hover_thrust
				+ self._divergence_gain * divergence_error
			)

		# ------------------------------------------------------------
		# 4. Final thrust saturation.
		# ------------------------------------------------------------
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
	