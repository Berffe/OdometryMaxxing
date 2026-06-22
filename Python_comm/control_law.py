"""
Low-frequency vision-only control law.

This module turns a TargetEstimate into a desired attitude/thrust
setpoint. It deliberately does not read PX4 position or velocity; PX4
keeps the fast inner stabilization loop, while this law only produces
slow attitude/thrust references from vision-derived quantities.
"""

from .state import AttitudeSetpoint, TargetEstimate


class ControlLaw:
	"""
	Simple first implementation of the target-centering law:

		roll_cmd  = -Kx * (cx - cx*)
		pitch_cmd = -Ky * (cy - cy*)
		thrust    = hover_thrust
		yaw       = yaw_setpoint

	TargetAcquisition returns normalized image offsets, not raw pixels:
		offset_x = (cx - image_center_x) / (image_width  / 2)
		offset_y = (cy - image_center_y) / (image_height / 2)

	So offset_x and offset_y are approximately in [-1, 1]. This makes the
	gains easy to interpret: Kx and Ky are roughly the maximum attitude
	command [rad] requested for a target at the image border, before
	saturation.

	The signs follow the mathematical convention proposed in the control
	design. If the first flight test moves away from the target instead of
	towards it, flip the sign of roll_gain and/or pitch_gain rather than
	changing the rest of the code.
	"""

	def __init__(
		self,
		hover_thrust: float = 0.5,
		yaw_setpoint: float = 0.0,
		roll_gain: float = 0.10,
		pitch_gain: float = 0.10,
		roll_limit: float = 0.15,
		pitch_limit: float = 0.15,
		thrust_min: float = 0.20,
		thrust_max: float = 0.80,
	):
		self._hover_thrust = hover_thrust
		self._yaw_setpoint = yaw_setpoint

		self._roll_gain = roll_gain
		self._pitch_gain = pitch_gain

		self._roll_limit = abs(roll_limit)
		self._pitch_limit = abs(pitch_limit)
		self._thrust_min = thrust_min
		self._thrust_max = thrust_max

	def compute(self, target: TargetEstimate, dt: float) -> AttitudeSetpoint:
		"""
		Compute the latest attitude/thrust setpoint.

		dt is accepted now so the public API is already compatible with the
		future PI/PID version, but the current first implementation is purely
		proportional and therefore does not use dt.
		"""
		thrust = self._clamp(self._hover_thrust, self._thrust_min, self._thrust_max)

		if not target.found:
			# No reliable target yet: stay neutral and hover-ish instead of
			# integrating or commanding a blind correction.
			return AttitudeSetpoint(
				timestamp=target.timestamp,
				roll=0.0,
				pitch=0.0,
				yaw=self._yaw_setpoint,
				thrust=thrust,
			)

		roll_cmd = -self._roll_gain * target.offset_x
		pitch_cmd = -self._pitch_gain * target.offset_y

		roll_cmd = self._clamp(roll_cmd, -self._roll_limit, self._roll_limit)
		pitch_cmd = self._clamp(pitch_cmd, -self._pitch_limit, self._pitch_limit)

		return AttitudeSetpoint(
			timestamp=target.timestamp,
			roll=roll_cmd,
			pitch=pitch_cmd,
			yaw=self._yaw_setpoint,
			thrust=thrust,
		)

	@staticmethod
	def _clamp(value: float, lower: float, upper: float) -> float:
		return max(lower, min(upper, float(value)))
