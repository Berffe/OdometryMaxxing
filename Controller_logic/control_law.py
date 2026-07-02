"""
Closed-loop visual controller: constant-gain PD lateral + probe-driven thrust.

VISUAL-ONLY by design: once active, commands use ONLY visual data (target
offset, normalized optical flow, divergence). No PX4 state feeds the control
law after handoff.

This is the committed, simplified controller. The earlier LQR + per-
area_fraction gain-scheduling machinery (ScheduledLQR/ScalarSchedule, the
identified (A,B) state models, the phase-lead branches, and the fov-saturation
area_fraction latch) has been removed now that the constant-gain architecture
is validated in flight. What remains:

  LATERAL (roll, pitch): one fixed PD gain set per axis,
        u = -(k_p * offset + k_d * optical_flow) * lateral_gain_scale
  scaled per-tick by lateral_gain_scale (the mission routine ramps this on the
  same de Croon 2016 App. B height schedule as the thrust gain).

  THRUST: a direct Herisse (2012, eq. 32) / de Croon accel-domain law,
        a_cmd        = thrust_gain_override * (D - D*)      [m/s^2]
        thrust_delta = hover_thrust * a_cmd / G_ACCEL       [normalized]
  where thrust_gain_override ("k") is supplied per-tick by the mission routine
  (a hand-tuned exploration value, decayed through descent). The sign is
  POSITIVE on (D - D*) because the identified plant has B<0 (more thrust
  REDUCES divergence), so arresting an approach requires MORE thrust. An
  optional constant integral term (thrust_integral_gain_const) may add a slow
  bias correction on top; it is 0 by default.

Both the lateral command and the thrust command are passed through a first-
order low-pass + slew-rate limiter + clamp (_shape_commands) before output.

The divergence used for control is a fixed blend of the filtered and raw
divergence (raw_divergence_weight); the mission's divergence_setpoint D* is
supplied per-tick (0 for the hover/probe hold, a small positive value to
descend).
"""

import math
from typing import Optional

try:
	from .state import AttitudeSetpoint, FlowResult, TargetEstimate
except ImportError:
	from state import AttitudeSetpoint, FlowResult, TargetEstimate


# Standard gravity: converts between normalized collective thrust and world-
# vertical acceleration for the Herisse/de Croon accel-domain thrust law.
G_ACCEL = 9.80665


class ControlLaw:
	def __init__(
		self,
		hover_thrust: float = 0.73,
		yaw_setpoint: float = 0.0,
		divergence_setpoint: float = 0.0,  # 0 = visual hover; raised to descend.

		# --- Lateral constant PD (offset P + optical-flow D), per axis. ---
		roll_kp: float = 0.22,
		roll_kd: float = 0.11,
		pitch_kp: float = 0.15,
		pitch_kd: float = 0.07,

		# --- Thrust axis. ---
		thrust_integral_gain_const: float = 0.01,
		divergence_integral_limit: float = 1.2,
		max_visual_thrust_delta_from_hover: float = 0.18,
		raw_divergence_weight: float = 0.10,
		enable_divergence_control: bool = True,
		require_target_for_descent: bool = True,

		# --- Output limits + command shaping (low-pass -> slew -> clamp). ---
		roll_limit: float = 0.20,
		pitch_limit: float = 0.20,
		thrust_min: float = 0.62,
		thrust_max: float = 0.84,
		roll_output_sign: float = 1.0,
		pitch_output_sign: float = 1.0,
		roll_slew_rate_rad_s: float = 0.35,
		pitch_slew_rate_rad_s: float = 0.20,
		thrust_slew_rate_per_s: float = 0.40,
		command_filter_alpha: float = 0.85,
	):
		self._hover_thrust = float(hover_thrust)
		self._yaw_setpoint = float(yaw_setpoint)
		self._divergence_setpoint = float(divergence_setpoint)

		self._roll_kp = float(roll_kp)
		self._roll_kd = float(roll_kd)
		self._pitch_kp = float(pitch_kp)
		self._pitch_kd = float(pitch_kd)

		self._thrust_integral_gain_const = float(thrust_integral_gain_const)
		self._divergence_integral_limit = abs(float(divergence_integral_limit))
		self._max_visual_thrust_delta = abs(float(max_visual_thrust_delta_from_hover))
		self._raw_divergence_weight = self._clamp(raw_divergence_weight, 0.0, 1.0)
		self._enable_divergence_control = bool(enable_divergence_control)
		self._require_target_for_descent = bool(require_target_for_descent)

		self._roll_limit = abs(float(roll_limit))
		self._pitch_limit = abs(float(pitch_limit))
		self._thrust_min = float(thrust_min)
		self._thrust_max = float(thrust_max)
		self._roll_output_sign = 1.0 if roll_output_sign >= 0.0 else -1.0
		self._pitch_output_sign = 1.0 if pitch_output_sign >= 0.0 else -1.0
		self._roll_slew_rate_rad_s = abs(float(roll_slew_rate_rad_s))
		self._pitch_slew_rate_rad_s = abs(float(pitch_slew_rate_rad_s))
		self._thrust_slew_rate_per_s = abs(float(thrust_slew_rate_per_s))
		self._command_filter_alpha = self._clamp(command_filter_alpha, 0.0, 1.0)

		# Runtime state.
		self._divergence_integral = 0.0
		self._previous_roll_cmd = 0.0
		self._previous_pitch_cmd = 0.0
		self._previous_thrust_cmd = self._hover_thrust
		self._has_previous_command = False

	@property
	def hover_thrust(self) -> float:
		return self._hover_thrust

	@property
	def divergence_integral(self) -> float:
		return self._divergence_integral

	def reset_visual_integrators(self):
		self._divergence_integral = 0.0
		self._previous_roll_cmd = 0.0
		self._previous_pitch_cmd = 0.0
		self._previous_thrust_cmd = self._hover_thrust
		self._has_previous_command = False

	def compute(
		self,
		target: TargetEstimate,
		flow: FlowResult,
		dt: float,
		*,
		divergence_setpoint: Optional[float] = None,
		thrust_gain_override: Optional[float] = None,
		lateral_gain_scale: float = 1.0,
	) -> AttitudeSetpoint:
		"""Desired roll/pitch/yaw/thrust from visual data only.

		divergence_setpoint: per-call override of D* (constructor value left
		    untouched). 0 for the probe/hover hold, small positive to descend.
		thrust_gain_override: "k" in a_cmd = k*(D - D*). Supplied per-tick by
		    the mission (hand-tuned exploration gain, decayed through descent).
		lateral_gain_scale: multiplies the constant lateral PD gains this call.
		"""
		dt = max(1e-3, float(dt))

		roll_cmd = 0.0
		pitch_cmd = 0.0
		visual_thrust_delta = 0.0

		flow_valid = flow is not None and bool(getattr(flow, "valid", False))
		target_found = target is not None and bool(getattr(target, "found", False))

		# --- Lateral axes: constant PD (offset P + optical-flow D). ---
		if target_found:
			flow_x = float(getattr(flow, "mean_flow_x_norm", 0.0)) if flow_valid else 0.0
			flow_y = float(getattr(flow, "mean_flow_y_norm", 0.0)) if flow_valid else 0.0
			offset_x = float(target.offset_x)
			offset_y = float(target.offset_y)

			s = max(0.0, float(lateral_gain_scale))
			roll_u = -(self._roll_kp * offset_x + self._roll_kd * flow_x) * s
			pitch_u = -(self._pitch_kp * offset_y + self._pitch_kd * flow_y) * s

			roll_cmd = self._soft_limit(self._roll_output_sign * roll_u, self._roll_limit)
			pitch_cmd = self._soft_limit(self._pitch_output_sign * pitch_u, self._pitch_limit)

		# --- Thrust axis: accel-domain law on divergence error + integral. ---
		can_use_divergence = self._enable_divergence_control and flow_valid
		if self._require_target_for_descent:
			can_use_divergence = can_use_divergence and target_found

		effective_divergence_setpoint = (
			self._divergence_setpoint if divergence_setpoint is None
			else float(divergence_setpoint)
		)

		if can_use_divergence:
			error = self._divergence_for_control(flow) - effective_divergence_setpoint

			self._divergence_integral = self._clamp(
				self._divergence_integral + error * dt,
				-self._divergence_integral_limit,
				self._divergence_integral_limit,
			)

			if thrust_gain_override is not None:
				# Herisse eq. 32 / de Croon accel-domain law: a_cmd = +k*(D-D*).
				# POSITIVE sign because the plant has B<0 (more thrust reduces
				# divergence): arresting a sink (D>D*) needs MORE thrust.
				accel_cmd = float(thrust_gain_override) * error
				proportional_delta = self._hover_thrust * accel_cmd / G_ACCEL
			else:
				proportional_delta = 0.0

			integral_delta = self._thrust_integral_gain_const * self._divergence_integral
			visual_thrust_delta = self._soft_limit(
				proportional_delta + integral_delta, self._max_visual_thrust_delta
			)
		else:
			# No visual measurement: decay the integral (don't hard-reset) so
			# one dropped frame is not a discontinuity.
			self._divergence_integral *= 0.90

		thrust_cmd = self._clamp(
			self._hover_thrust + visual_thrust_delta, self._thrust_min, self._thrust_max
		)

		roll_cmd, pitch_cmd, thrust_cmd = self._shape_commands(roll_cmd, pitch_cmd, thrust_cmd, dt)

		return AttitudeSetpoint(
			timestamp=getattr(target, "timestamp", 0.0),
			roll=roll_cmd,
			pitch=pitch_cmd,
			yaw=self._yaw_setpoint,
			thrust=thrust_cmd,
		)

	def _shape_commands(self, roll: float, pitch: float, thrust: float, dt: float):
		"""
		filter:  c_f = (1-a) c_prev + a c       (first-order low-pass)
		slew:    c_s = c_prev + clip(c_f - c_prev, +-rate*dt)
		clamp:   to the axis limits.
		"""
		if not self._has_previous_command:
			self._previous_roll_cmd = 0.0
			self._previous_pitch_cmd = 0.0
			self._previous_thrust_cmd = self._hover_thrust
			self._has_previous_command = True

		a = self._command_filter_alpha
		roll_f = (1.0 - a) * self._previous_roll_cmd + a * roll
		pitch_f = (1.0 - a) * self._previous_pitch_cmd + a * pitch
		thrust_f = (1.0 - a) * self._previous_thrust_cmd + a * thrust

		roll_s = self._slew_limit(self._previous_roll_cmd, roll_f, self._roll_slew_rate_rad_s * dt)
		pitch_s = self._slew_limit(self._previous_pitch_cmd, pitch_f, self._pitch_slew_rate_rad_s * dt)
		thrust_s = self._slew_limit(self._previous_thrust_cmd, thrust_f, self._thrust_slew_rate_per_s * dt)

		roll_s = self._clamp(roll_s, -self._roll_limit, self._roll_limit)
		pitch_s = self._clamp(pitch_s, -self._pitch_limit, self._pitch_limit)
		thrust_s = self._clamp(thrust_s, self._thrust_min, self._thrust_max)

		self._previous_roll_cmd = roll_s
		self._previous_pitch_cmd = pitch_s
		self._previous_thrust_cmd = thrust_s
		return roll_s, pitch_s, thrust_s

	def _divergence_for_control(self, flow: FlowResult) -> float:
		"""Blend filtered and raw divergence: (1-w) d_filt + w d_raw."""
		filtered = self._safe_float(getattr(flow, "divergence", 0.0))
		raw = self._safe_float(getattr(flow, "raw_divergence", filtered), default=filtered)
		w = self._raw_divergence_weight
		return (1.0 - w) * filtered + w * raw

	@staticmethod
	def _soft_limit(value: float, limit: float) -> float:
		"""L * tanh(v / L): smooth, bounded by +-L, ~linear near 0."""
		limit = abs(float(limit))
		return 0.0 if limit <= 1e-12 else limit * math.tanh(float(value) / limit)

	@staticmethod
	def _slew_limit(previous: float, desired: float, max_step: float) -> float:
		max_step = abs(float(max_step))
		return previous + max(-max_step, min(max_step, desired - previous))

	@staticmethod
	def _safe_float(value, default: float = 0.0) -> float:
		try:
			return float(value)
		except (TypeError, ValueError):
			return float(default)

	@staticmethod
	def _clamp(value: float, lower: float, upper: float) -> float:
		return max(lower, min(upper, float(value)))