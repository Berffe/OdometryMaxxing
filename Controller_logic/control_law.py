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
        u = -(k_p * lateral_p_scale * offset + k_d * lateral_d_scale * optical_flow)
  scaled per-tick by INDEPENDENT P and D scales (the mission routine ramps
  these on the same de Croon 2016 App. B height schedule as the thrust gain
  during DESCEND; during CENTER they instead hold fixed at mission-phase-
  specific values -- see mission_routine.py's CENTER docstrings). Split
  because kp and kd have been scaled by different historical factors, so a
  single shared scale cannot reproduce an earlier validated (kp, kd) pair.

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
order low-pass + optional slew-rate limiter + clamp (_shape_commands) before output.

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
		# Scaled from the previous validated values (roll 0.22/0.11, pitch
		# 0.15/0.07) to raise tracking BANDWIDTH without adding aggressiveness.
		#
		# Measured directly (cross-spectral analysis of relative_y vs
		# platform_y over a steady-lateral-gain window): at the
		# platform's own oscillation frequency (~0.06 Hz here), closed-loop
		# tracking gain was only ~3.6% -- the vehicle was essentially parked
		# near a compromise position while the platform oscillated under it,
		# with roll/pitch nowhere close to saturating (max 0.047/0.017 rad
		# against a 0.20 rad limit) -- so the limit was never the bottleneck.
		#
		# The lateral loop is a PD compensator on what is effectively a double
		# integrator (roll -> tilt -> horizontal accel -> velocity ->
		# position). For that plant, kp sets the closed-loop natural
		# frequency/bandwidth and kd sets the damping ratio
		# (zeta ~ kd/sqrt(kp)); a properly bandwidth-matched loop tracks a
		# disturbance well below its own bandwidth near-perfectly, so ~3.6%
		# tracking at a comparatively SLOW 0.06 Hz means the loop's effective
		# bandwidth was sitting below the platform's own frequency, not just
		# "a bit low."
		#
		# Fix: raise kp (x2) to push bandwidth up past the platform's
		# frequency, and raise kd by LESS (x sqrt(2)) to hold the SAME damping
		# ratio rather than let the loop go underdamped/oscillatory as
		# bandwidth increases -- the standard "same shape, faster" 2nd-order
		# scaling. There is large saturation headroom (measured max commands
		# were ~25% and ~8% of roll_limit/pitch_limit respectively), so this
		# is a conservative first step, not a max-authority change; re-measure
		# the same tracking-gain diagnostic on the next log before scaling
		# further.
		#
		# ^ That step (kp x2, kd x sqrt(2)) preserved the EXISTING damping
		# ratio while raising bandwidth -- it did not fix damping itself. The
		# next step below addresses that directly.
		# REVERTED from a 3x kd trial (0.46/0.30) -- direct A/B comparison
		# against the two most recent logs (identical except kd) showed the
		# higher kd made things WORSE, not better: envelope decay half-life
		# went from 50.5s to 179.7s (~3.6x SLOWER), and peak commands moved
		# closer to saturating (roll 0.152->0.193 against a 0.20 limit)
		# instead of settling faster. This contradicts the idealized clean-
		# derivative 2nd-order prediction -- the likely reason is that kd
		# multiplies REAL optical flow, which carries measurement noise and
		# processing latency, not a clean derivative; past some point a
		# derivative term on a noisy/lagged signal injects excitation and
		# effectively adds dead-time rather than damping. The data overrides
		# the theory here: do not raise kd further as the next lever. See the
		# comment block above for what WAS validated (the kp/kd(sqrt2) step
		# and the gain blend, both measured as improvements).
		roll_kp: float = 0.22,
		roll_kd: float = 0.1,
		pitch_kp: float = 0.15,
		pitch_kd: float = 0.075,

		# --- Error-magnitude gain blend (large-offset transient damping) ---
		# roll_kp/kd/pitch_kp/kd above are the FULL gains, used once |offset|
		# is small (near-centered, oscillation-tracking regime -- this is what
		# was tuned/validated against the platform-tracking measurement).
		#
		# Problem this fixes: roll_u = -(kp*offset + kd*flow)*s is summed
		# BEFORE _soft_limit's L*tanh(v/L) saturation. At large offset (e.g.
		# during CENTER, before anything has converged), the compound P+D
		# signal can be several times roll_limit, deep in tanh's saturating
		# region. There, EFFECTIVE gain (d(output)/d(input) = sech^2(v/L))
		# collapses -- e.g. at offset=0.9 with the gains above, effective gain
		# is only ~7% of nominal. The loop becomes nearly deaf to error
		# exactly when the error is largest, which is a standard mechanism for
		# large swings/overshoot severe enough to push the target out of the
		# camera FOV. This is also why raising kd:kp alone barely helped: both
		# terms are summed before the SAME saturating nonlinearity, so no
		# ratio between them changes how hard the compound signal is squashed.
		#
		# Fix: smoothly reduce the gain BEFORE the P+D sum as |offset| grows,
		# so the signal feeding _soft_limit never has to travel as deep into
		# the collapsing-gain region during a large-offset transient. Blended
		# with a raised-cosine (same shape as the D* ramp, for the same
		# reason: a hard switch between gain sets would itself be a
		# derivative-discontinuity kick). Uses the RADIAL offset magnitude
		# (hypot of both axes), not separate per-axis thresholds, since "close
		# to centered" is inherently a 2D notion -- this keeps roll/pitch
		# scaled down together rather than asymmetrically.
		large_offset_gain_scale: float = 0.45,   # multiplier applied at/beyond large_offset_threshold
		small_offset_threshold: float = 0.15,    # |offset| below this: FULL gain (scale=1.0)
		large_offset_threshold: float = 0.55,    # |offset| at/above this: large_offset_gain_scale

		# --- Thrust axis. ---
		thrust_integral_gain_const: float = 0.1,
		divergence_integral_limit: float = 1.2,
		max_visual_thrust_delta_from_hover: float = 0.18,
		raw_divergence_weight: float = 0.10,
		enable_divergence_control: bool = True,
		require_target_for_descent: bool = True,

		# --- Output limits + command shaping (low-pass -> optional slew -> clamp). ---
		# Set enable_slew_rate_limits=False for latency/phase-margin tests: the
		# first-order command filter and hard clamps stay active, but roll/pitch/
		# thrust no longer have per-tick rate limiting. Do NOT turn the numeric
		# slew rates to zero to disable them -- zero means "no motion allowed"
		# in _slew_limit().
		enable_slew_rate_limits: bool = False,
		roll_limit: float = 0.20,
		pitch_limit: float = 0.20,
		thrust_min: float = 0.57,
		thrust_max: float = 0.90,
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

		self._large_offset_gain_scale = max(0.0, float(large_offset_gain_scale))
		self._small_offset_threshold = max(0.0, float(small_offset_threshold))
		self._large_offset_threshold = max(
			self._small_offset_threshold + 1e-6, float(large_offset_threshold)
		)

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
		self._enable_slew_rate_limits = bool(enable_slew_rate_limits)
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
	def slew_rate_limits_enabled(self) -> bool:
		return self._enable_slew_rate_limits

	def set_slew_rate_limits_enabled(self, enabled: bool) -> None:
		"""Enable/disable roll, pitch, and thrust slew-rate limiting at runtime.

		When disabled, _shape_commands still applies the first-order command
		filter and final clamps; it only bypasses the per-axis rate limiter.
		"""
		self._enable_slew_rate_limits = bool(enabled)

	@property
	def divergence_integral(self) -> float:
		return self._divergence_integral

	def reset_visual_integrators(self):
		self._divergence_integral = 0.0
		self._previous_roll_cmd = 0.0
		self._previous_pitch_cmd = 0.0
		self._previous_thrust_cmd = self._hover_thrust
		self._has_previous_command = False

	def reset_divergence_integral(self):
		"""Clear ONLY the divergence integral -- unlike reset_visual_integrators,
		this leaves the command-shaping filter/slew state untouched, so it does
		NOT introduce a step discontinuity in the next commanded roll/pitch/
		thrust. Use this at a mid-flight phase transition (e.g. CENTER->PROBE)
		where you want to discard integral bias accumulated during a noisy
		period without rebasing the live command trajectory to hover/zero.
		reset_visual_integrators() remains the right call for a true cold start
		(no previous command exists yet, e.g. at mission.start())."""
		self._divergence_integral = 0.0

	def compute(
		self,
		target: TargetEstimate,
		flow: FlowResult,
		dt: float,
		*,
		divergence_setpoint: Optional[float] = None,
		thrust_gain_override: Optional[float] = None,
		lateral_p_scale: float = 1.0,
		lateral_d_scale: float = 1.0,
		enable_integral: bool = True,
	) -> AttitudeSetpoint:
		"""Desired roll/pitch/yaw/thrust from visual data only.

		divergence_setpoint: per-call override of D* (constructor value left
		    untouched). 0 for the probe/hover hold, small positive to descend.
		thrust_gain_override: "k" in a_cmd = k*(D - D*). Supplied per-tick by
		    the mission (hand-tuned exploration gain, decayed through descent).
		lateral_p_scale / lateral_d_scale: INDEPENDENT multipliers on the
		    offset (P) and optical-flow (D) terms respectively. Split into two
		    knobs rather than one because kp and kd have, over this project's
		    tuning history, been scaled by DIFFERENT factors relative to
		    earlier validated baselines (e.g. kp x2, kd x sqrt(2) when tuned
		    for platform-oscillation tracking) -- a single shared scale cannot
		    reverse two different historical factors at once, so it can only
		    ever land on the RIGHT kp or the right kd, not both (verified
		    directly: scale=0.5 landed kp exactly on an earlier baseline but
		    left kd under-damped relative to that same baseline). Both apply
		    on top of _offset_magnitude_gain_scale (which still scales P and D
		    together, since that mechanism is about compound-signal saturation,
		    not P/D balance -- see that method's docstring).
		enable_integral: when False, the integral neither accumulates nor
		    contributes to thrust this tick (frozen at its current value, not
		    reset -- see reset_divergence_integral for an explicit clear). Use
		    this during DESCEND: the scheduled gain k(t) deliberately decays
		    toward k_min as height decreases, trading tracking authority for
		    stability margin, so divergence necessarily undershoots D* by a
		    structural (not noise, not a fixable bias) amount for most of the
		    descent. An always-on integral has no way to tell that gap is
		    intentional and keeps accumulating against it for the whole
		    descent, dragging a persistent thrust bias into the final approach
		    when precision matters most. Integral action is for correcting a
		    fixable bias around a stable operating point (e.g. hover during
		    CENTER/PROBE/PROBE_HOLD, where it's left enabled by default) --
		    not for chasing a gap the gain schedule has deliberately decided
		    not to close.
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

			# Error-magnitude blend still scales P and D TOGETHER (compound-
			# signal saturation protection, not a P/D balance concern).
			err_scale = self._offset_magnitude_gain_scale(offset_x, offset_y)
			p_scale = max(0.0, float(lateral_p_scale)) * err_scale
			d_scale = max(0.0, float(lateral_d_scale)) * err_scale
			roll_u = -(self._roll_kp * p_scale * offset_x + self._roll_kd * d_scale * flow_x)
			pitch_u = -(self._pitch_kp * p_scale * offset_y + self._pitch_kd * d_scale * flow_y)

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

			if enable_integral:
				self._divergence_integral = self._clamp(
					self._divergence_integral + error * dt,
					-self._divergence_integral_limit,
					self._divergence_integral_limit,
				)
			# else: frozen at its current value -- neither accumulates nor
			# contributes below (integral_delta is forced to 0.0).

			if thrust_gain_override is not None:
				# Herisse eq. 32 / de Croon accel-domain law: a_cmd = +k*(D-D*).
				# POSITIVE sign because the plant has B<0 (more thrust reduces
				# divergence): arresting a sink (D>D*) needs MORE thrust.
				accel_cmd = float(thrust_gain_override) * error
				proportional_delta = self._hover_thrust * accel_cmd / G_ACCEL
			else:
				proportional_delta = 0.0

			integral_delta = (
				self._thrust_integral_gain_const * self._divergence_integral
				if enable_integral else 0.0
			)
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
		         bypassed entirely when enable_slew_rate_limits=False
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

		if self._enable_slew_rate_limits:
			roll_s = self._slew_limit(self._previous_roll_cmd, roll_f, self._roll_slew_rate_rad_s * dt)
			pitch_s = self._slew_limit(self._previous_pitch_cmd, pitch_f, self._pitch_slew_rate_rad_s * dt)
			thrust_s = self._slew_limit(self._previous_thrust_cmd, thrust_f, self._thrust_slew_rate_per_s * dt)
		else:
			roll_s = roll_f
			pitch_s = pitch_f
			thrust_s = thrust_f

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

	def _offset_magnitude_gain_scale(self, offset_x: float, offset_y: float) -> float:
		"""1.0 (full gain) when the radial offset is small; smoothly falls to
		large_offset_gain_scale as it grows past small_offset_threshold,
		reaching that floor at large_offset_threshold. Raised-cosine blend
		(same shape/reasoning as the D* ramp): zero slope at both ends, no
		derivative-discontinuity kick at either threshold. Applied to BOTH
		lateral_p_scale and lateral_d_scale equally (compound pre-saturation
		signal protection, not a P/D balance concern -- see compute()'s
		docstring for that distinction) -- upstream of _soft_limit."""
		err = math.hypot(float(offset_x), float(offset_y))
		if err <= self._small_offset_threshold:
			return 1.0
		if err >= self._large_offset_threshold:
			return self._large_offset_gain_scale
		span = self._large_offset_threshold - self._small_offset_threshold
		frac = (err - self._small_offset_threshold) / span
		shaped = 0.5 * (1.0 - math.cos(math.pi * frac))
		return 1.0 + (self._large_offset_gain_scale - 1.0) * shaped

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