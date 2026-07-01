"""
Closed-loop visual controller: single-baseline LQR + manual per-altitude trim.

Constraint: once the visual controller is active, commands use ONLY visual data
(target offset, normalized optical flow, divergence, area_fraction). No PX4
position/velocity enters the control law.

Per lateral axis (roll <- x, pitch <- y), the identified open-loop model is
	x[k+1] = A(af) x[k] + B(af) u[k],     x = [offset, flow_norm]^T
with A, B (and the scalar thrust divergence model's a, b) coming from a
SINGLE trusted open-loop measurement (af=0.133, ~2m relative altitude) --
see ROLL_STATE_MODELS/PITCH_STATE_MODELS/THRUST_DIVERGENCE_MODELS' comment
for why the other open-loop knots that used to live there were dropped
rather than kept as a multi-point schedule. solve_discrete_lqr gives the
optimal baseline gain K_lqr = [k_p, k_d] from that one measurement; the
command is
	u = sign * ( -(k_p_eff*offset + k_d_eff*flow) )
with k_p_eff/k_d_eff derived from K_lqr via the manual trim knobs below.
A small optional offset-prediction lead branch can be added on top of this
PD-like visual feedback. It finite-differences the target-center offset, filters
and bounds that trend, then adds k_lead*T_lead*d(offset)/dt to the same feedback
sum. The optical-flow term is kept as the primary damping/velocity feedback;
the lead branch is only a modest phase-advance/prediction correction.
This single baseline is a STARTING POINT only -- it sets the gain SHAPE and
sign/order-of-magnitude at one operating point, not the final numbers at
every altitude.

The trim knobs (roll_prop_scale, pitch_damp_ratio, thrust_gain_scale, ...) are
where altitude-dependent behavior actually lives now, and each one accepts
EITHER:
	a single number     -> applied uniformly at every area_fraction
	[(af, value), ...]   -> a value tuned PER ALTITUDE, interpolated the
							same way ScheduledLQR's table used to be (see
							lqr.ScalarSchedule) -- this is now the ONLY
							place that kind of schedule lives in this file
The intended workflow for the second form is closed-loop, not open-loop:
1. Hover (divergence_setpoint=0) at a fixed altitude, hand-tune
	roll/pitch_prop_scale and damp_ratio against the REAL closed-loop
	response, repeat at each altitude of interest.
2. With the platform's own z_amplitude/z_frequency turned on (see
	bee_platform.sdf's OscillatingPlatformController -- all zero by default),
	repeat in hover to tune thrust_gain_scale/divergence_integral_gain: a
	stationary vehicle hovering over a stationary platform has ~zero relative
	z-motion, so divergence ~0 and the thrust gain has nothing to react
	to -- the platform's own oscillation is what makes the thrust loop's
	dynamics observable without the vehicle itself having to descend.
Collect a few (af, value) pairs this way and pass them straight in -- e.g.
roll_prop_scale/roll_damp_ratio default to a single point at af=0.215 (the
one altitude validated so far); add more points as more altitudes get their
own closed-loop run, the same way roll/pitch's damping was tuned.

Note for anyone re-deriving these numbers from a NEW measurement at a
different af: since there's only one baseline now, changing it changes the
baseline gain EVERYWHERE, not just locally -- a prop_scale/damp_ratio tuned
against the old multi-knot baseline will not transfer exactly (damp_ratio,
being a k_d/k_p RATIO, transfers far more robustly than prop_scale, which
multiplies an absolute k_p that moves with the baseline).

Damping note: in the open-loop fit the optimal k_d can be near zero or even
wrong-signed relative to k_p (pitch's baseline at af=0.133 is ~0 -- see
PITCH_STATE_MODELS' comment), so the flow (velocity) term would barely act,
or act backwards, left alone. damp_ratio (k_d = ratio*k_p) exists to
deliberately set it instead, since it doesn't depend on the LQR's own
unreliable k_d. If the loop still oscillates after raising damping, relax
*_slew_rate/command_filter_alpha: a tight slew rate rate-limits the damping
command itself and reintroduces the lag the damping was meant to remove.

Lateral lead note: roll/pitch still keep the optical-flow term as the true
visual damping path. The lateral lead branch is intentionally separate: it
uses a filtered finite-difference of the target offset to predict the near-
future offset and adds only a small fraction of k_p times that prediction.
That makes it a phase-advance correction rather than a simple replacement for
or duplicate of the optical-flow damping term.

Thrust uses the scalar divergence model d[k+1] = a d[k] + b (thrust - hover),
with LQR feedback on a lead-compensated divergence error plus a small
visual-only integral term. The lead branch is applied only to the fast
proportional/LQR path; the integral still accumulates the true divergence
error so it remains a slow bias corrector.

Past target.fov_saturated (target exceeds the camera's FOV, not just fills
it -- see target_acquisition.py), area_fraction/box geometry are clamped at
the frame's own pixel size and stop tracking true range. _scheduling_area_fraction
freezes the area_fraction fed to every schedule lookup (LQR gain_at() AND the
trim ScalarSchedules) at its last good pre-saturation value once that happens;
the actual control inputs (offset, flow, divergence) keep updating live.
Divergence stays valid through this regime (it's a rate, not a size
measurement), so descent control continues on flow alone with fixed gains
rather than an undefined/frozen schedule input.

Commands are passed through a purely internal shaper (soft saturation ->
first-order filter -> slew limit -> clamp). This uses only previous commands,
never PX4 state, and removes bang-bang excitation of the slow image dynamics.
"""

import math
from typing import Optional

import numpy as np

try:
	from .lqr import ScheduledLQR, ScalarSchedule
	from .state import AttitudeSetpoint, FlowResult, TargetEstimate
except ImportError:
	from lqr import ScheduledLQR, ScalarSchedule
	from state import AttitudeSetpoint, FlowResult, TargetEstimate


# Open-loop model from the reconstructed calibration.
# Entry: (area_fraction, A, B) for [offset[k+1], flow[k+1]]^T = A x + B u.
#
# Single knot, deliberately -- only the af=0.133 (~2m relative altitude)
# measurement is trusted. The other three knots that used to live here
# (0.066, 0.215, 0.511) came from open-loop calibration runs with known
# quality problems (FOV saturation, drift, contact-detection edge cases --
# see calibration_node.py/fit_axis_models.py's quality gates, built
# specifically because of these) and are not trusted enough to anchor the
# LQR's gain SHAPE at those operating points.
#
# With one knot, ScheduledLQR.gain_at() returns this SAME baseline at every
# area_fraction (confirmed: its clamp-to-nearest-knot logic degenerates
# correctly to "always this one" when there's nothing to interpolate
# between -- no special-casing needed). All actual area_fraction-dependent
# behavior now comes entirely from the manual trim ScalarSchedules below
# (roll/pitch_prop_scale, *_damp_ratio, thrust_gain_scale,
# divergence_integral_gain) -- hand-tuned via closed-loop testing at each
# altitude of interest, the same way roll/pitch's damp_ratio values were
# already tuned, rather than read off more open-loop knots of uncertain
# quality. Add more knots back here only if a SPECIFIC af is independently
# re-measured and trusted enough to justify reshaping the baseline itself,
# as opposed to just adding another (af, value) point to a trim schedule.
ROLL_STATE_MODELS = (
	(0.133, [[0.6785, 0.2522], [-0.7866, 0.0246]], [[-0.40704], [-1.06228]]),
)

PITCH_STATE_MODELS = (
	(0.133, [[0.7885, 0.0946], [-0.4753, -0.2675]], [[-0.88419], [-1.93529]]),
)

# Scalar divergence model: d[k+1] = a d[k] + b (thrust[k] - hover).
THRUST_DIVERGENCE_MODELS = (
	(0.133, [[0.9302]], [[-0.1294]]),
)

# Standard gravity, used only to convert between normalized collective thrust
# and world-vertical acceleration for the optional Herisse/de Croon gain
# override path below (compute()'s thrust_gain_override). Kept as a local
# constant rather than importing from mission_routine/estimators so this file
# stays ROS-free and dependency-free on its own, per its visual-only-by-design
# constraint.
G_ACCEL = 9.80665


class ControlLaw:
	def __init__(
		self,
		hover_thrust: float = 0.73,
		yaw_setpoint: float = 0.0,
		divergence_setpoint: float = 0.0,  # 0 = visual hover; raise slowly to descend.

		# --- Baseline LQR cost (gain SHAPE). Larger R -> smaller gains; the
		#     second Q entry weights the flow/velocity state -> damping. ---
		roll_q=((1.0, 0.0), (0.0, 0.25)),
		roll_r=((2.0,),),
		pitch_q=((1.0, 0.0), (0.0, 0.25)),
		pitch_r=((2.0,),),
		thrust_q=((1.0,),),
		thrust_r=((0.9,),),

		# --- Manual gain trim (the experimental surface). Each accepts a
		#     single number (uniform) OR [(area_fraction, value), ...] (tuned
		#     per altitude via closed-loop hover testing -- see module
		#     docstring). prop_scale multiplies the LQR proportional gain k_p.
		#     Damping is set ONE of two ways, per axis:
		#       damp_ratio is not None -> k_d = damp_ratio * k_p  (RECOMMENDED).
		#         Guarantees damping is non-zero and same-signed as k_p
		#         regardless of the baseline's own k_d -- needed because the
		#         open-loop flow row is noisy and the LQR k_d it produces can
		#         be near zero or wrong-signed (see ROLL/PITCH_STATE_MODELS:
		#         pitch's baseline k_d/k_p at the sole af=0.133 knot is ~0).
		#       damp_ratio is None -> k_d = damp_scale * (LQR k_d)  (legacy).
		#
		# Roll: switched to damp_ratio after a closed-loop hover run with the
		# legacy path (prop=0.5, damp_scale=15) showed a persistent, non-
		# decaying ~28.5s oscillation -- the SAME mode pitch had, now
		# destabilized from the opposite direction. damp_ratio=4.5
		# (ratio*omega~1, same target as pitch's fix) confirmed in closed-loop
		# simulation settling cleanly where the old config only sustained.
		# Validated by closed-loop testing at af=0.215 (climb=5.0m, ~3m
		# relative altitude) -- written as a single-point schedule (not a
		# plain float) specifically so it's obvious this is a STARTING point
		# for one altitude, ready to grow into a real schedule as more
		# altitudes get their own closed-loop validation, the same way the
		# baseline model above now leans on these schedules instead of more
		# open-loop knots. A single point behaves identically to a plain
		# float (ScalarSchedule's clamp-to-nearest-knot degenerates the same
		# way ScheduledLQR's does -- see ROLL_STATE_MODELS' comment) --
		# changing nothing here, just making the TODO visible in the code
		# rather than only in a comment.
		roll_prop_scale = [
			(0.05, 2.10),
			(0.10, 1.90),
			(0.215, 1.30),
			(0.45, 0.85),
			(0.70, 0.95),
			(0.85, 0.85),
		],
		roll_damp_scale=1.5,            # inert while roll_damp_ratio is set; kept for the legacy path.
		roll_damp_ratio = [
			(0.05, 3.20),
			(0.10, 3.00),
			(0.215, 2.20),
			(0.45, 1.55),
			(0.70, 1.55),
			(0.85, 1.35),
		],
		# Pitch: damp_ratio=10 confirmed in a long closed-loop hover/descent
		# run (std(offset_y)~0.005, zero saturation) -- working well.
		# CRITICAL, not just recommended, now that ROLL_STATE_MODELS/
		# PITCH_STATE_MODELS collapsed to the single af=0.133 knot: pitch's
		# own LQR k_d/k_p at that exact knot is ~0.0001, i.e. zero baseline
		# damping (the same dead zone flagged in the trim-knob comment
		# above, previously just one knot among several, now the ONLY
		# knot). damp_ratio=None here would mean pitch runs with
		# essentially no damping at ANY area_fraction, not just locally --
		# do not disable without replacing the baseline model's own (A,B).
		pitch_prop_scale = [
			(0.05, 0.78),
			(0.10, 0.62),
			(0.215, 0.42),
			(0.45, 0.22),
			(0.70, 0.14),
			(0.85, 0.12),
		],
		pitch_damp_scale=10.0,          # inert while pitch_damp_ratio is set.
		pitch_damp_ratio = [
			(0.05, 2.60),
			(0.10, 2.40),
			(0.215, 2.10),
			(0.45, 1.80),
			(0.70, 1.40),
			(0.85, 1.25),
		],

		# --- Lateral phase lead / short-horizon offset prediction. ---
		# Keep the optical-flow term as the damping path. This branch uses the
		# finite-differenced target offset, filtered and bounded, to add a small
		# phase-advance correction: k_lead * T_lead * d(offset)/dt, where
		# k_lead = lead_gain_ratio * k_p. Each lead_time/gain_ratio accepts the
		# same scalar-or-schedule format as the other trim knobs.
		roll_lead_time = [
			(0.10, 0.35),
			(0.45, 0.35),
			(0.70, 0.40),
			(0.85, 0.45),
		],
		roll_lead_gain_ratio = [
			(0.10, 0.55),
			(0.45, 0.55),
			(0.70, 0.75),
			(0.85, 0.85),
		],
		pitch_lead_time=0.35,
		pitch_lead_gain_ratio=0.55,
		lateral_lead_filter_alpha: float = 0.85,
		lateral_lead_rate_limit: float = 1.5,
		lateral_lead_correction_limit: float = 0.40,

		# Thrust: still at its original conservative value -- this is the
		# next axis to tune, via hover with the platform's own z oscillation
		# turned on (see module docstring step 2), not yet exercised.
		thrust_gain_scale = [
			(0.10, 1.00),
			(0.215, 0.90),
			(0.45, 0.75),
			(0.70, 0.62),
			(0.85, 0.55),
		],
		divergence_integral_gain = [
			(0.10, 0.020),
			(0.215, 0.018),
			(0.45, 0.015),
			(0.70, 0.012),
			(0.85, 0.010),
		],

		# --- Command limits [rad] / normalized thrust. ---
		roll_limit: float = 0.20,
		pitch_limit: float = 0.20,
		thrust_min: float = 0.62,
		thrust_max: float = 0.84,

		# --- Command shaping. Slew rates relaxed vs the first run so the
		#     damping command is not itself rate-limited away. ---
		roll_slew_rate_rad_s: float = 0.35,
		pitch_slew_rate_rad_s: float = 0.20,
		thrust_slew_rate_per_s: float = 0.40,
		command_filter_alpha: float = 0.85,

		# --- Visual thrust loop. Positive divergence = target expanding =
		#     approach -> increase thrust. Stays purely visual. ---
		enable_divergence_control: bool = True,
		require_target_for_descent: bool = True,
		max_visual_thrust_delta_from_hover: float = 0.18,
		divergence_integral_limit: float = 1.2,
		raw_divergence_weight: float = 0.10,

		# --- Phase lead for the vertical/divergence loop. ---
		# The platform injects a periodic relative vertical motion. Pure P+I on
		# divergence reacts after the oscillation appears; this lead branch adds a
		# filtered derivative of the divergence error to the proportional/LQR path
		# only. Keep the integral on the original error so it remains a slow bias
		# corrector rather than a noise amplifier. divergence_lead_time accepts the
		# same scalar-or-schedule format as the other trim knobs.
		divergence_lead_time=0.25,
		divergence_lead_filter_alpha: float = 0.80,
		divergence_lead_rate_limit: float = 2.0,

		# Sign convention confirmed by closed-loop tests.
		roll_output_sign: float = 1.0,
		pitch_output_sign: float = 1.0,

		# Master switch for ALL phase-lead / offset-prediction branches (both
		# the lateral offset-lead and the divergence-error lead). False ->
		# lateral becomes pure PD (offset P + optical-flow D) and thrust becomes
		# pure PI (divergence P + I), with no derivative/prediction terms at all.
		# Use False to test the Herisse/de Croon gain-bound idea against the
		# simplest possible feedback law; the lead code paths are left intact and
		# merely forced to contribute zero, so this is fully reversible.
		enable_lead_branches: bool = False,

		# --- Simplified constant-gain path (the live path for the bounds test) ---
		# When use_constant_pd is True (default), the lateral axes use ONE fixed
		# PD gain set per axis, NOT the LQR/area_fraction schedule below, and the
		# thrust integral uses one constant gain. The LQR machinery, the
		# identified (A,B) state models, the per-area_fraction ScalarSchedules,
		# and _scheduling_area_fraction() are all left in the file but are no
		# longer in the live path -- kept only as a dormant fallback (set
		# use_constant_pd=False to fall back to them). Rationale: area_fraction
		# saturates at the FOV near touchdown (so a schedule keyed on it goes
		# undefined exactly at the deck), and the previously calibrated gains
		# were tuned at a different platform frequency and are not worth
		# preserving as the schedule shape.
		#
		# UNTUNED PLACEHOLDERS. These four numbers are NOT validated at the probe
		# frequency -- they are order-of-magnitude starting points only. Tune
		# them in hover (HOVER_PROBE_ONLY) before trusting any descent. k_d is
		# the optical-flow (velocity) damping gain; k_p the offset gain.
		use_constant_pd: bool = True,
		roll_kp: float = 0.22,
		roll_kd: float = 0.11,
		pitch_kp: float = 0.18,
		pitch_kd: float = 0.075,
		# Constant thrust integral gain used by the override (probe-driven)
		# thrust path, replacing the per-area_fraction divergence_integral_gain
		# schedule. Slow bias corrector only; keep small.
		thrust_integral_gain_const: float = 0.00,
	):
		self._hover_thrust = float(hover_thrust)
		self._yaw_setpoint = float(yaw_setpoint)
		self._divergence_setpoint = float(divergence_setpoint)

		self._roll_limit = abs(float(roll_limit))
		self._pitch_limit = abs(float(pitch_limit))
		self._thrust_min = float(thrust_min)
		self._thrust_max = float(thrust_max)

		self._roll_output_sign = 1.0 if roll_output_sign >= 0.0 else -1.0
		self._pitch_output_sign = 1.0 if pitch_output_sign >= 0.0 else -1.0
		self._enable_lead = bool(enable_lead_branches)

		# Simplified constant-gain live path (see the constructor docstring).
		self._use_constant_pd = bool(use_constant_pd)
		self._roll_kp = float(roll_kp)
		self._roll_kd = float(roll_kd)
		self._pitch_kp = float(pitch_kp)
		self._pitch_kd = float(pitch_kd)
		self._thrust_integral_gain_const = float(thrust_integral_gain_const)

		# Damp-ratio mode: k_d is synthesized from k_p in compute(), bypassing
		# the unreliable LQR k_d entirely. Otherwise k_d = damp_scale * LQR k_d.
		self._roll_damp_ratio = None if roll_damp_ratio is None else ScalarSchedule(roll_damp_ratio)
		self._pitch_damp_ratio = None if pitch_damp_ratio is None else ScalarSchedule(pitch_damp_ratio)
		self._roll_prop_scale = ScalarSchedule(roll_prop_scale)
		self._roll_damp_scale = ScalarSchedule(roll_damp_scale)
		self._pitch_prop_scale = ScalarSchedule(pitch_prop_scale)
		self._pitch_damp_scale = ScalarSchedule(pitch_damp_scale)
		self._roll_lead_time = ScalarSchedule(roll_lead_time)
		self._roll_lead_gain_ratio = ScalarSchedule(roll_lead_gain_ratio)
		self._pitch_lead_time = ScalarSchedule(pitch_lead_time)
		self._pitch_lead_gain_ratio = ScalarSchedule(pitch_lead_gain_ratio)
		self._lateral_lead_filter_alpha = self._clamp(lateral_lead_filter_alpha, 0.0, 1.0)
		self._lateral_lead_rate_limit = abs(float(lateral_lead_rate_limit))
		self._lateral_lead_correction_limit = abs(float(lateral_lead_correction_limit))
		self._previous_roll_offset: Optional[float] = None
		self._previous_pitch_offset: Optional[float] = None
		self._filtered_roll_offset_rate = 0.0
		self._filtered_pitch_offset_rate = 0.0
		self._thrust_gain_scale = ScalarSchedule(thrust_gain_scale)

		# Raw LQR baseline gains -- the manual trim knobs above are applied at
		# RUNTIME in compute() (via the ScalarSchedules), not baked in here, so
		# each one can vary with area_fraction (tuned per altitude) rather than
		# being a single constant across the whole table.
		self._roll_lqr = ScheduledLQR(self._schedule(ROLL_STATE_MODELS, roll_q, roll_r))
		self._pitch_lqr = ScheduledLQR(self._schedule(PITCH_STATE_MODELS, pitch_q, pitch_r))
		self._thrust_lqr = ScheduledLQR(self._schedule(THRUST_DIVERGENCE_MODELS, thrust_q, thrust_r))

		self._roll_slew_rate_rad_s = abs(float(roll_slew_rate_rad_s))
		self._pitch_slew_rate_rad_s = abs(float(pitch_slew_rate_rad_s))
		self._thrust_slew_rate_per_s = abs(float(thrust_slew_rate_per_s))
		self._command_filter_alpha = self._clamp(command_filter_alpha, 0.0, 1.0)

		self._enable_divergence_control = bool(enable_divergence_control)
		self._require_target_for_descent = bool(require_target_for_descent)
		self._max_visual_thrust_delta = abs(float(max_visual_thrust_delta_from_hover))
		self._divergence_integral_gain = ScalarSchedule(divergence_integral_gain)
		self._divergence_integral_limit = abs(float(divergence_integral_limit))
		self._raw_divergence_weight = self._clamp(raw_divergence_weight, 0.0, 1.0)
		self._divergence_integral = 0.0

		self._divergence_lead_time = ScalarSchedule(divergence_lead_time)
		self._divergence_lead_filter_alpha = self._clamp(divergence_lead_filter_alpha, 0.0, 1.0)
		self._divergence_lead_rate_limit = abs(float(divergence_lead_rate_limit))
		self._previous_divergence_error: Optional[float] = None
		self._filtered_divergence_error_rate = 0.0

		self._previous_roll_cmd = 0.0
		self._previous_pitch_cmd = 0.0
		self._previous_thrust_cmd = self._hover_thrust
		self._has_previous_command = False

		# See _scheduling_area_fraction(): the last area_fraction seen before
		# fov_saturated, latched and reused for gain_at() while saturated.
		self._frozen_area_fraction: Optional[float] = None

	@property
	def hover_thrust(self) -> float:
		return self._hover_thrust

	@property
	def divergence_integral(self) -> float:
		return self._divergence_integral

	@property
	def frozen_area_fraction(self) -> Optional[float]:
		"""Latched pre-saturation area_fraction, or None if never latched/active."""
		return self._frozen_area_fraction

	def reset_visual_integrators(self):
		self._divergence_integral = 0.0
		self._previous_divergence_error = None
		self._filtered_divergence_error_rate = 0.0
		self._previous_roll_offset = None
		self._previous_pitch_offset = None
		self._filtered_roll_offset_rate = 0.0
		self._filtered_pitch_offset_rate = 0.0
		self._previous_roll_cmd = 0.0
		self._previous_pitch_cmd = 0.0
		self._previous_thrust_cmd = self._hover_thrust
		self._has_previous_command = False
		self._frozen_area_fraction = None

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

		divergence_setpoint: if given, overrides the constructor's D* for THIS
		    call only (the constructor value is left untouched, so a caller that
		    never passes this argument sees no behavior change at all). Used by
		    the mission routine to switch between the D*=0 probe hold and the
		    constant-divergence descent without re-constructing ControlLaw.

		thrust_gain_override: if given, REPLACES the baseline-LQR thrust path
		    (thrust_gain_scale * baseline LQR gain) with a direct Herisse
		    (2012, eq. 32)/de Croon-style proportional law in ACCELERATION
		    units:
		        a_cmd = -thrust_gain_override * lead_error      [m/s^2]
		        thrust_delta = hover_thrust * a_cmd / G_ACCEL    [normalized]
		    i.e. thrust_gain_override is "k" in m/s, exactly the quantity
		    Herisse's floor (eq. 33) and de Croon's ceiling (eq. 25, K_cr=2Z/dt)
		    both bound -- the mission routine computes k once from a probe and
		    schedules it down through descent, and this is the injection point.
		    The lead-compensated proportional path is replaced; the divergence
		    INTEGRAL term (and its own schedule/limit) is untouched and still
		    adds on top, exactly as in the legacy path, so the override only
		    replaces the fast term, not the slow bias corrector.
		    None (default) -> identical to the old behavior: the baseline LQR
		    thrust gain (LQR * thrust_gain_scale schedule) is used, unchanged.

		lateral_gain_scale: multiplies the constant lateral PD gains this call
		    (only when use_constant_pd is True). The mission routine sets this to
		    k(t)/k_explore so the lateral axes ride the SAME height ramp as the
		    thrust gain (de Croon 2016 App. B: the ventral-flow loop has the same
		    2Z/dt ceiling as the divergence loop, so the same schedule shape is
		    the correct one for both). 1.0 (default) -> full lateral gains.
		"""
		dt = max(1e-3, float(dt))

		roll_cmd = 0.0
		pitch_cmd = 0.0
		visual_thrust_delta = 0.0

		area_fraction = self._safe_area_fraction(target)
		flow_valid = flow is not None and bool(getattr(flow, "valid", False))
		target_found = target is not None and bool(getattr(target, "found", False))

		scheduling_area_fraction = self._scheduling_area_fraction(target, target_found, area_fraction)

		# --- Lateral axes. ---
		if target_found:
			flow_x = float(getattr(flow, "mean_flow_x_norm", 0.0)) if flow_valid else 0.0
			flow_y = float(getattr(flow, "mean_flow_y_norm", 0.0)) if flow_valid else 0.0
			offset_x = float(target.offset_x)
			offset_y = float(target.offset_y)

			if self._use_constant_pd:
				# Live path: one constant PD gain set per axis (offset P +
				# optical-flow D), scaled by lateral_gain_scale so the lateral
				# gains ride the same height ramp as the thrust gain. No LQR, no
				# area_fraction, no lead term. K_eff = lateral_gain_scale * K_0.
				s = max(0.0, float(lateral_gain_scale))
				roll_u = -(self._roll_kp * offset_x + self._roll_kd * flow_x) * s
				pitch_u = -(self._pitch_kp * offset_y + self._pitch_kd * flow_y) * s
			else:
				# Dormant fallback: the original LQR + per-area_fraction schedule.
				roll_lead_correction = self._lead_offset_correction(
					"roll", offset_x, dt, scheduling_area_fraction
				)
				pitch_lead_correction = self._lead_offset_correction(
					"pitch", offset_y, dt, scheduling_area_fraction
				)
				roll_u = self._axis_command(
					self._roll_lqr.gain_at(scheduling_area_fraction), offset_x, flow_x,
					self._roll_prop_scale.value_at(scheduling_area_fraction),
					self._roll_damp_scale.value_at(scheduling_area_fraction),
					self._roll_damp_ratio.value_at(scheduling_area_fraction) if self._roll_damp_ratio else None,
					roll_lead_correction,
					self._roll_lead_gain_ratio.value_at(scheduling_area_fraction),
				)
				pitch_u = self._axis_command(
					self._pitch_lqr.gain_at(scheduling_area_fraction), offset_y, flow_y,
					self._pitch_prop_scale.value_at(scheduling_area_fraction),
					self._pitch_damp_scale.value_at(scheduling_area_fraction),
					self._pitch_damp_ratio.value_at(scheduling_area_fraction) if self._pitch_damp_ratio else None,
					pitch_lead_correction,
					self._pitch_lead_gain_ratio.value_at(scheduling_area_fraction),
				)

			# Smooth saturation toward the limit (not a hard clip).
			roll_cmd = self._soft_limit(self._roll_output_sign * roll_u, self._roll_limit)
			pitch_cmd = self._soft_limit(self._pitch_output_sign * pitch_u, self._pitch_limit)
		else:
			# No target center: reset the lateral prediction memory so reacquisition
			# cannot create a finite-difference spike.
			self._previous_roll_offset = None
			self._previous_pitch_offset = None
			self._filtered_roll_offset_rate *= 0.90
			self._filtered_pitch_offset_rate *= 0.90

		# --- Thrust axis: feedback on divergence error + visual integral. ---
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

			lead_error = self._lead_compensated_divergence_error(
				error, dt, scheduling_area_fraction
			)

			if thrust_gain_override is not None:
				# Direct Herisse/de Croon accel-domain law:
				#     a_cmd = k * (D - D*)   (Herisse eq. 32: thrust rises with
				#                              positive divergence error).
				# The sign is POSITIVE, not the naive regulator -k*e, because the
				# identified plant has B<0 (more thrust REDUCES divergence), so
				# arresting an approach (D>D*, sinking) requires MORE thrust. The
				# legacy LQR path reaches the same +sign via a negative LQR gain
				# (K<0 since B<0); this override must match it explicitly. A
				# negative sign here inverts the vertical loop into positive
				# feedback on sink -- the drone descends when it should hold.
				accel_cmd = float(thrust_gain_override) * lead_error
				lqr_delta = self._hover_thrust * accel_cmd / G_ACCEL
				integral_gain = self._thrust_integral_gain_const
			else:
				thrust_gain_scale = self._thrust_gain_scale.value_at(scheduling_area_fraction)
				baseline_thrust_gain = float(self._thrust_lqr.gain_at(scheduling_area_fraction)[0, 0])
				# Lead compensation is deliberately applied only to the fast LQR/P path.
				# The integral below keeps using the true error accumulated above.
				lqr_delta = -(thrust_gain_scale * baseline_thrust_gain) * lead_error
				integral_gain = self._divergence_integral_gain.value_at(scheduling_area_fraction)

			integral_delta = integral_gain * self._divergence_integral
			visual_thrust_delta = self._soft_limit(
				lqr_delta + integral_delta, self._max_visual_thrust_delta
			)
		else:
			# No visual measurement: decay (don't hard-reset) so one dropped
			# frame is not a discontinuity, while stale info is forgotten. Reset
			# the lead memory so a later reacquisition does not create a derivative spike.
			self._divergence_integral *= 0.90
			self._previous_divergence_error = None
			self._filtered_divergence_error_rate *= 0.90

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

	def _scheduling_area_fraction(self, target: TargetEstimate, target_found: bool, area_fraction: float) -> float:
		"""
		area_fraction to feed gain_at(), latching it once the target saturates
		the camera's field of view.

		Once target.fov_saturated is True, the target's true size exceeds the
		frame, and area_fraction/detection box are clamped at the image's own
		pixel dimensions (cv2.boundingRect cannot report bigger than the
		array it's computed on) -- they stop tracking true range entirely.
		Divergence does NOT have this problem: it is a per-pixel velocity
		gradient, not a size measurement, so it stays meaningful through the
		same regime (confirmed: real Farneback recovery on a textured target
		holds at 95-100% of true divergence at this range, see optical_flow.py).

		So once saturated: freeze the SCHEDULING variable (every gain_at()
		call below uses this), while the actual control inputs (offset, flow,
		divergence) keep updating live every tick as normal. This is an
		explicit latch, not reliance on ScheduledLQR's incidental clamp-to-
		endpoint behavior for out-of-range values -- that clamp only protects
		the controller today because the schedule's top knot happens to sit
		below the saturation point; it is not a designed connection between
		the two, and would silently stop protecting anything if the schedule
		were ever recalibrated with a knot placed past saturation.

		The unfreeze direction also matters: when fov_saturated clears (e.g.
		a moving platform), this resumes tracking the live area_fraction on
		the very next non-saturated sample -- it does not stay latched once
		the geometry is informative again.
		"""
		if target_found and not bool(getattr(target, "fov_saturated", False)):
			self._frozen_area_fraction = area_fraction
			return area_fraction

		if self._frozen_area_fraction is not None:
			return self._frozen_area_fraction

		return area_fraction  # never seen a good sample yet -- best available.

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

	def _lead_offset_correction(
		self, axis: str, offset: float, dt: float, scheduling_area_fraction: float
	) -> float:
		"""Return T_lead * filtered(d offset / dt) for roll/pitch prediction.

		This branch is deliberately separate from the optical-flow damping term.
		Optical flow remains the visual velocity feedback; this method uses the
		target-center trend as a bounded short-horizon offset predictor. On the
		first valid sample after reset/reacquisition, it returns zero so the
		controller does not kick from an artificial derivative.
		"""
		offset = float(offset)
		dt = max(1e-3, float(dt))

		# Lead disabled: contribute nothing, so the lateral law is pure PD
		# (offset P + optical-flow D). Memory is left untouched; it simply is
		# not consulted while disabled.
		if not self._enable_lead:
			return 0.0

		if axis == "roll":
			previous = self._previous_roll_offset
			lead_time = self._roll_lead_time.value_at(scheduling_area_fraction)
			filtered_rate = self._filtered_roll_offset_rate
		elif axis == "pitch":
			previous = self._previous_pitch_offset
			lead_time = self._pitch_lead_time.value_at(scheduling_area_fraction)
			filtered_rate = self._filtered_pitch_offset_rate
		else:
			raise ValueError(f"unknown lateral lead axis: {axis!r}")

		if previous is None:
			if axis == "roll":
				self._previous_roll_offset = offset
				self._filtered_roll_offset_rate = 0.0
			else:
				self._previous_pitch_offset = offset
				self._filtered_pitch_offset_rate = 0.0
			return 0.0

		raw_rate = (offset - previous) / dt
		if self._lateral_lead_rate_limit > 0.0:
			raw_rate = self._clamp(
				raw_rate,
				-self._lateral_lead_rate_limit,
				self._lateral_lead_rate_limit,
			)

		alpha = self._lateral_lead_filter_alpha
		filtered_rate = alpha * filtered_rate + (1.0 - alpha) * raw_rate
		correction = lead_time * filtered_rate
		if self._lateral_lead_correction_limit > 0.0:
			correction = self._clamp(
				correction,
				-self._lateral_lead_correction_limit,
				self._lateral_lead_correction_limit,
			)

		if axis == "roll":
			self._previous_roll_offset = offset
			self._filtered_roll_offset_rate = filtered_rate
		else:
			self._previous_pitch_offset = offset
			self._filtered_pitch_offset_rate = filtered_rate

		return correction

	def _lead_compensated_divergence_error(
		self, error: float, dt: float, scheduling_area_fraction: float
	) -> float:
		"""Return e + T_lead * filtered(de/dt) for the thrust P/LQR path.

		This is a bounded derivative / phase-lead branch, not an integral
		replacement. On the first valid sample after reset/reacquisition, it
		returns the raw error so the controller does not kick from an artificial
		derivative.
		"""
		error = float(error)
		dt = max(1e-3, float(dt))

		# Lead disabled: return the raw error so the thrust path is pure P+I
		# (the override's a_cmd = +k*(D-D*), plus the unchanged integral term).
		if not self._enable_lead:
			return error

		lead_time = self._divergence_lead_time.value_at(scheduling_area_fraction)

		if self._previous_divergence_error is None:
			self._previous_divergence_error = error
			self._filtered_divergence_error_rate = 0.0
			return error

		raw_rate = (error - self._previous_divergence_error) / dt
		self._previous_divergence_error = error

		if self._divergence_lead_rate_limit > 0.0:
			raw_rate = self._clamp(
				raw_rate,
				-self._divergence_lead_rate_limit,
				self._divergence_lead_rate_limit,
			)

		alpha = self._divergence_lead_filter_alpha
		self._filtered_divergence_error_rate = (
			alpha * self._filtered_divergence_error_rate
			+ (1.0 - alpha) * raw_rate
		)

		return error + lead_time * self._filtered_divergence_error_rate

	def _divergence_for_control(self, flow: FlowResult) -> float:
		"""Blend filtered and raw divergence: (1-w) d_filt + w d_raw."""
		filtered = self._safe_float(getattr(flow, "divergence", 0.0))
		raw = self._safe_float(getattr(flow, "raw_divergence", filtered), default=filtered)
		w = self._raw_divergence_weight
		return (1.0 - w) * filtered + w * raw

	@staticmethod
	def _schedule(models, q, r):
		"""Expand (af, A, B) models into ScheduledLQR (af, A, B, Q, R) tuples."""
		return ((af, A, B, q, r) for af, A, B in models)

	@staticmethod
	def _axis_command(
		baseline_gain: np.ndarray, offset: float, flow: float,
		prop_scale: float, damp_scale: float, damp_ratio: Optional[float],
		lead_correction: float = 0.0, lead_gain_ratio: float = 0.0,
	) -> float:
		"""
		Lateral feedback
			u = -(k_p_eff*offset + k_d_eff*flow + k_lead*lead_correction)
		with k_p_eff = prop_scale * baseline_gain[0,0], k_d_eff synthesized
		from damp_ratio when enabled, and k_lead = lead_gain_ratio*k_p_eff.

		lead_correction has offset units: T_lead * filtered(d offset / dt).
		It is a small prediction/phase-advance term, while flow remains the
		primary visual damping signal.
		"""
		k_p = prop_scale * float(baseline_gain[0, 0])
		k_d = damp_ratio * k_p if damp_ratio is not None else damp_scale * float(baseline_gain[0, 1])
		k_lead = lead_gain_ratio * k_p
		return -(k_p * offset + k_d * flow + k_lead * lead_correction)

	@staticmethod
	def _safe_area_fraction(target: TargetEstimate) -> float:
		if target is None:
			return 0.066
		try:
			return max(1e-4, float(getattr(target, "area_fraction", 0.066)))
		except (TypeError, ValueError):
			return 0.066

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