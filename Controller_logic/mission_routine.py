"""
Mission routine: PROBE -> DESCEND, the minimal version.

CLOCK: every timer in this module (center_dwell_sec, center_timeout_sec,
probe_min_duration_sec, d_star_ramp_in_sec, the gain schedule's elapsed time,
h_pred) is measured on ONE clock: `t`, the argument passed into update(), which
bee_node.py feeds from the vision/image timestamp -- i.e. SIMULATION time, not
wall-clock. This is deliberate, not an oversight: the gain schedule and h_pred
are derived from h(t)=h0*exp(-D* t), a statement about physical descent
dynamics, and the physics itself evolves in lockstep with sim time -- so
tying the WHOLE mission state machine to that same clock keeps every timer
physically consistent with the dynamics it reasons about, rather than each
using a different, disconnected notion of "time passing."

The consequence: under a simulator real-time-factor (RTF) < 1 (Gazebo running
slower than real time -- common under render/physics load), every one of
these "N seconds" parameters takes proportionally longer in WALL-clock terms
(elapsed_wall ~= N / RTF). A `probe_min_duration_sec=15.0` at RTF=0.2 is not
"15 seconds" to a human watching the clock; it is closer to 75. This is not a
bug and should NOT be fixed by switching these timers to wall-clock (that
would just reintroduce the exact problem the D* ramp was built to avoid: a
physically-meaningful trajectory parameter decoupled from the dynamics it
describes). See bee_node.py's EXPECTED_SIM_RTF for a log-only wall-time
estimate that makes this visible without touching any control-relevant math.

The ONE place a wall-clock quantity legitimately belongs is the de Croon
feasibility gate's dt (see MissionRoutine's stability_dt_sec) -- that
represents real physical dead-time between a sensor sample and the correction
it drives, which is a property the safety margin must hold on real hardware
(RTF=1 always, by definition), not a property of any particular sim run.
stability_dt_sec is therefore DELIBERATELY decoupled from both control_period_sec
(the ROS actuation loop's rate) and from this module's SIM-time mission clock.

This replaces the earlier (feedforward + phase-lock + EKF + mode-estimator)
design with the simplest thing that is still scientifically justified end to
end. The whole controller is three numbers, computed once, plus one clamped
formula evaluated every tick:

1. PROBE: hold divergence_setpoint=0 (true visual hover) for a fixed window.
	ControlLaw's OWN existing PI-on-divergence loop does the holding; this
	module does not add a new control path for the probe, it only WATCHES the
	thrust command that loop produces. Because the hold keeps the closing
	rate near zero, the loop's commanded acceleration is, each tick,
	approximately the platform's own vertical acceleration (it has to be,
	to keep the gap from changing) -- this is exactly the "thrust as a
	stand-in for the disturbance" reading used by Bergantin/de Croon, applied
	directly to efference instead of going through divergence at all. No
	dither, no frequency estimation, no observability subtlety: we are not
	trying to estimate height or phase here, only the SIZE of the worst
	acceleration the deck threw at the loop.

2. GATE: from peak_accel and the descent divergence setpoint D*, compute
		k_min  = peak_accel / D*                    (Herisse 2012 eq. 33,
														floor, disturbance/
														ground-effect terms
														dropped -- see below)
		h_crit = k_min * control_period_sec / 2      (de Croon 2016 eq. 25,
														K_cr=2Z/dt, inverted)
	h_crit is the height below which ANY fixed gain must drop below the
	Herisse floor to stay under the de Croon ceiling -- the two analytic
	bounds the whole project has been about. If h_crit exceeds the vehicle's
	own leg clearance, there is no constant gain that is both stabilizing and
	non-oscillatory all the way to contact at this control rate: report
	infeasible and hold. Otherwise, proceed.

3. DESCEND: schedule the thrust gain as an EXPLICIT FUNCTION OF TIME -- the
	descent reads a stopwatch, never a height estimate:
		K(t) = clamp( k_explore * exp(-D* * t),  k_min,  k_explore )
	where t is the elapsed time since descent began. This is identical to
	clamp(safety*2*h(t)/dt, ...) with the open-loop trajectory
	h(t)=h0*exp(-D* t) (Herisse section V / Ho eq. 3) substituted in and the
	height cancelled out: h0 enters ONCE, frozen inside
	k_explore = safety*2*h0/dt at descent start. So the live path needs only
	the clock and the two frozen constants k_min, k_explore -- no runtime
	height, divergence, or integrator. It is the Ho eq. 20 /
	de Croon gain-proportional-to-height law written on a stopwatch, with the
	k_min clamp (Herisse floor) added because neither faced a moving deck.
	The lateral axes ride the same ramp, normalized to k(t)/k_explore.

	The COMMANDED D* itself ramps smoothly (raised-cosine, zero slope at both
	endpoints) from 0 over d_star_ramp_in_sec at
	descent entry, rather than stepping instantly to its target. The step was
	found to be the actual source of a "violent" transient at PROBE->DESCEND:
	k(t) was already smooth, but the error (D_measured - D*) snapping from ~0
	to the full target in one tick drove a sharp thrust cut (and, with a
	nonzero thrust_integral_gain_const, risked integral saturation on that same
	step). See the constructor's d_star_ramp_in_sec docstring for why this is
	applied only to the commanded setpoint, not to k(t)'s decay rate.

	CLOCK-FOLLOWING TRADE (accepted for now). Scheduling on time assumes the
	real descent tracks h(t). If it LAGS (you are higher than predicted), the
	clock has decayed the gain early -> conservative, safe-but-slow. If it runs
	AHEAD (you are lower than predicted), K is too high for the true height ->
	toward de Croon oscillation, backstopped by the k_min clamp. A future
	refinement (planned) re-anchors k_safety de-Croon-style by following the
	oscillation and raising gain until relative motion grows; until then the
	naive clock is accepted. critical_time() gives t_crit = (1/D*)ln(h0/h_crit)
	for an optional descent-duration cap (abort/re-probe if touchdown has not
	fired well past it). height_prediction.png (predicted h(t) vs SITL ground
	truth) is the instrument that says whether the clock assumption is holding.

NOT estimated here, by design: platform frequency, phase, or amplitude;
live height (h0 is a one-time seed, not tracked online -- see the seeding note
below); any feedforward. All of that is available to add back later (the
earlier estimators.py sketch is not lost, just not part of this minimal
version) once this version is flying and its assumptions have been checked
against real logs.

WHAT THE FLOOR DROPS, HONESTLY. Herisse's full condition (eq. 33) is
	k > (|delta|_max + m|z_ddot_G|_max + mg|b_max-1|) / (m * omega*)
i.e. unmodeled disturbance and ground-effect terms ride along with the
platform's own acceleration. This module keeps only the platform term
(m|z_ddot_G|_max / m = peak_accel), which is the dominant one away from the
deck and the only one this probe can actually measure. If ground effect or a
known disturbance bound matters for your airframe, pad peak_accel before
calling compute_gate(), or extend it explicitly -- don't silently assume zero.

H0 SEEDING (the one open question flagged for later). h0 is currently passed
in by the caller (bee_node) as the commanded takeoff altitude -- a known
constant at the moment visual control takes over, not a live measurement, so
reading it does not reopen the visual-only constraint on the control LAW (only
on this one-time mission seed). The open-loop height prediction below is only
as good as that seed and as good as D* actually being tracked; see the module
docstring in bee_node.py's patch for the calibration note.

CONTROL-RATE CAVEAT. h_crit = k_min * control_period_sec / 2 is directly
proportional to your control PERIOD. At CONTROL_PERIOD_SEC=0.5 (2 Hz) it is
7.5x larger than it would be at Herisse's own 15 Hz hardware rate for the same
platform violence -- meaning this gate will legitimately flag many realistic
"rough platform" scenarios as infeasible at the current control rate. That is
not a bug in the gate; it is the gate doing its job and surfacing the same
bandwidth ceiling this whole design has been about. Raising CONTROL_PERIOD_SEC
(running compute() faster) is the first lever if real platforms keep failing
this check.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


G_ACCEL = 9.80665

CENTER = "center"
PROBE = "probe"
PROBE_HOLD = "probe_hold"
DESCEND = "descend"
INFEASIBLE = "infeasible"


# --------------------------------------------------------------------------- #
#  Thrust <-> acceleration (the efference copy)                                #
# --------------------------------------------------------------------------- #
class ThrustModel:
	"""a_drone = g * (u / u_hover - 1), world-up, tilt ignored (first version)."""

	def __init__(self, hover_thrust: float, g: float = G_ACCEL):
		self._u_hover = max(1e-3, float(hover_thrust))
		self._g = float(g)

	def accel_from_thrust(self, u: float) -> float:
		return self._g * (float(u) / self._u_hover - 1.0)


# --------------------------------------------------------------------------- #
#  Probe: peak platform acceleration from thrust efference                     #
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
	peak_accel: float = 0.0
	n_samples: int = 0
	duration_sec: float = 0.0
	ready: bool = False


class PlatformProbe:
	"""Watches thrust commands during a D*=0 hold; reports peak |acceleration|.

	High-passes with a running EMA mean before tracking the peak, so a
	mis-calibrated hover_thrust (a slow bias, not a platform oscillation)
	does not inflate the estimate -- only deviations from the hold's own
	running mean count as "the platform moved".

	Peak, not a percentile: simplest possible first version. A single noisy
	spike only makes the gate MORE conservative (raises k_min, which is the
	safe direction), so this is an acceptable place to start; consider a
	robust percentile (e.g. P95) later if probe noise causes too many false
	"infeasible" calls.
	"""

	def __init__(self, thrust_model: ThrustModel, highpass_tau_sec: float = 2.5):
		# highpass_tau_sec sets the EMA time constant used to subtract the
		# slow mean before tracking peak |accel|. It must be long enough to
		# remove the thrust loop's own slow resonance (documented ~23s ring in
		# platform_motion.py) while short enough to pass the platform's actual
		# oscillation. 4s was too short: the 23s ring aliased into peak_accel,
		# inflating k_min and making the gate falsely infeasible on a stationary
		# platform. 15s sits between the two periods and removes the ring.
		# If the real platform oscillates slower than ~5s, raise this further.
		self._tm = thrust_model
		self._tau = float(highpass_tau_sec)
		self._mean = 0.0
		self._has_mean = False
		self._peak = 0.0
		self._n = 0
		self._elapsed = 0.0

	def reset(self) -> None:
		self._mean = 0.0
		self._has_mean = False
		self._peak = 0.0
		self._n = 0
		self._elapsed = 0.0

	def update(self, thrust_cmd: float, dt: float) -> None:
		dt = max(1e-3, float(dt))
		a = self._tm.accel_from_thrust(thrust_cmd)

		if not self._has_mean:
			self._mean = a
			self._has_mean = True
		else:
			alpha = math.exp(-dt / max(1e-3, self._tau))
			self._mean = alpha * self._mean + (1.0 - alpha) * a

		ac = abs(a - self._mean)
		self._peak = max(self._peak, ac)
		self._n += 1
		self._elapsed += dt

	def result(self, min_duration_sec: float) -> ProbeResult:
		return ProbeResult(
			peak_accel=self._peak,
			n_samples=self._n,
			duration_sec=self._elapsed,
			ready=self._elapsed >= min_duration_sec,
		)


# --------------------------------------------------------------------------- #
#  Gate + gain schedule (Herisse floor / de Croon ceiling)                     #
# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
	k_min: float = 0.0
	h_crit: float = 0.0
	k_explore: float = 0.0
	feasible: bool = False


def critical_height(k_min: float, control_period_sec: float, safety: float = 1.0) -> float:
	"""Height where the (safety-scaled) de Croon ceiling drops to k_min.

	Live gain is safety*2h/dt, so it reaches k_min at h = k_min*dt/(2*safety).
	A smaller safety (more conservative, gain held further below the ceiling)
	makes h_crit LARGER -- you hit the floor higher up -- which correctly makes
	the feasibility gate stricter.
	"""
	s = max(1e-3, float(safety))
	return k_min * float(control_period_sec) / (2.0 * s)


def compute_gate(
	peak_accel: float,
	descent_divergence_setpoint: float,
	initial_thrust_gain: float,
	control_period_sec: float,
	leg_clearance_m: float,
	ceiling_safety_factor: float = 0.5,
	min_divergence_setpoint: float = 0.01,
) -> GateResult:
	"""Herisse floor -> k_min; de Croon ceiling (safety-scaled), inverted ->
	h_crit; gate against leg clearance.

	k_explore is now the HAND-TUNED initial thrust gain (initial_thrust_gain),
	NOT the height-derived de Croon ceiling. Rationale: the ATTITUDE_HOLD height
	reference is relative to the ground while the platform sits ~2 m above it, so
	h0 (takeoff altitude) is not the true height above the deck and the
	ceiling-at-h0 gain was wrong -- it needed an ad-hoc divisor to fly. The
	exploration/descent-start gain is therefore set by hand like the lateral PD
	gains. The probe still sets the k_min FLOOR and the h_crit feasibility gate;
	only the START gain is decoupled from height."""
	d_star = max(float(min_divergence_setpoint), float(descent_divergence_setpoint))
	s = max(1e-3, float(ceiling_safety_factor))
	k_min = max(0.0, float(peak_accel)) / d_star
	h_crit = critical_height(k_min, control_period_sec, s)
	k_explore = max(0.0, float(initial_thrust_gain))
	feasible = h_crit <= float(leg_clearance_m)
	return GateResult(k_min=k_min, h_crit=h_crit, k_explore=k_explore, feasible=feasible)


def scheduled_gain_at_time(
	elapsed_sec: float,
	descent_divergence_setpoint: float,
	k_min: float,
	k_explore: float,
	safety: float = 0.5,
) -> float:
	"""The live descent gain, as an EXPLICIT FUNCTION OF TIME -- no height.

		K(t) = clamp( k_explore * exp(-D* * t),  k_min,  k_explore )

	This is exactly clamp(safety*2*h(t)/dt, ...) with h(t)=h0*exp(-D* t)
	substituted in, but written so it is obvious that the only live input is
	the elapsed-time clock since descent began. h0 enters once, frozen inside
	k_explore (= safety*2*h0/dt, computed at descent start); the schedule reads
	NO height estimate at runtime. It is the Ho eq. 20 / de Croon
	gain-proportional-to-height law, expressed on a stopwatch.

	Safe-but-slow by construction: if the real descent lags the ideal
	exponential, the clock has decayed the gain more than the true height
	warrants, so K is conservatively low. The dangerous case (descending AHEAD
	of prediction) is backstopped by the k_min clamp, below which the gain stops
	following the exponential and holds the Herisse floor.
	"""
	decay = math.exp(-float(descent_divergence_setpoint) * max(0.0, float(elapsed_sec)))
	return max(float(k_min), min(float(k_explore), float(k_explore) * decay))


def critical_time(
	h0: float, descent_divergence_setpoint: float, h_crit: float
) -> float:
	"""Elapsed descent time at which K(t) reaches the k_min floor, i.e. when the
	predicted height crosses h_crit:  t_crit = (1/D*) * ln(h0 / h_crit).

	Known BEFORE descent from frozen constants. Use it to cap descent duration:
	if touchdown has not fired by some margin past t_crit, the open-loop clock
	has drifted from the true descent (you are higher than predicted) -- a clean
	place to abort or re-probe rather than keep trusting a stale h0. Returns inf
	if h_crit is non-positive or >= h0 (floor never reached on the way down)."""
	d = float(descent_divergence_setpoint)
	if h_crit <= 0.0 or h_crit >= h0 or d <= 1e-9:
		return float("inf")
	return (1.0 / d) * math.log(float(h0) / float(h_crit))


def predicted_height(h0: float, descent_divergence_setpoint: float, elapsed_sec: float) -> float:
	"""h(t) = h0 * exp(-D* t): the exact trajectory of a perfectly-tracked
	constant-divergence descent (Herisse section V / Ho eq. 3).

	DERIVATION / DIAGNOSTIC ONLY -- not used by the live gain schedule, which is
	now a pure function of time (see scheduled_gain_at_time). Kept so the log
	can record the height the clock-based schedule IMPLIES, for offline
	comparison against SITL ground truth (height_prediction.png)."""
	return max(0.0, float(h0)) * math.exp(-float(descent_divergence_setpoint) * max(0.0, elapsed_sec))


# --------------------------------------------------------------------------- #
#  Mission orchestration                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class MissionControl:
	"""What the mission tells ControlLaw to do this tick. Passed straight
	through as control_law.compute()'s new keyword-only overrides."""
	divergence_setpoint: float = 0.0
	thrust_gain_override: Optional[float] = None
	lateral_p_scale: float = 1.0
	lateral_d_scale: float = 1.0
	enable_integral: bool = True
	substate: str = PROBE
	info: dict = field(default_factory=dict)


class MissionRoutine:
	def __init__(
		self,
		hover_thrust: float,
		control_period_sec: float = 0.5,
		# NOTE: this default is DEAD CODE when constructed via bee_node.py,
		# which always passes an explicit value -- see bee_node.py's WARNING
		# block above DESCENT_DIVERGENCE_SETPOINT. Kept in sync with that
		# constant so standalone construction (e.g. this module's own __main__
		# smoke test) reflects the same tuning, not because this default is
		# itself read during real flight.
		descent_divergence_setpoint: float = 0.30,
		# SIM-TIME, not wall-clock -- see the module-level CLOCK note. Under a
		# simulator real-time-factor (RTF) < 1, this takes proportionally
		# longer in wall-clock terms (e.g. RTF=0.2 -> ~5x the real-world wait).
		probe_min_duration_sec: float = 15.0,
		leg_clearance_m: float = 0.20,
		enable_descent: bool = True,
		probe_only: bool = False,
		ceiling_safety_factor: float = 0.5,
		# dt fed into the de Croon feasibility gate (critical_height: h_crit =
		# k_min*dt/(2*safety)), DECOUPLED from control_period_sec above.
		#
		# control_period_sec is the ROS actuation loop's WALL-clock publish
		# period -- how often a setpoint is SENT to PX4. That is NOT the same
		# quantity as de Croon's dt, which is the dead-time between a fresh
		# SENSOR sample and the correction it drives: republishing an unchanged
		# command because vision hasn't updated yet adds no new corrective
		# information, so control_period_sec understates the true dead-time
		# whenever fresh vision arrives slower than the ROS loop ticks (which it
		# does under any sim real-time-factor < 1 -- see the module-level RTF
		# note below).
		#
		# The tempting alternative -- measure the WALL-clock gap between fresh
		# vision samples in THIS run and use that -- is also wrong, for the
		# opposite reason: it bakes today's simulator slowdown into a safety
		# margin that is supposed to hold on real hardware, where RTF is always
		# 1 and the camera delivers frames at its true configured rate with no
		# such inflation. That would make the gate needlessly strict in sim
		# (and would silently stop reflecting hardware reality if the sim's
		# load/RTF ever changed) without validating anything true about the
		# deployed system.
		#
		# The physically correct number is the camera's SIM-time frame period --
		# what it delivers at RTF=1, which is the regime the safety margin must
		# hold in. Default 1/30s matches this project's camera plugin config;
		# override if yours differs. If None, falls back to control_period_sec
		# (the old, WALL-loop-period behavior) for backward compatibility.
		stability_dt_sec: Optional[float] = 1.0 / 30.0,
		# HAND-TUNED initial/exploration thrust gain "k" (m/s), set like the
		# lateral PD gains rather than derived from height + Ho's ceiling. This
		# is the gain the CENTER/PROBE/HOLD hover runs at and the value the
		# descent schedule decays from. Decoupled from h0 because the
		# ATTITUDE_HOLD height reference is ground-relative while the platform
		# sits ~2 m up, so the height-derived ceiling gain was wrong. Default
		# preserves the validated hover (old safety*ceiling_gain(h0)/12 = 0.833).
		# NOTE: this default is DEAD CODE when constructed via bee_node.py,
		# which always passes an explicit value -- see bee_node.py's WARNING
		# block above INITIAL_THRUST_GAIN. Kept in sync with that constant for
		# the same reason as descent_divergence_setpoint above.
		initial_thrust_gain: float = 6.5,
		# --- CENTER phase (runs BEFORE probe) ---
		# Center the drone over the target first, then probe. Probing while the
		# lateral loop is still banking contaminates the thrust->accel reading
		# (ThrustModel ignores tilt: vertical thrust is u*cos(roll)cos(pitch),
		# not u), so peak_accel would pick up the lateral maneuver, not the
		# deck. We hold D*=0 while centering and only start the probe once the
		# target is centered AND that has held for a debounce dwell.
		center_offset_threshold: float = 0.10,   # |offset_x|,|offset_y| both under this
		# Both SIM-TIME, not wall-clock -- same caveat as probe_min_duration_sec
		# above. All of CENTER/PROBE/descent share ONE clock (the mission's `t`,
		# fed from the vision timestamp) by design, since that is what keeps
		# the gain schedule and D* ramp physically consistent with the actual
		# descent dynamics -- see the module docstring's CLOCK note.
		center_dwell_sec: float = 2.0,           # sustained-centered debounce
		center_timeout_sec: float = 20.0,        # fallback if never centers
		enable_center: bool = True,
		# Ramp the COMMANDED D* setpoint smoothly from 0 -> descent_divergence_
		# setpoint over this many seconds of descent-elapsed time, instead of
		# stepping it instantly at PROBE->DESCEND. The instant step was the
		# actual source of the "violent" transient at descent entry: k(t) was
		# already smooth (starts at k_explore, decays exponentially), but the
		# ERROR fed to control_law (D_measured - D*) snapped from ~0 to the full
		# negative D* in one tick, driving a sharp thrust cut and, with a
		# nonzero thrust_integral_gain_const, risking integral saturation on
		# that same step. Ramping D* keeps the error small throughout, so
		# neither mechanism fires. Set to 0 to recover the old instant-step
		# behavior.
		#
		# SHAPE: raised-cosine (0.5*(1-cos(pi*x))), not linear. A linear ramp
		# removes the VALUE step but still has a corner in its SLOPE at both
		# endpoints (the commanded rate of change jumps from 0 to a constant at
		# ramp-start, and drops back to 0 at ramp-end) -- measured directly: a
		# linear ramp produced a real, if modest, climb-then-sink wobble in vz
		# during the ramp window. The raised-cosine blend has zero slope at
		# BOTH endpoints, removing that corner while keeping the same total
		# ramp duration and reaching the exact target at elapsed>=this value.
		#
		# Deliberately NOT applied to the gain schedule's decay rate (still
		# scheduled_gain_at_time(..., self._d_star, ...), the FINAL target) or
		# to h_pred's prediction: during the ramp the true physical descent is
		# slightly slower than h0*exp(-d_star*t) assumes (since the commanded
		# D* is briefly below its target), so using the final d_star for k(t)'s
		# decay makes k(t) decay slightly FASTER than strictly necessary in
		# that window -- conservative, consistent with the project's existing
		# "safe-but-slow" clock-following choice, not a new risk. Expect
		# height_prediction.png to show a small, self-correcting bias during
		# just the ramp window; that is this trade-off made visible, not a bug.
		d_star_ramp_in_sec: float = 3.0,
		# CENTER's vertical (thrust) loop runs at the SAME gain as everywhere
		# else (k_explore) -- no reduced CENTER-phase thrust gain and no
		# CENTER->PROBE thrust ramp. (An earlier version decoupled the
		# vertical loop from CENTER's heavy lateral banking with a reduced
		# gain here; that mechanism has been removed -- it is not used.)
		#
		# Time (SIM-time, like every other mission timer -- see the CLOCK
		# note) to smoothly restore the LATERAL gains (center_lateral_p_scale/
		# center_lateral_d_scale -> probe_lateral_p_scale/probe_lateral_d_scale)
		# once CENTER hands off, using the SAME raised-cosine shape as the D*
		# ramp and for the same reason: a hard step risks a kick right at the
		# handoff, exactly where we've already found transients matter most.
		# Thrust is NOT part of this ramp (see above) -- this timer now
		# governs the lateral transition only. The probe's own peak_accel/
		# min-duration measurement is deliberately NOT started until this
		# ramp completes -- starting it mid-ramp would measure against a
		# still-changing lateral gain, not a clean, steady one, the same
		# contamination concern that motivated resetting the probe's baseline
		# in the first place. 0 recovers an instant switch.
		center_to_probe_lateral_ramp_sec: float = 2.0,
		# LATERAL gain, mission-PHASE-based rather than instantaneous-offset-
		# based. This is a DIFFERENT mechanism from ControlLaw's
		# large_offset_gain_scale blend (which reduces gain as a function of
		# the CURRENT |offset|, restoring full gain every time the vehicle
		# happens to swing near center -- including repeatedly, mid-oscillation,
		# throughout CENTER). This scale instead stays fixed for the WHOLE
		# CENTER phase regardless of how many times offset crosses near zero,
		# and only transitions ONCE, at the CENTER->PROBE handoff -- matching
		# the originally-requested design ("small gain to center the image,
		# then increase once centered, THEN start following the
		# disturbance"), which a purely offset-magnitude-keyed blend cannot
		# express (it cannot distinguish "still early in centering, currently
		# near zero mid-swing" from "genuinely converged, entering PROBE").
		# The two compose multiplicatively (this sets the phase's baseline
		# regime; the offset-magnitude blend still provides extra saturation
		# protection within either regime). Ramps to the PROBE steady-state
		# scales (probe_lateral_p_scale/probe_lateral_d_scale) over
		# center_to_probe_lateral_ramp_sec -- see that ramp's docstring.
		# 1.0 disables this (CENTER at full lateral gain throughout, the old
		# behavior).
		#
		# SPLIT INTO TWO SCALES, not one -- a single shared scale CANNOT
		# reproduce an earlier validated (kp, kd) pair, because kp and kd were
		# boosted by DIFFERENT factors when tuned for platform-oscillation
		# tracking (kp x2, kd x sqrt(2) -- see control_law.py's gain history).
		# Verified directly: a single scale=0.5 landed kp exactly on the
		# earlier validated baseline (0.44*0.5=0.22) but left kd under-damped
		# relative to that SAME baseline (0.155*0.5=0.078 vs the validated
		# 0.11) -- CENTER was less damped than the historical baseline it was
		# meant to recover, not more. These two values EXACTLY reverse the two
		# historical factors: p_scale=1/2 reverses "kp x2"; d_scale=1/sqrt(2)
		# reverses "kd x sqrt(2)". Verified to reproduce the original
		# validated (roll_kp=0.22, roll_kd=0.11, pitch_kp=0.15, pitch_kd=0.07)
		# gains to within <0.4% (the residual is rounding in the stored
		# decimal constants, not a modeling error).
		center_lateral_p_scale: float = 0.5,
		center_lateral_d_scale: float = 1.0 / math.sqrt(2.0),
		# STEADY PROBE lateral P scale (kd left at 1.0 -- deliberately NOT
		# raised further, since raising kd was already A/B-tested and found
		# to make tracking WORSE, not better: it operates on optical flow, a
		# real noisy/lagged signal, not a clean derivative).
		#
		# Motivated by a cross-spectral measurement of the closed loop at full
		# gain against the platform's own oscillation: amplitude ratio 2.229
		# (residual EXCEEDS the platform's own motion -- not simple
		# under-tracking, which would show ratio->1 at ~180 deg phase, the
		# original pre-kp-boost signature) and phase -98.9 deg (close to -90,
		# the textbook signature of a lightly-damped 2nd-order system driven
		# AT its own resonant frequency, not below or above it). This means
		# the platform's ~0.055 Hz oscillation is landing almost exactly on
		# the closed loop's resonant peak -- the worst possible frequency for
		# an underdamped loop, producing amplification rather than tracking
		# OR simple non-response.
		#
		# kp sets the loop's natural frequency; raising it further pushes the
		# resonant peak AWAY from the platform's low frequency, moving 0.055 Hz
		# into the loop's flat, well-tracked low-frequency region instead of
		# onto its peak -- unlike raising kd, this isn't undermined by
		# derivative noise, since kp acts on offset (position), a clean
		# signal. 1.5 is a MODERATE first step (kp_eff: roll 0.44->0.66, pitch
		# 0.30->0.45), not a confident final answer -- re-measure the SAME
		# amplitude-ratio/phase diagnostic on the next log: ratio should fall
		# and phase should move away from -90 deg (toward 0) if this is
		# working. 1.0 disables this (PROBE at the current tracking-tuned
		# kp, unchanged).
		probe_lateral_p_scale: float = 1.5,
		probe_lateral_d_scale: float = 1.0,
	):
		self._dt = float(control_period_sec)
		# Gate/feasibility dt is DELIBERATELY separate from the ROS loop's
		# self._dt above -- see the constructor docstring. Falls back to
		# control_period_sec if not given, for backward compatibility.
		self._stability_dt = (
			float(stability_dt_sec) if stability_dt_sec is not None else self._dt
		)
		self._d_star = float(descent_divergence_setpoint)
		self._probe_min = float(probe_min_duration_sec)
		self._leg_clearance = float(leg_clearance_m)
		self._enable_descent = bool(enable_descent)
		self._probe_only = bool(probe_only)
		self._enable_center = bool(enable_center)
		self._center_offset_thr = float(center_offset_threshold)
		self._center_dwell = float(center_dwell_sec)
		self._center_timeout = float(center_timeout_sec)
		self._d_star_ramp_in = max(0.0, float(d_star_ramp_in_sec))
		self._lateral_ramp = max(0.0, float(center_to_probe_lateral_ramp_sec))
		self._center_lateral_p_scale = max(0.0, float(center_lateral_p_scale))
		self._center_lateral_d_scale = max(0.0, float(center_lateral_d_scale))
		self._probe_lateral_p_scale = max(0.0, float(probe_lateral_p_scale))
		self._probe_lateral_d_scale = max(0.0, float(probe_lateral_d_scale))
		# Hold the live gain this fraction below the hard de Croon ceiling 2h/dt.
		# 0.5 is "really conservative" per the design discussion; raise toward
		# 1.0 for more aggressive (closer-to-ceiling) gains once validated.
		self._safety = max(1e-3, min(1.0, float(ceiling_safety_factor)))
		self._initial_thrust_gain = max(0.0, float(initial_thrust_gain))

		self._tm = ThrustModel(hover_thrust)
		self._probe = PlatformProbe(self._tm)

		self._substate = CENTER if self._enable_center else PROBE
		self._t0: Optional[float] = None
		self._h0: Optional[float] = None
		self._t_descend_start: Optional[float] = None
		self._centered_since: Optional[float] = None   # dwell timer start
		self._center_start_t: Optional[float] = None    # for timeout
		self._t_probe_entry: Optional[float] = None      # for the center->probe gain ramp

		self.gate: GateResult = GateResult()
		self.probe_result: ProbeResult = ProbeResult()

	def reset(self) -> None:
		self._probe.reset()
		self._substate = CENTER if self._enable_center else PROBE
		self._t0 = None
		self._t_descend_start = None
		self._centered_since = None
		self._center_start_t = None
		self._t_probe_entry = None
		self._k_explore = 0.0
		self.gate = GateResult()
		self.probe_result = ProbeResult()

	def start(self, t: float, start_height_m: float) -> None:
		"""start_height_m: the one-time h0 seed -- see module docstring's
		H0 SEEDING note. Not re-read after this call."""
		self.reset()
		self._t0 = float(t)
		self._h0 = max(1e-3, float(start_height_m))
		# Exploration gain is HAND-TUNED (see initial_thrust_gain), not derived
		# from h0. h0 is still seeded here for the descent height PREDICTION
		# (diagnostic h_pred) and critical_time only -- not for any gain.
		self._k_explore = self._initial_thrust_gain

	@property
	def substate(self) -> str:
		return self._substate

	@property
	def feasible(self) -> bool:
		return self.gate.feasible

	def update(
		self,
		t: float,
		dt: float,
		last_thrust_cmd: float,
		offset_x: float = 0.0,
		offset_y: float = 0.0,
		target_found: bool = False,
	) -> MissionControl:
		t = float(t)
		dt = max(1e-3, float(dt))
		if self._t0 is None:
			self._t0 = t
		if self._h0 is None:
			self._h0 = 5.0  # defensive fallback; start() should always set this.

		if self._substate == CENTER:
			return self._do_center(t, offset_x, offset_y, target_found)
		if self._substate == PROBE:
			return self._do_probe(t, dt, last_thrust_cmd)
		if self._substate == PROBE_HOLD:
			return self._do_probe_hold(t)
		if self._substate == INFEASIBLE:
			return self._do_infeasible(t)
		return self._do_descend(t)

	def _do_center(
		self, t: float, offset_x: float, offset_y: float, target_found: bool
	) -> MissionControl:
		"""Station-keep (D*=0) at the full vertical exploration gain (k_explore
		-- CENTER no longer runs a reduced thrust gain, see the constructor
		note) and REDUCED, independently-scaled lateral gains
		(center_lateral_p_scale, center_lateral_d_scale) -- fixed for the
		WHOLE phase, not just when momentarily near zero mid-swing -- see the
		constructor docstrings for the full reasoning) until the target is
		centered and has stayed centered for a debounce dwell, THEN hand off to
		the probe. Probing is deliberately NOT run here -- see the CENTER config
		note. On handoff the probe is reset so its slow-baseline high-pass
		starts fresh from the settled hover, not from the centering transient."""
		if self._center_start_t is None:
			self._center_start_t = t

		centered = (
			target_found
			and abs(float(offset_x)) <= self._center_offset_thr
			and abs(float(offset_y)) <= self._center_offset_thr
		)
		if centered:
			if self._centered_since is None:
				self._centered_since = t
			dwell = t - self._centered_since
		else:
			self._centered_since = None
			dwell = 0.0

		elapsed = t - self._center_start_t
		settled = centered and dwell >= self._center_dwell
		timed_out = elapsed >= self._center_timeout

		if settled or timed_out:
			self._probe.reset()          # fresh high-pass baseline for the probe
			self._t_probe_entry = t      # starts the gain ramp back to k_explore
			self._substate = PROBE
			# Do NOT run a probe sample this frame -- we don't have the real
			# last_thrust_cmd here, and a fake one would corrupt peak_accel.
			# Emit a clean station-keep hold at full thrust gain (k_explore --
			# CENTER's thrust loop is not reduced) and the still-reduced
			# center lateral gains; the lateral ramp to full/probe gain begins
			# on the next tick in _do_probe.
			return MissionControl(
				divergence_setpoint=0.0,
				thrust_gain_override=self._k_explore,
				lateral_p_scale=self._center_lateral_p_scale,
				lateral_d_scale=self._center_lateral_d_scale,
				substate=PROBE,
				info={
					"event": "center_done",
					"centered_ok": bool(settled),
					"center_timed_out": bool(timed_out),
					"center_elapsed_sec": elapsed,
				},
			)

		# still centering: hold altitude at the FULL exploration gain
		# (k_explore -- CENTER's vertical loop is not reduced/ramped), and
		# REDUCED, INDEPENDENTLY-scaled lateral P/D gains (center_lateral_p_
		# scale/center_lateral_d_scale) -- gentle for the WHOLE centering
		# approach, not just when momentarily near zero mid-swing.
		return MissionControl(
			divergence_setpoint=0.0,
			thrust_gain_override=self._k_explore,
			lateral_p_scale=self._center_lateral_p_scale,
			lateral_d_scale=self._center_lateral_d_scale,
			substate=CENTER,
			info={
				"offset_x": offset_x,
				"offset_y": offset_y,
				"target_found": target_found,
				"centered": centered,
				"center_dwell_sec": dwell,
				"center_elapsed_sec": elapsed,
			},
		)

	def _do_probe(self, t: float, dt: float, last_thrust_cmd: float) -> MissionControl:
		if self._t_probe_entry is None:
			self._t_probe_entry = t  # defensive: should already be set at the CENTER handoff

		ramp = self._lateral_ramp
		ramp_elapsed = t - self._t_probe_entry
		ramp_frac = 1.0 if ramp <= 1e-9 else min(1.0, ramp_elapsed / ramp)
		# Same raised-cosine shape as the D* ramp (zero slope at both ends: no
		# derivative-discontinuity kick restoring gain right after handoff).
		# LATERAL only -- thrust runs at k_explore throughout CENTER and PROBE,
		# so there is no vertical quantity to ramp here anymore.
		ramp_shaped = 0.5 * (1.0 - math.cos(math.pi * ramp_frac))
		current_lateral_p = self._center_lateral_p_scale + (self._probe_lateral_p_scale - self._center_lateral_p_scale) * ramp_shaped
		current_lateral_d = self._center_lateral_d_scale + (self._probe_lateral_d_scale - self._center_lateral_d_scale) * ramp_shaped

		if ramp_frac < 1.0:
			# Lateral gain is still being restored toward its PROBE steady-
			# state value; thrust is already at k_explore (no ramp). Do NOT
			# feed the probe yet -- accumulating peak_accel or elapsed
			# duration against a CHANGING lateral gain would contaminate the
			# measurement the same way banking does (see PROBE's module note).
			# The probe's own clock only starts once the lateral gain is
			# fully settled.
			return MissionControl(
				divergence_setpoint=0.0,
				thrust_gain_override=self._k_explore,
				lateral_p_scale=current_lateral_p,
				lateral_d_scale=current_lateral_d,
				substate=PROBE,
				info={
					"event": "center_lateral_ramp",
					"lateral_ramp_frac": ramp_shaped,
					"k_explore": self._k_explore,
					"center_lateral_p_scale": self._center_lateral_p_scale,
					"center_lateral_d_scale": self._center_lateral_d_scale,
				},
			)

		self._probe.update(last_thrust_cmd, dt)
		self.probe_result = self._probe.result(self._probe_min)

		if self.probe_result.ready:
			self.gate = compute_gate(
				peak_accel=self.probe_result.peak_accel,
				descent_divergence_setpoint=self._d_star,
				initial_thrust_gain=self._initial_thrust_gain,
				control_period_sec=self._stability_dt,
				leg_clearance_m=self._leg_clearance,
				ceiling_safety_factor=self._safety,
			)

			# Hover/probe-only test mode: the gate numbers are now computed and
			# available for inspection (k_min, h_crit, feasible), but we never
			# descend and never abort -- just keep hovering so the probe and the
			# bounds can be validated in isolation before the descent is trusted.
			if self._probe_only:
				self._substate = PROBE_HOLD
				return self._do_probe_hold(t, just_entered=True)

			if self.gate.feasible and self._enable_descent:
				self._substate = DESCEND
				self._t_descend_start = t
				return self._do_descend(t, just_entered=True)

			self._substate = INFEASIBLE
			return self._do_infeasible(t, just_entered=True)

		# Still probing: hold D*=0 at the conservative exploration gain, using
		# the new accel-domain thrust law (k_explore) rather than the dormant
		# LQR. Lateral at the PROBE steady-state scale (see
		# probe_lateral_p_scale/probe_lateral_d_scale). The probe watches the
		# thrust THIS produces to estimate peak platform acceleration.
		return MissionControl(
			divergence_setpoint=0.0,
			thrust_gain_override=self._k_explore,
			lateral_p_scale=self._probe_lateral_p_scale,
			lateral_d_scale=self._probe_lateral_d_scale,
			substate=PROBE,
			info={
				"peak_accel": self.probe_result.peak_accel,
				"probe_elapsed_sec": self.probe_result.duration_sec,
				"probe_min_sec": self._probe_min,
				"k_explore": self._k_explore,
			},
		)

	def _do_descend(self, t: float, just_entered: bool = False) -> MissionControl:
		elapsed = t - (self._t_descend_start if self._t_descend_start is not None else t)
		# LIVE gain: explicit function of the descent clock only -- no height.
		# Decay rate uses the FINAL d_star even during the D* ramp below -- see
		# the d_star_ramp_in_sec constructor note for why that is the safe
		# choice, not an oversight.
		k = scheduled_gain_at_time(
			elapsed, self._d_star, self.gate.k_min, self.gate.k_explore, self._safety
		)
		# Lateral rides the SAME normalized ramp fraction as thrust: 1.0 at
		# t=0 -> k_min/k_explore at t_crit (de Croon App. B -- same 2Z/dt
		# ceiling on the lateral loop). Applied as a MULTIPLIER on the PROBE
		# steady-state gains (probe_lateral_p_scale/probe_lateral_d_scale)
		# below, not on a bare 1.0 -- DESCEND is entered directly from PROBE,
		# so the schedule should decay from wherever PROBE actually left the
		# lateral gains, not silently reset them to CENTER-equivalent values
		# first (multiplying a bare 1.0 previously made the lateral gains
		# jump from PROBE's steady-state values back up to 1.0/1.0 right at
		# DESCEND entry -- an unintended step at exactly the handoff this
		# design otherwise takes care to smooth).
		scale = (k / self.gate.k_explore) if self.gate.k_explore > 1e-9 else 1.0
		# Diagnostic only: the height this clock-based schedule IMPLIES, logged
		# for offline comparison against ground truth. NOT used to compute k.
		h_pred = predicted_height(self._h0, self._d_star, elapsed)

		# COMMANDED D*: ramps 0 -> self._d_star over d_star_ramp_in_sec, instead
		# of stepping instantly at descent entry (see constructor note). Shaped
		# with a raised-cosine blend, NOT linear: a linear ramp is continuous in
		# VALUE but has a corner in its SLOPE at both endpoints (the commanded
		# rate of change jumps from 0 to a constant at ramp-start, and drops
		# back to 0 at ramp-end/plateau) -- a real system with inertia and
		# filtered actuation reacts to that corner as a small kick. Measured
		# directly: a linear ramp produced a real (if modest) climb-then-sink
		# wobble in vz during the ramp window, distinct from the separate,
		# already-diagnosed steady vz climb that continues well after the ramp
		# plateaus (a k_min-authority-loss issue near touchdown, not a ramp
		# artifact -- do not conflate the two). The raised-cosine blend below
		# has ZERO slope at both x=0 and x=1, removing the corner at both ends
		# while keeping the same total ramp duration and exact target value at
		# elapsed>=d_star_ramp_in_sec. Setting d_star_ramp_in_sec=0 still
		# recovers the old instant-step behavior exactly (linear_frac clamps to
		# 1.0 immediately, and the raised-cosine of 1.0 is exactly 1.0).
		if self._d_star_ramp_in <= 1e-9:
			linear_frac = 1.0
		else:
			linear_frac = min(1.0, elapsed / self._d_star_ramp_in)
		# Raised-cosine (half-cosine) blend: 0.5*(1-cos(pi*x)). f(0)=0, f(1)=1,
		# f'(0)=f'(1)=0 -- classic, simple, single-trig-call ease-in/ease-out.
		ramp_frac = 0.5 * (1.0 - math.cos(math.pi * linear_frac))
		d_star_cmd = self._d_star * ramp_frac

		return MissionControl(
			divergence_setpoint=d_star_cmd,
			thrust_gain_override=k,
			# DESCEND's height-based ramp FRACTION (scale) applies uniformly
			# to both P and D (de Croon App. B: lateral rides the same
			# schedule as thrust), but the BASELINE it multiplies is now the
			# PROBE steady-state gain for each axis (probe_lateral_p_scale,
			# probe_lateral_d_scale) rather than a shared bare 1.0 -- so at
			# descent entry (scale=1.0) these equal exactly what PROBE was
			# already commanding, with no step. This is a different concern
			# from CENTER's P/D split above, which exists only to reproduce
			# an earlier validated (kp, kd) PAIR exactly.
			lateral_p_scale=self._probe_lateral_p_scale * scale,
			lateral_d_scale=self._probe_lateral_d_scale * scale,
			# k(t) deliberately decays toward k_min as height decreases (the
			# whole point of the schedule), so divergence necessarily
			# undershoots D* by a STRUCTURAL amount for most of descent -- not
			# a fixable bias an integral should chase. Left on, the integral
			# accumulates against that intentional gap for the whole descent,
			# dragging a growing thrust bias into the final approach. Integral
			# stays enabled during CENTER/PROBE/PROBE_HOLD (default True),
			# where it corrects a genuine slow hover-thrust bias instead.
			enable_integral=False,
			substate=DESCEND,
			info={
				"just_entered": just_entered,
				"h_pred": h_pred,
				"k": k,
				"lateral_scale": scale,
				"lateral_p_scale": self._probe_lateral_p_scale * scale,
				"lateral_d_scale": self._probe_lateral_d_scale * scale,
				"k_min": self.gate.k_min,
				"k_explore": self.gate.k_explore,
				"h_crit": self.gate.h_crit,
				"elapsed_sec": elapsed,
				"t_crit_sec": critical_time(self._h0, self._d_star, self.gate.h_crit),
				"d_star_ramp_frac": ramp_frac,
				"d_star_ramp_linear_frac": linear_frac,
				"d_star_target": self._d_star,
			},
		)

	def _do_probe_hold(self, t: float, just_entered: bool = False) -> MissionControl:
		"""Hover/probe-only terminal hold: keep D*=0 at the exploration gain
		(the new accel-domain thrust law, not the dormant LQR), never descend,
		never abort. Surfaces the computed gate so k_min / h_crit / feasibility
		can be read off in isolation."""
		return MissionControl(
			divergence_setpoint=0.0,
			thrust_gain_override=self._k_explore,
			lateral_p_scale=self._probe_lateral_p_scale,
			lateral_d_scale=self._probe_lateral_d_scale,
			substate=PROBE_HOLD,
			info={
				"just_entered": just_entered,
				"probe_only": True,
				"peak_accel": self.probe_result.peak_accel,
				"k_min": self.gate.k_min,
				"h_crit": self.gate.h_crit,
				"k_explore": self.gate.k_explore,
				"feasible_if_descended": self.gate.feasible,
				"leg_clearance_m": self._leg_clearance,
			},
		)

	def _do_infeasible(self, t: float, just_entered: bool = False) -> MissionControl:
		return MissionControl(
			divergence_setpoint=0.0,
			thrust_gain_override=self._k_explore,
			lateral_p_scale=1.0,
			lateral_d_scale=1.0,
			substate=INFEASIBLE,
			info={
				"just_entered": just_entered,
				"reason": "h_crit exceeds leg clearance at this control rate",
				"h_crit": self.gate.h_crit,
				"leg_clearance_m": self._leg_clearance,
				"k_min": self.gate.k_min,
				"peak_accel": self.probe_result.peak_accel,
			},
		)

	def status_line(self) -> str:
		if self._substate == CENTER:
			return (f"[center] waiting for target within +/-{self._center_offset_thr:.2f} "
					f"for {self._center_dwell:.1f}s (timeout {self._center_timeout:.0f}s)")
		if self._substate == PROBE:
			return (f"[probe] elapsed={self.probe_result.duration_sec:.1f}s/"
					f"{self._probe_min:.0f}s peak_accel={self.probe_result.peak_accel:.3f} m/s^2")
		if self._substate == PROBE_HOLD:
			verdict = "WOULD-LAND" if self.gate.feasible else "WOULD-ABORT"
			return (f"[probe_hold] peak_accel={self.probe_result.peak_accel:.3f} m/s^2 "
					f"k_min={self.gate.k_min:.2f} h_crit={self.gate.h_crit:.2f}m "
					f"vs leg={self._leg_clearance:.2f}m -> {verdict} (hovering, no descent)")
		if self._substate == INFEASIBLE:
			return (f"[infeasible] h_crit={self.gate.h_crit:.2f}m > "
					f"leg_clearance={self._leg_clearance:.2f}m "
					f"(k_min={self.gate.k_min:.2f}, peak_accel={self.probe_result.peak_accel:.3f})")
		elapsed = 0.0 if self._t_descend_start is None else max(0.0, self._t0)
		return (f"[descend] h_crit={self.gate.h_crit:.2f}m "
				f"k_min={self.gate.k_min:.2f} k_explore={self.gate.k_explore:.2f}")


# --------------------------------------------------------------------------- #
def _smoke_test() -> None:
	"""Synthetic check: feasible and infeasible platforms behave as expected."""
	dt = 0.5
	hover = 0.73

	for label, a_peak in (("calm", 0.05), ("moderate", 0.30), ("violent", 1.50)):
		m = MissionRoutine(
			hover_thrust=hover, control_period_sec=dt,
			descent_divergence_setpoint=0.15, probe_min_duration_sec=3.0,
			leg_clearance_m=0.20,
		)
		m.start(0.0, start_height_m=5.0)
		t = 0.0
		u = hover
		tm = ThrustModel(hover)
		# Window extended to cover center_dwell + the center->probe gain ramp
		# (both now sequential BEFORE the probe's own probe_min_duration_sec
		# timer starts -- see center_to_probe_lateral_ramp_sec) + the probe window
		# itself, so the probe actually reaches `ready` in this synthetic run.
		while t < 10.0:
			# synthetic thrust command: hover plus a sinusoidal "platform" kick
			a_plat = a_peak * math.cos(2 * math.pi * 0.2 * t)
			u = hover + hover * a_plat / G_ACCEL
			mc = m.update(t, dt, u, offset_x=0.0, offset_y=0.0, target_found=True)
			t += dt
		print(f"{label:9s} a_peak={a_peak:.2f} -> substate={mc.substate:10s} "
			f"k_min={m.gate.k_min:.2f} h_crit={m.gate.h_crit:.2f} feasible={m.feasible}")

		if mc.substate == DESCEND:
			for _ in range(10):
				mc = m.update(t, dt, u, offset_x=0.0, offset_y=0.0, target_found=True)
				t += dt
			print(f"  after 10 more ticks: {m.status_line()}  k_now={mc.thrust_gain_override:.2f}")


if __name__ == "__main__":
	_smoke_test()