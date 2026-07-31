"""
Bio-inspired near-field mission routine.

Sequence:
    CENTER -> APPROACH_PROBE -> FINAL_PROBE -> DESCEND

The control law remains in control_law.py. This module selects the divergence
setpoint, vertical gain, lateral gain scales, integral enable flag, and the final
landing-feasibility decision.

Three acceleration probes run in parallel from APPROACH_PROBE through
FINAL_PROBE:
    - vertical command acceleration;
    - roll-channel command acceleration;
    - pitch-channel command acceleration.

Each probe uses the same de-biasing, rolling-percentile and leaky-peak logic. The
far-field estimates carry into FINAL_PROBE, where all three probes are retuned to
the near-field time constants without resetting their accumulated envelopes.
The final transition requires all three probes to be ready and all three gain
windows to be feasible.

Vertical feasibility compares the Herisse disturbance-rejection floor with the
safety-scaled de Croon ceiling at landing-gear height. Roll and pitch feasibility
use the acceleration-domain lateral bound

    K_min = c_max / kappa + peak_accel / omega_adm

and verify that the planned D gain at touchdown lies below the corresponding
discrete lateral ceiling. No online height estimate is used.

All timers use the camera/image timestamp supplied as ``t`` to update().
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


G_ACCEL = 9.80665

CENTER = "center"
PROBE = "probe"
APPROACH_PROBE = "approach_probe"
FINAL_PROBE = "final_probe"
# Terminal state, latched by bee_node via mark_landed().
LANDED = "landed"
PROBE_HOLD = "probe_hold"
DESCEND = "descend"
INFEASIBLE = "infeasible"


def clamp(value: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, float(value)))


def raised_cosine01(x: float) -> float:
	"""Smooth 0->1 blend with zero slope at both ends."""
	x = clamp(x, 0.0, 1.0)
	return 0.5 * (1.0 - math.cos(math.pi * x))


class ThrustModel:
	"""a_drone = g * (u / u_hover - 1), world-up, tilt ignored."""

	def __init__(self, hover_thrust: float, g: float = G_ACCEL):
		self._u_hover = max(1e-3, float(hover_thrust))
		self._g = float(g)

	def accel_from_thrust(self, u: float) -> float:
		return self._g * (float(u) / self._u_hover - 1.0)


@dataclass
class ProbeResult:
	peak_accel: float = 0.0
	n_samples: int = 0
	duration_sec: float = 0.0        # elapsed in the CURRENT phase (since the last retune)
	total_duration_sec: float = 0.0  # elapsed since the probe started (across the retune)
	ready: bool = False              # both duration requirements met


class PlatformProbe:
	"""Estimate a robust acceleration envelope from a command history.

	The EMA removes the slow command bias, the rolling percentile rejects isolated
	outliers, and the leaky maximum retains the largest recent excursion. The same
	object can consume normalized thrust through update() or an acceleration
	directly through update_accel().
	"""

	def __init__(
		self,
		thrust_model: ThrustModel,
		highpass_tau_sec: float = 7.0,
		percentile_window_sec: float = 2.0,
		peak_decay_tau_sec: float = 3.0,
		peak_percentile: float = 0.95,
		accel_margin: float = 0.05,
		probe_attenuation_comp: float = 1.2,
	):
		self._tm = thrust_model
		self._tau = max(1e-3, float(highpass_tau_sec))

		self._mean = 0.0
		self._has_mean = False
		self._peak = 0.0
		self._n = 0
		self._elapsed = 0.0        # since the last retune() (the CURRENT phase)
		self._total_elapsed = 0.0  # since reset() (across the retune)

		self._percentile_window_sec = max(1e-3, float(percentile_window_sec))
		self._peak_percentile = clamp(peak_percentile, 0.0, 1.0)
		self._accel_margin = max(0.0, float(accel_margin))
		self._probe_attenuation_comp = max(1.0, float(probe_attenuation_comp))
		self._peak_decay_tau = max(1e-3, float(peak_decay_tau_sec))
		self._residual_window: Deque[tuple[float, float]] = deque()

		# Per-step values from the LAST update(), exposed for logging. They are what
		# the peak envelope is actually built from, so plotting them against
		# peak_accel shows whether the envelope is tracking real excursions or
		# coasting on a stale peak.
		self._last_accel = 0.0       # a = accel_from_thrust(thrust_cmd)
		self._last_residual = 0.0    # |a - mean|, the de-biased signal
		self._last_percentile = 0.0  # the window percentile the peak chases

	def reset(self) -> None:
		self._mean = 0.0
		self._has_mean = False
		self._peak = 0.0
		self._n = 0
		self._elapsed = 0.0
		self._total_elapsed = 0.0
		self._residual_window.clear()
		self._last_accel = 0.0
		self._last_residual = 0.0
		self._last_percentile = 0.0

	def retune(
		self,
		highpass_tau_sec: Optional[float] = None,
		percentile_window_sec: Optional[float] = None,
		peak_decay_tau_sec: Optional[float] = None,
	) -> None:
		"""Apply new time constants while preserving the mean and peak.

		The phase clock and rolling window restart, while the accumulated envelope
		carries into the near-field probe.
		"""
		if highpass_tau_sec is not None:
			self._tau = max(1e-3, float(highpass_tau_sec))
		if percentile_window_sec is not None:
			self._percentile_window_sec = max(1e-3, float(percentile_window_sec))
		if peak_decay_tau_sec is not None:
			self._peak_decay_tau = max(1e-3, float(peak_decay_tau_sec))

		# Phase clock restarts (so `ready` can require a minimum NEAR-field hold),
		# but the total clock and the peak/mean state carry through.
		self._elapsed = 0.0
		self._residual_window.clear()

	@property
	def peak_accel(self) -> float:
		"""The leaky-max envelope -- the number the gate consumes."""
		return float(self._peak)

	@property
	def accel(self) -> float:
		"""Instantaneous commanded acceleration used by the last probe update."""
		return float(self._last_accel)

	@property
	def mean_accel(self) -> float:
		"""The EMA bias being subtracted (hover trim + any slow descent term)."""
		return float(self._mean)

	@property
	def residual_accel(self) -> float:
		"""|a - mean|: the de-biased per-step signal the envelope is built from."""
		return float(self._last_residual)

	@property
	def percentile_accel(self) -> float:
		"""The rolling-window percentile the peak chases each step."""
		return float(self._last_percentile)

	@property
	def peak_decay_tau_sec(self) -> float:
		return float(self._peak_decay_tau)

	def update(self, thrust_cmd: float, dt: float) -> None:
		"""Update from normalized collective thrust (vertical probe compatibility)."""
		self.update_accel(self._tm.accel_from_thrust(thrust_cmd), dt)

	def update_accel(self, accel_cmd: float, dt: float) -> None:
		"""Update directly from a commanded acceleration in m/s^2."""
		dt = max(1e-3, float(dt))
		a = float(accel_cmd)

		if not self._has_mean:
			self._mean = a
			self._has_mean = True
		else:
			alpha = math.exp(-dt / self._tau)
			self._mean = alpha * self._mean + (1.0 - alpha) * a

		residual = abs(a - self._mean)
		self._last_accel = a
		self._last_residual = residual

		self._n += 1
		self._elapsed += dt
		self._total_elapsed += dt
		self._residual_window.append((self._elapsed, residual))

		window_sec = max(1e-3, self._percentile_window_sec)
		while (
			self._residual_window
			and self._elapsed - self._residual_window[0][0] > window_sec
		):
			self._residual_window.popleft()

		values = sorted(v for _, v in self._residual_window)
		if not values:
			target_peak = residual
		elif len(values) == 1:
			target_peak = values[0]
		else:
			q = clamp(self._peak_percentile, 0.0, 1.0)
			idx = q * (len(values) - 1)
			lo = int(math.floor(idx))
			hi = int(math.ceil(idx))
			if lo == hi:
				target_peak = values[lo]
			else:
				w = idx - lo
				target_peak = (1.0 - w) * values[lo] + w * values[hi]

		# Raw percentile stays the diagnostic ("what the probe actually saw").
		self._last_percentile = float(target_peak)

		# Two margins, two shapes: factor inverts the dB attenuation (scales with the
		# value); additive covers unmodeled/ground-effect terms AND guarantees a
		# nonzero floor when the percentile is small, where a bare factor collapses.
		protected_peak = target_peak * self._probe_attenuation_comp + self._accel_margin

		# Leaky max, unchanged. Note the additive term now also acts as a hard lower
		# bound: _peak can never leak below accel_margin.
		decay = math.exp(-dt / max(1e-3, self._peak_decay_tau))
		self._peak = max(protected_peak, decay * self._peak)

	def result(
		self,
		min_duration_sec: float,
		min_total_duration_sec: float = 0.0,
	) -> ProbeResult:
		"""ready requires BOTH: enough time in the current phase (min_duration_sec,
		e.g. the near-field hold) AND enough total probing (min_total_duration_sec,
		e.g. about one platform period, which the near-field hold alone is far too
		short to provide)."""
		return ProbeResult(
			peak_accel=float(self._peak),
			n_samples=int(self._n),
			duration_sec=float(self._elapsed),
			total_duration_sec=float(self._total_elapsed),
			ready=(
				self._elapsed >= float(min_duration_sec)
				and self._total_elapsed >= float(min_total_duration_sec)
			),
		)

@dataclass
class GateResult:
	k_min: float = 0.0          # Herisse floor: peak_accel / D*
	h_crit: float = 0.0         # height at which the safety-scaled ceiling == k_min
	k_explore: float = 0.0      # hand-tuned exploration gain (schedule's start value)
	feasible: bool = False      # landing window exists and FINAL_PROBE gain reaches k_min

	# --- Ceiling-riding descent target (see module docstring). ---
	k_ceiling_leg: float = 0.0    # de Croon safety-scaled ceiling AT LEG HEIGHT
	k_target: float = 0.0         # ceiling_margin * k_ceiling_leg
	k_floor: float = 0.0          # max(k_min, k_target) -- what the schedule decays TO
	ceiling_margin: float = 0.0
	k_descend_start: float = 0.0  # gain at DESCEND entry = k_probe, NOT k_explore.
	                              # The descent continues down from where FINAL_PROBE
	                              # left the gain; it never jumps back up to the
	                              # far-field value.
	accel_capacity_floor: float = 0.0
	accel_capacity_ceiling: float = 0.0
	window_exists: bool = False
	start_above_floor: bool = False


@dataclass
class LateralGateResult:
	peak_accel: float = 0.0
	k_min: float = 0.0
	k_probe: float = 0.0
	k_touchdown: float = 0.0
	k_target: float = 0.0
	k_floor: float = 0.0
	k_descend_start: float = 0.0
	k_ceiling_probe: float = 0.0
	k_ceiling_leg: float = 0.0
	ceiling_margin: float = 0.0
	accel_capacity_floor: float = 0.0
	accel_capacity_ceiling: float = 0.0
	probe_within_ceiling: bool = False
	window_exists: bool = False
	start_above_floor: bool = False
	floor_within_ceiling: bool = False
	feasible: bool = False


def lateral_ceiling_gain_at_height(
	height_m: float,
	control_period_sec: float,
	kappa: float,
	max_closing_speed_m_s: float,
	safety: float = 1.0,
) -> float:
	"""Safety-scaled discrete lateral ceiling with bounded closing speed."""
	s = max(1e-3, float(safety))
	dt = max(1e-6, float(control_period_sec))
	kappa = max(1e-6, float(kappa))
	c_max = max(0.0, float(max_closing_speed_m_s))
	return max(0.0, 2.0 * s * max(0.0, float(height_m)) / (kappa * dt) - c_max / kappa)


def lateral_accel_capacity(
	gain: float,
	flow_admissible_norm_s: float,
	max_closing_speed_m_s: float,
	kappa: float,
) -> float:
	"""Maximum lateral platform acceleration rejected by ``gain``."""
	omega_adm = max(1e-6, float(flow_admissible_norm_s))
	kappa = max(1e-6, float(kappa))
	closing_term = max(0.0, float(max_closing_speed_m_s)) / kappa
	return omega_adm * max(0.0, float(gain) - closing_term)


def compute_lateral_gate(
	peak_accel: float,
	flow_admissible_norm_s: float,
	max_closing_speed_m_s: float,
	kappa: float,
	probe_gain: float,
	near_field_height_m: float,
	leg_clearance_m: float,
	control_period_sec: float,
	ceiling_safety_factor: float = 0.5,
	ceiling_margin: float = 0.8,
) -> LateralGateResult:
	"""Build an independent lateral gain floor and verify its gain window."""
	kappa = max(1e-6, float(kappa))
	omega_adm = max(1e-6, float(flow_admissible_norm_s))
	c_max = max(0.0, float(max_closing_speed_m_s))
	margin = max(0.0, float(ceiling_margin))

	k_min = c_max / kappa + max(0.0, float(peak_accel)) / omega_adm
	k_probe = max(0.0, float(probe_gain))
	k_ceiling_probe = lateral_ceiling_gain_at_height(
		near_field_height_m, control_period_sec, kappa, c_max,
		ceiling_safety_factor,
	)
	k_ceiling_leg = lateral_ceiling_gain_at_height(
		leg_clearance_m, control_period_sec, kappa, c_max,
		ceiling_safety_factor,
	)

	k_target = margin * k_ceiling_leg
	k_floor = max(k_min, k_target)
	# A scheduled floor may never exceed the gain at descent entry: the
	# trajectory is a monotone decay from the FINAL_PROBE gain.
	k_floor = min(k_floor, k_probe) if k_probe > 0.0 else k_floor
	k_touchdown = k_floor

	probe_within_ceiling = k_probe <= k_ceiling_probe
	window_exists = k_min <= k_ceiling_leg
	start_above_floor = k_probe >= k_min
	floor_within_ceiling = k_floor <= k_ceiling_leg
	feasible = (
		probe_within_ceiling
		and window_exists
		and start_above_floor
		and floor_within_ceiling
	)

	return LateralGateResult(
		peak_accel=max(0.0, float(peak_accel)),
		k_min=float(k_min),
		k_probe=float(k_probe),
		k_touchdown=float(k_touchdown),
		k_target=float(k_target),
		k_floor=float(k_floor),
		k_descend_start=float(k_probe),
		k_ceiling_probe=float(k_ceiling_probe),
		k_ceiling_leg=float(k_ceiling_leg),
		ceiling_margin=float(margin),
		accel_capacity_floor=lateral_accel_capacity(
			k_floor, omega_adm, c_max, kappa
		),
		accel_capacity_ceiling=lateral_accel_capacity(
			k_ceiling_leg, omega_adm, c_max, kappa
		),
		probe_within_ceiling=bool(probe_within_ceiling),
		window_exists=bool(window_exists),
		start_above_floor=bool(start_above_floor),
		floor_within_ceiling=bool(floor_within_ceiling),
		feasible=bool(feasible),
	)


def critical_height(k_min: float, control_period_sec: float, safety: float = 1.0) -> float:
	"""Height where the safety-scaled de Croon ceiling reaches k_min."""
	s = max(1e-3, float(safety))
	return float(k_min) * float(control_period_sec) / (2.0 * s)


def ceiling_gain_at_height(
	height_m: float, control_period_sec: float, safety: float = 1.0
) -> float:
	"""Safety-scaled de Croon stability ceiling on k, at a given height.

	k_ceiling(h) = 2*s*h/dt -- the exact inverse of critical_height(), which
	solves k_ceiling(h_crit) = k_min. Evaluated at leg_clearance_m this is the
	largest gain that is still admissible at touchdown, and (scaled by
	ceiling_margin) the value the descent schedule now decays toward instead of
	k_min.
	"""
	s = max(1e-3, float(safety))
	dt = max(1e-6, float(control_period_sec))
	return 2.0 * s * max(0.0, float(height_m)) / dt


def compute_gate(
	peak_accel: float,
	descent_divergence_setpoint: float,
	initial_thrust_gain: float,
	control_period_sec: float,
	leg_clearance_m: float,
	ceiling_safety_factor: float = 0.5,
	min_divergence_setpoint: float = 0.01,
	ceiling_margin: float = 0.8,
	descend_start_gain: Optional[float] = None,
) -> GateResult:
	"""Turn a probed peak_accel into the descent gain window.

	ceiling_margin: how close to the safety-scaled ceiling AT LEG HEIGHT the
	    descent should settle (the Bode/bandwidth argument -- higher gain at
	    touchdown means better synchronization with the platform). This is a
	    SECOND multiplicative margin on top of ceiling_safety_factor; see the
	    module docstring before reading 0.8 as "80% of the stability limit".

	The returned k_floor -- not k_min -- is what scheduled_gain_at_time() decays
	toward. k_min survives as a hard floor UNDER k_target so that a marginally-
	feasible mission (k_min close to k_ceiling_leg, where margin<1 would place
	k_target below the disturbance-rejection floor) safely falls back to the old
	conservative behavior instead of under-gaining.
	"""
	d_star = max(float(min_divergence_setpoint), float(descent_divergence_setpoint))
	s = max(1e-3, float(ceiling_safety_factor))
	margin = max(0.0, float(ceiling_margin))

	k_min = max(0.0, float(peak_accel)) / d_star
	h_crit = critical_height(k_min, control_period_sec, s)
	k_explore = max(0.0, float(initial_thrust_gain))

	k_ceiling_leg = ceiling_gain_at_height(leg_clearance_m, control_period_sec, s)
	k_target = margin * k_ceiling_leg

	# The descent starts from wherever FINAL_PROBE left the gain (k_probe), not
	# from the far-field k_explore -- the gain has already been walked down the
	# ceiling during the approach and must not step back up.
	k_start = float(k_explore if descend_start_gain is None else descend_start_gain)
	window_exists = h_crit <= float(leg_clearance_m)
	start_above_floor = k_start >= k_min
	feasible = window_exists and start_above_floor

	# Never below the Herisse floor, and never above the gain the schedule starts
	# from (it only ever decays -- a k_floor above the start would turn the
	# "decay" into a step up, which is not what the trajectory means).
	k_floor = max(float(k_min), float(k_target))
	k_floor = min(k_floor, k_start) if k_start > 0.0 else k_floor

	return GateResult(
		k_min=float(k_min),
		h_crit=float(h_crit),
		k_explore=float(k_explore),
		feasible=bool(feasible),
		k_ceiling_leg=float(k_ceiling_leg),
		k_target=float(k_target),
		k_floor=float(k_floor),
		ceiling_margin=float(margin),
		k_descend_start=k_start,
		accel_capacity_floor=float(d_star * k_floor),
		accel_capacity_ceiling=float(d_star * k_ceiling_leg),
		window_exists=bool(window_exists),
		start_above_floor=bool(start_above_floor),
	)


def commanded_divergence_integral(
	elapsed_sec: float,
	divergence_setpoint: float,
	ramp_in_sec: float = 0.0,
) -> float:
	"""Integral of a raised-cosine ramp from 0 to D* followed by constant D*."""
	t = max(0.0, float(elapsed_sec))
	d = max(0.0, float(divergence_setpoint))
	T = max(0.0, float(ramp_in_sec))

	if d <= 1e-12 or t <= 0.0:
		return 0.0

	if T <= 1e-9:
		return d * t

	if t < T:
		return d * (0.5 * t - 0.5 * T / math.pi * math.sin(math.pi * t / T))

	return d * (t - 0.5 * T)


def scheduled_gain_at_time(
	elapsed_sec: float,
	descent_divergence_setpoint: float,
	k_floor: float,
	k_explore: float,
	d_star_ramp_in_sec: float = 0.0,
) -> float:
	"""K(t) = clamp(k_explore * exp(-integral(D*_cmd dt)), k_floor, k_explore).

	The TRAJECTORY is unchanged from before (a conservative exponential that, by
	design, decays faster than the height does under the same commanded D*). What
	changed is the ASYMPTOTE: k_floor is now GateResult.k_floor = max(k_min,
	ceiling_margin * k_ceiling_leg), so the descent settles just under the de
	Croon ceiling at leg height instead of sinking all the way to the Herisse
	floor k_min. See the module docstring.

	Depends only on elapsed time and the commanded D* -- NOT on any height
	estimate. h_pred/h0 stay diagnostic-only.

	The ceiling margin enters through k_floor, which compute_gate() derives from
	the known landing-gear height and the stability period.
	"""
	exponent = commanded_divergence_integral(
		elapsed_sec, descent_divergence_setpoint, d_star_ramp_in_sec
	)
	decay = math.exp(-exponent)
	return max(float(k_floor), min(float(k_explore), float(k_explore) * decay))


def critical_time(
	h0: float,
	descent_divergence_setpoint: float,
	h_crit: float,
	d_star_ramp_in_sec: float = 0.0,
) -> float:
	"""Predicted time for h(t)=h_crit under the same ramp-aware D* schedule."""
	d = float(descent_divergence_setpoint)
	h0 = float(h0)
	h_crit = float(h_crit)

	if h_crit <= 0.0 or h_crit >= h0 or d <= 1e-9:
		return float("inf")

	target_integral = math.log(h0 / h_crit)
	T = max(0.0, float(d_star_ramp_in_sec))

	if T <= 1e-9:
		return target_integral / d

	ramp_integral = commanded_divergence_integral(T, d, T)
	if target_integral >= ramp_integral:
		return target_integral / d + 0.5 * T

	lo, hi = 0.0, T
	for _ in range(40):
		mid = 0.5 * (lo + hi)
		if commanded_divergence_integral(mid, d, T) < target_integral:
			lo = mid
		else:
			hi = mid
	return 0.5 * (lo + hi)


def predicted_height(
	h0: float,
	descent_divergence_setpoint: float,
	elapsed_sec: float,
	d_star_ramp_in_sec: float = 0.0,
) -> float:
	exponent = commanded_divergence_integral(
		elapsed_sec, descent_divergence_setpoint, d_star_ramp_in_sec
	)
	return max(0.0, float(h0)) * math.exp(-exponent)


@dataclass
class MissionControl:
	divergence_setpoint: float = 0.0
	thrust_gain_override: Optional[float] = None
	lateral_p_scale: float = 1.0
	lateral_d_scale: float = 1.0
	roll_p_scale: Optional[float] = None
	roll_d_scale: Optional[float] = None
	pitch_p_scale: Optional[float] = None
	pitch_d_scale: Optional[float] = None
	enable_integral: bool = True
	substate: str = CENTER
	info: dict = field(default_factory=dict)


class MissionRoutine:
	def __init__(
		self,
		hover_thrust: float,
		control_period_sec: float = 0.5,
		descent_divergence_setpoint: float = 0.30,
		approach_divergence_setpoint: float = 0.12,
		final_probe_duration_sec: float = 6.0,
		final_probe_entry_ramp_sec: float = 1.5,
		fov_near_area_fraction: float = 0.55,
		probe_min_duration_sec: float = 15.0,
		leg_clearance_m: float = 0.20,
		enable_descent: bool = True,
		probe_only: bool = False,
		ceiling_safety_factor: float = 0.5,
		stability_dt_sec: Optional[float] = 1.0 / 30.0,
		initial_thrust_gain: float = 6.5,
		roll_d_gain: float = 3.5,
		pitch_d_gain: float = 3.5,
		roll_kappa: float = 1.0,
		pitch_kappa: float = 1.0,
		roll_flow_admissible_norm_s: float = 0.25,
		pitch_flow_admissible_norm_s: float = 0.25,
		max_closing_speed_m_s: float = 0.20,
		# --- Descent gain target: ride the ceiling at leg height, not the floor. ---
		# k(t) decays from k_explore toward max(k_min, ceiling_margin *
		# k_ceiling_leg) instead of toward k_min. 0.8 = settle at 80% of the
		# ALREADY-safety-derated ceiling (see module docstring: this is the second
		# of two multiplicative margins, not the only one).
		ceiling_margin: float = 0.8,
		probe_vert_accel_margin_m_s2: float = 0.05,
		probe_roll_accel_margin_m_s2: float = 0.01,
		probe_pitch_accel_margin_m_s2: float = 0.01,

		probe_attenuation_comp: float = 1.2,
		# Height at which the near-field trigger (FOV saturation / area_fraction)
		# actually fires. This is the ANCHOR for the whole gain schedule -- see
		# _compute_probe_gain(). It is a CAMERA-GEOMETRY constant (target diameter
		# vs FOV), not a live estimate, and unlike h0 it is directly calibratable
		# from a log: read relative_z_m at the FINAL_PROBE entry (analyse_log's
		# mission summary now prints it) and put that number here.
		near_field_height_m: float = 0.40,
		# --- Probe time constants: FAR field (APPROACH_PROBE). ---
		# Sized against the PLATFORM PERIOD (~17-18 s at the measured ~0.055-0.06
		# Hz), not the control rate: this phase exists precisely to see the slow
		# oscillation the short near-field hold structurally cannot.
		# probe_min_duration_sec is the TOTAL probing time the gate requires (across
		# both phases) and should likewise cover about a full platform period.
		far_probe_window_sec: float = 20.0,
		far_probe_decay_tau_sec: float = 20.0,
		far_probe_highpass_tau_sec: float = 40.0,
		# --- Probe time constants: NEAR field (FINAL_PROBE hold, after retune). ---
		# The SAME probe continues with these; the accumulated peak carries over and
		# is progressively superseded by the (more trustworthy) near-field samples.
		# near_probe_decay_tau_sec is the handoff knob -- it sets how fast the
		# carried far-field estimate is forgotten. Defaulted LONG (~ one platform
		# period) on purpose: an under-estimated peak_accel produces a too-low k_min
		# and a too-permissive gate, so forgetting the old estimate slowly is the
		# conservative direction. See PlatformProbe.retune().
		near_probe_window_sec: float = 2.0,
		near_probe_decay_tau_sec: float = 18.0,
		near_probe_highpass_tau_sec: float = 7.0,
		center_offset_threshold: float = 0.10,
		center_dwell_sec: float = 2.0,
		center_timeout_sec: float = 20.0,
		enable_center: bool = True,

		# Enhanced visual-only CENTER exit gate.
		enable_center_condition_gate: bool = True,
		center_condition_dwell_sec: float = 0.75,
		center_offset_radius_max: float = 0.12,
		center_flow_radius_max_norm_s: float = 0.25,
		center_timeout_allows_handoff: bool = False,
		d_star_ramp_in_sec: float = 3.0,
		center_to_probe_lateral_ramp_sec: float = 2.0,
		center_lateral_p_scale: float = 1.0,
		center_lateral_d_scale: float = 1.0,
		probe_lateral_p_scale: float = 1.0,
		probe_lateral_d_scale: float = 1.0,
	):
		self._dt = float(control_period_sec)
		self._stability_dt = (
			float(stability_dt_sec) if stability_dt_sec is not None else self._dt
		)

		self._d_star = max(0.0, float(descent_divergence_setpoint))
		self._approach_d_star = max(0.0, float(approach_divergence_setpoint))
		self._final_probe_duration = max(0.0, float(final_probe_duration_sec))
		self._final_probe_entry_ramp = max(0.0, float(final_probe_entry_ramp_sec))
		self._fov_near_area_fraction = clamp(fov_near_area_fraction, 0.0, 1.0)

		self._probe_min = max(0.0, float(probe_min_duration_sec))
		self._leg_clearance = float(leg_clearance_m)
		self._enable_descent = bool(enable_descent)
		self._probe_only = bool(probe_only)

		self._safety = max(1e-3, min(1.0, float(ceiling_safety_factor)))
		self._initial_thrust_gain = max(0.0, float(initial_thrust_gain))
		self._roll_d_gain = max(0.0, float(roll_d_gain))
		self._pitch_d_gain = max(0.0, float(pitch_d_gain))
		self._roll_kappa = max(1e-6, float(roll_kappa))
		self._pitch_kappa = max(1e-6, float(pitch_kappa))
		self._roll_flow_admissible = max(1e-6, float(roll_flow_admissible_norm_s))
		self._pitch_flow_admissible = max(1e-6, float(pitch_flow_admissible_norm_s))
		self._max_closing_speed = max(0.0, float(max_closing_speed_m_s))
		self._ceiling_margin = max(0.0, float(ceiling_margin))
		self._near_field_height = max(1e-3, float(near_field_height_m))

		self._enable_center = bool(enable_center)
		self._center_offset_thr = max(0.0, float(center_offset_threshold))
		self._center_dwell = max(0.0, float(center_dwell_sec))
		self._center_timeout = max(0.0, float(center_timeout_sec))

		self._enable_center_condition_gate = bool(enable_center_condition_gate)
		self._center_condition_dwell = max(0.0, float(center_condition_dwell_sec))
		self._center_offset_radius_max = max(0.0, float(center_offset_radius_max))
		self._center_flow_radius_max = max(0.0, float(center_flow_radius_max_norm_s))
		self._center_timeout_allows_handoff = bool(center_timeout_allows_handoff)

		self._d_star_ramp_in = max(0.0, float(d_star_ramp_in_sec))
		self._lateral_ramp = max(0.0, float(center_to_probe_lateral_ramp_sec))

		self._center_lateral_p_scale = max(0.0, float(center_lateral_p_scale))
		self._center_lateral_d_scale = max(0.0, float(center_lateral_d_scale))
		self._probe_lateral_p_scale = max(0.0, float(probe_lateral_p_scale))
		self._probe_lateral_d_scale = max(0.0, float(probe_lateral_d_scale))

		self._tm = ThrustModel(hover_thrust)

		# Three parallel probes share the same far/near timing design.
		self._probe = PlatformProbe(
			self._tm,
			highpass_tau_sec=float(far_probe_highpass_tau_sec),
			percentile_window_sec=float(far_probe_window_sec),
			peak_decay_tau_sec=float(far_probe_decay_tau_sec),
			accel_margin=float(probe_vert_accel_margin_m_s2),
			probe_attenuation_comp=float(probe_attenuation_comp),
		)
		self._roll_probe = PlatformProbe(
			self._tm,
			highpass_tau_sec=float(far_probe_highpass_tau_sec),
			percentile_window_sec=float(far_probe_window_sec),
			peak_decay_tau_sec=float(far_probe_decay_tau_sec),
			accel_margin=float(probe_roll_accel_margin_m_s2),
			probe_attenuation_comp=float(probe_attenuation_comp),
		)
		self._pitch_probe = PlatformProbe(
			self._tm,
			highpass_tau_sec=float(far_probe_highpass_tau_sec),
			percentile_window_sec=float(far_probe_window_sec),
			peak_decay_tau_sec=float(far_probe_decay_tau_sec),
			accel_margin=float(probe_pitch_accel_margin_m_s2),
			probe_attenuation_comp=float(probe_attenuation_comp),
		)
		self._far_probe_window = float(far_probe_window_sec)
		self._far_probe_decay_tau = float(far_probe_decay_tau_sec)
		self._far_probe_highpass_tau = float(far_probe_highpass_tau_sec)

		self._near_probe_window = float(near_probe_window_sec)
		self._near_probe_decay_tau = float(near_probe_decay_tau_sec)
		self._near_probe_highpass_tau = float(near_probe_highpass_tau_sec)

		self.gate = GateResult()
		self.roll_gate = LateralGateResult()
		self.pitch_gate = LateralGateResult()
		self.probe_result = ProbeResult()
		self.roll_probe_result = ProbeResult()
		self.pitch_probe_result = ProbeResult()
		# peak_accel at the instant of the far->near handoff, frozen for diagnostics:
		# comparing it against the final peak_accel shows how much the near field
		# actually revised the far-field estimate (and in which direction).
		self.peak_accel_at_handoff: Optional[float] = None
		self.roll_peak_accel_at_handoff: Optional[float] = None
		self.pitch_peak_accel_at_handoff: Optional[float] = None

		self._substate = CENTER if self._enable_center else APPROACH_PROBE
		self._t0: Optional[float] = None
		self._h0: Optional[float] = None
		self._k_explore = self._initial_thrust_gain

		self._centered_since: Optional[float] = None
		self._center_start_t: Optional[float] = None
		self._t_approach_entry: Optional[float] = None
		self._t_final_probe_entry: Optional[float] = None
		self._t_final_probe_hold_start: Optional[float] = None
		self._t_descend_start: Optional[float] = None
		self._t_landed: Optional[float] = None

	def mark_landed(self, t: float) -> None:
		"""Latch the terminal LANDED substate. Called by bee_node when its own
		touchdown detector fires (_enter_landed_phase).

		The mission routine cannot detect touchdown itself -- it is visual-only and
		has no height -- so bee_node owns the detection and simply TELLS us. Without
		this, mission_substate stayed "descend" through the entire post-touchdown
		zero-thrust hold, and every statistic computed per-phase from the log mixed
		flying rows with sitting-on-the-platform rows.

		Idempotent: the first call wins, so a repeated latch cannot move t_landed.
		"""
		if self._substate == LANDED:
			return
		self._substate = LANDED
		self._t_landed = float(t)

	def reset(self) -> None:
		for probe in (self._probe, self._roll_probe, self._pitch_probe):
			probe.reset()
			probe.retune(
				highpass_tau_sec=self._far_probe_highpass_tau,
				percentile_window_sec=self._far_probe_window,
				peak_decay_tau_sec=self._far_probe_decay_tau,
			)
		self.gate = GateResult()
		self.roll_gate = LateralGateResult()
		self.pitch_gate = LateralGateResult()
		self.probe_result = ProbeResult()
		self.roll_probe_result = ProbeResult()
		self.pitch_probe_result = ProbeResult()
		self.peak_accel_at_handoff = None
		self.roll_peak_accel_at_handoff = None
		self.pitch_peak_accel_at_handoff = None

		self._substate = CENTER if self._enable_center else APPROACH_PROBE
		self._t0 = None
		self._h0 = None
		self._k_explore = self._initial_thrust_gain

		self._centered_since = None
		self._center_start_t = None
		self._t_approach_entry = None
		self._t_final_probe_entry = None
		self._t_final_probe_hold_start = None
		self._t_descend_start = None
		self._t_landed = None

	def start(self, t: float, start_height_m: float) -> None:
		self.reset()
		self._t0 = float(t)
		self._h0 = max(1e-3, float(start_height_m))
		self._k_explore = self._initial_thrust_gain

	@property
	def substate(self) -> str:
		return self._substate

	def probe_telemetry(self) -> dict:
		"""Return per-step vertical, roll and pitch probe diagnostics."""
		active = self._substate in (APPROACH_PROBE, FINAL_PROBE)
		near = self._t_final_probe_hold_start is not None

		# The PER-STEP fields are None whenever the probe is not running (CENTER,
		# DESCEND, INFEASIBLE) -- the probe simply stops being updated there, so its
		# last values would otherwise persist and draw a flat line that looks like a
		# measurement but is just a frozen register. None -> blank CSV cell -> a gap
		# in the plot, which is the truth.
		#
		# The ENVELOPE (probe_peak_accel) and the handoff value are NOT blanked:
		# they are the gate's inputs and stay meaningful for the whole descent
		# (k_min = peak/D* is what the schedule's floor was built from).
		return {
			"probe_active": active,
			"probe_phase": ("near" if near else "far") if active else "",
			"probe_accel": self._probe.accel if active else None,
			"probe_mean_accel": self._probe.mean_accel if active else None,
			"probe_residual_accel": self._probe.residual_accel if active else None,
			"probe_percentile_accel": self._probe.percentile_accel if active else None,
			"probe_peak_accel": self._probe.peak_accel,
			"probe_peak_decay_tau_sec": self._probe.peak_decay_tau_sec,
			"probe_elapsed_sec": self.probe_result.duration_sec,
			"probe_total_elapsed_sec": self.probe_result.total_duration_sec,
			"probe_peak_accel_at_handoff": self.peak_accel_at_handoff,
			"roll_probe_accel": self._roll_probe.accel if active else None,
			"roll_probe_mean_accel": self._roll_probe.mean_accel if active else None,
			"roll_probe_residual_accel": self._roll_probe.residual_accel if active else None,
			"roll_probe_percentile_accel": self._roll_probe.percentile_accel if active else None,
			"roll_probe_peak_accel": self._roll_probe.peak_accel,
			"roll_probe_peak_accel_at_handoff": self.roll_peak_accel_at_handoff,
			"pitch_probe_accel": self._pitch_probe.accel if active else None,
			"pitch_probe_mean_accel": self._pitch_probe.mean_accel if active else None,
			"pitch_probe_residual_accel": self._pitch_probe.residual_accel if active else None,
			"pitch_probe_percentile_accel": self._pitch_probe.percentile_accel if active else None,
			"pitch_probe_peak_accel": self._pitch_probe.peak_accel,
			"pitch_probe_peak_accel_at_handoff": self.pitch_peak_accel_at_handoff,
			"vertical_probe_feasible": self.gate.feasible,
			"roll_probe_feasible": self.roll_gate.feasible,
			"pitch_probe_feasible": self.pitch_gate.feasible,
			"landing_feasible": self.feasible,
			"k_probe": self._compute_probe_gain(),
			"near_field_height_m": self._near_field_height,
		}

	def _compute_probe_gain(self) -> float:
		"""The gain flown through FINAL_PROBE -- and the anchor of the schedule.

		k_explore (6.5) is a FAR-FIELD gain. The de Croon ceiling shrinks with
		height, k_ceiling(h) = 2*s*h/dt, so k_explore is only admissible above
		h = k_explore*dt/(2*s). At the flown constants that is ~0.46 m -- but
		FINAL_PROBE fires on FOV saturation, i.e. WELL BELOW it. Holding k_explore
		into the near field therefore probes ABOVE the stability ceiling, and since
		the probe measures the THRUST-COMMAND RESIDUAL, any resulting self-induced
		oscillation is counted as platform acceleration. That does not merely risk
		instability: it corrupts the one number the feasibility gate rests on.

		So the near-field probe is flown at the same fraction of the ceiling that
		the descent targets, evaluated at the height the probe actually happens at:

		    k_probe = min(k_explore, ceiling_margin * k_ceiling(near_field_height))

		The min() means that if the near-field trigger fires while still high enough
		that k_explore is already under the ceiling, nothing is dropped -- the gain
		is only ever reduced to become admissible, never raised.
		"""
		ceiling = ceiling_gain_at_height(
			self._near_field_height, self._stability_dt, self._safety
		)
		return min(self._initial_thrust_gain, self._ceiling_margin * ceiling)

	@property
	def probe_gain(self) -> float:
		"""Gain held flat through FINAL_PROBE; the descent schedule starts here."""
		return self._compute_probe_gain()

	@property
	def feasible(self) -> bool:
		return self.gate.feasible and self.roll_gate.feasible and self.pitch_gate.feasible

	@property
	def vertical_feasible(self) -> bool:
		return bool(self.gate.feasible)

	@property
	def roll_feasible(self) -> bool:
		return bool(self.roll_gate.feasible)

	@property
	def pitch_feasible(self) -> bool:
		return bool(self.pitch_gate.feasible)

	def update(
		self,
		t: float,
		dt: float,
		last_thrust_cmd: float,
		offset_x: float = 0.0,
		offset_y: float = 0.0,
		target_found: bool = False,
		flow_x_norm_s: float = 0.0,
		flow_y_norm_s: float = 0.0,
		flow_valid: bool = False,
		area_fraction: float = 0.0,
		fov_saturated: bool = False,
		last_vertical_accel_cmd: Optional[float] = None,
		last_roll_accel_cmd: float = 0.0,
		last_pitch_accel_cmd: float = 0.0,
	) -> MissionControl:
		t = float(t)
		dt = max(1e-3, float(dt))

		if self._t0 is None:
			self._t0 = t
		if self._h0 is None:
			self._h0 = 5.0

		# TERMINAL. Checked first: once landed, nothing else runs -- no probe update,
		# no gain schedule, no phase logic. bee_node overrides the setpoint with its
		# own zero-thrust command anyway, so the values here only ever reach the log.
		if self._substate == LANDED:
			return self._do_landed(t)

		if self._substate == CENTER:
			return self._do_center(
				t, offset_x, offset_y, target_found,
				flow_x_norm_s, flow_y_norm_s, flow_valid,
			)

		if self._substate == APPROACH_PROBE:
			return self._do_approach_probe(
				t,
				dt,
				last_thrust_cmd,
				last_vertical_accel_cmd,
				last_roll_accel_cmd,
				last_pitch_accel_cmd,
				offset_x,
				offset_y,
				target_found,
				area_fraction,
				fov_saturated,
			)

		if self._substate == FINAL_PROBE:
			return self._do_final_probe(
				t, dt, last_thrust_cmd,
				last_vertical_accel_cmd, last_roll_accel_cmd, last_pitch_accel_cmd,
			)

		if self._substate == PROBE_HOLD:
			return self._do_probe_hold(t)

		if self._substate == INFEASIBLE:
			return self._do_infeasible(t)

		return self._do_descend(t)

	def _update_probes(
		self,
		last_thrust_cmd: float,
		last_vertical_accel_cmd: Optional[float],
		last_roll_accel_cmd: float,
		last_pitch_accel_cmd: float,
		dt: float,
	) -> None:
		if last_vertical_accel_cmd is None:
			self._probe.update(last_thrust_cmd, dt)
		else:
			self._probe.update_accel(last_vertical_accel_cmd, dt)
		self._roll_probe.update_accel(last_roll_accel_cmd, dt)
		self._pitch_probe.update_accel(last_pitch_accel_cmd, dt)

	def _retune_probes(self) -> None:
		for probe in (self._probe, self._roll_probe, self._pitch_probe):
			probe.retune(
				highpass_tau_sec=self._near_probe_highpass_tau,
				percentile_window_sec=self._near_probe_window,
				peak_decay_tau_sec=self._near_probe_decay_tau,
			)

	def _refresh_probe_results(
		self,
		min_duration_sec: float,
		min_total_duration_sec: float,
	) -> None:
		kwargs = dict(
			min_duration_sec=min_duration_sec,
			min_total_duration_sec=min_total_duration_sec,
		)
		self.probe_result = self._probe.result(**kwargs)
		self.roll_probe_result = self._roll_probe.result(**kwargs)
		self.pitch_probe_result = self._pitch_probe.result(**kwargs)

	def _compute_lateral_gates(self) -> None:
		roll_probe_gain = self._roll_d_gain * self._probe_lateral_d_scale
		pitch_probe_gain = self._pitch_d_gain * self._probe_lateral_d_scale
		self.roll_gate = compute_lateral_gate(
			peak_accel=self.roll_probe_result.peak_accel,
			flow_admissible_norm_s=self._roll_flow_admissible,
			max_closing_speed_m_s=self._max_closing_speed,
			kappa=self._roll_kappa,
			probe_gain=roll_probe_gain,
			near_field_height_m=self._near_field_height,
			leg_clearance_m=self._leg_clearance,
			control_period_sec=self._stability_dt,
			ceiling_safety_factor=self._safety,
			ceiling_margin=self._ceiling_margin,
		)
		self.pitch_gate = compute_lateral_gate(
			peak_accel=self.pitch_probe_result.peak_accel,
			flow_admissible_norm_s=self._pitch_flow_admissible,
			max_closing_speed_m_s=self._max_closing_speed,
			kappa=self._pitch_kappa,
			probe_gain=pitch_probe_gain,
			near_field_height_m=self._near_field_height,
			leg_clearance_m=self._leg_clearance,
			control_period_sec=self._stability_dt,
			ceiling_safety_factor=self._safety,
			ceiling_margin=self._ceiling_margin,
		)


	def _gate_failure_reasons(self) -> list[str]:
		"""Return concise, axis-specific reasons for an INFEASIBLE decision."""
		reasons: list[str] = []

		if not self.gate.window_exists:
			reasons.append(
				"VERTICAL no gain window: "
				f"K_min={self.gate.k_min:.3f} > "
				f"K_ceiling_leg={self.gate.k_ceiling_leg:.3f}"
			)
		elif not self.gate.start_above_floor:
			reasons.append(
				"VERTICAL FINAL_PROBE gain below floor: "
				f"K_probe={self.gate.k_descend_start:.3f} < "
				f"K_min={self.gate.k_min:.3f}"
			)

		for label, gate in (("ROLL", self.roll_gate), ("PITCH", self.pitch_gate)):
			if not gate.probe_within_ceiling:
				reasons.append(
					f"{label} probe gain above near-field ceiling: "
					f"K_probe={gate.k_probe:.3f} > "
					f"K_ceiling_probe={gate.k_ceiling_probe:.3f}"
				)
			if not gate.window_exists:
				reasons.append(
					f"{label} no gain window: "
					f"K_min={gate.k_min:.3f} > "
					f"K_ceiling_leg={gate.k_ceiling_leg:.3f}"
				)
			elif not gate.start_above_floor:
				reasons.append(
					f"{label} FINAL_PROBE gain below floor: "
					f"K_probe={gate.k_probe:.3f} < K_min={gate.k_min:.3f}"
				)
			elif not gate.floor_within_ceiling:
				reasons.append(
					f"{label} scheduled floor above touchdown ceiling: "
					f"K_floor={gate.k_floor:.3f} > "
					f"K_ceiling_leg={gate.k_ceiling_leg:.3f}"
				)

		if not self._enable_descent:
			reasons.append("DESCENT disabled by configuration")

		return reasons


	def _do_center(
		self,
		t: float,
		offset_x: float,
		offset_y: float,
		target_found: bool,
		flow_x_norm_s: float,
		flow_y_norm_s: float,
		flow_valid: bool,
	) -> MissionControl:
		"""Hold visual hover until the target is centred AND laterally settled.

		This is mission-phase logic, not command formation. ControlLaw continues
		receiving the real target/flow throughout CENTER; this method only decides
		when the mission is allowed to enter APPROACH_PROBE.
		"""
		if self._center_start_t is None:
			self._center_start_t = t

		offset_radius = math.hypot(float(offset_x), float(offset_y))
		flow_radius = (
			math.hypot(float(flow_x_norm_s), float(flow_y_norm_s))
			if flow_valid else float("inf")
		)

		if self._enable_center_condition_gate:
			centered = (
				bool(target_found)
				and bool(flow_valid)
				and offset_radius <= self._center_offset_radius_max
				and flow_radius <= self._center_flow_radius_max
			)
			required_dwell = self._center_condition_dwell
		else:
			centered = self._is_centered(offset_x, offset_y, target_found)
			required_dwell = self._center_dwell

		if centered:
			if self._centered_since is None:
				self._centered_since = t
			dwell = t - self._centered_since
		else:
			self._centered_since = None
			dwell = 0.0

		elapsed = t - self._center_start_t
		settled = centered and dwell >= required_dwell
		timed_out = (
			self._center_timeout > 0.0
			and elapsed >= self._center_timeout
		)
		timeout_handoff = (
			timed_out and self._center_timeout_allows_handoff
		)

		if settled or timeout_handoff:
			for probe in (self._probe, self._roll_probe, self._pitch_probe):
				probe.reset()
				probe.retune(
					highpass_tau_sec=self._far_probe_highpass_tau,
					percentile_window_sec=self._far_probe_window,
					peak_decay_tau_sec=self._far_probe_decay_tau,
				)
			self.probe_result = ProbeResult()
			self.roll_probe_result = ProbeResult()
			self.pitch_probe_result = ProbeResult()
			self.peak_accel_at_handoff = None
			self.roll_peak_accel_at_handoff = None
			self.pitch_peak_accel_at_handoff = None
			self._t_approach_entry = t
			self._substate = APPROACH_PROBE

			return MissionControl(
				divergence_setpoint=0.0,
				thrust_gain_override=self._k_explore,
				lateral_p_scale=self._center_lateral_p_scale,
				lateral_d_scale=self._center_lateral_d_scale,
				enable_integral=True,
				substate=APPROACH_PROBE,
				info={
					"event": "center_done",
					"centered_ok": bool(settled),
					"center_timed_out": bool(timed_out),
					"center_timeout_handoff": bool(timeout_handoff),
					"center_elapsed_sec": elapsed,
					"center_dwell_sec": dwell,
					"center_offset_radius": offset_radius,
					"center_flow_radius_norm_s": flow_radius,
				},
			)

		return MissionControl(
			divergence_setpoint=0.0,
			thrust_gain_override=self._k_explore,
			lateral_p_scale=self._center_lateral_p_scale,
			lateral_d_scale=self._center_lateral_d_scale,
			enable_integral=True,
			substate=CENTER,
			info={
				"offset_x": offset_x,
				"offset_y": offset_y,
				"target_found": target_found,
				"flow_valid": flow_valid,
				"centered": centered,
				"center_dwell_sec": dwell,
				"center_required_dwell_sec": required_dwell,
				"center_elapsed_sec": elapsed,
				"center_timed_out": bool(timed_out),
				"center_timeout_allows_handoff": bool(
					self._center_timeout_allows_handoff
				),
				"center_offset_radius": offset_radius,
				"center_flow_radius_norm_s": flow_radius,
				"center_offset_radius_max": self._center_offset_radius_max,
				"center_flow_radius_max_norm_s": self._center_flow_radius_max,
			},
		)

	def _do_approach_probe(
		self,
		t: float,
		dt: float,
		last_thrust_cmd: float,
		last_vertical_accel_cmd: Optional[float],
		last_roll_accel_cmd: float,
		last_pitch_accel_cmd: float,
		offset_x: float,
		offset_y: float,
		target_found: bool,
		area_fraction: float,
		fov_saturated: bool,
	) -> MissionControl:
		if self._t_approach_entry is None:
			self._t_approach_entry = t

		elapsed = t - self._t_approach_entry

		lateral_frac = (
			1.0 if self._lateral_ramp <= 1e-9 else min(1.0, elapsed / self._lateral_ramp)
		)
		lateral_blend = raised_cosine01(lateral_frac)

		lateral_p = self._center_lateral_p_scale + (
			self._probe_lateral_p_scale - self._center_lateral_p_scale
		) * lateral_blend

		lateral_d = self._center_lateral_d_scale + (
			self._probe_lateral_d_scale - self._center_lateral_d_scale
		) * lateral_blend

		approach_frac = (
			1.0 if self._d_star_ramp_in <= 1e-9 else min(1.0, elapsed / self._d_star_ramp_in)
		)
		approach_blend = raised_cosine01(approach_frac)
		d_approach_cmd = self._approach_d_star * approach_blend

		# GAIN DROP, far -> near. k(t) decays from k_explore toward k_probe on the
		# integral of the COMMANDED approach D*. Because h(t) also decays as
		# exp(-integral D* dt), and k_ceiling is proportional to h, this decay is
		# PARALLEL to the shrinking stability ceiling -- the gain comes down at the
		# same rate the ceiling does, instead of holding at a far-field value while
		# the ceiling collapses beneath it. It bottoms out at k_probe, the gain the
		# near-field probe is flown at (see _compute_probe_gain).
		k_approach = scheduled_gain_at_time(
			elapsed_sec=elapsed,
			descent_divergence_setpoint=self._approach_d_star,
			k_floor=self._compute_probe_gain(),
			k_explore=self._k_explore,
			d_star_ramp_in_sec=self._d_star_ramp_in,
		)

		# FAR-field probing supplies the long observation interval needed to resolve
		# the platform motion. The three envelopes carry into FINAL_PROBE.
		#
		# CAVEAT, deliberate: k is RAMPING during this phase, so the loop gain that
		# shapes the thrust-command residual is itself moving, and the far-field
		# peak_accel is contaminated by that. This is accepted rather than fixed,
		# because (a) the far estimate exists to be SUPERSEDED by the near-field one
		# (which is flown at a constant k_probe -- see _do_final_probe), and (b) the
		# alternative, dropping the gain before probing starts, would fly the whole
		# approach far below the ceiling and throw away the bandwidth this design is
		# built to exploit.
		self._update_probes(
			last_thrust_cmd, last_vertical_accel_cmd,
			last_roll_accel_cmd, last_pitch_accel_cmd, dt,
		)
		self._refresh_probe_results(0.0, self._probe_min)

		near_field = self._near_field_reached(
			offset_x=offset_x,
			offset_y=offset_y,
			target_found=target_found,
			area_fraction=area_fraction,
			fov_saturated=fov_saturated,
		)

		if near_field:
			self._substate = FINAL_PROBE
			self._t_final_probe_entry = t
			self._t_final_probe_hold_start = None
			# All three accumulated envelopes carry into the near-field hold.

			return MissionControl(
				divergence_setpoint=d_approach_cmd,
				thrust_gain_override=k_approach,
				lateral_p_scale=lateral_p,
				lateral_d_scale=lateral_d,
				enable_integral=True,
				substate=FINAL_PROBE,
				info={
					"event": "final_probe_start",
					"k": k_approach,
					"area_fraction": float(area_fraction),
					"fov_saturated": bool(fov_saturated),
					"near_field": True,
					"approach_elapsed_sec": elapsed,
					"approach_d_star_cmd": d_approach_cmd,
				},
			)

		return MissionControl(
			divergence_setpoint=d_approach_cmd,
			thrust_gain_override=k_approach,
			lateral_p_scale=lateral_p,
			lateral_d_scale=lateral_d,
			enable_integral=True,
			substate=APPROACH_PROBE,
			info={
				"k": k_approach,
				"k_probe": self._compute_probe_gain(),
				"area_fraction": float(area_fraction),
				"fov_saturated": bool(fov_saturated),
				"near_field": False,
				"approach_elapsed_sec": elapsed,
				"approach_d_star_cmd": d_approach_cmd,
				"approach_ramp_frac": approach_blend,
				"lateral_ramp_frac": lateral_blend,
				"peak_accel": self.probe_result.peak_accel,
				"roll_peak_accel": self.roll_probe_result.peak_accel,
				"pitch_peak_accel": self.pitch_probe_result.peak_accel,
				"probe_elapsed_sec": self.probe_result.total_duration_sec,
			},
		)

	def _do_final_probe(
		self,
		t: float,
		dt: float,
		last_thrust_cmd: float,
		last_vertical_accel_cmd: Optional[float],
		last_roll_accel_cmd: float,
		last_pitch_accel_cmd: float,
	) -> MissionControl:
		if self._t_final_probe_entry is None:
			self._t_final_probe_entry = t

		entry_elapsed = t - self._t_final_probe_entry

		# Smooth D* from approach descent back to zero before the near-field probe.
		if self._final_probe_entry_ramp > 1e-9 and entry_elapsed < self._final_probe_entry_ramp:
			frac = raised_cosine01(entry_elapsed / self._final_probe_entry_ramp)
			d_cmd = self._approach_d_star * (1.0 - frac)
			self._update_probes(
				last_thrust_cmd, last_vertical_accel_cmd,
				last_roll_accel_cmd, last_pitch_accel_cmd, dt,
			)
			self._refresh_probe_results(0.0, self._probe_min)

			return MissionControl(
				divergence_setpoint=d_cmd,
				thrust_gain_override=self._compute_probe_gain(),
				lateral_p_scale=self._probe_lateral_p_scale,
				lateral_d_scale=self._probe_lateral_d_scale,
				enable_integral=True,
				substate=FINAL_PROBE,
				info={
					"event": "final_probe_entry_ramp",
					"k": self._compute_probe_gain(),
					"final_probe_entry_elapsed_sec": entry_elapsed,
					"final_probe_entry_ramp_frac": frac,
					"probe_elapsed_sec": self.probe_result.total_duration_sec,
				},
			)

		if self._t_final_probe_hold_start is None:
			self._t_final_probe_hold_start = t
			# Retune all probes together while preserving their far-field envelopes.
			self.peak_accel_at_handoff = self._probe.peak_accel
			self.roll_peak_accel_at_handoff = self._roll_probe.peak_accel
			self.pitch_peak_accel_at_handoff = self._pitch_probe.peak_accel
			self._retune_probes()

		self._update_probes(
			last_thrust_cmd, last_vertical_accel_cmd,
			last_roll_accel_cmd, last_pitch_accel_cmd, dt,
		)
		# ready requires BOTH: a full near-field hold, AND enough TOTAL probing
		# across both phases (~ one platform period) -- the hold alone is far too
		# short to supply the latter, so this is what keeps a fast FOV saturation
		# from gating on a fraction of one platform cycle.
		self._refresh_probe_results(self._final_probe_duration, self._probe_min)

		if (
			self.probe_result.ready
			and self.roll_probe_result.ready
			and self.pitch_probe_result.ready
		):
			self.gate = compute_gate(
				peak_accel=self.probe_result.peak_accel,
				descent_divergence_setpoint=self._d_star,
				initial_thrust_gain=self._initial_thrust_gain,
				control_period_sec=self._stability_dt,
				leg_clearance_m=self._leg_clearance,
				ceiling_safety_factor=self._safety,
				ceiling_margin=self._ceiling_margin,
				descend_start_gain=self._compute_probe_gain(),
			)
			self._compute_lateral_gates()

			if self._probe_only:
				self._substate = PROBE_HOLD
				return self._do_probe_hold(t, just_entered=True)

			vertical_probe_ok = self.vertical_feasible
			roll_probe_ok = self.roll_feasible
			pitch_probe_ok = self.pitch_feasible
			if (
				vertical_probe_ok
				and roll_probe_ok
				and pitch_probe_ok
				and self._enable_descent
			):
				self._substate = DESCEND
				self._t_descend_start = t
				return self._do_descend(t, just_entered=True)

			self._substate = INFEASIBLE
			return self._do_infeasible(t, just_entered=True)

		# THE PROBE ITSELF. k is held FLAT at k_probe here -- deliberately, twice
		# over: (1) it is the admissible gain at this height (k_explore is above the
		# ceiling in the near field, and probing there would feed self-induced
		# oscillation straight into peak_accel), and (2) a gain that MOVES during
		# the probe moves the closed-loop transfer function with it, so the residual
		# would mix platform motion with our own gain sweep. D*=0 in this phase, so
		# the exp(-integral D*) schedule holds k constant here of its own accord --
		# the flatness is structural, not a special case.
		return MissionControl(
			divergence_setpoint=0.0,
			thrust_gain_override=self._compute_probe_gain(),
			lateral_p_scale=self._probe_lateral_p_scale,
			lateral_d_scale=self._probe_lateral_d_scale,
			enable_integral=True,
			substate=FINAL_PROBE,
			info={
				"event": "final_probe_hold",
				"k": self._compute_probe_gain(),
				"peak_accel": self.probe_result.peak_accel,
				"roll_peak_accel": self.roll_probe_result.peak_accel,
				"pitch_peak_accel": self.pitch_probe_result.peak_accel,
				"peak_accel_at_handoff": self.peak_accel_at_handoff,
				"roll_peak_accel_at_handoff": self.roll_peak_accel_at_handoff,
				"pitch_peak_accel_at_handoff": self.pitch_peak_accel_at_handoff,
				"hold_elapsed_sec": self.probe_result.duration_sec,
				"probe_total_elapsed_sec": self.probe_result.total_duration_sec,
				"hold_min_sec": self._final_probe_duration,
				"probe_total_min_sec": self._probe_min,
				"k_explore": self._k_explore,
			},
		)

	def _do_descend(self, t: float, just_entered: bool = False) -> MissionControl:
		elapsed = t - (self._t_descend_start if self._t_descend_start is not None else t)

		# Each translational axis now follows the same exponential trajectory but
		# keeps its own independently probed disturbance floor.
		k = scheduled_gain_at_time(
			elapsed_sec=elapsed,
			descent_divergence_setpoint=self._d_star,
			k_floor=self.gate.k_floor,
			k_explore=self.gate.k_descend_start,
			d_star_ramp_in_sec=self._d_star_ramp_in,
		)
		roll_k = scheduled_gain_at_time(
			elapsed_sec=elapsed,
			descent_divergence_setpoint=self._d_star,
			k_floor=self.roll_gate.k_floor,
			k_explore=self.roll_gate.k_descend_start,
			d_star_ramp_in_sec=self._d_star_ramp_in,
		)
		pitch_k = scheduled_gain_at_time(
			elapsed_sec=elapsed,
			descent_divergence_setpoint=self._d_star,
			k_floor=self.pitch_gate.k_floor,
			k_explore=self.pitch_gate.k_descend_start,
			d_star_ramp_in_sec=self._d_star_ramp_in,
		)

		roll_ratio = (
			roll_k / self.roll_gate.k_descend_start
			if self.roll_gate.k_descend_start > 1e-9 else 1.0
		)
		pitch_ratio = (
			pitch_k / self.pitch_gate.k_descend_start
			if self.pitch_gate.k_descend_start > 1e-9 else 1.0
		)
		roll_p_scale = self._probe_lateral_p_scale * roll_ratio
		roll_d_scale = self._probe_lateral_d_scale * roll_ratio
		pitch_p_scale = self._probe_lateral_p_scale * pitch_ratio
		pitch_d_scale = self._probe_lateral_d_scale * pitch_ratio

		# Legacy shared fields remain populated for compatibility and represent the
		# most demanding active lateral axis. ControlLaw receives the specific
		# per-axis fields below.
		lateral_p_scale = max(roll_p_scale, pitch_p_scale)
		lateral_d_scale = max(roll_d_scale, pitch_d_scale)
		h_pred = predicted_height(self._h0, self._d_star, elapsed, self._d_star_ramp_in)

		if self._d_star_ramp_in <= 1e-9:
			linear_frac = 1.0
		else:
			linear_frac = min(1.0, elapsed / self._d_star_ramp_in)

		ramp_frac = raised_cosine01(linear_frac)
		d_star_cmd = self._d_star * ramp_frac

		return MissionControl(
			divergence_setpoint=d_star_cmd,
			thrust_gain_override=k,
			lateral_p_scale=lateral_p_scale,
			lateral_d_scale=lateral_d_scale,
			roll_p_scale=roll_p_scale,
			roll_d_scale=roll_d_scale,
			pitch_p_scale=pitch_p_scale,
			pitch_d_scale=pitch_d_scale,
			enable_integral=False,
			substate=DESCEND,
			info={
				"just_entered": just_entered,
				"event": "descent_start" if just_entered else "",
				"h_pred": h_pred,
				"k": k,
				"k_min": self.gate.k_min,
				"k_floor": self.gate.k_floor,
				"k_target": self.gate.k_target,
				"k_ceiling_leg": self.gate.k_ceiling_leg,
				"k_explore": self.gate.k_explore,
				"k_descend_start": self.gate.k_descend_start,
				"h_crit": self.gate.h_crit,
				"vertical_accel_capacity_floor": self.gate.accel_capacity_floor,
				"vertical_accel_capacity_ceiling": self.gate.accel_capacity_ceiling,
				"roll_k": roll_k,
				"roll_k_floor": self.roll_gate.k_floor,
				"roll_k_target": self.roll_gate.k_target,
				"roll_k_ceiling_leg": self.roll_gate.k_ceiling_leg,
				"roll_accel_capacity_floor": self.roll_gate.accel_capacity_floor,
				"roll_accel_capacity_ceiling": self.roll_gate.accel_capacity_ceiling,
				"pitch_k": pitch_k,
				"pitch_k_floor": self.pitch_gate.k_floor,
				"pitch_k_target": self.pitch_gate.k_target,
				"pitch_k_ceiling_leg": self.pitch_gate.k_ceiling_leg,
				"pitch_accel_capacity_floor": self.pitch_gate.accel_capacity_floor,
				"pitch_accel_capacity_ceiling": self.pitch_gate.accel_capacity_ceiling,
				"roll_p_scale": roll_p_scale,
				"roll_d_scale": roll_d_scale,
				"pitch_p_scale": pitch_p_scale,
				"pitch_d_scale": pitch_d_scale,
				"elapsed_sec": elapsed,
				"t_crit_sec": critical_time(
					self._h0,
					self._d_star,
					self.gate.h_crit,
					self._d_star_ramp_in,
				),
				"d_star_ramp_frac": ramp_frac,
				"d_star_ramp_linear_frac": linear_frac,
				"d_star_target": self._d_star,
			},
		)

	def _do_probe_hold(self, t: float, just_entered: bool = False) -> MissionControl:
		# Near-field hover (HOVER_PROBE_ONLY). Same ceiling argument as FINAL_PROBE:
		# we are close to the platform, so k_explore is not admissible here.
		return MissionControl(
			divergence_setpoint=0.0,
			thrust_gain_override=self._compute_probe_gain(),
			lateral_p_scale=self._probe_lateral_p_scale,
			lateral_d_scale=self._probe_lateral_d_scale,
			enable_integral=True,
			substate=PROBE_HOLD,
			info={
				"just_entered": just_entered,
				"probe_only": True,
				"peak_accel": self.probe_result.peak_accel,
				"roll_peak_accel": self.roll_probe_result.peak_accel,
				"pitch_peak_accel": self.pitch_probe_result.peak_accel,
				"k_min": self.gate.k_min,
				"h_crit": self.gate.h_crit,
				"k_explore": self.gate.k_explore,
				"vertical_feasible": self.vertical_feasible,
				"roll_feasible": self.roll_feasible,
				"pitch_feasible": self.pitch_feasible,
				"feasible_if_descended": self.feasible,
				"leg_clearance_m": self._leg_clearance,
			},
		)

	def _do_landed(self, t: float) -> MissionControl:
		"""Terminal hold after touchdown. Emits no gain and no divergence setpoint.

		thrust_gain_override is None (not k_probe, not 0.0) on purpose: "no opinion".
		bee_node is publishing its own zero-thrust landed setpoint by now, so any
		number we put here would be a fiction that only shows up in the log and in
		the gain-schedule plot -- exactly the kind of phantom that made the last
		run's descent statistics unreadable.
		"""
		return MissionControl(
			divergence_setpoint=0.0,
			thrust_gain_override=None,
			lateral_p_scale=0.0,
			lateral_d_scale=0.0,
			enable_integral=False,
			substate=LANDED,
			info={
				"event": "landed",
				"landed_since_sec": (t - self._t_landed) if self._t_landed is not None else 0.0,
				"peak_accel": self.probe_result.peak_accel,
				"roll_peak_accel": self.roll_probe_result.peak_accel,
				"pitch_peak_accel": self.pitch_probe_result.peak_accel,
				"k_min": self.gate.k_min,
				"k_floor": self.gate.k_floor,
				"feasible": self.feasible,
			},
		)

	def _do_infeasible(self, t: float, just_entered: bool = False) -> MissionControl:
		# Active visual hover at the near-field probe gains.
		reasons = self._gate_failure_reasons()
		reason = "; ".join(reasons) if reasons else "unknown feasibility failure"
		return MissionControl(
			divergence_setpoint=0.0,
			thrust_gain_override=self._compute_probe_gain(),
			lateral_p_scale=self._probe_lateral_p_scale,
			lateral_d_scale=self._probe_lateral_d_scale,
			enable_integral=True,
			substate=INFEASIBLE,
			info={
				"just_entered": just_entered,
				"reason": reason,
				"infeasible_reason": reason,
				"h_crit": self.gate.h_crit,
				"leg_clearance_m": self._leg_clearance,
				"k_min": self.gate.k_min,
				"peak_accel": self.probe_result.peak_accel,
				"vertical_accel_capacity_floor": self.gate.accel_capacity_floor,
				"vertical_accel_capacity_ceiling": self.gate.accel_capacity_ceiling,
				"roll_k_min": self.roll_gate.k_min,
				"roll_k_floor": self.roll_gate.k_floor,
				"roll_k_ceiling_leg": self.roll_gate.k_ceiling_leg,
				"roll_accel_capacity_floor": self.roll_gate.accel_capacity_floor,
				"roll_accel_capacity_ceiling": self.roll_gate.accel_capacity_ceiling,
				"pitch_k_min": self.pitch_gate.k_min,
				"pitch_k_floor": self.pitch_gate.k_floor,
				"pitch_k_ceiling_leg": self.pitch_gate.k_ceiling_leg,
				"pitch_accel_capacity_floor": self.pitch_gate.accel_capacity_floor,
				"pitch_accel_capacity_ceiling": self.pitch_gate.accel_capacity_ceiling,
				"vertical_feasible": self.vertical_feasible,
				"roll_feasible": self.roll_feasible,
				"pitch_feasible": self.pitch_feasible,
			},
		)

	def _is_centered(self, offset_x: float, offset_y: float, target_found: bool) -> bool:
		return (
			bool(target_found)
			and abs(float(offset_x)) <= self._center_offset_thr
			and abs(float(offset_y)) <= self._center_offset_thr
		)

	def _near_field_reached(
		self,
		offset_x: float,
		offset_y: float,
		target_found: bool,
		area_fraction: float,
		fov_saturated: bool,
	) -> bool:
		centered = self._is_centered(offset_x, offset_y, target_found)

		# Area fraction is the monotone near-field trigger. fov_saturated remains a
		# diagnostic but does not drive the phase transition.
		visually_close = float(area_fraction) >= self._fov_near_area_fraction
		return centered and visually_close

	def status_line(self) -> str:
		if self._substate == LANDED:
			since = (
				f"{(self._t_landed):.1f}s"
				if self._t_landed is not None else "--"
			)
			return f"[landed] touchdown latched at t={since}; mission complete"

		if self._substate == CENTER:
			return (
				f"[center] waiting for target within +/-{self._center_offset_thr:.2f} "
				f"for {self._center_dwell:.1f}s (timeout {self._center_timeout:.0f}s)"
			)

		if self._substate == APPROACH_PROBE:
			return (
				f"[approach_probe] D*_approach={self._approach_d_star:.2f} "
				f"probe={self.probe_result.total_duration_sec:.1f}/{self._probe_min:.1f}s "
				f"peaks z/r/p={self.probe_result.peak_accel:.3f}/"
				f"{self.roll_probe_result.peak_accel:.3f}/"
				f"{self.pitch_probe_result.peak_accel:.3f} m/s^2 "
				f"near_area_thr={self._fov_near_area_fraction:.2f}"
			)

		if self._substate == FINAL_PROBE:
			handoff = (
				f"{self.peak_accel_at_handoff:.3f}"
				if self.peak_accel_at_handoff is not None else "--"
			)
			return (
				f"[final_probe] hold={self.probe_result.duration_sec:.1f}/"
				f"{self._final_probe_duration:.1f}s "
				f"total={self.probe_result.total_duration_sec:.1f}/{self._probe_min:.1f}s "
				f"peaks z/r/p={self.probe_result.peak_accel:.3f}/"
				f"{self.roll_probe_result.peak_accel:.3f}/"
				f"{self.pitch_probe_result.peak_accel:.3f} m/s^2 "
				f"(at handoff {handoff})"
			)

		if self._substate == PROBE_HOLD:
			verdict = "WOULD-LAND" if self.feasible else "WOULD-ABORT"
			handoff = (
				f"{self.peak_accel_at_handoff:.3f}"
				if self.peak_accel_at_handoff is not None else "--"
			)
			return (
				f"[probe_hold] peak_accel={self.probe_result.peak_accel:.3f} m/s^2 "
				f"(at handoff {handoff}) "
				f"k_min={self.gate.k_min:.2f} k_floor={self.gate.k_floor:.2f} "
				f"k_ceiling_leg={self.gate.k_ceiling_leg:.2f} "
				f"h_crit={self.gate.h_crit:.2f}m "
				f"flags z/r/p={int(self.vertical_feasible)}/"
				f"{int(self.roll_feasible)}/{int(self.pitch_feasible)} "
				f"vs leg={self._leg_clearance:.2f}m -> {verdict} (hovering, no descent)"
			)

		if self._substate == INFEASIBLE:
			reasons = "; ".join(self._gate_failure_reasons())
			return (
				f"[infeasible] flags z/r/p={int(self.vertical_feasible)}/"
				f"{int(self.roll_feasible)}/{int(self.pitch_feasible)}: {reasons}"
			)

		return (
			f"[descend] h_crit={self.gate.h_crit:.2f}m "
			f"k: {self.gate.k_descend_start:.2f} (probe) -> {self.gate.k_floor:.2f} "
			f"(floor = max(k_min {self.gate.k_min:.2f}, "
			f"{self.gate.ceiling_margin:.2f} x k_ceiling_leg "
			f"{self.gate.k_ceiling_leg:.2f}))"
		)


def _smoke_test() -> None:
	dt = 0.05
	hover = 0.73

	for label, a_peak in (("calm", 0.05), ("moderate", 0.30), ("violent", 1.50)):
		m = MissionRoutine(
			hover_thrust=hover,
			enable_center=False,
			control_period_sec=dt,
			descent_divergence_setpoint=0.30,
			approach_divergence_setpoint=0.12,
			final_probe_duration_sec=3.0,
			final_probe_entry_ramp_sec=0.5,
			fov_near_area_fraction=0.55,
			probe_min_duration_sec=8.0,
			leg_clearance_m=0.20,
			center_dwell_sec=0.2,
			center_to_probe_lateral_ramp_sec=0.1,
			d_star_ramp_in_sec=1.0,
			stability_dt_sec=0.05,
			ceiling_margin=0.8,
			far_probe_window_sec=6.0,
			far_probe_decay_tau_sec=6.0,
			far_probe_highpass_tau_sec=12.0,
			near_probe_window_sec=1.0,
			near_probe_decay_tau_sec=5.0,
			near_probe_highpass_tau_sec=3.0,
		)
		m.start(0.0, start_height_m=3.0)

		t = 0.0
		mc = MissionControl()

		while t < 30.0:
			a_plat = a_peak * math.cos(2.0 * math.pi * 0.4 * t)
			u = hover + hover * a_plat / G_ACCEL
			area = 0.65 if t > 2.0 else 0.20

			mc = m.update(
				t,
				dt,
				u,
				offset_x=0.0,
				offset_y=0.0,
				target_found=True,
				flow_valid=True,
				area_fraction=area,
				fov_saturated=False,
				last_vertical_accel_cmd=a_plat,
				last_roll_accel_cmd=0.6 * a_plat,
				last_pitch_accel_cmd=0.4 * a_plat,
			)
			t += dt

		print(
			f"{label:9s} a_peak={a_peak:.2f} -> substate={mc.substate:14s} "
			f"peak={m.probe_result.peak_accel:.3f} "
			f"(handoff={m.peak_accel_at_handoff or float('nan'):.3f}) "
			f"k_min={m.gate.k_min:6.2f} k_ceil_leg={m.gate.k_ceiling_leg:5.2f} "
			f"k_floor={m.gate.k_floor:5.2f} h_crit={m.gate.h_crit:.3f} "
			f"feasible={m.feasible}"
		)


if __name__ == "__main__":
	_smoke_test()