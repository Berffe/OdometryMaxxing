"""
Open-loop test-command sequencing for axis identification, decoupled
from ROS.

This builds a piecewise-constant (roll, pitch, thrust) command sequence
that excites one axis at a time (step up, hold, step down, hold,
repeat) while holding the other axes at trim. Driving the vehicle with
this — instead of ControlLaw — is what makes the resulting log usable
for identifying the per-axis discrete-time model in control_law.py: the
command is independent of the measured state, so cause (u) and effect
(e) aren't entangled through feedback the way they would be with the
closed-loop controller running.

Used by calibration_node.py; kept here, ROS-free, so it can be
unit-tested directly (see the project's existing split between
ROS-coupled nodes and plain-Python algorithm modules).
"""

from typing import List, Optional, Tuple
import math

Segment = Tuple[float, float, float, float]  # duration_sec, roll, pitch, thrust


class StepSequence:
	"""
	A fixed, piecewise-constant (roll, pitch, thrust) command timeline.

	segments: list of (duration_sec, roll, pitch, thrust). command_at(t)
	returns whichever segment's time window contains `t`; past the end
	of the sequence, it holds the last segment's command (so a node that
	keeps running after the test finishes doesn't suddenly command
	something undefined).

	labels (optional): one string per segment (e.g. "settle", "roll",
	"pitch", "thrust") — axis_at(t) looks this up the same way
	command_at(t) looks up the command, so a caller can tell which axis
	is currently being exercised without re-deriving it from the
	command values (which doesn't work: a "roll" segment's neutral
	sub-steps and a "thrust" segment's neutral sub-steps both have
	roll=pitch=0, so you can't tell them apart from the numbers alone).
	"""

	def __init__(self, segments: List[Segment], labels: Optional[List[str]] = None):
		if not segments:
			raise ValueError("StepSequence needs at least one segment")

		self._segments = list(segments)
		self._labels = list(labels) if labels is not None else ["test"] * len(self._segments)

		if len(self._labels) != len(self._segments):
			raise ValueError("labels must have the same length as segments")

		self._boundaries = []

		elapsed = 0.0
		for duration, _, _, _ in self._segments:
			elapsed += float(duration)
			self._boundaries.append(elapsed)

		self._total_duration = elapsed

	def command_at(self, t: float) -> Tuple[float, float, float]:
		"""Return (roll, pitch, thrust) commanded at elapsed time t."""
		if t < 0.0:
			t = 0.0

		for (duration, roll, pitch, thrust), boundary in zip(
			self._segments, self._boundaries
		):
			if t < boundary:
				return roll, pitch, thrust

		# Past the end: hold the final segment's command.
		_, roll, pitch, thrust = self._segments[-1]
		return roll, pitch, thrust

	def axis_at(self, t: float) -> str:
		"""Return the label of whichever segment contains elapsed time t."""
		if t < 0.0:
			t = 0.0

		for boundary, label in zip(self._boundaries, self._labels):
			if t < boundary:
				return label

		return self._labels[-1]

	def is_finished(self, t: float) -> bool:
		return t >= self._total_duration

	@property
	def total_duration(self) -> float:
		return self._total_duration


def build_axis_step_train(
	axis: str,
	amplitude: float,
	hover_thrust: float,
	hold_sec: float = 2.0,
	repeats: int = 3,
	reset_sec: float = 0.0,
) -> Tuple[List[Segment], List[str]]:
	"""
	Build the segments (and matching labels) for one axis' step train:
	alternating +amplitude / neutral / -amplitude / neutral, held
	`hold_sec` each, repeated `repeats` times, with the other two axes
	pinned at trim (roll=0, pitch=0, thrust=hover_thrust) throughout.

	axis: "roll", "pitch", or "thrust".
	amplitude: deviation from trim used for the +/- steps. For roll/pitch
		this is in radians (keep it comfortably inside roll_limit /
		pitch_limit); for thrust it's added to/subtracted from
		hover_thrust (keep it comfortably inside thrust_min/thrust_max).

	reset_sec: if > 0, inserts a brief return-to-trim segment (labeled
	"settle", not `axis`) between each pair of consecutive repeats.
	Exists specifically for thrust: that's the one axis where the
	vertical damper is fully disabled during its own test (since thrust
	IS what's being identified, so it can't also be corrected), so any
	residual hover_thrust error has nothing checking it for the entire,
	now much longer (repeats * 4 * hold_sec) duration of the test. A
	real run showed exactly this: vz climbing from ~0 to 0.47 m/s
	(monotonically, not oscillating) over a single uninterrupted
	~30s thrust test, consistent with a small uncorrected bias
	compounding the whole time rather than averaging out. Labeling the
	reset "settle" rather than "thrust" does two things: the vertical
	damper activates for it (same condition as the inter-axis gaps,
	`current_axis != "thrust"`), and fit_axis_models.py excludes it from
	the thrust fit, the same way it already excludes the inter-axis
	gaps -- so resets bound how far this can drift without touching the
	identification data at all.
	"""
	if axis not in ("roll", "pitch", "thrust"):
		raise ValueError(f"unknown axis: {axis!r}")

	def command(value: float) -> Tuple[float, float, float]:
		roll, pitch, thrust = 0.0, 0.0, hover_thrust

		if axis == "roll":
			roll = value
		elif axis == "pitch":
			pitch = value
		else:
			thrust = hover_thrust + value

		return roll, pitch, thrust

	segments: List[Segment] = []
	labels: List[str] = []

	for i in range(repeats):
		for value in (amplitude, 0.0, -amplitude, 0.0):
			roll, pitch, thrust = command(value)
			segments.append((hold_sec, roll, pitch, thrust))
			labels.append(axis)

		if reset_sec > 0.0 and i < repeats - 1:
			segments.append((reset_sec, 0.0, 0.0, hover_thrust))
			labels.append("settle")

	return segments, labels


class VerticalVelocityDamper:
	"""
	Stateful PID-ish damper driving thrust to cancel vz and pull back
	toward a reference altitude:

		thrust = hover_thrust + kp*vz + ki*integral(vz dt)
		                      + kz*(z - z_target) + kiz*integral((z - z_target) dt),
		clamped.

	PX4 local-position velocity and position both use NED convention:
	vz > 0 means descending, z increasing means lower altitude. So a
	positive vz (descending) must INCREASE thrust, and being below the
	target altitude (z > z_target) must also increase thrust — all
	terms are added, not subtracted. Getting a sign backwards doesn't
	just under-correct, it's active positive feedback: descending
	reduces thrust further, which accelerates the descent further. A
	wrong sign here looks exactly like "almost free-falling" the instant
	offboard engages, because that's what it is.

	Why PI and not just P, for vz: a proportional-only damper reaches an
	equilibrium against any *constant* disturbance rather than driving
	vz to zero — it doesn't almost-work, it structurally can't fully
	cancel a steady bias, no matter how small. Real calibration data
	showed exactly that: a P-only formula computing correctly while vz
	still sat at a persistent nonzero mean throughout.

	Why kiz, on top of kz, for altitude: the exact same limitation shows
	up one level up. kz alone is P-only on position, so it ALSO reaches
	an equilibrium against a constant hover_thrust error instead of
	closing it — a real 45+ second settle attempt measured this
	directly: kz=0.10 left a steady ~1.18m altitude error (a settle
	timeout, correctly, rather than a false "settled"), which implies
	kz was stuck fighting a ~0.118 thrust gap it could never fully
	close on its own. kiz integrates the altitude error over time the
	same way ki integrates the velocity error, and is what actually
	drives that steady-state error toward zero rather than just
	bounding it. kiz=0.0 (the default) reproduces the kz-only behavior.

	Both integral_limit and integral_z_limit need to be sized to the
	kind of disturbance you actually expect to correct, not just "some
	bound to prevent windup." A limit so tight that ki*integral_limit
	(or kiz*integral_z_limit) is tiny compared to a plausible
	hover_thrust error means that integral term can NEVER close that
	error, no matter how long it has — it isn't a slower correction,
	it's a structurally incomplete one.

	One instance is meant to be shared continuously from the moment
	offboard thrust commands start being sent through the open-loop
	roll/pitch test phases (see calibration_node.py) — both integral
	terms are exactly the part that benefits from not being reset at
	each phase boundary, since the disturbance they're correcting for
	doesn't reset either.
	"""

	def __init__(
		self,
		hover_thrust: float,
		kp: float = 0.08,
		ki: float = 0.02,
		kz: float = 0.0,
		kiz: float = 0.0,
		thrust_min: float = 0.35,
		thrust_max: float = 0.65,
		integral_limit: float = 0.05,
		integral_z_limit: float = 10.0,
	):
		self._hover_thrust = hover_thrust
		self._kp = kp
		self._ki = ki
		self._kz = kz
		self._kiz = kiz
		self._thrust_min = thrust_min
		self._thrust_max = thrust_max
		self._integral_limit = abs(integral_limit)
		self._integral_z_limit = abs(integral_z_limit)

		self._integral = 0.0
		self._integral_z = 0.0
		self._last_time: Optional[float] = None
		self._target_z: Optional[float] = None

	def reset(self):
		self._integral = 0.0
		self._integral_z = 0.0
		self._last_time = None

	def set_altitude_target(self, z_target: Optional[float]):
		"""
		Set (or clear, with None) the altitude this damper actively pulls
		back toward. Typically captured once, right after takeoff
		reaches its intended altitude, and kept for the rest of the
		pre-test handoff -- see calibration_node.py. Resets the altitude
		integral too: it's only meaningful relative to a specific target,
		so a new target should not inherit whatever the old one
		accumulated.
		"""
		self._target_z = z_target
		self._integral_z = 0.0

	@property
	def integral(self) -> float:
		return self._integral

	@property
	def integral_z(self) -> float:
		return self._integral_z

	@property
	def altitude_target(self) -> Optional[float]:
		return self._target_z

	def step(self, now: float, vz: float, z: Optional[float] = None) -> float:
		altitude_error = None
		if self._target_z is not None and z is not None:
			altitude_error = z - self._target_z

		if self._last_time is not None:
			dt = now - self._last_time
			if dt > 0.0:
				self._integral += vz * dt
				self._integral = max(
					-self._integral_limit, min(self._integral_limit, self._integral)
				)

				if altitude_error is not None:
					self._integral_z += altitude_error * dt
					self._integral_z = max(
						-self._integral_z_limit, min(self._integral_z_limit, self._integral_z)
					)

		self._last_time = now

		thrust = self._hover_thrust + self._kp * vz + self._ki * self._integral

		if altitude_error is not None:
			thrust += self._kz * altitude_error + self._kiz * self._integral_z

		return max(self._thrust_min, min(self._thrust_max, thrust))


class LateralVelocityDamper:
	"""
	Stateful PI damper driving roll/pitch to cancel lateral drift (vx,
	vy) during the pre-test handoff phases only -- NEVER during the
	open-loop roll/pitch test itself, which would reintroduce exactly
	the cause/effect entanglement this whole calibration setup exists
	to avoid for those two axes. See calibration_node.py for exactly
	which phases apply its output.

	Why this exists: a real run showed target_offset_x starting around
	0.35-0.37 (well off-center) and continuing to drift through the
	whole test. roll=pitch=0 zeroes lateral *acceleration*, not
	velocity, and OFFBOARD_PROBE deliberately commands a real roll to
	verify attitude response -- nothing has corrected the resulting
	lateral velocity since. Same gap that motivated the vertical
	damper, one axis over.

	Sign derivation -- not assumed, derived from the same ZYX Euler
	convention _euler_to_quaternion/_quaternion_to_euler already use,
	and checked numerically before trusting it: rotating the body-frame
	thrust vector (0,0,-1) by roll alone gives (0, sin(roll), -cos(roll))
	-- positive roll tilts thrust toward +Y (East, in NED, for yaw=0).
	Rotating by pitch alone gives (-sin(pitch), 0, -cos(pitch)) --
	positive pitch tilts thrust toward -X (South). So countering a
	positive vy (drifting East) needs NEGATIVE roll; countering a
	positive vx (drifting North) needs POSITIVE pitch. For nonzero yaw,
	vx/vy are rotated into the body forward/right frame first (using
	the vehicle's current heading), so the same two rules still apply
	regardless of which way the vehicle happens to be facing -- this
	project never actively controls yaw, so it can't be assumed to stay
	at 0.

	roll_limit/pitch_limit default well inside the test sequence's own
	ROLL_LIMIT_RAD/PITCH_LIMIT_RAD -- this is meant to gently hold
	position during settle phases, not aggressively chase it.
	"""

	def __init__(
		self,
		kp: float = 0.10,
		ki: float = 0.03,
		roll_limit: float = 0.05,
		pitch_limit: float = 0.05,
		integral_limit: float = 0.3,
	):
		self._kp = kp
		self._ki = ki
		self._roll_limit = abs(roll_limit)
		self._pitch_limit = abs(pitch_limit)
		self._integral_limit = abs(integral_limit)

		self._integral_vx = 0.0
		self._integral_vy = 0.0
		self._last_time: Optional[float] = None

	def reset(self):
		self._integral_vx = 0.0
		self._integral_vy = 0.0
		self._last_time = None

	def step(self, now: float, vx: float, vy: float, yaw: float = 0.0) -> Tuple[float, float]:
		"""Return (roll, pitch) correction for this tick."""
		if self._last_time is not None:
			dt = now - self._last_time
			if dt > 0.0:
				self._integral_vx += vx * dt
				self._integral_vx = max(
					-self._integral_limit, min(self._integral_limit, self._integral_vx)
				)
				self._integral_vy += vy * dt
				self._integral_vy = max(
					-self._integral_limit, min(self._integral_limit, self._integral_vy)
				)

		self._last_time = now

		cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

		# Rotate NED (north, east) velocity and its integral into body
		# (forward, right). Rotating the accumulated integral this way
		# (rather than accumulating in body frame directly) is only
		# exact if yaw stays constant -- true here, since this project
		# never actively commands yaw changes.
		v_forward = vx * cos_yaw + vy * sin_yaw
		v_right = -vx * sin_yaw + vy * cos_yaw
		integral_forward = self._integral_vx * cos_yaw + self._integral_vy * sin_yaw
		integral_right = -self._integral_vx * sin_yaw + self._integral_vy * cos_yaw

		pitch = self._kp * v_forward + self._ki * integral_forward
		roll = -self._kp * v_right - self._ki * integral_right

		roll = max(-self._roll_limit, min(self._roll_limit, roll))
		pitch = max(-self._pitch_limit, min(self._pitch_limit, pitch))

		return roll, pitch


class VerticalSettler:
	"""
	"Have we been quiet for long enough, AND actually near where we
	wanted to be" detector wrapping a VerticalVelocityDamper. See
	VerticalVelocityDamper for the actual thrust computation; this class
	only adds the is-it-time-to-proceed state machine on top of it.

	Why a settle phase exists at all: commanding exactly hover_thrust
	makes vertical *acceleration* zero, not velocity. If the vehicle
	enters calibration with any residual vz (left over from arming,
	mode-switching, or wherever the previous run ended), holding thrust
	constant preserves that velocity instead of correcting it.

	Why settling requires altitude too, not just velocity: vz reading
	near zero is necessary but not sufficient for "genuinely hovering" —
	a vehicle resting on the ground or platform also reads vz≈0, since
	the surface's normal force, not a real hover equilibrium, is what
	stopped it. That's a real failure mode, not a hypothetical one: a
	hover_thrust significantly off can let the vehicle descend for
	several seconds before a velocity-only check ever notices anything
	wrong, by which point "settled" can mean "landed," not "hovering."
	If `altitude_tolerance_m` is given and the damper has an altitude
	target set, settling additionally requires z to be within that
	tolerance of the target — so a vehicle that's quietly sitting on the
	ground far below where it started does NOT get waved through.

	Pure state machine, no ROS dependency, so it's unit-testable without
	a simulator — see calibration_node.py for how it's driven.
	"""

	def __init__(
		self,
		damper: VerticalVelocityDamper,
		vz_threshold: float = 0.05,
		min_duration_sec: float = 1.0,
		timeout_sec: float = 15.0,
		altitude_tolerance_m: Optional[float] = None,
	):
		self._damper = damper
		self._vz_threshold = vz_threshold
		self._min_duration_sec = min_duration_sec
		self._timeout_sec = timeout_sec
		self._altitude_tolerance_m = altitude_tolerance_m

		self._start_time: Optional[float] = None
		self._ok_since: Optional[float] = None
		self._settled = False
		self._timed_out = False

	@property
	def is_settled(self) -> bool:
		return self._settled

	@property
	def timed_out(self) -> bool:
		return self._timed_out

	def step(self, now: float, vz: float, z: Optional[float] = None) -> float:
		"""
		Advance the settle state machine by one tick and return the
		thrust command to apply this tick. Stop calling step() once
		is_settled is True (the caller should move on to the real
		sequence at that point — and can keep using the same `damper`
		instance, continuously, for roll/pitch's ongoing damping).
		"""
		if self._start_time is None:
			self._start_time = now

		elapsed = now - self._start_time

		altitude_ok = True
		target_z = self._damper.altitude_target
		if self._altitude_tolerance_m is not None and target_z is not None and z is not None:
			altitude_ok = abs(z - target_z) <= self._altitude_tolerance_m

		if abs(vz) < self._vz_threshold and altitude_ok:
			if self._ok_since is None:
				self._ok_since = now
			elif now - self._ok_since >= self._min_duration_sec:
				self._settled = True
		else:
			self._ok_since = None

		if not self._settled and elapsed >= self._timeout_sec:
			self._settled = True
			self._timed_out = True

		return self._damper.step(now, vz, z=z)


def exceeds_safety_bounds(
	vz: float,
	area_fraction: float,
	vz_limit: float = 1.0,
	area_fraction_max: float = 0.97,
	vx: float = 0.0,
	vy: float = 0.0,
	lateral_velocity_limit: float = 2.0,
) -> bool:
	"""
	True if vz, area_fraction, or lateral velocity is clearly outside
	what a calibration run should ever see in normal operation — vz or
	lateral velocity this large means something (a damper, a bad
	command) is actively driving a runaway rather than gently exciting
	one axis; area_fraction this high means the vehicle is on top of
	the target. Any of these is reason to abort rather than continue
	the sequence. vx/vy default to 0.0 so existing callers that don't
	pass them can't trip the new check by omission.
	"""
	return (
		abs(vz) > vz_limit
		or area_fraction > area_fraction_max
		or abs(vx) > lateral_velocity_limit
		or abs(vy) > lateral_velocity_limit
	)


def build_calibration_sequence(
	hover_thrust: float,
	roll_amplitude: float = 0.08,
	pitch_amplitude: float = 0.08,
	thrust_amplitude: float = 0.05,
	roll_hold_sec: float = 6.0,
	pitch_hold_sec: float = 6.0,
	thrust_hold_sec: float = 2.0,
	roll_repeats: int = 8,
	pitch_repeats: int = 8,
	thrust_repeats: int = 8,
	thrust_reset_sec: float = 0.0,
	settle_sec: float = 2.0,
	axes: Tuple[str, ...] = ("roll", "pitch", "thrust"),
) -> StepSequence:
	"""
	Build the full calibration sequence: a settle period at trim, then
	one step train per axis in `axes`, back to back, each preceded by a
	short return-to-trim so every axis' train starts from the same
	(roll=0, pitch=0, thrust=hover_thrust) condition.

	Test one axis at a time on purpose — see the module docstring.

	hold_sec and repeats are both per axis, not shared. For hold_sec:
	roll/pitch want it long enough that a held tilt actually has time to
	build up real velocity (the response scales with hold_sec^2 for a
	roughly double-integrator system — position from velocity from
	acceleration — so doubling hold_sec is a much bigger lever on signal
	strength than the same relative increase in amplitude). Thrust
	deliberately keeps a short hold: it already commands real altitude
	excursions, and a longer hold there directly widens the
	area_fraction range swept within one file (see fit_axis_models.py's
	wide-range warning) — don't fix one axis by making this worse.

	thrust_reset_sec (thrust only — roll_repeats/pitch_repeats don't
	need this, see build_axis_step_train) inserts a brief, damped
	return-to-trim between thrust repeats, bounding how far a residual
	hover_thrust error can drift the whole test rather than requiring
	hover_thrust to be guessed exactly right.
	"""
	segments: List[Segment] = [(settle_sec, 0.0, 0.0, hover_thrust)]
	labels: List[str] = ["settle"]

	amplitudes = {
		"roll": roll_amplitude,
		"pitch": pitch_amplitude,
		"thrust": thrust_amplitude,
	}
	hold_secs = {
		"roll": roll_hold_sec,
		"pitch": pitch_hold_sec,
		"thrust": thrust_hold_sec,
	}
	repeats_map = {
		"roll": roll_repeats,
		"pitch": pitch_repeats,
		"thrust": thrust_repeats,
	}
	reset_secs = {
		"roll": 0.0,
		"pitch": 0.0,
		"thrust": thrust_reset_sec,
	}

	for axis in axes:
		axis_segments, axis_labels = build_axis_step_train(
			axis=axis,
			amplitude=amplitudes[axis],
			hover_thrust=hover_thrust,
			hold_sec=hold_secs[axis],
			repeats=repeats_map[axis],
			reset_sec=reset_secs[axis],
		)
		segments.extend(axis_segments)
		labels.extend(axis_labels)

		segments.append((settle_sec, 0.0, 0.0, hover_thrust))
		labels.append("settle")

	return StepSequence(segments, labels)