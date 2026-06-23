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
) -> List[Segment]:
	"""
	Build the segments for one axis' step train: alternating
	+amplitude / neutral / -amplitude / neutral, held `hold_sec` each,
	repeated `repeats` times, with the other two axes pinned at trim
	(roll=0, pitch=0, thrust=hover_thrust) throughout.

	axis: "roll", "pitch", or "thrust".
	amplitude: deviation from trim used for the +/- steps. For roll/pitch
		this is in radians (keep it comfortably inside roll_limit /
		pitch_limit); for thrust it's added to/subtracted from
		hover_thrust (keep it comfortably inside thrust_min/thrust_max).
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

	for _ in range(repeats):
		for value in (amplitude, 0.0, -amplitude, 0.0):
			roll, pitch, thrust = command(value)
			segments.append((hold_sec, roll, pitch, thrust))

	return segments


class VerticalVelocityDamper:
	"""
	Stateful PI damper driving thrust to cancel vz:

		thrust = hover_thrust - kp*vz - ki*integral(vz dt), clamped.

	Why PI and not just P (which is what this replaced): a proportional-
	only damper reaches an equilibrium against any *constant* disturbance
	rather than driving vz to zero — it doesn't almost-work, it
	structurally can't fully cancel a steady bias, no matter how small.
	This isn't hypothetical: real calibration data showed the P-only
	version computing exactly as designed (formula match to floating-
	point precision) while vz still sat at a persistent, nonzero mean in
	every phase (settle, roll, pitch), growing over the course of the
	run. Whatever that bias's source — hover_thrust not being exactly
	exact, an EKF2 quirk, anything else constant or slowly varying —
	integral action is the standard fix: it accumulates exactly the
	correction needed to cancel a steady disturbance over time, which
	proportional action alone cannot do.

	The integral is clamped (`integral_limit`) to bound windup: without
	that, a sustained large vz (e.g. during the settle phase's initial
	transient) could otherwise accumulate a correction so large it
	overshoots once the real disturbance is gone.

	One instance is meant to be shared continuously across the settle
	phase and the roll/pitch test phases (see calibration_node.py) — the
	integral term is exactly the part that benefits from not being reset
	at each phase boundary, since the disturbance it's correcting for
	doesn't reset either.
	"""

	def __init__(
		self,
		hover_thrust: float,
		kp: float = 0.08,
		ki: float = 0.02,
		thrust_min: float = 0.35,
		thrust_max: float = 0.65,
		integral_limit: float = 0.05,
	):
		self._hover_thrust = hover_thrust
		self._kp = kp
		self._ki = ki
		self._thrust_min = thrust_min
		self._thrust_max = thrust_max
		self._integral_limit = abs(integral_limit)

		self._integral = 0.0
		self._last_time: Optional[float] = None

	def reset(self):
		self._integral = 0.0
		self._last_time = None

	@property
	def integral(self) -> float:
		return self._integral

	def step(self, now: float, vz: float) -> float:
		if self._last_time is not None:
			dt = now - self._last_time
			if dt > 0.0:
				self._integral += vz * dt
				self._integral = max(
					-self._integral_limit, min(self._integral_limit, self._integral)
				)

		self._last_time = now

		thrust = self._hover_thrust - self._kp * vz - self._ki * self._integral
		return max(self._thrust_min, min(self._thrust_max, thrust))


class VerticalSettler:
	"""
	"Have we been quiet for long enough" detector wrapping a
	VerticalVelocityDamper, so the open-loop test sequence doesn't start
	until vz has genuinely settled — not just "this one sample happened
	to be small". See VerticalVelocityDamper for the actual thrust
	computation; this class only adds the is-it-time-to-proceed state
	machine on top of it.

	Why a settle phase exists at all: commanding exactly hover_thrust
	makes vertical *acceleration* zero, not velocity. If the vehicle
	enters calibration with any residual vz (left over from arming,
	mode-switching, or wherever the previous run ended), holding thrust
	constant preserves that velocity instead of correcting it — the
	sequence then climbs or crashes depending on the sign of whatever vz
	happened to exist at t=0, even with a perfectly calibrated
	hover_thrust.

	Pure state machine, no ROS dependency, so it's unit-testable without
	a simulator — see calibration_node.py for how it's driven.
	"""

	def __init__(
		self,
		damper: VerticalVelocityDamper,
		vz_threshold: float = 0.05,
		min_duration_sec: float = 1.0,
		timeout_sec: float = 15.0,
	):
		self._damper = damper
		self._vz_threshold = vz_threshold
		self._min_duration_sec = min_duration_sec
		self._timeout_sec = timeout_sec

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

	def step(self, now: float, vz: float) -> float:
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

		if abs(vz) < self._vz_threshold:
			if self._ok_since is None:
				self._ok_since = now
			elif now - self._ok_since >= self._min_duration_sec:
				self._settled = True
		else:
			self._ok_since = None

		if not self._settled and elapsed >= self._timeout_sec:
			self._settled = True
			self._timed_out = True

		return self._damper.step(now, vz)


def exceeds_safety_bounds(
	vz: float,
	area_fraction: float,
	vz_limit: float = 1.0,
	area_fraction_max: float = 0.97,
) -> bool:
	"""
	True if vz or area_fraction is clearly outside what a calibration
	run should ever see in normal operation — vz this large means
	something (the settle damper, a bad command) is actively driving a
	runaway rather than gently exciting one axis; area_fraction this
	high means the vehicle is on top of the target. Either is reason to
	abort rather than continue the sequence.
	"""
	return abs(vz) > vz_limit or area_fraction > area_fraction_max


def build_calibration_sequence(
	hover_thrust: float,
	roll_amplitude: float = 0.08,
	pitch_amplitude: float = 0.08,
	thrust_amplitude: float = 0.05,
	roll_hold_sec: float = 6.0,
	pitch_hold_sec: float = 6.0,
	thrust_hold_sec: float = 2.0,
	repeats: int = 3,
	settle_sec: float = 2.0,
	axes: Tuple[str, ...] = ("roll", "pitch", "thrust"),
) -> StepSequence:
	"""
	Build the full calibration sequence: a settle period at trim, then
	one step train per axis in `axes`, back to back, each preceded by a
	short return-to-trim so every axis' train starts from the same
	(roll=0, pitch=0, thrust=hover_thrust) condition.

	Test one axis at a time on purpose — see the module docstring.

	hold_sec is per axis, not shared: roll/pitch want it long enough
	that a held tilt actually has time to build up real velocity (the
	response scales with hold_sec^2 for a roughly double-integrator
	system — position from velocity from acceleration — so doubling
	hold_sec is a much bigger lever on signal strength than the same
	relative increase in amplitude). Thrust deliberately keeps a short
	hold: it already commands real altitude excursions, and a longer
	hold there directly widens the area_fraction range swept within one
	file, which is its own separate problem (see fit_axis_models.py's
	wide-range warning) — don't fix one axis by making this worse.
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

	for axis in axes:
		axis_segments = build_axis_step_train(
			axis=axis,
			amplitude=amplitudes[axis],
			hover_thrust=hover_thrust,
			hold_sec=hold_secs[axis],
			repeats=repeats,
		)
		segments.extend(axis_segments)
		labels.extend([axis] * len(axis_segments))

		segments.append((settle_sec, 0.0, 0.0, hover_thrust))
		labels.append("settle")

	return StepSequence(segments, labels)