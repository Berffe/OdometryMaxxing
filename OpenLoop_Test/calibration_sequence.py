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

from typing import List, Tuple

Segment = Tuple[float, float, float, float]  # duration_sec, roll, pitch, thrust


class StepSequence:
	"""
	A fixed, piecewise-constant (roll, pitch, thrust) command timeline.

	segments: list of (duration_sec, roll, pitch, thrust). command_at(t)
	returns whichever segment's time window contains `t`; past the end
	of the sequence, it holds the last segment's command (so a node that
	keeps running after the test finishes doesn't suddenly command
	something undefined).
	"""

	def __init__(self, segments: List[Segment]):
		if not segments:
			raise ValueError("StepSequence needs at least one segment")

		self._segments = list(segments)
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


def build_calibration_sequence(
	hover_thrust: float,
	roll_amplitude: float = 0.04,
	pitch_amplitude: float = 0.04,
	thrust_amplitude: float = 0.05,
	hold_sec: float = 2.0,
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
	"""
	segments: List[Segment] = [(settle_sec, 0.0, 0.0, hover_thrust)]

	amplitudes = {
		"roll": roll_amplitude,
		"pitch": pitch_amplitude,
		"thrust": thrust_amplitude,
	}

	for axis in axes:
		segments.extend(
			build_axis_step_train(
				axis=axis,
				amplitude=amplitudes[axis],
				hover_thrust=hover_thrust,
				hold_sec=hold_sec,
				repeats=repeats,
			)
		)
		segments.append((settle_sec, 0.0, 0.0, hover_thrust))

	return StepSequence(segments)
