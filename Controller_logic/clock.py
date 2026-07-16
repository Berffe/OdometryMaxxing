"""Clock ownership and diagnostic alignment for BEE_LAND.

The node uses four distinct notions of time:

SYSTEM WALL
	Unix-epoch seconds. Used only for outgoing PX4/uXRCE-DDS timestamps and for
	logging callback receipt instants that need to be correlated with external
	processes.

MONOTONIC
	Local steady time. Used for durations, callback periods and latency. It is
	immune to system-clock corrections and must never be sent to PX4.

SIM
	Gazebo simulation seconds, normally carried by Image.header.stamp. This is
	the physical time base for optical flow, mission timing and offline
	differentiation of simulated trajectories.

PX4
	PX4 hrt seconds since boot, carried by incoming /fmu/out messages.

Raw source timestamps and raw receipt timestamps are the authoritative log
fields. The affine fits maintained here are diagnostics-only conveniences for
placing unstamped data (notably /platform/pose) approximately on the SIM axis.
They never feed the controller and never alter outgoing PX4 timestamps.

A single additive ``wall - sim`` offset is NOT a valid conversion when Gazebo
runs at RTF != 1. Consequently this module fits a local affine relation instead:

	source_time ~= source_ref + rate * (wall_time - wall_ref)

where ``rate`` is the local clock rate (RTF for SIM). Both the local-fit rate and
the run-wide first-to-latest rate are exposed; their names make the distinction
explicit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Deque, Optional

try:
	from rclpy.clock import Clock, ClockType
except ImportError:  # Allows ROS-free import for tests.
	Clock = None
	ClockType = None


@dataclass(frozen=True)
class ClockObservation:
	"""One raw source timestamp paired with callback receipt wall time."""

	source_sec: float
	receipt_wall_sec: float


class _AffineClockFit:
	"""Rolling source-vs-wall affine fit used only for diagnostics.

	The fit accepts only strictly newer source stamps. This prevents duplicate
	diagnostics rows and out-of-order messages from corrupting the rate. The
	latest raw observation is still stored by TimeManager separately, so no
	source timestamp is hidden from the CSV just because it was unsuitable for
	the fit.
	"""

	def __init__(self, window_wall_sec: float = 8.0, max_samples: int = 1024):
		self._window_wall_sec = max(0.5, float(window_wall_sec))
		self._samples: Deque[ClockObservation] = deque(maxlen=max(8, int(max_samples)))
		self._first: Optional[ClockObservation] = None
		self._latest_accepted: Optional[ClockObservation] = None
		self._rate: Optional[float] = None
		self._wall_ref: Optional[float] = None
		self._source_ref: Optional[float] = None
		self._latest_residual_source_sec: Optional[float] = None

	def add(self, observation: ClockObservation) -> None:
		if self._latest_accepted is not None:
			if observation.source_sec <= self._latest_accepted.source_sec + 1e-12:
				return
			if observation.receipt_wall_sec <= self._latest_accepted.receipt_wall_sec + 1e-12:
				return

		if self._first is None:
			self._first = observation
		self._latest_accepted = observation
		self._samples.append(observation)

		cutoff = observation.receipt_wall_sec - self._window_wall_sec
		while len(self._samples) > 2 and self._samples[0].receipt_wall_sec < cutoff:
			self._samples.popleft()
		self._refit()

	def _refit(self) -> None:
		if len(self._samples) < 2:
			return

		wall_mean = sum(s.receipt_wall_sec for s in self._samples) / len(self._samples)
		source_mean = sum(s.source_sec for s in self._samples) / len(self._samples)
		var_wall = sum((s.receipt_wall_sec - wall_mean) ** 2 for s in self._samples)
		if var_wall <= 1e-12:
			return
		cov = sum(
			(s.receipt_wall_sec - wall_mean) * (s.source_sec - source_mean)
			for s in self._samples
		)
		rate = cov / var_wall
		if not math.isfinite(rate) or not (1e-6 < rate < 20.0):
			return

		self._rate = rate
		self._wall_ref = wall_mean
		self._source_ref = source_mean
		latest = self._latest_accepted
		if latest is not None:
			predicted = self.source_at_wall(latest.receipt_wall_sec)
			self._latest_residual_source_sec = (
				latest.source_sec - predicted if predicted is not None else None
			)

	def source_at_wall(self, wall_sec: float) -> Optional[float]:
		if self._rate is None or self._wall_ref is None or self._source_ref is None:
			return None
		return self._source_ref + self._rate * (float(wall_sec) - self._wall_ref)

	def wall_at_source(self, source_sec: float) -> Optional[float]:
		if (
			self._rate is None
			or self._wall_ref is None
			or self._source_ref is None
			or abs(self._rate) <= 1e-12
		):
			return None
		return self._wall_ref + (float(source_sec) - self._source_ref) / self._rate

	def local_rate(self) -> Optional[float]:
		return self._rate

	def run_rate(self) -> Optional[float]:
		if self._first is None or self._latest_accepted is None:
			return None
		d_wall = self._latest_accepted.receipt_wall_sec - self._first.receipt_wall_sec
		d_source = self._latest_accepted.source_sec - self._first.source_sec
		if d_wall <= 1e-12 or d_source <= 0.0:
			return None
		return d_source / d_wall

	def snapshot(self) -> dict:
		return {
			"sample_count": len(self._samples),
			"local_rate": self.local_rate(),
			"run_rate": self.run_rate(),
			"fit_wall_reference_sec": self._wall_ref,
			"fit_source_reference_sec": self._source_ref,
			"latest_fit_residual_source_sec": self._latest_residual_source_sec,
		}


class TimeManager:
	def __init__(self, node=None, fit_window_wall_sec: float = 8.0):
		self._node = node
		if node is not None and Clock is not None:
			self._system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
		else:
			self._system_clock = None

		self._sim_fit = _AffineClockFit(window_wall_sec=fit_window_wall_sec)
		self._px4_fit = _AffineClockFit(window_wall_sec=fit_window_wall_sec)
		self._latest_sim_observation: Optional[ClockObservation] = None
		self._latest_px4_observation: Optional[ClockObservation] = None

	# ---- Local clocks -----------------------------------------------------

	def wall_ns(self) -> int:
		"""Unix-epoch nanoseconds, independent of ROS use_sim_time."""
		if self._system_clock is not None:
			return int(self._system_clock.now().nanoseconds)
		return time.time_ns()

	def wall_sec(self) -> float:
		return self.wall_ns() / 1e9

	@staticmethod
	def monotonic_sec() -> float:
		"""Steady local time for durations and latency measurements."""
		return time.monotonic()

	@staticmethod
	def perf_counter_sec() -> float:
		"""High-resolution steady clock, suitable across local processes."""
		return time.perf_counter()

	def px4_tx_timestamp_us(self) -> int:
		"""Timestamp for outgoing PX4 messages: SYSTEM WALL microseconds."""
		return self.wall_ns() // 1000

	# ---- Raw source observations ----------------------------------------

	def observe_sim_timestamp(
		self, sim_sec, receipt_wall_sec: Optional[float] = None
	) -> Optional[ClockObservation]:
		parsed = self._parse_positive_seconds(sim_sec, scale=1.0)
		if parsed is None:
			return None
		obs = ClockObservation(
			source_sec=parsed,
			receipt_wall_sec=self.wall_sec() if receipt_wall_sec is None else float(receipt_wall_sec),
		)
		self._latest_sim_observation = obs
		self._sim_fit.add(obs)
		return obs

	def observe_px4_timestamp(
		self, px4_timestamp_us, receipt_wall_sec: Optional[float] = None
	) -> Optional[ClockObservation]:
		parsed = self._parse_positive_seconds(px4_timestamp_us, scale=1e-6)
		if parsed is None:
			return None
		obs = ClockObservation(
			source_sec=parsed,
			receipt_wall_sec=self.wall_sec() if receipt_wall_sec is None else float(receipt_wall_sec),
		)
		self._latest_px4_observation = obs
		self._px4_fit.add(obs)
		return obs

	@staticmethod
	def _parse_positive_seconds(raw, scale: float) -> Optional[float]:
		try:
			value = float(raw) * float(scale)
		except (TypeError, ValueError):
			return None
		return value if math.isfinite(value) and value > 0.0 else None

	# ---- Diagnostic-only affine conversion ------------------------------

	def sim_at_wall_sec(self, wall_sec) -> Optional[float]:
		"""Approximate SIM time at a wall instant using the local affine fit."""
		try:
			return self._sim_fit.source_at_wall(float(wall_sec))
		except (TypeError, ValueError):
			return None

	def wall_at_sim_sec(self, sim_sec) -> Optional[float]:
		"""Approximate wall time for a SIM stamp using the local affine fit."""
		try:
			return self._sim_fit.wall_at_source(float(sim_sec))
		except (TypeError, ValueError):
			return None

	def px4_at_wall_sec(self, wall_sec) -> Optional[float]:
		try:
			return self._px4_fit.source_at_wall(float(wall_sec))
		except (TypeError, ValueError):
			return None

	def wall_at_px4_sec(self, px4_sec) -> Optional[float]:
		try:
			return self._px4_fit.wall_at_source(float(px4_sec))
		except (TypeError, ValueError):
			return None

	# Backward-compatible aliases. Their affine nature is now explicit in the
	# implementation; callers should prefer wall_at_* / *_at_wall names.
	def sim_to_wall_sec(self, sim_sec):
		return self.wall_at_sim_sec(sim_sec)

	def px4_to_wall_sec(self, px4_timestamp_us):
		parsed = self._parse_positive_seconds(px4_timestamp_us, scale=1e-6)
		return None if parsed is None else self.wall_at_px4_sec(parsed)

	# ---- Diagnostic snapshots -------------------------------------------

	def sim_clock_snapshot(self) -> dict:
		result = self._sim_fit.snapshot()
		latest = self._latest_sim_observation
		result.update({
			"latest_source_sec": None if latest is None else latest.source_sec,
			"latest_receipt_wall_sec": None if latest is None else latest.receipt_wall_sec,
		})
		return result

	def px4_clock_snapshot(self) -> dict:
		result = self._px4_fit.snapshot()
		latest = self._latest_px4_observation
		result.update({
			"latest_source_sec": None if latest is None else latest.source_sec,
			"latest_receipt_wall_sec": None if latest is None else latest.receipt_wall_sec,
		})
		return result

	def clock_snapshot(self) -> dict:
		return {
			"sim": self.sim_clock_snapshot(),
			"px4": self.px4_clock_snapshot(),
		}

	# Established method names retained for callers; they now have precise
	# meanings. ``sim_rtf_estimate`` is LOCAL fit RTF, while run-wide RTF is
	# available separately.
	def sim_rtf_estimate(self):
		return self._sim_fit.local_rate()

	def sim_run_rtf_estimate(self):
		return self._sim_fit.run_rate()

	def px4_rtf_estimate(self):
		return self._px4_fit.local_rate()

	def px4_run_rtf_estimate(self):
		return self._px4_fit.run_rate()

	# Deprecated diagnostic values: these are only the latest raw differences,
	# not conversion constants. Kept so older auxiliary code does not crash.
	def sim_wall_offset_sec(self):
		obs = self._latest_sim_observation
		return None if obs is None else obs.receipt_wall_sec - obs.source_sec

	def px4_wall_offset_sec(self):
		obs = self._latest_px4_observation
		return None if obs is None else obs.receipt_wall_sec - obs.source_sec
