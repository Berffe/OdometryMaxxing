"""
Single source of truth for time across the BEE_LAND node.

The node straddles THREE clocks that must never be compared by absolute value
to one another. Before this module they were sampled ad hoc -- some callbacks
used time.time(), px4_interface used node.get_clock().now() -- which is only
safe while use_sim_time is false, and silently breaks the offboard stream the
moment it isn't. Everything that needs "now" now goes through one TimeManager,
so the clock family is an explicit, auditable choice rather than an accident of
which call happened to be nearby.

THE THREE FAMILIES
------------------
WALL  (epoch seconds / microseconds)
    The uXRCE-DDS agent runs its PX4<->companion timesync against the agent's
    WALL clock, NOT the ROS clock; the two coincide only when use_sim_time is
    false (confirmed by PX4's maintainers, and the cause of the "time jump
    detected" / offboard-drop class of bug in sim). Therefore EVERY timestamp
    we publish TO PX4 -- OffboardControlMode, VehicleAttitudeSetpoint,
    VehicleCommand -- is stamped on this clock via px4_tx_timestamp_us(), which
    reads a dedicated SYSTEM_TIME clock that ignores use_sim_time. This keeps
    the bridge's own timesync valid no matter how use_sim_time is configured.
    WALL is also the only clock used to build the diagnostics t_sec origin.

SIM   (Gazebo seconds)
    The camera Image.header.stamp lives here, and through it optical flow,
    target acquisition, and the control-law dt (see bee_node._control_dt_sec).
    PX4 SITL advances lockstep with this clock, but its absolute value is
    offset from WALL by a roughly constant amount (PX4 boot vs node start), so
    SIM and WALL are related ONLY through the measured offset below, never by
    raw subtraction.

PX4   (hrt microseconds since boot)
    Carried on incoming /fmu/out messages (VehicleLocalPosition.timestamp,
    VehicleAttitude.timestamp). Useful for diagnostics and cross-checks;
    converted into WALL seconds through the measured offset.

WHAT THE MEASURED OFFSETS ARE FOR (AND NOT FOR)
-----------------------------------------------
observe_px4_timestamp()/observe_sim_timestamp() maintain smoothed estimates of
(WALL - PX4) and (WALL - SIM). These exist purely so diagnostics can place all
three families on one axis and so desync is OBSERVABLE (log px4_wall_offset_sec
and watch it drift). They are deliberately NOT used to alter outgoing PX4
timestamps: those stay raw WALL, because the bridge's timesync -- not this
node -- is the authority on the PX4<->companion relationship. Folding our own
offset into outgoing stamps would double-correct and fight the bridge.

The offsets are sampled at callback-entry wall time, so each observation is
biased by transport+callback latency. That bias is small (sub-millisecond to a
few ms in SITL) and roughly constant, which is fine for alignment/observability
but is why these offsets must not feed a control path.
"""

import time

try:
	from rclpy.clock import Clock, ClockType
except ImportError:  # Allows ROS-free import for tests.
	Clock = None
	ClockType = None


class TimeManager:
	def __init__(self, node=None, offset_smoothing: float = 0.98):
		"""
		node: the rclpy Node, used only to anchor a SYSTEM_TIME clock. If None
		      (or rclpy unavailable), falls back to time.time_ns(); both yield
		      the same epoch clock, so behavior is identical in tests.
		offset_smoothing: EMA factor in [0, 1) for the WALL<->PX4 / WALL<->SIM
		      offset estimates. Higher = steadier, slower to react.
		"""
		self._node = node
		self._offset_smoothing = max(0.0, min(1.0 - 1e-6, float(offset_smoothing)))

		if node is not None and Clock is not None:
			# A dedicated wall clock, independent of the node's ROS clock and
			# therefore independent of use_sim_time. This is the clock the
			# uXRCE-DDS agent's timesync expects for outgoing PX4 stamps.
			self._system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
		else:
			self._system_clock = None

		# wall = px4 + offset ;  wall = sim + offset.  None until first sample.
		self._px4_to_wall_offset_sec = None
		self._sim_to_wall_offset_sec = None

		# Rate diagnostics: remote clock seconds advanced per wall second.
		# In SITL, sim_rtf_estimate is the measured real-time factor. PX4 should
		# normally track the same time base as SITL; a px4_rtf_estimate that drifts
		# away from sim_rtf_estimate is a clock/timesync diagnostic, not a control
		# input.
		self._px4_rtf_estimate = None
		self._sim_rtf_estimate = None
		self._last_px4_rate_sample = None  # (remote_sec, wall_sec)
		self._last_sim_rate_sample = None  # (remote_sec, wall_sec)

	# ---- WALL: the clock for PX4 I/O and diagnostics origin ------------------

	def wall_ns(self) -> int:
		if self._system_clock is not None:
			return int(self._system_clock.now().nanoseconds)
		return time.time_ns()

	def wall_sec(self) -> float:
		return self.wall_ns() / 1e9

	def px4_tx_timestamp_us(self) -> int:
		"""
		The timestamp to stamp on OUTGOING PX4 messages (microseconds).

		Sample this ONCE per publish cycle and pass the same value to the
		heartbeat and the setpoint so PX4 sees a single coherent instant for
		the pair, rather than two stamps a few microseconds apart.
		"""
		return self.wall_ns() // 1000

	# ---- Offset estimation (diagnostics / observability only) ---------------

	def observe_px4_timestamp(self, px4_timestamp_us) -> None:
		"""Feed an incoming /fmu/out msg.timestamp (microseconds) to track the
		WALL<->PX4 offset and PX4-clock rate relative to wall time. No-op for
		missing/zero stamps."""
		parsed = self._parse_remote_seconds(px4_timestamp_us, scale=1e-6)
		if parsed is None:
			return
		remote_sec = parsed
		wall_sec = self.wall_sec()
		self._px4_to_wall_offset_sec = self._update_offset_at_wall(
			self._px4_to_wall_offset_sec, remote_sec, wall_sec
		)
		self._px4_rtf_estimate, self._last_px4_rate_sample = self._update_rate_estimate(
			self._px4_rtf_estimate, self._last_px4_rate_sample, remote_sec, wall_sec
		)

	def observe_sim_timestamp(self, sim_sec) -> None:
		"""Feed a vision/sim timestamp (seconds, e.g. the image stamp) to track
		the WALL<->SIM offset and measured simulator real-time factor. No-op for
		missing/zero stamps."""
		parsed = self._parse_remote_seconds(sim_sec, scale=1.0)
		if parsed is None:
			return
		remote_sec = parsed
		wall_sec = self.wall_sec()
		self._sim_to_wall_offset_sec = self._update_offset_at_wall(
			self._sim_to_wall_offset_sec, remote_sec, wall_sec
		)
		self._sim_rtf_estimate, self._last_sim_rate_sample = self._update_rate_estimate(
			self._sim_rtf_estimate, self._last_sim_rate_sample, remote_sec, wall_sec
		)

	@staticmethod
	def _parse_remote_seconds(raw_remote, scale):
		try:
			remote_sec = float(raw_remote) * scale
		except (TypeError, ValueError):
			return None
		return remote_sec if remote_sec > 0.0 else None

	def _update_offset_at_wall(self, current, remote_sec, wall_sec):
		observed = wall_sec - remote_sec
		if current is None:
			return observed
		a = self._offset_smoothing
		return a * current + (1.0 - a) * observed

	def _update_rate_estimate(self, current, last_sample, remote_sec, wall_sec):
		if last_sample is None:
			return current, (remote_sec, wall_sec)

		last_remote, last_wall = last_sample
		d_remote = remote_sec - last_remote
		d_wall = wall_sec - last_wall

		# Ignore duplicate/out-of-order stamps and tiny wall intervals. This is
		# diagnostics-only, so it is better to leave the previous estimate untouched
		# than to inject a spike.
		if d_remote <= 0.0 or d_wall <= 1e-4:
			return current, last_sample

		instant = d_remote / d_wall
		if not (0.0 < instant < 10.0):
			return current, last_sample

		if current is None:
			return instant, (remote_sec, wall_sec)

		a = self._offset_smoothing
		return a * current + (1.0 - a) * instant, (remote_sec, wall_sec)

	# ---- Cross-family conversion (diagnostics) ------------------------------

	def px4_to_wall_sec(self, px4_timestamp_us):
		"""PX4 microseconds -> WALL seconds, or None if no offset seeded yet."""
		if self._px4_to_wall_offset_sec is None or px4_timestamp_us is None:
			return None
		try:
			return float(px4_timestamp_us) / 1e6 + self._px4_to_wall_offset_sec
		except (TypeError, ValueError):
			return None

	def sim_to_wall_sec(self, sim_sec):
		"""SIM seconds -> WALL seconds, or None if no offset seeded yet."""
		if self._sim_to_wall_offset_sec is None or sim_sec is None:
			return None
		try:
			return float(sim_sec) + self._sim_to_wall_offset_sec
		except (TypeError, ValueError):
			return None

	def px4_wall_offset_sec(self):
		"""Smoothed (WALL - PX4) in seconds, or None if unseeded. Watch this for
		drift to detect bridge desync."""
		return self._px4_to_wall_offset_sec

	def sim_wall_offset_sec(self):
		"""Smoothed (WALL - SIM) in seconds, or None if unseeded."""
		return self._sim_to_wall_offset_sec

	def px4_rtf_estimate(self):
		"""Smoothed PX4-clock seconds per wall second, or None if unseeded."""
		return self._px4_rtf_estimate

	def sim_rtf_estimate(self):
		"""Smoothed SIM seconds per wall second: the measured real-time factor."""
		return self._sim_rtf_estimate
