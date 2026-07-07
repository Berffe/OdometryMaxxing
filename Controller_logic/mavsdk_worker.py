"""
MAVSDK worker: automatic takeoff and terminal motor-stop, isolated from the ROS
node.

This is the async/threaded MAVLink-side subsystem, extracted from bee_node.py so
the node itself stays synchronous and readable. It owns exactly one job: use
MAVSDK to connect, arm, and climb to the takeoff altitude, then sit idle until
asked to stop the motors after a confirmed touchdown. Closed-loop attitude/thrust
setpoints do NOT go through here -- they are published directly to PX4 over
ROS 2/uXRCE-DDS by the node. MAVSDK is kept only for the two things it does more
robustly than raw offboard: the initial guided takeoff, and the terminal
disarm/kill.

INTERFACE (what the node uses):
    worker = MavsdkWorker(logger, on_pre_motor_stop=..., <config...>)
    worker.start()                 # spawn the worker thread (idempotent)
    worker.takeoff_started         # bool
    worker.takeoff_done            # bool  -> node advances past MAVSDK_TAKEOFF
    worker.takeoff_error           # None | str -> node aborts if set
    worker.request_motor_stop()    # after confirmed touchdown
    worker.motor_stop_done         # bool
    worker.motor_stop_error        # None | str
    worker.request_stop()          # tell the worker loop to exit (shutdown)

on_pre_motor_stop is an optional zero-argument callback invoked once, on the
worker thread, immediately before the disarm/kill attempt. The node uses it to
latch its own outgoing setpoint to a zero-thrust hold so nothing fights the
disarm. It must be thread-safe from the node's side (a single attribute
assignment is fine).

THREADING: the worker runs its own asyncio event loop on a daemon thread. The
status fields (takeoff_done, motor_stop_done, ...) are plain attributes written
on that thread and read on the node's main thread. They are single-writer
booleans/strings used only as one-way latches, so this is safe without a lock
(the node never writes them; it only writes the request_* intents, which are
likewise single-writer from its side).

TOUCHDOWN-TO-MOTOR-STOP LATENCY. request_motor_stop()->motor_stop_done used to
be gated purely by a fixed 50ms poll (`await asyncio.sleep(0.05)` in the main
loop), and _try_motor_stop() always attempted a normal disarm() FIRST, even
though on a moving platform PX4's own land detector is expected to refuse it
-- meaning the common path paid for a full doomed MAVLink round-trip before
ever reaching kill(), the command that actually works here. Both were real,
stacking, previously unmeasured contributors to the gap between physical leg
contact and motors actually stopping (see bee_node.py's touchdown handling +
platform_motion.py's PLATFORM_TOP_SURFACE_OFFSET_M for how that gap was
found: ~19cm of apparent leg interpenetration in one logged landing). Fixed
here two ways:
  1. An asyncio.Event wakes the worker loop the instant request_motor_stop()
     fires, instead of waiting out the next poll tick -- see _wake_event.
  2. When enable_kill_fallback is set (the SITL/moving-platform case this
     whole mechanism exists for), the doomed disarm() attempt is skipped
     entirely and kill() is issued directly -- see _try_motor_stop.
  3. Every stage now gets a monotonic timestamp (self.timing_*), so the next
     landing log can show which stage actually dominates rather than assume
     it -- these are diagnostics-only, read by nothing in this file, plain
     time.monotonic() (not clock.py's WALL/SIM/PX4 family: this is a
     same-process duration measurement, not a cross-system correlation, so
     it doesn't need that taxonomy).
"""

import asyncio
import os
import signal
import subprocess
import threading
import time
from typing import Callable, Optional

try:
	from mavsdk import System
except ImportError:  # Allows ROS-free / mavsdk-free import for tests.
	System = None


class MavsdkWorker:
	def __init__(
		self,
		logger,
		on_pre_motor_stop: Optional[Callable[[], None]] = None,
		*,
		system_address: str,
		port_to_free: int,
		takeoff_altitude_m: float,
		connect_timeout_sec: float,
		health_timeout_sec: float,
		takeoff_altitude_timeout_sec: float,
		ekf2_settle_time_sec: float,
		enable_kill_fallback: bool,
	):
		self._logger = logger
		self._on_pre_motor_stop = on_pre_motor_stop

		self._system_address = str(system_address)
		self._port_to_free = int(port_to_free)
		self._takeoff_altitude_m = float(takeoff_altitude_m)
		self._connect_timeout = float(connect_timeout_sec)
		self._health_timeout = float(health_timeout_sec)
		self._takeoff_timeout = float(takeoff_altitude_timeout_sec)
		self._ekf2_settle = float(ekf2_settle_time_sec)
		self._enable_kill_fallback = bool(enable_kill_fallback)

		self._thread: Optional[threading.Thread] = None
		# Set once the worker's event loop is running (see _worker_async);
		# lets request_motor_stop()/request_stop(), called from the node's
		# thread, wake the worker immediately instead of waiting out a poll.
		self._loop: Optional[asyncio.AbstractEventLoop] = None
		self._wake_event: Optional[asyncio.Event] = None

		# One-way status latches (written on the worker thread, read on main).
		self.takeoff_started = False
		self.takeoff_done = False
		self.takeoff_error: Optional[str] = None

		# Intents (written on the node's thread, read on the worker thread).
		self.stop_requested = False
		self.motor_stop_requested = False

		# Motor-stop status latches (written on the worker thread).
		self.motor_stop_attempted = False
		self.motor_stop_done = False
		self.motor_stop_error: Optional[str] = None

		# Diagnostics-only timing (time.monotonic() seconds; None until each
		# stage happens). See module docstring's TOUCHDOWN-TO-MOTOR-STOP
		# LATENCY note -- lets a landing log show which stage actually
		# dominates the touchdown-to-motor-stop gap instead of assuming it.
		self.timing_motor_stop_requested = None      # request_motor_stop() called
		self.timing_motor_stop_picked_up = None      # worker loop woke and saw the request
		self.timing_pre_motor_stop_done = None       # on_pre_motor_stop() callback returned
		self.timing_disarm_attempted = None          # None if skipped (see _try_motor_stop)
		self.timing_disarm_result = None
		self.timing_kill_attempted = None
		self.timing_kill_result = None

	# ---- node-facing controls ------------------------------------------------
	def start(self) -> None:
		"""Spawn the worker thread (idempotent). Sets takeoff_error instead of
		raising if MAVSDK is unavailable, so the node can abort cleanly."""
		if self.takeoff_started:
			return
		if System is None:
			self.takeoff_error = "mavsdk is not installed in this Python environment"
			return
		self.takeoff_started = True
		self._logger.info(f"Starting MAVSDK takeoff to {self._takeoff_altitude_m:.2f} m.")
		self._thread = threading.Thread(
			target=self._run_thread, name="mavsdk_takeoff", daemon=True
		)
		self._thread.start()

	def request_motor_stop(self) -> None:
		self.timing_motor_stop_requested = time.monotonic()
		self.motor_stop_requested = True
		self._wake()

	def request_stop(self) -> None:
		self.stop_requested = True
		self._wake()

	def _wake(self) -> None:
		"""Nudge the worker's poll loop to check intents immediately, from
		whatever thread this is called on (the node's thread, normally).
		Best-effort: the flags above remain the actual source of truth, so a
		missed wake (e.g. called before the worker's loop exists yet) just
		falls back to the loop's own short timeout poll -- never incorrect,
		only possibly up to one poll period slower. See _worker_async."""
		loop, event = self._loop, self._wake_event
		if loop is not None and event is not None:
			loop.call_soon_threadsafe(event.set)

	# ---- worker thread -------------------------------------------------------
	def _run_thread(self) -> None:
		try:
			asyncio.run(self._worker_async())
		except Exception as exc:
			if not self.takeoff_done:
				self.takeoff_error = repr(exc)
			else:
				self.motor_stop_error = repr(exc)

	async def _worker_async(self) -> None:
		# Created here (not in __init__) so it's bound to THIS running loop --
		# request_motor_stop()/request_stop() reach it via call_soon_threadsafe
		# from the node's thread (see _wake()).
		self._wake_event = asyncio.Event()
		self._loop = asyncio.get_running_loop()

		self._free_port(self._port_to_free)
		drone = System()
		await drone.connect(system_address=self._system_address)

		self._logger.info("MAVSDK: waiting for drone connection...")
		await self._wait_for_condition(
			drone.core.connection_state(),
			lambda state: state.is_connected,
			self._connect_timeout,
			"MAVSDK connection",
		)
		self._logger.info("MAVSDK: connected.")

		self._logger.info("MAVSDK: waiting for global/home/local position estimates...")
		await self._wait_for_condition(
			drone.telemetry.health(),
			lambda h: h.is_global_position_ok and h.is_home_position_ok and h.is_local_position_ok,
			self._health_timeout,
			"global/home/local position health",
		)
		self._logger.info("MAVSDK: all position estimates OK.")

		await asyncio.sleep(self._ekf2_settle)
		home_position = await self._wait_for_condition(
			drone.telemetry.position(),
			lambda position: True,
			self._connect_timeout,
			"initial position reading",
		)
		home_baro_offset = home_position.relative_altitude_m
		self._logger.info(f"MAVSDK: home altitude offset {home_baro_offset:.2f} m.")

		self._logger.info(f"MAVSDK: setting MIS_TAKEOFF_ALT={self._takeoff_altitude_m:.2f} m.")
		await drone.param.set_param_float("MIS_TAKEOFF_ALT", float(self._takeoff_altitude_m))
		await asyncio.sleep(0.5)

		self._logger.info("MAVSDK: arming.")
		await drone.action.arm()
		self._logger.info("MAVSDK: takeoff command.")
		await drone.action.takeoff()

		await self._wait_for_condition(
			drone.telemetry.position(),
			lambda p: (p.relative_altitude_m - home_baro_offset) >= self._takeoff_altitude_m - 0.20,
			self._takeoff_timeout,
			"takeoff altitude reached",
			progress_fn=lambda p, elapsed: self._logger.info(
				f"MAVSDK: still climbing after {elapsed:.0f}s -- altitude={p.relative_altitude_m - home_baro_offset:.2f} m"
			),
			progress_interval=5.0,
		)
		self._logger.info("MAVSDK: reached takeoff altitude; hovering.")
		self.takeoff_done = True

		# MAVSDK is kept only for takeoff and terminal motor-stop actions.
		# Closed-loop attitude/thrust setpoints are published directly to PX4 via
		# ROS 2/uXRCE-DDS by the node's setpoint timer.
		#
		# Woken immediately by request_motor_stop()/request_stop() via
		# _wake_event (see module docstring) instead of waiting out a fixed
		# poll; the 0.05s timeout below is now only a safety net for the
		# (should-never-happen) case of a missed wake, so this is never
		# SLOWER than the old fixed-poll design, only potentially faster.
		while not self.stop_requested:
			if self.motor_stop_requested and not self.motor_stop_done:
				self.timing_motor_stop_picked_up = time.monotonic()
				await self._try_motor_stop(drone)
				if self.motor_stop_done:
					break
			try:
				await asyncio.wait_for(self._wake_event.wait(), timeout=0.05)
			except asyncio.TimeoutError:
				pass
			self._wake_event.clear()

	async def _try_motor_stop(self, drone) -> None:
		"""Stop motors after confirmed Gazebo touchdown.

		When enable_kill_fallback is set -- the SITL/moving-platform case this
		mechanism exists for -- disarm() is SKIPPED entirely and kill() is
		issued directly. This used to try disarm() first always, but PX4's own
		land detector is expected to refuse a normal disarm on a moving deck
		(that expectation is exactly WHY the kill fallback exists), so trying
		it first on this path was paying for one full doomed MAVLink
		round-trip, serially, before the command that actually works -- a
		real, previously unmeasured contributor to the gap between physical
		leg contact and motors actually stopping. When enable_kill_fallback is
		NOT set (the real-hardware-oriented default, where forcibly killing
		motors is a much bigger deal than in sim), the original disarm-only
		behavior is unchanged -- there is no fallback to skip to.

		This method attempts motor stop only once to avoid spamming MAVSDK/PX4
		with repeated commands if it fails.
		"""
		if self.motor_stop_attempted:
			return

		self.motor_stop_attempted = True
		if self._on_pre_motor_stop is not None:
			# Node latches its outgoing setpoint to a zero-thrust hold.
			self._on_pre_motor_stop()
		self.timing_pre_motor_stop_done = time.monotonic()

		if not self._enable_kill_fallback:
			try:
				self._logger.warning("MAVSDK: touchdown confirmed, requesting disarm.")
				self.timing_disarm_attempted = time.monotonic()
				await drone.action.disarm()
				self.timing_disarm_result = time.monotonic()
				self._logger.warning("MAVSDK: disarm accepted after touchdown.")
				self.motor_stop_done = True
				self.stop_requested = True
			except Exception as exc:
				self.timing_disarm_result = time.monotonic()
				self.motor_stop_error = repr(exc)
				self._logger.error(
					f"MAVSDK: disarm failed after touchdown: {self.motor_stop_error}"
				)
				self._logger.error(
					"MAVSDK: kill fallback disabled; keeping zero-thrust offboard stream alive."
				)
			return

		# enable_kill_fallback is set: go straight to kill(), see docstring.
		try:
			self._logger.error(
				"MAVSDK: using kill fallback after confirmed Gazebo contact. SITL only."
			)
			self.timing_kill_attempted = time.monotonic()
			await drone.action.kill()
			self.timing_kill_result = time.monotonic()
			self._logger.warning("MAVSDK: kill accepted after touchdown.")
			self.motor_stop_done = True
			self.stop_requested = True
		except Exception as kill_exc:
			self.timing_kill_result = time.monotonic()
			self.motor_stop_error = repr(kill_exc)
			self._logger.error(
				f"MAVSDK: kill fallback failed: {self.motor_stop_error}"
			)

	# ---- helpers -------------------------------------------------------------
	@staticmethod
	async def _wait_for_condition(
		async_iterable, condition, timeout: float, label: str,
		progress_fn=None, progress_interval: float = 5.0,
	):
		async def _inner():
			loop = asyncio.get_event_loop()
			start = loop.time()
			last_progress = start
			async for item in async_iterable:
				if condition(item):
					return item
				if progress_fn is not None:
					now = loop.time()
					if now - last_progress >= progress_interval:
						last_progress = now
						progress_fn(item, now - start)
		try:
			return await asyncio.wait_for(_inner(), timeout=timeout)
		except asyncio.TimeoutError:
			raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {label}")

	@staticmethod
	def _free_port(port: int):
		try:
			result = subprocess.run(
				["lsof", "-t", f"-i:UDP:{port}"],
				capture_output=True, text=True, check=False,
			)
		except FileNotFoundError:
			return
		for pid in result.stdout.strip().split():
			try:
				os.kill(int(pid), signal.SIGKILL)
			except ProcessLookupError:
				pass