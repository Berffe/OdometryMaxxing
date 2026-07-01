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
"""

import asyncio
import os
import signal
import subprocess
import threading
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
		self.motor_stop_requested = True

	def request_stop(self) -> None:
		self.stop_requested = True

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
		while not self.stop_requested:
			if self.motor_stop_requested and not self.motor_stop_done:
				await self._try_motor_stop(drone)
				if self.motor_stop_done:
					break
			await asyncio.sleep(0.05)

	async def _try_motor_stop(self, drone) -> None:
		"""Stop motors after confirmed Gazebo touchdown.

		Normal disarm is tried first. In SITL, kill() is used as a fallback when
		PX4's internal land detector refuses the disarm on the moving platform.
		This method attempts motor stop only once to avoid spamming MAVSDK/PX4
		with repeated commands if both methods fail.
		"""
		if self.motor_stop_attempted:
			return

		self.motor_stop_attempted = True
		if self._on_pre_motor_stop is not None:
			# Node latches its outgoing setpoint to a zero-thrust hold.
			self._on_pre_motor_stop()

		try:
			self._logger.warning("MAVSDK: touchdown confirmed, requesting disarm.")
			await drone.action.disarm()
			self._logger.warning("MAVSDK: disarm accepted after touchdown.")
			self.motor_stop_done = True
			self.stop_requested = True
			return
		except Exception as exc:
			self.motor_stop_error = repr(exc)
			self._logger.error(
				f"MAVSDK: disarm failed after touchdown: {self.motor_stop_error}"
			)

		if not self._enable_kill_fallback:
			self._logger.error(
				"MAVSDK: kill fallback disabled; keeping zero-thrust offboard stream alive."
			)
			return

		try:
			self._logger.error(
				"MAVSDK: using kill fallback after confirmed Gazebo contact. SITL only."
			)
			await drone.action.kill()
			self._logger.warning("MAVSDK: kill accepted after touchdown.")
			self.motor_stop_done = True
			self.stop_requested = True
		except Exception as kill_exc:
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
