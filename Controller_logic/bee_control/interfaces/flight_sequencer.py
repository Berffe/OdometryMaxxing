"""Outer controller lifecycle: takeoff -> offboard -> handoff -> terminal.

This is the second state machine in the system.  ``MissionRoutine`` owns the
visual landing sequence *inside* CLOSED_LOOP; this owns everything around it --
waiting for MAVSDK takeoff, pre-streaming neutral setpoints so PX4 will accept
OFFBOARD, confirming ARMED+OFFBOARD, settling, and gating the handoff on one
fresh valid vision result.

It used to be a ~120-line ``if``-chain inside ``bee_node.on_supervisor_timer``,
which is why it could not be tested without a live DDS stack.  Here it talks
only to :class:`SequencerPorts` -- plain callables -- so a test can drive a full
takeoff-to-touchdown sequence with a fake clock and no ROS at all.

Setpoint authority
------------------
The sequencer does not build setpoints; it declares who is allowed to.
``setpoint_policy`` returns one of:

``INHIBIT``      publish nothing (PX4 is still flying the MAVSDK takeoff)
``NEUTRAL_HOLD`` publish hover thrust, level attitude -- visual control inhibited
``CONTROL``      publish whatever the control law last produced
``ZERO_THRUST``  publish zero thrust (post-touchdown)

``bee_node.on_px4_timer`` applies that policy and nothing else, so there is
exactly one place in the system that decides when visual commands may reach the
vehicle.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from bee_control.core.config import BeeConfig
from bee_control.core.controller_state import PX4Status
# Controller phases. Distinct from mission substates: these describe the outer
# PX4 / handoff lifecycle, those describe the visual landing sequence. They are
# logged in separate columns on purpose and must not be conflated.
WAIT_TAKEOFF = "mavsdk_takeoff"
PRESTREAM = "prestream_offboard"
WAIT_OFFBOARD = "wait_offboard"
OFFBOARD_SETTLE = "offboard_settle"
CLOSED_LOOP = "closed_loop"
LANDED = "landed"
ABORTED = "aborted"

TERMINAL_PHASES = (LANDED, ABORTED)


class SetpointPolicy(Enum):
    INHIBIT = "inhibit"
    NEUTRAL_HOLD = "neutral_hold"
    CONTROL = "control"
    ZERO_THRUST = "zero_thrust"


_POLICY_BY_PHASE = {
    WAIT_TAKEOFF: SetpointPolicy.INHIBIT,
    PRESTREAM: SetpointPolicy.NEUTRAL_HOLD,
    WAIT_OFFBOARD: SetpointPolicy.NEUTRAL_HOLD,
    OFFBOARD_SETTLE: SetpointPolicy.NEUTRAL_HOLD,
    CLOSED_LOOP: SetpointPolicy.CONTROL,
    LANDED: SetpointPolicy.ZERO_THRUST,
    ABORTED: SetpointPolicy.NEUTRAL_HOLD,
}


@dataclass
class SequencerPorts:
    """Everything the sequencer needs from the outside world.

    Deliberately callables rather than an object: it keeps the dependency
    explicit and one-directional, and it means a test supplies six lambdas
    instead of a mock ROS node.
    """

    #: MAVSDK takeoff progress.
    takeoff_done: Callable[[], bool]
    takeoff_error: Callable[[], Optional[str]]

    #: Ask PX4 to switch to OFFBOARD. Called repeatedly until confirmed.
    request_offboard: Callable[[], None]

    #: Latest ``(target, flow, metrics)`` bundle, or None if none yet.
    latest_vision: Callable[[], Optional[tuple]]

    #: Console logging.
    log_info: Callable[[str], None]
    log_error: Callable[[str], None]
    #: Write one diagnostics row: ``(event, detail)``.
    log_event: Callable[[str, str], None]

    #: Called once, with the flow SIM timestamp that authorised the handoff.
    on_handoff: Callable[[float], None]
    #: Terminal transitions. The node owns what they mean (motor stop, etc).
    on_landed: Callable[[str], None]
    on_aborted: Callable[[str], None]

    #: Monotonic seconds. Injected so tests can drive a fake clock.
    monotonic: Callable[[], float]


class FlightSequencer:
    def __init__(self, config: BeeConfig, px4_status: PX4Status,
                 ports: SequencerPorts):
        self._cfg = config
        self._sched = config.scheduling
        self._px4 = px4_status
        self._ports = ports

        self._phase = WAIT_TAKEOFF
        self._phase_start_mono = ports.monotonic()
        self._offboard_request_mono: Optional[float] = None
        self._last_offboard_request_mono: Optional[float] = None
        self._settle_flow_floor: Optional[float] = None
        self._waiting_for_fresh_handoff_logged = False
        self._last_landed_log_mono: Optional[float] = None

    # ------------------------------------------------------------------ state
    @property
    def phase(self) -> str:
        return self._phase

    @property
    def is_closed_loop(self) -> bool:
        return self._phase == CLOSED_LOOP

    @property
    def is_terminal(self) -> bool:
        return self._phase in TERMINAL_PHASES

    @property
    def setpoint_policy(self) -> SetpointPolicy:
        return _POLICY_BY_PHASE.get(self._phase, SetpointPolicy.NEUTRAL_HOLD)

    def _set_phase(self, phase: str) -> None:
        if phase != self._phase:
            self._ports.log_info(f"Controller phase: {self._phase} -> {phase}")
        self._phase = phase
        self._phase_start_mono = self._ports.monotonic()

    def _elapsed(self, now: float) -> float:
        return now - self._phase_start_mono

    # -------------------------------------------------------- terminal states
    def enter_landed(self, reason: str) -> None:
        """Latch touchdown. Idempotent; safe to call from any thread context."""
        if self._phase == LANDED:
            return
        self._set_phase(LANDED)
        self._last_landed_log_mono = None
        self._ports.on_landed(reason)
        self._ports.log_event("landed", reason)

    def abort(self, reason: str) -> None:
        """Latch an abort. Never overrides an already-latched touchdown."""
        if self._phase in TERMINAL_PHASES:
            return
        self._set_phase(ABORTED)
        self._ports.log_error(f"ABORT: {reason}")
        self._ports.on_aborted(reason)
        self._ports.log_event("aborted", reason)

    # --------------------------------------------------------------- the tick
    def update(self) -> None:
        """One supervisor tick. Advances the outer lifecycle by at most one step."""
        error = self._ports.takeoff_error()
        if error:
            self.abort(f"MAVSDK takeoff failed: {error}")
            return

        now = self._ports.monotonic()
        handler = getattr(self, f"_in_{self._phase_key()}", None)
        if handler is not None:
            handler(now)

    def _phase_key(self) -> str:
        return {
            WAIT_TAKEOFF: "wait_takeoff",
            PRESTREAM: "prestream",
            WAIT_OFFBOARD: "wait_offboard",
            OFFBOARD_SETTLE: "offboard_settle",
            LANDED: "landed",
        }.get(self._phase, "noop")

    def _in_noop(self, now: float) -> None:
        """CLOSED_LOOP and ABORTED need nothing from the supervisor."""

    def _in_landed(self, now: float) -> None:
        # Keep the controller log alive after PX4 disarms. The truth stream
        # continues independently; these sparse snapshots preserve the
        # controller / mission terminal state for offline merging.
        period = self._sched.post_landing_log_period_sec
        if (self._last_landed_log_mono is None
                or now - self._last_landed_log_mono >= period):
            self._last_landed_log_mono = now
            self._ports.log_event("landed_state", "post-touchdown controller state")

    def _in_wait_takeoff(self, now: float) -> None:
        if not self._ports.takeoff_done():
            return
        self._set_phase(PRESTREAM)
        self._ports.log_info(
            "MAVSDK takeoff complete. Starting neutral PX4 offboard prestream.")
        self._ports.log_event("takeoff_complete", "")

    def _in_prestream(self, now: float) -> None:
        # PX4 rejects OFFBOARD unless a setpoint stream is already flowing, so
        # the neutral hold must run for a while BEFORE the mode request.
        if self._elapsed(now) < self._sched.offboard_prestream_sec:
            return
        self._ports.log_info(
            "Requesting PX4 offboard mode; visual control remains inhibited.")
        self._ports.request_offboard()
        self._offboard_request_mono = now
        self._last_offboard_request_mono = now
        self._set_phase(WAIT_OFFBOARD)
        self._ports.log_event("offboard_requested", "")

    def _in_wait_offboard(self, now: float) -> None:
        if self._px4.offboard_confirmed:
            bundle = self._ports.latest_vision()
            flow = bundle[1] if bundle else None
            # Freeze the current flow timestamp: the handoff below requires a
            # STRICTLY NEWER result, so a stale bundle cannot authorise it.
            self._settle_flow_floor = float(getattr(flow, "timestamp", 0.0) or 0.0)
            self._waiting_for_fresh_handoff_logged = False
            self._set_phase(OFFBOARD_SETTLE)
            self._ports.log_info(
                "PX4 confirms ARMED+OFFBOARD. Holding neutral thrust for settle time.")
            self._ports.log_event("offboard_confirmed", "")
            return

        waited = now - (self._offboard_request_mono or now)
        since_request = now - (self._last_offboard_request_mono or now)
        if since_request >= self._sched.px4_offboard_reengage_interval_sec:
            self._ports.request_offboard()
            self._last_offboard_request_mono = now
        if waited >= self._sched.px4_offboard_confirm_timeout_sec:
            self.abort(
                "PX4 did not confirm ARMED+OFFBOARD within "
                f"{self._sched.px4_offboard_confirm_timeout_sec:.1f} s")

    def _in_offboard_settle(self, now: float) -> None:
        if not self._px4.offboard_confirmed:
            self.abort("PX4 left offboard during the neutral settle")
            return
        if self._elapsed(now) < self._sched.px4_offboard_switch_settle_sec:
            return

        bundle = self._ports.latest_vision()
        if bundle is None:
            return
        target, flow, _ = bundle
        flow_stamp = float(getattr(flow, "timestamp", 0.0) or 0.0)
        fresh = (
            flow_stamp > float(self._settle_flow_floor or 0.0)
            and bool(getattr(target, "found", False))
            and bool(getattr(flow, "valid", False))
        )
        if not fresh:
            if not self._waiting_for_fresh_handoff_logged:
                self._ports.log_info(
                    "Neutral settle complete; waiting for one fresh valid vision result.")
                self._waiting_for_fresh_handoff_logged = True
            return

        # The result that AUTHORISES the handoff stays neutral. The first visual
        # command is formed from the following camera result.
        self._ports.on_handoff(flow_stamp)
        self._set_phase(CLOSED_LOOP)
        detail = (
            f"flow_sim={flow_stamp:.6f}, divergence={float(flow.divergence):+.4f}, "
            f"offset=({float(target.offset_x):+.3f},{float(target.offset_y):+.3f})")
        self._ports.log_info(f"VISION-CONTROLLER HANDOFF COMPLETE: {detail}")
        self._ports.log_event("vision_controller_handoff", detail)
