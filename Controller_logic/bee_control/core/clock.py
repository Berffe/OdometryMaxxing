"""Clock utilities for the BEE_LAND controller.

Three time bases, and they are not interchangeable:

* camera stamps are Gazebo SIM time and stay untouched;
* monotonic time measures local durations;
* Unix wall time stamps outgoing PX4 messages and external events.

No clock fitting or physical-state reconstruction belongs in the live node.

Why the wall clock is projected, not read
-----------------------------------------
``time.time()`` is the system clock, and the system clock can STEP.  On a VM
guest (WSL2, Hyper-V, any hypervisor with periodic host time sync) it is
corrected against the host every few tens of seconds, and the correction is a
discontinuity, not a rate adjustment.

A forward step is harmless.  A BACKWARD step is not:

* every uORB message stamped from it lands in PX4's past, where PX4 may treat
  it as stale;
* any deadline computed from it -- including ROS timer deadlines -- moves into
  the future by the size of the step, stalling the loop for that long.

A 2.2 s backward step therefore stops the offboard setpoint stream for 2.2 s,
which is four times PX4's ``COM_OF_LOSS_T``, and the vehicle drops to its
offboard-loss failsafe. This was observed: three failsafes per flight, each one
landing exactly on a backward step in the log.

``SteadyWallClock`` removes that failure mode from the outgoing message path by
anchoring the Unix epoch to the monotonic clock ONCE, at construction, and
projecting forward from there. The value stays a real Unix timestamp -- so PX4
still sees the epoch it expects, and merging against the truth log still works
-- but it cannot jump. It will slowly diverge from true UTC, which is the right
trade: a stable timebase matters far more here than agreeing with the host to
the millisecond over a two-minute flight.

The timers need the same treatment, and that is done separately by passing a
``STEADY_TIME`` rclpy clock in ``bee_node`` -- a projected Python value cannot
fix rclpy's own internal deadline arithmetic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ReceiptStamp:
    wall_sec: float
    monotonic_sec: float


@dataclass(frozen=True)
class ClockStep:
    """One detected discontinuity of the system clock."""

    monotonic_sec: float
    step_sec: float          # signed: negative is the dangerous direction

    @property
    def backward(self) -> bool:
        return self.step_sec < 0.0


class SteadyWallClock:
    """Unix-epoch wall time that advances monotonically.

    ``wall_sec()`` returns ``anchor_wall + (monotonic - anchor_monotonic)``.
    Same units and epoch as ``time.time()``, but immune to steps.

    ``check_step()`` compares the projection against the real system clock and
    reports a discontinuity. It never re-anchors: silently absorbing a step
    would reintroduce the jump into the outgoing timestamps, which is the whole
    thing being avoided. The step is reported so it lands in the log, and the
    operator fixes the host.
    """

    def __init__(self, step_threshold_sec: float = 0.05):
        self._anchor_wall = time.time()
        self._anchor_monotonic = time.monotonic()
        self._threshold = max(1e-3, float(step_threshold_sec))
        self._steps: list[ClockStep] = []
        self._last_skew_sec = 0.0

    def wall_sec(self) -> float:
        return self._anchor_wall + (time.monotonic() - self._anchor_monotonic)

    @property
    def skew_sec(self) -> float:
        """system clock minus projected clock. Drifts slowly; steps abruptly."""
        return self._last_skew_sec

    @property
    def steps(self) -> Sequence[ClockStep]:
        return tuple(self._steps)

    @property
    def backward_steps(self) -> int:
        return sum(1 for s in self._steps if s.backward)

    def check_step(self) -> Optional[ClockStep]:
        """Sample the system clock and report a discontinuity, if any.

        Cheap enough to call on the supervisor tick. Returns the step so the
        caller can log it loudly -- a backward step means the HOST is
        misbehaving, and no amount of code makes that not worth knowing.
        """
        skew = time.time() - self.wall_sec()
        delta = skew - self._last_skew_sec
        self._last_skew_sec = skew
        if abs(delta) < self._threshold:
            return None
        step = ClockStep(monotonic_sec=time.monotonic(), step_sec=delta)
        self._steps.append(step)
        return step


class TimeManager:
    """The controller's clocks, and a telemetry source describing them."""

    TELEMETRY_PREFIX: ClassVar[str] = "clock"

    def __init__(self, node=None, *, steady_wall: bool = True,
                 step_threshold_sec: float = 0.05):
        self._node = node
        self._steady = SteadyWallClock(step_threshold_sec=step_threshold_sec)
        self._use_steady = bool(steady_wall)

    # ------------------------------------------------------------------ time
    def wall_sec(self) -> float:
        """Unix wall seconds. Step-immune unless explicitly disabled."""
        return self._steady.wall_sec() if self._use_steady else time.time()

    @staticmethod
    def monotonic_sec() -> float:
        return time.monotonic()

    def receipt_stamp(self) -> ReceiptStamp:
        return ReceiptStamp(self.wall_sec(), self.monotonic_sec())

    def px4_timestamp_us(self) -> int:
        return int(self.wall_sec() * 1_000_000)

    @staticmethod
    def image_stamp_sec(msg) -> float:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return 0.0
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    # ------------------------------------------------------------ monitoring
    def check_clock_step(self) -> Optional[ClockStep]:
        return self._steady.check_step()

    @property
    def clock(self) -> SteadyWallClock:
        return self._steady

    # ------------------------------------------------------------- telemetry
    @classmethod
    def telemetry_fields(cls) -> Sequence[str]:
        return (
            "steady_wall_active",
            "system_minus_steady_skew_sec",
            "steps_detected",
            "backward_steps_detected",
            "last_step_sec",
        )

    def telemetry(self) -> Mapping[str, object]:
        steps = self._steady.steps
        return {
            "steady_wall_active": int(self._use_steady),
            "system_minus_steady_skew_sec": self._steady.skew_sec,
            "steps_detected": len(steps),
            "backward_steps_detected": self._steady.backward_steps,
            "last_step_sec": steps[-1].step_sec if steps else "",
        }