"""Minimal clock utilities for the BEE_LAND controller.

Only three operations remain:
- camera stamps are Gazebo SIM time and stay untouched;
- monotonic time measures local durations;
- Unix wall time stamps outgoing PX4 messages and external events.

No clock fitting or physical-state reconstruction belongs in the live node.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ReceiptStamp:
	wall_sec: float
	monotonic_sec: float


class TimeManager:
	def __init__(self, node=None):
		self._node = node

	@staticmethod
	def wall_sec() -> float:
		return time.time()

	@staticmethod
	def monotonic_sec() -> float:
		return time.monotonic()

	@classmethod
	def receipt_stamp(cls) -> ReceiptStamp:
		return ReceiptStamp(cls.wall_sec(), cls.monotonic_sec())

	@classmethod
	def px4_timestamp_us(cls) -> int:
		return int(cls.wall_sec() * 1_000_000)

	@staticmethod
	def image_stamp_sec(msg) -> float:
		stamp = getattr(getattr(msg, "header", None), "stamp", None)
		if stamp is None:
			return 0.0
		return float(stamp.sec) + 1e-9 * float(stamp.nanosec)
