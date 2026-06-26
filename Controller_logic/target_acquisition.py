"""
Lightweight, NN-free target acquisition, decoupled from ROS.

	update(frame_bgr, timestamp) -> TargetEstimate

Pipeline: blur -> HSV saliency mask | Canny edges -> morphological cleanup ->
contour selection -> centroid -> normalized offsets + bounding box. The box is
returned so optical_flow can restrict divergence to the target ROI;
area_fraction (box area / frame area) is the controller's scheduling variable.

Two robustness behaviors:
  - Large detections are down-weighted, never rejected (_large_area_penalty):
    a near-full-frame flower at touchdown is the success condition, not an
    outlier.
  - A short temporal hold bridges single-frame dropouts (_held_target_or_lost):
    within loss_grace_period_sec the last good estimate is reused with confidence
    decayed toward zero, instead of immediately reporting found=False.

TargetEstimate.fov_saturated flags when the box touches all four image
borders: the target's true size meets or exceeds the camera's field of
view, not just fills it. Past that point area_fraction/detection_width/
height are a frame-size artifact, not a measurement -- see state.py.

Run `python target_acquisition.py` (or `-m bee_control.target_acquisition`) to
launch the visual debug test.
"""

from typing import Optional, Sequence, Tuple
from dataclasses import replace

import cv2
import numpy as np

try:
	from .state import FlowResult, TargetEstimate
except ImportError:
	from state import FlowResult, TargetEstimate


HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


class TargetAcquisition:
	def __init__(
		self,
		hsv_ranges: Optional[Sequence[HSVRange]] = None,
		min_area_px: float = 80.0,
		max_area_fraction: float = 0.60,
		absolute_max_area_fraction: float = 0.95,
		min_large_area_penalty: float = 0.5,
		blur_kernel_size: int = 5,
		morph_kernel_size: int = 5,
		min_saturation: int = 60,
		min_value: int = 45,
		canny_low: int = 60,
		canny_high: int = 140,
		loss_grace_period_sec: float = 0.15,
		fov_saturation_margin_px: int = 2,
	):
		self._hsv_ranges = list(hsv_ranges) if hsv_ranges is not None else None

		self._min_area_px = float(min_area_px)
		self._max_area_fraction = float(max_area_fraction)
		self._absolute_max_area_fraction = float(absolute_max_area_fraction)
		self._min_large_area_penalty = float(min_large_area_penalty)

		self._blur_kernel_size = self._make_odd(blur_kernel_size)
		self._morph_kernel_size = self._make_odd(morph_kernel_size)

		self._min_saturation = int(min_saturation)
		self._min_value = int(min_value)

		self._canny_low = int(canny_low)
		self._canny_high = int(canny_high)

		# Temporal hold: lets a brief detection dropout reuse the last
		# known-good TargetEstimate instead of immediately going to
		# found=False. See _held_target_or_lost().
		self._loss_grace_period_sec = float(loss_grace_period_sec)
		self._last_found_target: Optional[TargetEstimate] = None
		self._last_found_time: Optional[float] = None
		self._fov_saturation_margin_px = max(0, int(fov_saturation_margin_px))

	def update(
		self,
		frame_bgr,
		flow_result: Optional[FlowResult] = None,
		timestamp: Optional[float] = None,
	) -> TargetEstimate:
		"""
		Production function used by bee_node.py.

		Returns only TargetEstimate:

			timestamp
			found
			offset_x
			offset_y
			confidence
			detection_width
			detection_height
			area_fraction
		"""
		timestamp = self._resolve_timestamp(flow_result, timestamp)

		if frame_bgr is None:
			return TargetEstimate(timestamp=timestamp, found=False)

		height, width = frame_bgr.shape[:2]
		if width <= 0 or height <= 0:
			return TargetEstimate(timestamp=timestamp, found=False)

		masks = self._build_masks(frame_bgr)
		contour = self._select_best_contour(masks["clean_mask"], width, height)

		return self._resolve_target(contour, width, height, timestamp)

	def process_debug(
		self,
		frame_bgr,
		flow_result: Optional[FlowResult] = None,
		timestamp: Optional[float] = None,
	) -> dict:
		"""
		Debug helper used by target_acquisition_debug.py.

		This function is not used by the controller. It exposes the
		intermediate masks and selected contour for visualization.
		"""
		timestamp = self._resolve_timestamp(flow_result, timestamp)

		if frame_bgr is None:
			return {
				"frame": None,
				"blurred": None,
				"hsv_mask": None,
				"edges": None,
				"combined_mask": None,
				"clean_mask": None,
				"contour": None,
				"target": TargetEstimate(timestamp=timestamp, found=False),
			}

		height, width = frame_bgr.shape[:2]
		if width <= 0 or height <= 0:
			return {
				"frame": frame_bgr,
				"blurred": frame_bgr.copy(),
				"hsv_mask": None,
				"edges": None,
				"combined_mask": None,
				"clean_mask": None,
				"contour": None,
				"target": TargetEstimate(timestamp=timestamp, found=False),
			}

		masks = self._build_masks(frame_bgr)
		contour = self._select_best_contour(masks["clean_mask"], width, height)

		target = self._resolve_target(contour, width, height, timestamp)

		return {
			"frame": frame_bgr,
			"blurred": masks["blurred"],
			"hsv_mask": masks["hsv_mask"],
			"edges": masks["edges"],
			"combined_mask": masks["combined_mask"],
			"clean_mask": masks["clean_mask"],
			"contour": contour,
			"target": target,
		}

	def reset(self):
		"""Clear temporal memory (the held last-known-good target)."""
		self._last_found_target = None
		self._last_found_time = None

	def _resolve_target(
		self,
		contour,
		width: int,
		height: int,
		timestamp: float,
	) -> TargetEstimate:
		"""
		Turn a selected contour (or lack of one) into a TargetEstimate,
		updating/consulting the temporal hold as needed.
		"""
		if contour is None:
			return self._held_target_or_lost(timestamp)

		target = self._target_from_contour(contour, width, height, timestamp)
		self._last_found_target = target
		self._last_found_time = timestamp

		return target

	def _held_target_or_lost(self, timestamp: float) -> TargetEstimate:
		"""
		Bridge brief detection dropouts. Within loss_grace_period_sec of the
		last good detection, reuse it with confidence linearly decayed toward
		zero (decay = 1 - elapsed/grace), so downstream consumers can tell the
		estimate is aging rather than freshly detected. Otherwise report lost.
		"""
		if (
			self._loss_grace_period_sec > 0.0
			and self._last_found_target is not None
			and self._last_found_time is not None
		):
			elapsed = timestamp - self._last_found_time

			if 0.0 <= elapsed <= self._loss_grace_period_sec:
				decay = 1.0 - (elapsed / self._loss_grace_period_sec)

				return replace(
					self._last_found_target,
					timestamp=timestamp,
					confidence=self._last_found_target.confidence * decay,
				)

		return TargetEstimate(timestamp=timestamp, found=False)

	def _build_masks(self, frame_bgr) -> dict:
		blurred = cv2.GaussianBlur(
			frame_bgr,
			(self._blur_kernel_size, self._blur_kernel_size),
			0,
		)

		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		if self._hsv_ranges:
			hsv_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

			for lower, upper in self._hsv_ranges:
				lower_arr = np.array(lower, dtype=np.uint8)
				upper_arr = np.array(upper, dtype=np.uint8)

				hsv_mask = cv2.bitwise_or(
					hsv_mask,
					cv2.inRange(hsv, lower_arr, upper_arr),
				)
		else:
			hsv_mask = cv2.inRange(
				hsv,
				np.array((0, self._min_saturation, self._min_value), dtype=np.uint8),
				np.array((179, 255, 255), dtype=np.uint8),
			)

		gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

		edges = cv2.Canny(gray, self._canny_low, self._canny_high)
		edges = cv2.dilate(
			edges,
			np.ones((3, 3), dtype=np.uint8),
			iterations=1,
		)

		combined_mask = cv2.bitwise_or(hsv_mask, edges)

		kernel = np.ones(
			(self._morph_kernel_size, self._morph_kernel_size),
			dtype=np.uint8,
		)

		clean_mask = cv2.morphologyEx(
			combined_mask,
			cv2.MORPH_OPEN,
			kernel,
			iterations=1,
		)

		clean_mask = cv2.morphologyEx(
			clean_mask,
			cv2.MORPH_CLOSE,
			kernel,
			iterations=2,
		)

		return {
			"blurred": blurred,
			"hsv_mask": hsv_mask,
			"edges": edges,
			"combined_mask": combined_mask,
			"clean_mask": clean_mask,
		}

	def _select_best_contour(self, mask: np.ndarray, width: int, height: int):
		contours, _ = cv2.findContours(
			mask,
			cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE,
		)

		if not contours:
			return None

		image_area = float(width * height)

		best_contour = None
		best_score = -1.0

		for contour in contours:
			area = cv2.contourArea(contour)
			if area < self._min_area_px:
				continue

			area_fraction = area / image_area

			perimeter = cv2.arcLength(contour, closed=True)
			if perimeter <= 1e-6:
				continue

			moments = cv2.moments(contour)
			if abs(moments["m00"]) < 1e-6:
				continue

			cx = moments["m10"] / moments["m00"]
			cy = moments["m01"] / moments["m00"]

			compactness = 4.0 * np.pi * area / (perimeter * perimeter)
			compactness = max(0.0, min(1.0, compactness))

			nx = (cx - 0.5 * width) / (0.5 * width)
			ny = (cy - 0.5 * height) / (0.5 * height)

			center_distance = min(1.0, (nx * nx + ny * ny) ** 0.5)
			center_score = 1.0 - center_distance

			area_score = min(1.0, area_fraction / 0.05)
			large_area_penalty = self._large_area_penalty(area_fraction)

			score = (
				0.60 * area_score
				+ 0.25 * compactness
				+ 0.15 * center_score
			)

			score *= large_area_penalty

			if score > best_score:
				best_score = score
				best_contour = contour

		return best_contour

	def _target_from_contour(
		self,
		contour,
		width: int,
		height: int,
		timestamp: float,
	) -> TargetEstimate:
		moments = cv2.moments(contour)

		if abs(moments["m00"]) < 1e-6:
			return TargetEstimate(timestamp=timestamp, found=False)

		cx = moments["m10"] / moments["m00"]
		cy = moments["m01"] / moments["m00"]

		offset_x = (cx - 0.5 * width) / (0.5 * width)
		offset_y = (cy - 0.5 * height) / (0.5 * height)

		area = cv2.contourArea(contour)
		area_fraction = area / float(width * height)

		_, _, detection_width, detection_height = cv2.boundingRect(contour)

		confidence = self._estimate_confidence(contour, area_fraction)
		fov_saturated = self._is_fov_saturated(contour, width, height)

		return TargetEstimate(
			timestamp=timestamp,
			found=True,
			offset_x=float(offset_x),
			offset_y=float(offset_y),
			confidence=float(confidence),
			detection_width=float(detection_width),
			detection_height=float(detection_height),
			area_fraction=float(area_fraction),
			fov_saturated=fov_saturated,
		)

	def _large_area_penalty(self, area_fraction: float) -> float:
		"""
		Down-weight (never reject) contours covering a large frame fraction; a
		full-frame flower at touchdown is the success condition. The penalty
		ramps linearly 1.0 -> min_large_area_penalty as area_fraction goes from
		max_area_fraction to absolute_max_area_fraction, then holds at that floor.
		"""
		if area_fraction <= self._max_area_fraction:
			return 1.0

		span = max(
			self._absolute_max_area_fraction - self._max_area_fraction,
			1e-6,
		)

		ratio = (area_fraction - self._max_area_fraction) / span
		ratio = max(0.0, min(1.0, ratio))

		return 1.0 - (1.0 - self._min_large_area_penalty) * ratio

	def _is_fov_saturated(self, contour, width: int, height: int) -> bool:
		"""
		True if the contour's bounding box touches all four image borders
		(within fov_saturation_margin_px) -- the true target's projected size
		meets or exceeds the camera's field of view, not just fills it.
		cv2.boundingRect cannot report a box larger than the image array, so
		area_fraction/detection_width/height stop tracking true range past
		this point, regardless of how much closer the target actually gets.
		"""
		x, y, w, h = cv2.boundingRect(contour)
		m = self._fov_saturation_margin_px
		return (
			x <= m
			and y <= m
			and (x + w) >= width - m
			and (y + h) >= height - m
		)

	@staticmethod
	def _estimate_confidence(contour, area_fraction: float) -> float:
		area = cv2.contourArea(contour)
		perimeter = cv2.arcLength(contour, closed=True)

		if perimeter <= 1e-6:
			return 0.0

		compactness = 4.0 * np.pi * area / (perimeter * perimeter)
		compactness = max(0.0, min(1.0, compactness))

		area_score = min(1.0, area_fraction / 0.05)

		confidence = 0.70 * area_score + 0.30 * compactness
		return max(0.0, min(1.0, confidence))

	@staticmethod
	def _resolve_timestamp(
		flow_result: Optional[FlowResult],
		timestamp: Optional[float],
	) -> float:
		if timestamp is not None:
			return float(timestamp)

		if flow_result is not None:
			return float(flow_result.timestamp)

		return 0.0

	@staticmethod
	def _make_odd(value: int) -> int:
		value = max(1, int(value))
		return value if value % 2 == 1 else value + 1


if __name__ == "__main__":
	try:
		from ._target_acquisition_debug import test
	except ImportError:
		from _target_acquisition_debug import test

	test()