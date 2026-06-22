"""
Lightweight target acquisition, decoupled from ROS.

The first implementation intentionally avoids neural networks. It uses
classical image-processing steps that are cheap enough for the online
control loop:

1. HSV color/contrast segmentation.
2. Morphological cleanup.
3. Contour extraction.
4. Centroid + normalized image offset estimation.

The detector can be configured with explicit HSV ranges once the final
landing target color is fixed. Without explicit ranges, it uses a generic
"salient saturated object" mask, which is useful for early Gazebo tests
with a colored pad/marker.
"""

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .state import FlowResult, TargetEstimate


HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


class TargetAcquisition:
	"""
	Image-only target acquisition.

	No VehicleState is accepted here on purpose: target acquisition must not
	depend on PX4 position/velocity feedback. The optional FlowResult is
	kept for later fusion with optical-flow divergence, but this first
	version only uses the image.

	The output offsets are normalized:
		offset_x = (cx - image_center_x) / (image_width  / 2)
		offset_y = (cy - image_center_y) / (image_height / 2)

	With OpenCV image coordinates, offset_y is positive when the target is
	below the image center.
	BGR frame
	↓
	Gaussian blur
	↓
	HSV saliency mask
	↓
	Canny contrast cue
	↓
	morphological cleanup
	↓
	contour selection
	↓
	centroid cx, cy
	↓
	normalized offsets offset_x, offset_y
	"""

	def __init__(
		self,
		hsv_ranges: Optional[Sequence[HSVRange]] = None,
		min_area_px: float = 80.0,
		max_area_fraction: float = 0.60,
		blur_kernel_size: int = 5,
		morph_kernel_size: int = 5,
		min_saturation: int = 60,
		min_value: int = 45,
	):
		self._hsv_ranges = list(hsv_ranges) if hsv_ranges is not None else None
		self._min_area_px = float(min_area_px)
		self._max_area_fraction = float(max_area_fraction)
		self._blur_kernel_size = self._make_odd(blur_kernel_size)
		self._morph_kernel_size = self._make_odd(morph_kernel_size)
		self._min_saturation = int(min_saturation)
		self._min_value = int(min_value)

	def update(self, frame_bgr, flow_result: Optional[FlowResult] = None) -> TargetEstimate:
		"""
		Detect the landing target in the current camera frame.

		Returns found=False when no sufficiently reliable contour is detected.
		"""
		timestamp = flow_result.timestamp if flow_result is not None else 0.0

		if frame_bgr is None:
			return TargetEstimate(timestamp=timestamp, found=False)

		height, width = frame_bgr.shape[:2]
		if width <= 0 or height <= 0:
			return TargetEstimate(timestamp=timestamp, found=False)

		mask = self._build_mask(frame_bgr)
		contour = self._select_best_contour(mask, width, height)
		if contour is None:
			return TargetEstimate(timestamp=timestamp, found=False)

		moments = cv2.moments(contour)
		if abs(moments['m00']) < 1e-6:
			return TargetEstimate(timestamp=timestamp, found=False)

		cx = moments['m10'] / moments['m00']
		cy = moments['m01'] / moments['m00']

		offset_x = (cx - 0.5 * width) / (0.5 * width)
		offset_y = (cy - 0.5 * height) / (0.5 * height)

		area = cv2.contourArea(contour)
		area_fraction = area / float(width * height)
		confidence = self._estimate_confidence(contour, area_fraction)

		return TargetEstimate(
			timestamp=timestamp,
			found=True,
			offset_x=float(offset_x),
			offset_y=float(offset_y),
			confidence=float(confidence),
		)

	def _build_mask(self, frame_bgr) -> np.ndarray:
		blurred = cv2.GaussianBlur(
			frame_bgr,
			(self._blur_kernel_size, self._blur_kernel_size),
			0,
		)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		if self._hsv_ranges:
			mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
			for lower, upper in self._hsv_ranges:
				lower_arr = np.array(lower, dtype=np.uint8)
				upper_arr = np.array(upper, dtype=np.uint8)
				mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_arr, upper_arr))
		else:
			# Generic lightweight detector for early simulation: prefer regions
			# that are visually salient because they are saturated and not too
			# dark. This works well when the target/pad marker has a distinctive
			# color against the platform/background.
			saturation = hsv[:, :, 1]
			value = hsv[:, :, 2]
			mask = cv2.inRange(
				hsv,
				np.array((0, self._min_saturation, self._min_value), dtype=np.uint8),
				np.array((179, 255, 255), dtype=np.uint8),
			)

			# Add a cheap contrast cue so a high-contrast marker can still be
			# detected even if its saturation is moderate.
			gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray, 60, 140)
			edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
			mask = cv2.bitwise_or(mask, edges)

		kernel = np.ones((self._morph_kernel_size, self._morph_kernel_size), dtype=np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
		return mask

	def _select_best_contour(self, mask: np.ndarray, width: int, height: int):
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			return None

		image_area = float(width * height)
		max_area_px = self._max_area_fraction * image_area

		best_contour = None
		best_score = -1.0

		for contour in contours:
			area = cv2.contourArea(contour)
			if area < self._min_area_px or area > max_area_px:
				continue

			perimeter = cv2.arcLength(contour, closed=True)
			if perimeter <= 1e-6:
				continue

			# Compactness is 1.0 for a circle and smaller for long/noisy contours.
			compactness = 4.0 * np.pi * area / (perimeter * perimeter)
			compactness = max(0.0, min(1.0, compactness))

			moments = cv2.moments(contour)
			if abs(moments['m00']) < 1e-6:
				continue
			cx = moments['m10'] / moments['m00']
			cy = moments['m01'] / moments['m00']

			# During acquisition, prefer candidates closer to the center, but do
			# not make this dominant: a real target near the edge should still be
			# selected if it is the best blob.
			nx = (cx - 0.5 * width) / (0.5 * width)
			ny = (cy - 0.5 * height) / (0.5 * height)
			center_score = 1.0 - min(1.0, (nx * nx + ny * ny) ** 0.5)

			area_score = min(1.0, area / (0.05 * image_area))
			score = 0.65 * area_score + 0.25 * compactness + 0.10 * center_score

			if score > best_score:
				best_score = score
				best_contour = contour

		return best_contour

	@staticmethod
	def _estimate_confidence(contour, area_fraction: float) -> float:
		area = cv2.contourArea(contour)
		perimeter = cv2.arcLength(contour, closed=True)
		if perimeter <= 1e-6:
			return 0.0

		compactness = 4.0 * np.pi * area / (perimeter * perimeter)
		compactness = max(0.0, min(1.0, compactness))
		area_score = min(1.0, area_fraction / 0.05)
		return max(0.0, min(1.0, 0.70 * area_score + 0.30 * compactness))

	@staticmethod
	def _make_odd(value: int) -> int:
		value = max(1, int(value))
		return value if value % 2 == 1 else value + 1
