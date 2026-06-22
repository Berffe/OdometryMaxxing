"""
Lightweight target acquisition, decoupled from ROS.

The first implementation intentionally avoids neural networks. It uses
classical image-processing steps that are cheap enough for the online
control loop:

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

The detector can be configured with explicit HSV ranges once the final
landing target color is fixed. Without explicit ranges, it uses a generic
"salient saturated object" mask, which is useful for early Gazebo tests
with a colored pad/marker.
"""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

try:
	# Normal package import, used when running inside the ROS2 package:
	#   python -m bee_control.target_acquisition
	from .state import FlowResult, TargetEstimate
except ImportError:
	# Direct-script fallback, useful when running from the same folder:
	#   python target_acquisition.py
	from state import FlowResult, TargetEstimate


HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


class TargetAcquisition:
	"""
	Image-only target acquisition.

	No VehicleState is accepted here on purpose: target acquisition must
	not depend on PX4 position/velocity feedback.

	The optional FlowResult is kept for later fusion with optical-flow
	divergence, but this first version only uses the image.

	The output offsets are normalized:

		offset_x = (cx - image_center_x) / (image_width  / 2)
		offset_y = (cy - image_center_y) / (image_height / 2)

	With OpenCV image coordinates, offset_y is positive when the target
	is below the image center.
	"""

	def __init__(
		self,
		hsv_ranges: Optional[Sequence[HSVRange]] = None,
		min_area_px: float = 80.0,
		max_area_fraction: float = 0.60,
		absolute_max_area_fraction: float = 0.95,
		blur_kernel_size: int = 5,
		morph_kernel_size: int = 5,
		min_saturation: int = 60,
		min_value: int = 45,
		canny_low: int = 60,
		canny_high: int = 140,
	):
		self._hsv_ranges = list(hsv_ranges) if hsv_ranges is not None else None

		self._min_area_px = float(min_area_px)

		# Soft upper area threshold:
		# above max_area_fraction, large candidates are penalized but not
		# immediately rejected. This avoids losing the landing target when
		# the drone gets close and the target fills a large part of the image.
		self._max_area_fraction = float(max_area_fraction)

		# Hard upper area threshold:
		# above this, the blob is almost certainly the whole image/background
		# or a segmentation failure.
		self._absolute_max_area_fraction = float(absolute_max_area_fraction)

		self._blur_kernel_size = self._make_odd(blur_kernel_size)
		self._morph_kernel_size = self._make_odd(morph_kernel_size)

		self._min_saturation = int(min_saturation)
		self._min_value = int(min_value)

		self._canny_low = int(canny_low)
		self._canny_high = int(canny_high)

	def update(self, frame_bgr, flow_result: Optional[FlowResult] = None) -> TargetEstimate:
		"""
		Production function used by bee_node.py.

		This must always return the original TargetEstimate structure:
			timestamp
			found
			offset_x
			offset_y
			confidence

		No debug data is returned here.
		"""
		timestamp = flow_result.timestamp if flow_result is not None else 0.0

		if frame_bgr is None:
			return TargetEstimate(timestamp=timestamp, found=False)

		height, width = frame_bgr.shape[:2]
		if width <= 0 or height <= 0:
			return TargetEstimate(timestamp=timestamp, found=False)

		masks = self._build_masks(frame_bgr)
		clean_mask = masks["clean_mask"]

		contour = self._select_best_contour(clean_mask, width, height)

		if contour is None:
			return TargetEstimate(timestamp=timestamp, found=False)

		return self._target_from_contour(
			contour=contour,
			width=width,
			height=height,
			timestamp=timestamp,
		)

	def process_debug(self, frame_bgr, flow_result: Optional[FlowResult] = None) -> dict:
		"""
		Debug-only function.

		Returns intermediate images and selected contour for visualization.
		This function is not used by the controller.
		"""
		timestamp = flow_result.timestamp if flow_result is not None else 0.0

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
		clean_mask = masks["clean_mask"]

		contour = self._select_best_contour(clean_mask, width, height)

		if contour is None:
			target = TargetEstimate(timestamp=timestamp, found=False)
		else:
			target = self._target_from_contour(
				contour=contour,
				width=width,
				height=height,
				timestamp=timestamp,
			)

		return {
			"frame": frame_bgr,
			"blurred": masks["blurred"],
			"hsv_mask": masks["hsv_mask"],
			"edges": masks["edges"],
			"combined_mask": masks["combined_mask"],
			"clean_mask": clean_mask,
			"contour": contour,
			"target": target,
		}

	def draw_debug(self, frame_bgr, target: TargetEstimate, contour=None):
		"""
		Draw selected contour, image center, target centroid and normalized
		offsets on top of the image.
		"""
		if frame_bgr is None:
			return None

		debug_img = frame_bgr.copy()
		height, width = debug_img.shape[:2]

		image_cx = int(0.5 * width)
		image_cy = int(0.5 * height)

		# Image center.
		cv2.drawMarker(
			debug_img,
			(image_cx, image_cy),
			(255, 255, 255),
			markerType=cv2.MARKER_CROSS,
			markerSize=18,
			thickness=2,
		)

		if contour is not None:
			cv2.drawContours(debug_img, [contour], -1, (0, 255, 255), 2)

		if target.found:
			target_cx = int((target.offset_x * 0.5 + 0.5) * width)
			target_cy = int((target.offset_y * 0.5 + 0.5) * height)

			cv2.circle(debug_img, (target_cx, target_cy), 6, (0, 0, 255), -1)
			cv2.arrowedLine(
				debug_img,
				(image_cx, image_cy),
				(target_cx, target_cy),
				(0, 0, 255),
				2,
				tipLength=0.15,
			)

			text = (
				f"FOUND  "
				f"ox={target.offset_x:+.3f}  "
				f"oy={target.offset_y:+.3f}  "
				f"conf={target.confidence:.2f}"
			)
			color = (0, 255, 0)
		else:
			text = "NO TARGET"
			color = (0, 0, 255)

		cv2.putText(
			debug_img,
			text,
			(20, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.75,
			color,
			2,
			cv2.LINE_AA,
		)

		return debug_img

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
			# Generic lightweight detector for early simulation:
			# prefer regions that are visually salient because they are
			# saturated and not too dark.
			hsv_mask = cv2.inRange(
				hsv,
				np.array(
					(0, self._min_saturation, self._min_value),
					dtype=np.uint8,
				),
				np.array((179, 255, 255), dtype=np.uint8),
			)

		# Add a cheap contrast cue so a high-contrast marker can still be
		# detected even if its saturation is moderate.
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

			# Hard rejection only for almost-full-frame blobs.
			# This avoids selecting the entire background when segmentation
			# fails, but still allows a true target to become very large
			# close to touchdown.
			if area_fraction > self._absolute_max_area_fraction:
				continue

			perimeter = cv2.arcLength(contour, closed=True)
			if perimeter <= 1e-6:
				continue

			moments = cv2.moments(contour)
			if abs(moments["m00"]) < 1e-6:
				continue

			cx = moments["m10"] / moments["m00"]
			cy = moments["m01"] / moments["m00"]

			# Compactness is 1.0 for a circle and smaller for long/noisy
			# contours.
			compactness = 4.0 * np.pi * area / (perimeter * perimeter)
			compactness = max(0.0, min(1.0, compactness))

			# During acquisition, prefer candidates closer to the center,
			# but do not make this dominant: a real target near the edge
			# should still be selected if it is the best blob.
			nx = (cx - 0.5 * width) / (0.5 * width)
			ny = (cy - 0.5 * height) / (0.5 * height)
			center_distance = min(1.0, (nx * nx + ny * ny) ** 0.5)
			center_score = 1.0 - center_distance

			# Area score grows until around 5% of image area. Past that,
			# area alone should not dominate the selection.
			area_score = min(1.0, area_fraction / 0.05)

			# Large-area mitigation:
			# Above max_area_fraction, we reduce the score progressively
			# instead of discarding the contour immediately.
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

		confidence = self._estimate_confidence(contour, area_fraction)

		return TargetEstimate(
			timestamp=timestamp,
			found=True,
			offset_x=float(offset_x),
			offset_y=float(offset_y),
			confidence=float(confidence),
		)

	def _large_area_penalty(self, area_fraction: float) -> float:
		"""
		Penalize very large blobs softly.

		This is important for landing: when the drone gets close to the
		target, the correct target can occupy a large part of the image.
		We should not immediately reject it just because it is large.

		Below max_area_fraction:
			penalty = 1.0

		Between max_area_fraction and absolute_max_area_fraction:
			penalty decreases smoothly, but never reaches zero.

		Above absolute_max_area_fraction:
			the contour is already rejected before this function is used.
		"""
		if area_fraction <= self._max_area_fraction:
			return 1.0

		if self._absolute_max_area_fraction <= self._max_area_fraction:
			return 0.4

		ratio = (
			(area_fraction - self._max_area_fraction)
			/ (self._absolute_max_area_fraction - self._max_area_fraction)
		)

		ratio = max(0.0, min(1.0, ratio))

		# Keep a nonzero score for close-range large targets.
		return 1.0 - 0.6 * ratio

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
	def _make_odd(value: int) -> int:
		value = max(1, int(value))
		return value if value % 2 == 1 else value + 1


"""
FOR TESTING ONLY
"""
def _make_dummy_frame(
	width: int,
	height: int,
	step: int,
	with_target: bool = True,
	with_distractors: bool = True,
	target_radius: Optional[int] = None,
) -> np.ndarray:
	"""
	Create a synthetic test frame.

	The frame imitates a simple camera view with:
	- gray platform/background,
	- optional colored target,
	- optional distractor objects,
	- mild noise.

	This lets us test target acquisition without Gazebo or ROS.
	"""
	frame = np.zeros((height, width, 3), dtype=np.uint8)
	frame[:, :] = (70, 70, 70)

	# Add a slightly lighter "platform" rectangle.
	margin_x = int(0.10 * width)
	margin_y = int(0.12 * height)
	cv2.rectangle(
		frame,
		(margin_x, margin_y),
		(width - margin_x, height - margin_y),
		(95, 95, 95),
		thickness=-1,
	)

	# Add platform texture/grid lines.
	for x in range(margin_x, width - margin_x, 60):
		cv2.line(frame, (x, margin_y), (x, height - margin_y), (80, 80, 80), 1)

	for y in range(margin_y, height - margin_y, 60):
		cv2.line(frame, (margin_x, y), (width - margin_x, y), (80, 80, 80), 1)

	if with_distractors:
		cv2.circle(frame, (int(0.20 * width), int(0.25 * height)), 22, (255, 80, 30), -1)
		cv2.rectangle(
			frame,
			(int(0.70 * width), int(0.20 * height)),
			(int(0.82 * width), int(0.32 * height)),
			(30, 180, 255),
			-1,
		)
		cv2.line(
			frame,
			(int(0.15 * width), int(0.80 * height)),
			(int(0.85 * width), int(0.78 * height)),
			(210, 210, 210),
			4,
		)

	if with_target:
		# Moving target center.
		cx = int(0.5 * width + 0.28 * width * np.sin(0.045 * step))
		cy = int(0.5 * height + 0.22 * height * np.cos(0.035 * step))

		if target_radius is None:
			# Radius changes with time to imitate approach/retreat.
			target_radius = int(34 + 0.25 * step)

		# "Flower-like" target: colored petals + bright center.
		petal_color = (0, 0, 255)       # red in BGR
		center_color = (0, 255, 255)    # yellow in BGR

		for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
			px = int(cx + 0.58 * target_radius * np.cos(angle))
			py = int(cy + 0.58 * target_radius * np.sin(angle))
			cv2.circle(
				frame,
				(px, py),
				max(5, int(0.38 * target_radius)),
				petal_color,
				-1,
			)

		cv2.circle(frame, (cx, cy), target_radius, petal_color, 3)
		cv2.circle(frame, (cx, cy), max(5, int(0.35 * target_radius)), center_color, -1)

		# Add a high-contrast landing symbol.
		cv2.line(
			frame,
			(cx - int(0.45 * target_radius), cy),
			(cx + int(0.45 * target_radius), cy),
			(255, 255, 255),
			3,
		)
		cv2.line(
			frame,
			(cx, cy - int(0.45 * target_radius)),
			(cx, cy + int(0.45 * target_radius)),
			(255, 255, 255),
			3,
		)

	# Mild image noise.
	noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
	noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

	return noisy


def _to_bgr_for_display(image: Optional[np.ndarray]) -> np.ndarray:
	"""
	Convert a grayscale mask to BGR for side-by-side visualization.
	"""
	if image is None:
		return np.zeros((240, 320, 3), dtype=np.uint8)

	if len(image.shape) == 2:
		return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	return image.copy()


def _resize_for_tile(image: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
	return cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)


def _label_tile(image: np.ndarray, label: str) -> np.ndarray:
	labeled = image.copy()
	cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 28), (0, 0, 0), -1)
	cv2.putText(
		labeled,
		label,
		(8, 20),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		(255, 255, 255),
		1,
		cv2.LINE_AA,
	)
	return labeled


def _make_debug_canvas(debug: dict, detector: TargetAcquisition) -> np.ndarray:
	"""
	Build a 2x3 visualization canvas:

	1. Original BGR frame
	2. Gaussian blur
	3. HSV saliency mask
	4. Canny edges
	5. Morphological cleanup
	6. Final contour + centroid result
	"""
	frame = debug["frame"]
	blurred = debug["blurred"]
	hsv_mask = debug["hsv_mask"]
	edges = debug["edges"]
	clean_mask = debug["clean_mask"]
	contour = debug["contour"]
	target = debug["target"]

	final_debug = detector.draw_debug(frame, target, contour)

	tiles = [
		("1 BGR frame", _to_bgr_for_display(frame)),
		("2 Gaussian blur", _to_bgr_for_display(blurred)),
		("3 HSV saliency mask", _to_bgr_for_display(hsv_mask)),
		("4 Canny contrast cue", _to_bgr_for_display(edges)),
		("5 Morphological cleanup", _to_bgr_for_display(clean_mask)),
		("6 Detection result", _to_bgr_for_display(final_debug)),
	]

	tile_width = 420
	tile_height = 280

	resized_tiles = [
		_label_tile(_resize_for_tile(image, tile_width, tile_height), label)
		for label, image in tiles
	]

	row_1 = np.hstack(resized_tiles[:3])
	row_2 = np.hstack(resized_tiles[3:])

	return np.vstack([row_1, row_2])


def test():
	"""
	Standalone visual test for the target-acquisition pipeline.

	Run from the ROS2 package:

		python -m bee_control.target_acquisition

	Or directly from the folder containing target_acquisition.py and state.py:

		python target_acquisition.py

	Keyboard controls:

		q or Esc  -> quit
		n         -> toggle target on/off
		d         -> toggle distractors on/off
		c         -> toggle close-range large target mode
		s         -> save current preview image
	"""
	detector = TargetAcquisition(
		# For early generic tests, keep hsv_ranges=None.
		# Later, when the target color is fixed, define explicit HSV ranges.
		hsv_ranges=None,
		min_area_px=120.0,
		max_area_fraction=0.60,
		absolute_max_area_fraction=0.95,
		blur_kernel_size=5,
		morph_kernel_size=5,
		min_saturation=60,
		min_value=45,
	)

	width = 640
	height = 480

	with_target = True
	with_distractors = True
	close_range_mode = False

	step = 0

	window_name = "Target acquisition test"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	print("")
	print("Target acquisition test")
	print("-----------------------")
	print("q or Esc  -> quit")
	print("n         -> toggle target on/off")
	print("d         -> toggle distractors on/off")
	print("c         -> toggle close-range large target mode")
	print("s         -> save current preview image")
	print("")

	last_canvas = None

	while True:
		if close_range_mode:
			# Large target to test the near-touchdown case.
			target_radius = 190
		else:
			target_radius = None

		frame = _make_dummy_frame(
			width=width,
			height=height,
			step=step,
			with_target=with_target,
			with_distractors=with_distractors,
			target_radius=target_radius,
		)

		debug = detector.process_debug(frame)
		canvas = _make_debug_canvas(debug, detector)

		mode_text = (
			f"target={'ON' if with_target else 'OFF'} | "
			f"distractors={'ON' if with_distractors else 'OFF'} | "
			f"close_range={'ON' if close_range_mode else 'OFF'}"
		)

		cv2.putText(
			canvas,
			mode_text,
			(20, canvas.shape[0] - 18),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.65,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)

		cv2.imshow(window_name, canvas)
		last_canvas = canvas

		key = cv2.waitKey(40) & 0xFF

		if key in (ord("q"), 27):
			break

		if key == ord("n"):
			with_target = not with_target

		if key == ord("d"):
			with_distractors = not with_distractors

		if key == ord("c"):
			close_range_mode = not close_range_mode

		if key == ord("s") and last_canvas is not None:
			cv2.imwrite("target_acquisition_test_preview.png", last_canvas)
			print("Saved: target_acquisition_test_preview.png")

		step += 1

	cv2.destroyAllWindows()


if __name__ == "__main__":
	test()