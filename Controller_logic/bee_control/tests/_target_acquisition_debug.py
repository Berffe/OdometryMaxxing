"""
Standalone target-acquisition debug visualization.

This file is intentionally separate from target_acquisition.py so the
production detector stays small and readable.

Shows the SIX stages the detector actually runs. There is deliberately no Canny
panel: Canny is not part of the production candidate mask, and a permanently
black tile implied a pipeline stage that does not exist.

Run through target_acquisition.py, from the repo root:

	python -m bee_control.vision.target_acquisition

or directly:

	python -m bee_control.tests._target_acquisition_debug
"""

from typing import Optional

import cv2
import numpy as np

from bee_control.vision.target_acquisition import TargetAcquisition


def _make_dummy_frame(
	width: int,
	height: int,
	step: int,
	with_target: bool = True,
	with_distractors: bool = True,
	target_radius: Optional[int] = None,
) -> np.ndarray:
	frame = np.zeros((height, width, 3), dtype=np.uint8)
	frame[:, :] = (70, 70, 70)

	margin_x = int(0.10 * width)
	margin_y = int(0.12 * height)

	cv2.rectangle(
		frame,
		(margin_x, margin_y),
		(width - margin_x, height - margin_y),
		(95, 95, 95),
		thickness=-1,
	)

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
		cx = int(0.5 * width + 0.28 * width * np.sin(0.045 * step))
		cy = int(0.5 * height + 0.22 * height * np.cos(0.035 * step))

		if target_radius is None:
			target_radius = int(10 + 3 * step)

		petal_color = (0, 0, 255)
		center_color = (0, 255, 255)

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

	noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
	return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_detection_debug(frame_bgr, target, contour=None):
	if frame_bgr is None:
		return None

	debug_img = frame_bgr.copy()
	height, width = debug_img.shape[:2]

	image_cx = int(0.5 * width)
	image_cy = int(0.5 * height)

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
			(255, 255, 0),
			2,
			tipLength=0.15,
		)

		x0 = int(round(target_cx - 0.5 * target.detection_width))
		y0 = int(round(target_cy - 0.5 * target.detection_height))
		x1 = int(round(target_cx + 0.5 * target.detection_width))
		y1 = int(round(target_cy + 0.5 * target.detection_height))

		cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

		text = (
			f"FOUND  "
			f"ox={target.offset_x:+.3f}  "
			f"oy={target.offset_y:+.3f}  "
			f"conf={target.confidence:.2f}  "
			f"box={target.detection_width:.0f}x{target.detection_height:.0f}"
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


def _to_bgr_for_display(image: Optional[np.ndarray]) -> np.ndarray:
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


def _draw_contour_only(frame_bgr, contour):
	"""The winning contour on a dimmed frame.

	Replaces the old Canny tile. Canny showed an input the detector no longer
	uses; this shows the actual output of contour SELECTION, which is where a
	bad detection is normally traceable to -- a distractor winning the score, or
	the flower fragmenting into several small blobs.
	"""
	if frame_bgr is None:
		return None

	dimmed = (0.35 * frame_bgr.astype(np.float32)).astype(np.uint8)

	if contour is not None:
		cv2.drawContours(dimmed, [contour], -1, (0, 255, 255), 2)
		area = float(cv2.contourArea(contour))
		# y=70 at source resolution, so the text still clears the 28 px label
		# bar after the tile is resized down to 280 px tall.
		cv2.putText(
			dimmed, f"area={area:.0f}px", (14, 70),
			cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
		)
	else:
		cv2.putText(
			dimmed, "no contour selected", (14, 70),
			cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
		)

	return dimmed


def _make_debug_canvas(debug: dict) -> np.ndarray:
	"""Four panels: the production pipeline, and nothing that is not in it.

	The Canny panel that used to sit between the HSV mask and the cleanup was
	removed because Canny is no longer part of the candidate mask. Target
	identity is COLORFULNESS, not generic image structure -- a strong edge with
	no flower-like saturation must not be able to become a competing contour.
	``_build_masks`` still returns an ``edges`` entry, but it is an all-zero
	array, so the panel was a permanently black tile implying a pipeline stage
	that does not exist.
	"""
	frame = debug["frame"]
	blurred = debug["blurred"]
	hsv_mask = debug["hsv_mask"]
	clean_mask = debug["clean_mask"]
	contour = debug["contour"]
	target = debug["target"]

	final_debug = _draw_detection_debug(frame, target, contour)

	tiles = [
		("1 BGR frame", _to_bgr_for_display(frame)),
		("2 Gaussian blur", _to_bgr_for_display(blurred)),
		("3 HSV colorfulness mask", _to_bgr_for_display(hsv_mask)),
		("4 Morphological cleanup", _to_bgr_for_display(clean_mask)),
		("5 Selected contour", _to_bgr_for_display(_draw_contour_only(frame, contour))),
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
	detector = TargetAcquisition(
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
		target_radius = 190 if close_range_mode else None

		frame = _make_dummy_frame(
			width=width,
			height=height,
			step=step,
			with_target=with_target,
			with_distractors=with_distractors,
			target_radius=target_radius,
		)

		debug = detector.process_debug(frame)
		canvas = _make_debug_canvas(debug)

		mode_text = (
			f"target={'ON' if with_target else 'OFF'} | "
			f"distractors={'ON' if with_distractors else 'OFF'} | "
			f"close_range={'ON' if close_range_mode else 'OFF'}"
		)

		footer = np.zeros((30, canvas.shape[1], 3), dtype=np.uint8)
		cv2.putText(
			footer,
			f"{mode_text} | n: target | d: distractors | c: close range | "
			f"s: save | q/Esc: quit",
			(16, 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.52,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)
		canvas = np.vstack([canvas, footer])

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