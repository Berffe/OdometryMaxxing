"""
Standalone optical-flow debug visualization.

This file is intentionally separate from optical_flow.py so the production
estimator stays small and readable.

Run through optical_flow.py:

	python -m bee_control.optical_flow

or directly:

	python optical_flow_debug.py
"""

from typing import Optional

import cv2
import numpy as np

try:
	from .optical_flow import OpticalFlowEstimator
	from .state import TargetEstimate
except ImportError:
	from optical_flow import OpticalFlowEstimator
	from state import TargetEstimate


def _flow_to_color(flow_px_s: np.ndarray) -> np.ndarray:
	u = flow_px_s[:, :, 0]
	v = flow_px_s[:, :, 1]

	magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)

	hsv = np.zeros((flow_px_s.shape[0], flow_px_s.shape[1], 3), dtype=np.uint8)
	hsv[:, :, 0] = (angle / 2).astype(np.uint8)
	hsv[:, :, 1] = 255

	mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
	hsv[:, :, 2] = mag_norm.astype(np.uint8)

	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _draw_flow_arrows(
	frame_bgr: np.ndarray,
	flow_px_s: np.ndarray,
	roi=None,
	grid_step: int = 32,
	max_arrow_length: float = 18.0,
	min_arrow_length: float = 3.0,
	min_flow_magnitude: float = 1e-3,
) -> np.ndarray:
	vis = frame_bgr.copy()

	if roi is None:
		x_offset = 0
		y_offset = 0
	else:
		x_offset, y_offset, x1, y1 = roi
		cv2.rectangle(vis, (x_offset, y_offset), (x1, y1), (0, 255, 0), 2)

	roi_height, roi_width = flow_px_s.shape[:2]

	u = flow_px_s[:, :, 0]
	v = flow_px_s[:, :, 1]
	magnitude = np.sqrt(u * u + v * v)

	mag_ref = float(np.percentile(magnitude, 95))
	mag_ref = max(mag_ref, min_flow_magnitude)

	for y in range(grid_step // 2, roi_height, grid_step):
		for x in range(grid_step // 2, roi_width, grid_step):
			u_xy = float(flow_px_s[y, x, 0])
			v_xy = float(flow_px_s[y, x, 1])

			mag = (u_xy * u_xy + v_xy * v_xy) ** 0.5
			if mag < min_flow_magnitude:
				continue

			dir_x = u_xy / mag
			dir_y = v_xy / mag

			length = max_arrow_length * min(1.0, mag / mag_ref)
			length = max(min_arrow_length, length)

			x_start = int(x_offset + x)
			y_start = int(y_offset + y)
			x_end = int(round(x_start + length * dir_x))
			y_end = int(round(y_start + length * dir_y))

			cv2.circle(vis, (x_start, y_start), 2, (0, 0, 0), -1)

			cv2.arrowedLine(
				vis,
				(x_start, y_start),
				(x_end, y_end),
				(0, 0, 0),
				3,
				tipLength=0.35,
			)

			cv2.arrowedLine(
				vis,
				(x_start, y_start),
				(x_end, y_end),
				(0, 255, 255),
				1,
				tipLength=0.35,
			)

	return vis


def _divergence_to_heatmap(divergence_field: np.ndarray) -> np.ndarray:
	if divergence_field is None or divergence_field.size == 0:
		return None

	abs_max = float(np.percentile(np.abs(divergence_field), 98))
	abs_max = max(abs_max, 1e-6)

	normalized = np.clip(
		0.5 + 0.5 * divergence_field / abs_max,
		0.0,
		1.0,
	)

	image_u8 = (255.0 * normalized).astype(np.uint8)
	return cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)


def _make_base_scene(width: int, height: int) -> np.ndarray:
	scene = np.zeros((height, width, 3), dtype=np.uint8)
	scene[:, :] = (65, 65, 65)

	cv2.rectangle(
		scene,
		(int(0.08 * width), int(0.10 * height)),
		(int(0.92 * width), int(0.90 * height)),
		(95, 95, 95),
		-1,
	)

	for x in range(0, width, 45):
		cv2.line(scene, (x, 0), (x, height), (80, 80, 80), 1)

	for y in range(0, height, 45):
		cv2.line(scene, (0, y), (width, y), (80, 80, 80), 1)

	rng = np.random.default_rng(4)
	for _ in range(180):
		x = int(rng.integers(20, width - 20))
		y = int(rng.integers(20, height - 20))
		r = int(rng.integers(2, 6))
		color = int(rng.integers(90, 220))
		cv2.circle(scene, (x, y), r, (color, color, color), -1)

	cx = int(0.5 * width)
	cy = int(0.5 * height)
	radius = 50

	petal_color = (0, 0, 255)
	center_color = (0, 255, 255)

	for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
		px = int(cx + 0.65 * radius * np.cos(angle))
		py = int(cy + 0.65 * radius * np.sin(angle))
		cv2.circle(scene, (px, py), int(0.38 * radius), petal_color, -1)

	cv2.circle(scene, (cx, cy), radius, petal_color, 3)
	cv2.circle(scene, (cx, cy), int(0.35 * radius), center_color, -1)

	cv2.line(scene, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 3)
	cv2.line(scene, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 3)

	return scene


def _make_synthetic_frame(
	base_scene: np.ndarray,
	step: int,
	mode: str,
) -> np.ndarray:
	height, width = base_scene.shape[:2]
	center = (0.5 * width, 0.5 * height)

	if mode == "translate":
		scale = 1.0
		dx = 70.0 * np.sin(0.035 * step)
		dy = 45.0 * np.cos(0.030 * step)

	elif mode == "zoom":
		scale = 1.0 + 0.25 * np.sin(0.035 * step)
		dx = 0.0
		dy = 0.0

	else:
		scale = 1.0 + 0.18 * np.sin(0.030 * step)
		dx = 45.0 * np.sin(0.025 * step)
		dy = 30.0 * np.cos(0.020 * step)

	matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
	matrix[0, 2] += dx
	matrix[1, 2] += dy

	frame = cv2.warpAffine(
		base_scene,
		matrix,
		(width, height),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REFLECT,
	)

	rng = np.random.default_rng(step)
	noise = rng.normal(0, 2.5, frame.shape).astype(np.int16)

	return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _make_synthetic_target(width: int, height: int, step: int) -> TargetEstimate:
	"""
	Create a synthetic TargetEstimate matching the center flower region.
	This lets the optical-flow test exercise the ROI-based divergence logic.
	"""
	cx = int(0.5 * width)
	cy = int(0.5 * height)

	box_w = 160.0
	box_h = 160.0

	offset_x = (cx - 0.5 * width) / (0.5 * width)
	offset_y = (cy - 0.5 * height) / (0.5 * height)

	return TargetEstimate(
		timestamp=step / 30.0,
		found=True,
		offset_x=float(offset_x),
		offset_y=float(offset_y),
		confidence=1.0,
		detection_width=box_w,
		detection_height=box_h,
	)


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


def _make_debug_canvas(debug: dict, mode: str) -> np.ndarray:
	result = debug.get("result")

	current_frame = debug.get("current_frame")
	previous_frame = debug.get("previous_frame")
	flow_px_s = debug.get("flow_px_s")
	divergence_field = debug.get("divergence_field")
	roi = debug.get("roi")

	if current_frame is not None and flow_px_s is not None:
		flow_arrows = _draw_flow_arrows(current_frame, flow_px_s, roi=roi)
		flow_color = _flow_to_color(flow_px_s)
	else:
		flow_arrows = current_frame.copy() if current_frame is not None else None
		flow_color = None

	if divergence_field is not None:
		divergence_heatmap = _divergence_to_heatmap(divergence_field)
	else:
		divergence_heatmap = None

	tiles = [
		("1 Previous frame", _to_bgr_for_display(previous_frame)),
		("2 Current frame", _to_bgr_for_display(current_frame)),
		("3 ROI flow arrows", _to_bgr_for_display(flow_arrows)),
		("4 Local flow color", _to_bgr_for_display(flow_color)),
		("5 Local divergence heatmap", _to_bgr_for_display(divergence_heatmap)),
	]

	tile_width = 420
	tile_height = 280

	resized_tiles = [
		_label_tile(_resize_for_tile(image, tile_width, tile_height), label)
		for label, image in tiles
	]

	status = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)

	if result is None:
		lines = ["No FlowResult available"]
	else:
		lines = [
			f"mode: {mode}",
			f"valid: {result.valid}",
			f"roi: {roi}",
			f"mean_flow_x: {result.mean_flow_x:+.2f} px/s",
			f"mean_flow_y: {result.mean_flow_y:+.2f} px/s",
			f"raw divergence: {debug.get('raw_divergence', 0.0):+.4f} 1/s",
			f"filtered divergence: {result.divergence:+.4f} 1/s",
			debug.get("message", ""),
		]

	y = 40
	for line in lines:
		cv2.putText(
			status,
			line,
			(20, y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.65,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)
		y += 34

	status = _label_tile(status, "6 Scalar outputs")

	row_1 = np.hstack(resized_tiles[:3])
	row_2 = np.hstack([resized_tiles[3], resized_tiles[4], status])

	return np.vstack([row_1, row_2])


def test():
	width = 640
	height = 480
	dt = 1.0 / 30.0

	base_scene = _make_base_scene(width, height)

	estimator = OpticalFlowEstimator(
		require_target_roi=True,
		roi_margin_fraction=0.25,
		min_roi_size_px=32,
		divergence_smoothing=0.6,
		store_debug=True,
	)

	modes = ["translate", "zoom", "combined"]
	mode_index = 1

	step = 0
	last_canvas = None

	window_name = "Optical flow test"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	print("")
	print("Optical flow test")
	print("-----------------")
	print("q or Esc  -> quit")
	print("m         -> change synthetic motion mode")
	print("r         -> reset estimator memory")
	print("s         -> save current preview image")
	print("")

	while True:
		mode = modes[mode_index]

		frame = _make_synthetic_frame(
			base_scene=base_scene,
			step=step,
			mode=mode,
		)

		timestamp = step * dt
		target = _make_synthetic_target(width, height, step)

		estimator.update(frame, timestamp, target=target)
		debug = estimator.last_debug_data()

		canvas = _make_debug_canvas(debug, mode)

		cv2.imshow(window_name, canvas)
		last_canvas = canvas

		key = cv2.waitKey(40) & 0xFF

		if key in (ord("q"), 27):
			break

		if key == ord("m"):
			mode_index = (mode_index + 1) % len(modes)
			estimator.reset()
			step = 0
			continue

		if key == ord("r"):
			estimator.reset()
			step = 0
			continue

		if key == ord("s") and last_canvas is not None:
			cv2.imwrite("optical_flow_test_preview.png", last_canvas)
			print("Saved: optical_flow_test_preview.png")

		step += 1

	cv2.destroyAllWindows()


if __name__ == "__main__":
	test()
