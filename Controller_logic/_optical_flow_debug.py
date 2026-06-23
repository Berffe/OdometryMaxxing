"""
Standalone optical-flow debug visualization.

This file compares two divergence-estimation methods:

	1. Farneback ROI divergence:
		Dense optical flow is computed inside the target ROI, then
		divergence is estimated from the local flow field.

	2. Box-size divergence:
		Apparent target growth is estimated directly from the target
		box width and height.

This debug file does not change the controller. It only helps compare
candidate optical-flow / divergence strategies.

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


class BoxSizeDivergenceEstimator:
	"""
	Debug-only apparent-size divergence estimator.

	This estimates divergence from apparent growth of the detected target box:

		D_width  = (w_k - w_{k-1}) / (w_{k-1} * dt)
		D_height = (h_k - h_{k-1}) / (h_{k-1} * dt)

		D_box = 0.5 * (D_width + D_height)

	Positive D_box means the target appears larger in the image, which
	corresponds to approaching the target.

	This is not dense optical flow. It is a visual expansion estimator.
	"""

	def __init__(
		self,
		smoothing: float = 0.6,
		min_box_size_px: float = 3.0,
	):
		self._smoothing = float(smoothing)
		self._min_box_size_px = float(min_box_size_px)

		self._prev_width = None
		self._prev_height = None
		self._prev_timestamp = None

		self._filtered_divergence = 0.0
		self._has_filtered_divergence = False

	def reset(self):
		self._prev_width = None
		self._prev_height = None
		self._prev_timestamp = None

		self._filtered_divergence = 0.0
		self._has_filtered_divergence = False

	def update(self, target: TargetEstimate) -> dict:
		if target is None or not target.found:
			self.reset()

			return {
				"valid": False,
				"raw_divergence": 0.0,
				"filtered_divergence": 0.0,
				"width_divergence": 0.0,
				"height_divergence": 0.0,
				"width": 0.0,
				"height": 0.0,
				"message": "No valid target",
			}

		width = float(target.detection_width)
		height = float(target.detection_height)
		timestamp = float(target.timestamp)

		if width <= self._min_box_size_px or height <= self._min_box_size_px:
			self.reset()

			return {
				"valid": False,
				"raw_divergence": 0.0,
				"filtered_divergence": 0.0,
				"width_divergence": 0.0,
				"height_divergence": 0.0,
				"width": width,
				"height": height,
				"message": "Target box too small",
			}

		if (
			self._prev_width is None
			or self._prev_height is None
			or self._prev_timestamp is None
		):
			self._prev_width = width
			self._prev_height = height
			self._prev_timestamp = timestamp

			return {
				"valid": False,
				"raw_divergence": 0.0,
				"filtered_divergence": 0.0,
				"width_divergence": 0.0,
				"height_divergence": 0.0,
				"width": width,
				"height": height,
				"message": "Waiting for previous target box",
			}

		dt = timestamp - self._prev_timestamp

		if dt <= 1e-6:
			self._prev_width = width
			self._prev_height = height
			self._prev_timestamp = timestamp

			return {
				"valid": False,
				"raw_divergence": 0.0,
				"filtered_divergence": 0.0,
				"width_divergence": 0.0,
				"height_divergence": 0.0,
				"width": width,
				"height": height,
				"message": "Invalid dt",
			}

		width_divergence = (width - self._prev_width) / (self._prev_width * dt)
		height_divergence = (height - self._prev_height) / (self._prev_height * dt)

		raw_divergence = 0.5 * (width_divergence + height_divergence)
		filtered_divergence = self._filter(raw_divergence)

		self._prev_width = width
		self._prev_height = height
		self._prev_timestamp = timestamp

		return {
			"valid": True,
			"raw_divergence": float(raw_divergence),
			"filtered_divergence": float(filtered_divergence),
			"width_divergence": float(width_divergence),
			"height_divergence": float(height_divergence),
			"width": width,
			"height": height,
			"message": "",
		}

	def _filter(self, value: float) -> float:
		alpha = max(0.0, min(1.0, self._smoothing))

		if not self._has_filtered_divergence:
			self._filtered_divergence = float(value)
			self._has_filtered_divergence = True
		else:
			self._filtered_divergence = (
				alpha * self._filtered_divergence
				+ (1.0 - alpha) * float(value)
			)

		return self._filtered_divergence


def _motion_parameters(step: int, mode: str):
	"""
	Return the synthetic camera/target motion.

	The same scale and translation are used both to warp the synthetic
	image and to generate the synthetic TargetEstimate. This makes the
	box-size divergence meaningful.
	"""
	if mode == "translate":
		scale = 1.0
		dx = 70.0 * np.sin(0.035 * step)
		dy = 45.0 * np.cos(0.030 * step)

	elif mode == "zoom":
		scale = 1.0 + 0.25 * np.sin(0.09 * step)
		dx = 0.0
		dy = 0.0

	else:
		scale = 1.0 + 0.18 * np.sin(0.030 * step)
		dx = 45.0 * np.sin(0.025 * step)
		dy = 30.0 * np.cos(0.020 * step)

	return float(scale), float(dx), float(dy)


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
	"""
	Draw sparse arrows on top of the current frame.

	If roi is provided, flow_px_s is assumed to be local to the ROI.
	"""
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


def _draw_textured_flower_target(
	scene: np.ndarray,
	cx: int,
	cy: int,
	radius: int,
):
	"""
	Draw a flower-like landing target with internal texture.

	The goal is to keep roughly the same colors as before, but add enough
	local gradients and contrast so dense optical flow has more visual
	structure to track.
	"""
	height, width = scene.shape[:2]

	petal_base = np.array([0, 0, 220], dtype=np.uint8)      # dark red
	petal_alt  = np.array([20, 20, 255], dtype=np.uint8)    # lighter red
	center_base = np.array([0, 210, 210], dtype=np.uint8)   # yellow-ish
	center_alt  = np.array([40, 255, 255], dtype=np.uint8)  # lighter yellow

	# ------------------------------------------------------------
	# 1. Petals
	# ------------------------------------------------------------
	petal_centers = []
	for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
		px = int(cx + 0.65 * radius * np.cos(angle))
		py = int(cy + 0.65 * radius * np.sin(angle))
		petal_centers.append((px, py))

		petal_r = int(0.38 * radius)

		# Base filled petal
		cv2.circle(scene, (px, py), petal_r, tuple(int(v) for v in petal_base), -1)

		# Add inner stripes / rings to create texture
		for k in range(1, 5):
			rk = max(2, int(petal_r * k / 5))
			color = tuple(int(v) for v in (petal_alt if k % 2 == 0 else petal_base))
			cv2.circle(scene, (px, py), rk, color, 1)

		# Radial lines on each petal
		for beta in np.linspace(-0.8, 0.8, 5):
			x2 = int(px + 0.85 * petal_r * np.cos(angle + beta * 0.25))
			y2 = int(py + 0.85 * petal_r * np.sin(angle + beta * 0.25))
			cv2.line(scene, (px, py), (x2, y2), (30, 30, 255), 1)

	# ------------------------------------------------------------
	# 2. Outer flower ring
	# ------------------------------------------------------------
	cv2.circle(scene, (cx, cy), radius, (0, 0, 255), 3)

	# ------------------------------------------------------------
	# 3. Textured center disc
	# ------------------------------------------------------------
	center_r = int(0.35 * radius)
	cv2.circle(scene, (cx, cy), center_r, tuple(int(v) for v in center_base), -1)

	# Concentric texture rings
	for k in range(1, 5):
		rk = max(2, int(center_r * k / 5))
		color = tuple(int(v) for v in (center_alt if k % 2 == 0 else center_base))
		cv2.circle(scene, (cx, cy), rk, color, 1)

	# Dot texture inside the center
	rng = np.random.default_rng(1234)
	for _ in range(60):
		theta = rng.uniform(0.0, 2.0 * np.pi)
		rho = center_r * np.sqrt(rng.uniform(0.0, 1.0))
		x = int(cx + rho * np.cos(theta))
		y = int(cy + rho * np.sin(theta))

		if 0 <= x < width and 0 <= y < height:
			color = (0, 170 + int(rng.integers(0, 70)), 220 + int(rng.integers(0, 35)))
			cv2.circle(scene, (x, y), 1, color, -1)

	# ------------------------------------------------------------
	# 4. Strong cross marker kept from previous version
	# ------------------------------------------------------------
	cv2.line(scene, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 3)
	cv2.line(scene, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 3)

	# ------------------------------------------------------------
	# 5. Add subtle flower-wide texture spokes
	# ------------------------------------------------------------
	for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
		x2 = int(cx + 0.92 * radius * np.cos(angle))
		y2 = int(cy + 0.92 * radius * np.sin(angle))
		cv2.line(scene, (cx, cy), (x2, y2), (80, 80, 80), 1)

def _make_base_scene(width: int, height: int) -> np.ndarray:
	scene = np.zeros((height, width, 3), dtype=np.uint8)
	scene[:, :] = (65, 65, 65)

	# Platform area
	cv2.rectangle(
		scene,
		(int(0.08 * width), int(0.10 * height)),
		(int(0.92 * width), int(0.90 * height)),
		(95, 95, 95),
		-1,
	)

	# Background grid texture
	for x in range(0, width, 45):
		cv2.line(scene, (x, 0), (x, height), (80, 80, 80), 1)

	for y in range(0, height, 45):
		cv2.line(scene, (0, y), (width, y), (80, 80, 80), 1)

	# Deterministic textured dots on the platform
	rng = np.random.default_rng(4)
	for _ in range(180):
		x = int(rng.integers(20, width - 20))
		y = int(rng.integers(20, height - 20))
		r = int(rng.integers(2, 6))
		color = int(rng.integers(90, 220))
		cv2.circle(scene, (x, y), r, (color, color, color), -1)

	# Textured flower landing target
	cx = int(0.5 * width)
	cy = int(0.5 * height)
	radius = 50

	_draw_textured_flower_target(scene, cx, cy, radius)

	return scene


def _make_synthetic_frame(
	base_scene: np.ndarray,
	step: int,
	mode: str,
) -> np.ndarray:
	height, width = base_scene.shape[:2]
	center = (0.5 * width, 0.5 * height)

	scale, dx, dy = _motion_parameters(step, mode)

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


def _make_synthetic_target(
	width: int,
	height: int,
	step: int,
	mode: str,
	dt: float,
	base_box_width: float = 160.0,
	base_box_height: float = 160.0,
) -> TargetEstimate:
	"""
	Create a synthetic TargetEstimate consistent with the synthetic image warp.

	In zoom mode, the box size changes with the same scale used to warp the
	image. This allows the box-size divergence estimator to be compared
	against Farneback divergence.
	"""
	scale, dx, dy = _motion_parameters(step, mode)

	cx = 0.5 * width + dx
	cy = 0.5 * height + dy

	offset_x = (cx - 0.5 * width) / (0.5 * width)
	offset_y = (cy - 0.5 * height) / (0.5 * height)

	detection_width = base_box_width * scale
	detection_height = base_box_height * scale

	return TargetEstimate(
		timestamp=step * dt,
		found=True,
		offset_x=float(offset_x),
		offset_y=float(offset_y),
		confidence=1.0,
		detection_width=float(detection_width),
		detection_height=float(detection_height),
	)


def _draw_target_box(frame_bgr: np.ndarray, target: TargetEstimate) -> np.ndarray:
	vis = frame_bgr.copy()
	height, width = frame_bgr.shape[:2]

	if target is None or not target.found:
		return vis

	cx = int(round((0.5 * target.offset_x + 0.5) * width))
	cy = int(round((0.5 * target.offset_y + 0.5) * height))

	box_w = float(target.detection_width)
	box_h = float(target.detection_height)

	x0 = int(round(cx - 0.5 * box_w))
	y0 = int(round(cy - 0.5 * box_h))
	x1 = int(round(cx + 0.5 * box_w))
	y1 = int(round(cy + 0.5 * box_h))

	cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
	cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

	cv2.putText(
		vis,
		f"synthetic target box: {box_w:.0f}x{box_h:.0f}px",
		(20, 58),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.65,
		(0, 255, 255),
		2,
		cv2.LINE_AA,
	)

	return vis


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


def _make_farneback_panel(flow_debug: dict) -> np.ndarray:
	panel_width = 420
	panel_height = 280

	panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

	flow = flow_debug.get("result", None)
	roi = flow_debug.get("roi", None)
	message = flow_debug.get("message", "")

	lines = [
		"Farneback ROI divergence:",
		"",
	]

	if flow is None:
		lines += [
			"no FlowResult object",
		]
	else:
		lines += [
			f"valid: {flow.valid}",
			f"mean_flow_x: {flow.mean_flow_x:+.2f} px/s",
			f"mean_flow_y: {flow.mean_flow_y:+.2f} px/s",
			f"filtered divergence: {flow.divergence:+.4f} 1/s",
			f"raw divergence: {flow_debug.get('raw_divergence', 0.0):+.4f} 1/s",
			f"roi: {roi}",
			f"timestamp: {flow.timestamp:.3f} s",
		]

	if message:
		lines += [
			"",
			f"message: {message}",
		]

	y = 40
	for line in lines:
		cv2.putText(
			panel,
			line,
			(18, y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.56,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)
		y += 27

	return _label_tile(panel, "5 Farneback FlowResult")


def _make_box_size_panel(box_debug: dict) -> np.ndarray:
	panel_width = 420
	panel_height = 280

	panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

	message = box_debug.get("message", "")

	lines = [
		"Box-size divergence:",
		"",
		f"valid: {box_debug.get('valid', False)}",
		f"filtered D_box: {box_debug.get('filtered_divergence', 0.0):+.4f} 1/s",
		f"raw D_box: {box_debug.get('raw_divergence', 0.0):+.4f} 1/s",
		f"D_width: {box_debug.get('width_divergence', 0.0):+.4f} 1/s",
		f"D_height: {box_debug.get('height_divergence', 0.0):+.4f} 1/s",
		f"box width: {box_debug.get('width', 0.0):.1f} px",
		f"box height: {box_debug.get('height', 0.0):.1f} px",
	]

	if message:
		lines += [
			"",
			f"message: {message}",
		]

	y = 40
	for line in lines:
		cv2.putText(
			panel,
			line,
			(18, y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.56,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)
		y += 27

	return _label_tile(panel, "6 Box-size D_box")


def _make_debug_canvas(
	flow_debug: dict,
	box_debug: dict,
	target: TargetEstimate,
	mode: str,
	frame: np.ndarray,
) -> np.ndarray:
	previous_frame = flow_debug.get("previous_frame")
	current_frame = flow_debug.get("current_frame")
	flow_px_s = flow_debug.get("flow_px_s")
	divergence_field = flow_debug.get("divergence_field")
	roi = flow_debug.get("roi")

	current_with_box = _draw_target_box(current_frame, target) if current_frame is not None else None

	if current_frame is not None and flow_px_s is not None:
		flow_arrows = _draw_flow_arrows(current_frame, flow_px_s, roi=roi)
	else:
		flow_arrows = current_frame.copy() if current_frame is not None else None

	if divergence_field is not None:
		divergence_heatmap = _divergence_to_heatmap(divergence_field)
	else:
		divergence_heatmap = None

	farneback_panel = _make_farneback_panel(flow_debug)
	box_panel = _make_box_size_panel(box_debug)

	tiles = [
		("1 Previous frame", _to_bgr_for_display(previous_frame)),
		("2 Current frame + target box", _to_bgr_for_display(current_with_box)),
		("3 ROI flow arrows", _to_bgr_for_display(flow_arrows)),
		("4 ROI divergence heatmap", _to_bgr_for_display(divergence_heatmap)),
		("5 Farneback FlowResult", farneback_panel),
		("6 Box-size D_box", box_panel),
	]

	tile_width = 420
	tile_height = 280

	resized_tiles = [
		_label_tile(_resize_for_tile(image, tile_width, tile_height), label)
		for label, image in tiles
	]

	row_1 = np.hstack(resized_tiles[:3])
	row_2 = np.hstack(resized_tiles[3:])

	canvas = np.vstack([row_1, row_2])

	mode_text = f"mode: {mode} | m: switch mode | r: reset | s: save | q/Esc: quit"

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

	return canvas


def test():
	"""
	Standalone visual test for comparing divergence estimators.

	Keyboard controls:

		q or Esc  -> quit
		m         -> change synthetic motion mode
		r         -> reset estimators
		s         -> save current preview image
	"""
	width = 640
	height = 480
	dt = 1.0 / 30.0

	base_scene = _make_base_scene(width, height)

	flow_estimator = OpticalFlowEstimator(
		require_target_roi=True,
		roi_margin_fraction=0.05,
		min_roi_size_px=32,
		divergence_smoothing=0.6,
		store_debug=True,
	)

	box_estimator = BoxSizeDivergenceEstimator(
		smoothing=0.6,
		min_box_size_px=3.0,
	)

	modes = ["translate", "zoom", "combined"]
	mode_index = 1

	step = 0
	last_canvas = None

	window_name = "Optical flow divergence comparison"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	print("")
	print("Optical flow divergence comparison")
	print("----------------------------------")
	print("q or Esc  -> quit")
	print("m         -> change synthetic motion mode")
	print("r         -> reset estimators")
	print("s         -> save current preview image")
	print("")

	while True:
		mode = modes[mode_index]

		frame = _make_synthetic_frame(
			base_scene=base_scene,
			step=step,
			mode=mode,
		)

		target = _make_synthetic_target(
			width=width,
			height=height,
			step=step,
			mode=mode,
			dt=dt,
		)

		timestamp = step * dt

		flow_estimator.update(
			frame,
			timestamp,
			target=target,
		)

		box_debug = box_estimator.update(target)
		flow_debug = flow_estimator.last_debug_data()

		canvas = _make_debug_canvas(
			flow_debug=flow_debug,
			box_debug=box_debug,
			target=target,
			mode=mode,
			frame=frame,
		)

		cv2.imshow(window_name, canvas)
		last_canvas = canvas

		key = cv2.waitKey(40) & 0xFF

		if key in (ord("q"), 27):
			break

		if key == ord("m"):
			mode_index = (mode_index + 1) % len(modes)
			flow_estimator.reset()
			box_estimator.reset()
			step = 0
			continue

		if key == ord("r"):
			flow_estimator.reset()
			box_estimator.reset()
			step = 0
			continue

		if key == ord("s") and last_canvas is not None:
			cv2.imwrite("optical_flow_divergence_comparison.png", last_canvas)
			print("Saved: optical_flow_divergence_comparison.png")

		step += 1

	cv2.destroyAllWindows()


if __name__ == "__main__":
	test()