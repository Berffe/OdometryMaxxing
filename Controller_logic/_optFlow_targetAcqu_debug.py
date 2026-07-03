"""
Combined target-acquisition + optical-flow debug visualization.

This version uses a deterministic synthetic expanding target, so the optical-flow
pipeline can be tested without the controller, Gazebo, or camera artifacts.

The synthetic target is rendered as a centered circular surface whose image scale
increases exponentially:

    radius(t) = radius_0 * exp(0.5 * expected_divergence * t)

For a pure centered zoom field, the ideal normalized-image divergence is therefore
approximately `expected_divergence` [1/s]. Once the circle exceeds the image
field of view, the generator keeps rendering the same surface closer and closer,
so the FOV-saturated regime is tested too.

Run directly:

    python _optFlow_targetAcqu_debug_synthetic.py

or from the package root, after replacing the original file:

    python -m bee_control._optFlow_targetAcqu_debug
"""

from typing import Optional

import cv2
import numpy as np

try:
	from .target_acquisition import TargetAcquisition
	from .optical_flow import OpticalFlowEstimator

	from ._target_acquisition_debug import (
		_draw_detection_debug,
		_to_bgr_for_display,
		_resize_for_tile,
		_label_tile,
	)

	from ._optical_flow_debug import (
		_draw_flow_arrows,
		_divergence_to_heatmap,
	)

except ImportError:
	from target_acquisition import TargetAcquisition
	from optical_flow import OpticalFlowEstimator

	from _target_acquisition_debug import (
		_draw_detection_debug,
		_to_bgr_for_display,
		_resize_for_tile,
		_label_tile,
	)

	from _optical_flow_debug import (
		_draw_flow_arrows,
		_divergence_to_heatmap,
	)


# -----------------------------------------------------------------------------
# Synthetic target renderer
# -----------------------------------------------------------------------------

def _render_expanding_circle_target(
	width: int,
	height: int,
	timestamp: float,
	with_target: bool = True,
	textured: bool = True,
	expected_divergence: float = 0.85,
	initial_radius_px: float = 34.0,
	cycle_duration_sec: float = 8.0,
) -> tuple[np.ndarray, dict]:
	"""
	Create a camera-like frame containing a centered circular target that expands.

	The target is not simply a drawn circle whose edge grows; it is rendered as a
	surface pattern sampled at normalized coordinates (x/radius, y/radius). That
	means that after the outer rim has left the frame, the internal pattern still
	keeps expanding, which is exactly the close-range case we want to test.

	For this synthetic pure-zoom motion, the expected normalized-image divergence
	is approximately expected_divergence [1/s].
	"""
	# Low-saturation background so TargetAcquisition's broad HSV mask does not
	# classify the whole image as target.
	frame = np.full((height, width, 3), (42, 42, 42), dtype=np.uint8)

	if not with_target:
		return frame, {
			"radius_px": 0.0,
			"expected_divergence": 0.0,
			"phase_t": 0.0,
			"fov_radius_px": 0.5 * float(np.hypot(width, height)),
		}

	# Repeat the test automatically so a single run shows the whole transition:
	# small target -> large target -> FOV-saturated -> still approaching.
	phase_t = float(timestamp % cycle_duration_sec)
	radius = float(initial_radius_px * np.exp(0.5 * expected_divergence * phase_t))

	cx = 0.5 * (width - 1)
	cy = 0.5 * (height - 1)
	yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

	x = (xx - cx) / max(radius, 1e-6)
	y = (yy - cy) / max(radius, 1e-6)
	rho = np.sqrt(x * x + y * y)
	theta = np.arctan2(y, x)

	inside = rho <= 1.0
	if not np.any(inside):
		return frame, {
			"radius_px": radius,
			"expected_divergence": expected_divergence,
			"phase_t": phase_t,
			"fov_radius_px": 0.5 * float(np.hypot(width, height)),
		}

	# BGR colors: all target colors are saturated enough for the HSV detector.
	yellow = np.array((0, 190, 255), dtype=np.uint8)
	orange = np.array((0, 145, 210), dtype=np.uint8)
	dark_orange = np.array((0, 105, 155), dtype=np.uint8)

	frame[inside] = yellow

	if textured:
		# Normalized thicknesses. Because coordinates are normalized by radius,
		# these features grow in pixels as the target approaches.
		thin = 0.006
		medium = 0.012
		thick = 0.025

		# Checkerboard physical texture. The number of checks is fixed in target
		# coordinates, so check size grows with the target in the image.
		checker_region = inside & (rho >= 0.23) & (rho <= 0.74)
		checker = (
			(np.floor((x + 1.0) * 18.0) + np.floor((y + 1.0) * 18.0)).astype(np.int32) % 2
		) == 0
		frame[checker_region & checker] = orange
		frame[checker_region & ~checker] = dark_orange

		# Concentric rings. The ring positions are target-surface coordinates.
		for r in (0.16, 0.28, 0.36, 0.76, 0.84, 0.91, 0.98):
			mask = inside & (np.abs(rho - r) <= (thick if r in (0.16, 0.98) else medium))
			frame[mask] = dark_orange if r in (0.16, 0.36, 0.84) else yellow

		# Cardinal crosshair marks.
		cross = inside & (rho >= 0.16) & (
			(np.abs(x) <= medium) | (np.abs(y) <= medium)
		)
		frame[cross] = yellow

		# Dotted/tick ring. This gives Farneback many high-gradient features even
		# after the circular rim has left the FOV.
		sector = np.floor((theta + np.pi) / (2.0 * np.pi) * 96.0).astype(np.int32)
		dots = (sector % 2) == 0
		dot_ring = inside & (rho >= 0.86) & (rho <= 0.91) & dots
		frame[dot_ring] = yellow

		# Thin radial ticks in an inner band.
		sector_inner = np.floor((theta + np.pi) / (2.0 * np.pi) * 72.0).astype(np.int32)
		inner_ticks = inside & (rho >= 0.78) & (rho <= 0.83) & ((sector_inner % 2) == 0)
		frame[inner_ticks] = dark_orange

	else:
		# Deliberately texture-poor mode. This is useful to reproduce the classic
		# close-range failure: once the rim disappears, the interior has almost no
		# usable optical-flow support.
		for r in (0.98, 0.35, 0.18):
			frame[inside & (np.abs(rho - r) <= 0.012)] = dark_orange

	# Anti-aliased-looking softness without destroying geometry. This also makes
	# it closer to the real camera path, where the target is not a perfect binary
	# mask.
	frame = cv2.GaussianBlur(frame, (3, 3), 0)

	return frame, {
		"radius_px": radius,
		"expected_divergence": expected_divergence,
		"phase_t": phase_t,
		"fov_radius_px": 0.5 * float(np.hypot(width, height)),
	}


# -----------------------------------------------------------------------------
# Display panels
# -----------------------------------------------------------------------------

def _make_target_estimate_panel(target_debug: dict, synthetic_meta: Optional[dict] = None) -> np.ndarray:
	"""Text panel for TargetEstimate + synthetic ground truth."""
	panel_width = 420
	panel_height = 280
	panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

	target = target_debug.get("target", None)

	lines = ["TargetEstimate:", ""]

	if target is None:
		lines += ["no target object"]
	else:
		lines += [
			f"found: {target.found}",
			f"offset_x: {target.offset_x:+.3f}",
			f"offset_y: {target.offset_y:+.3f}",
			f"confidence: {target.confidence:.2f}",
			f"area_fraction: {getattr(target, 'area_fraction', 0.0):.3f}",
			f"fov_saturated: {getattr(target, 'fov_saturated', False)}",
			f"box: {target.detection_width:.0f} x {target.detection_height:.0f} px",
		]

	if synthetic_meta is not None:
		lines += [
			"",
			f"synthetic radius: {synthetic_meta['radius_px']:.1f} px",
			f"FOV corner radius: {synthetic_meta['fov_radius_px']:.1f} px",
		]

	y = 34
	for line in lines:
		cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
		y += 25

	return _label_tile(panel, "5 TargetEstimate")


def _make_flow_result_panel(flow_debug: dict) -> np.ndarray:
	"""Text panel for FlowResult only."""
	panel_width = 420
	panel_height = 280
	panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

	flow = flow_debug.get("result", None)
	roi = flow_debug.get("roi", None)
	message = flow_debug.get("message", "")

	lines = ["FlowResult:", ""]

	if flow is None:
		lines += ["no flow object"]
	else:
		lines += [
			f"valid: {flow.valid}",
			f"mean_flow_x: {flow.mean_flow_x:+.2f} px/s",
			f"mean_flow_y: {flow.mean_flow_y:+.2f} px/s",
			f"divergence: {flow.divergence:+.4f} 1/s",
			f"raw divergence: {getattr(flow, 'raw_divergence', flow_debug.get('raw_divergence', 0.0)):+.4f} 1/s",
			f"fit_quality: {getattr(flow, 'fit_quality', 0.0):+.3f}",
			f"roi: {roi}",
		]

	if message:
		lines += ["", f"message: {message}"]

	y = 34
	for line in lines:
		cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
		y += 24

	return _label_tile(panel, "6 FlowResult")


def _make_divergence_history_panel(history: list[dict], expected_divergence: float) -> np.ndarray:
	"""Small OpenCV line plot for measured divergence evolution."""
	panel_width = 420
	panel_height = 280
	panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

	left = 52
	right = panel_width - 18
	top = 26
	bottom = panel_height - 44

	cv2.rectangle(panel, (left, top), (right, bottom), (90, 90, 90), 1)
	cv2.putText(panel, "divergence history [1/s]", (18, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

	if len(history) < 2:
		return _label_tile(panel, "4 Divergence evolution")

	values = np.array([h["divergence"] for h in history if h.get("valid", False)], dtype=np.float32)
	raw_values = np.array([h["raw_divergence"] for h in history if h.get("valid", False)], dtype=np.float32)
	if values.size == 0:
		return _label_tile(panel, "4 Divergence evolution")

	y_abs = max(
		0.25,
		float(abs(expected_divergence)) * 1.4,
		float(np.nanmax(np.abs(values))) * 1.2,
		float(np.nanmax(np.abs(raw_values))) * 1.2 if raw_values.size else 0.25,
	)
	y_min = -0.15 * y_abs
	y_max = y_abs

	def x_of(i: int, n: int) -> int:
		if n <= 1:
			return left
		return int(left + (right - left) * i / float(n - 1))

	def y_of(v: float) -> int:
		alpha = (float(v) - y_min) / max(y_max - y_min, 1e-6)
		return int(bottom - alpha * (bottom - top))

	# Zero and expected lines.
	cv2.line(panel, (left, y_of(0.0)), (right, y_of(0.0)), (90, 90, 90), 1)
	cv2.line(panel, (left, y_of(expected_divergence)), (right, y_of(expected_divergence)), (80, 180, 255), 1)
	cv2.putText(panel, f"expected {expected_divergence:.2f}", (left + 4, y_of(expected_divergence) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 180, 255), 1, cv2.LINE_AA)

	valid_hist = [h for h in history if h.get("valid", False)]
	n = len(valid_hist)
	for i in range(1, n):
		p0_raw = (x_of(i - 1, n), y_of(valid_hist[i - 1]["raw_divergence"]))
		p1_raw = (x_of(i, n), y_of(valid_hist[i]["raw_divergence"]))
		cv2.line(panel, p0_raw, p1_raw, (120, 120, 120), 1)

		p0 = (x_of(i - 1, n), y_of(valid_hist[i - 1]["divergence"]))
		p1 = (x_of(i, n), y_of(valid_hist[i]["divergence"]))
		cv2.line(panel, p0, p1, (0, 255, 0), 2)

	# FOV saturation marker dots at the bottom.
	for i, h in enumerate(valid_hist):
		if h.get("fov_saturated", False):
			x = x_of(i, n)
			cv2.circle(panel, (x, bottom + 12), 2, (0, 180, 255), -1)

	last = valid_hist[-1]
	cv2.putText(panel, f"filtered: {last['divergence']:+.3f}", (left, panel_height - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1, cv2.LINE_AA)
	cv2.putText(panel, f"raw: {last['raw_divergence']:+.3f}", (left + 165, panel_height - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

	return _label_tile(panel, "4 Divergence evolution")


def _make_combined_canvas(
	target_debug: dict,
	flow_debug: dict,
	history: list[dict],
	synthetic_meta: dict,
	mode_text: str,
) -> np.ndarray:
	"""
	Build a 2x3 visualization canvas:

	1. BGR frame
	2. Target detection
	3. ROI flow arrows
	4. Divergence evolution
	5. TargetEstimate
	6. FlowResult
	"""
	frame = target_debug.get("frame")
	contour = target_debug.get("contour")
	target = target_debug.get("target")

	flow_px_s = flow_debug.get("flow_px_s")
	roi = flow_debug.get("roi")

	target_result = _draw_detection_debug(frame, target, contour)

	if frame is not None and flow_px_s is not None:
		flow_arrows = _draw_flow_arrows(frame, flow_px_s, roi=roi)
	else:
		flow_arrows = frame.copy() if frame is not None else None

	history_panel = _make_divergence_history_panel(
		history,
		expected_divergence=float(synthetic_meta.get("expected_divergence", 0.0)),
	)
	target_panel = _make_target_estimate_panel(target_debug, synthetic_meta)
	flow_panel = _make_flow_result_panel(flow_debug)

	tiles = [
		("1 BGR frame", _to_bgr_for_display(frame)),
		("2 Target detection", _to_bgr_for_display(target_result)),
		("3 ROI flow arrows", _to_bgr_for_display(flow_arrows)),
		("4 Divergence evolution", history_panel),
		("5 TargetEstimate", target_panel),
		("6 FlowResult", flow_panel),
	]

	tile_width = 420
	tile_height = 280

	resized_tiles = [
		_label_tile(
			_resize_for_tile(image, tile_width, tile_height),
			label,
		)
		for label, image in tiles
	]

	row_1 = np.hstack(resized_tiles[:3])
	row_2 = np.hstack(resized_tiles[3:])
	canvas = np.vstack([row_1, row_2])

	cv2.putText(canvas, mode_text, (20, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

	return canvas


# -----------------------------------------------------------------------------
# Main debug loop
# -----------------------------------------------------------------------------

def test():
	"""
	Combined visual test.

	Keyboard controls:

		q or Esc  -> quit
		space     -> pause/resume
		n         -> toggle target on/off
		t         -> toggle rich texture / poor texture
		[ and ]   -> decrease/increase expected synthetic divergence
		r         -> reset optical-flow estimator and history
		s         -> save current preview image
	"""
	width = 640
	height = 480
	dt = 1.0 / 30.0

	target_detector = TargetAcquisition(
		hsv_ranges=None,
		min_area_px=120.0,
		max_area_fraction=0.60,
		absolute_max_area_fraction=0.95,
		blur_kernel_size=5,
		morph_kernel_size=5,
		min_saturation=60,
		min_value=45,
		loss_grace_period_sec=0.0,
		fov_saturation_margin_px=2,
	)

	flow_estimator = OpticalFlowEstimator(
		require_target_roi=True,
		roi_margin_fraction=0.05,
		min_roi_size_px=32,
		divergence_smoothing=0.6,
		store_debug=True,
	)

	with_target = True
	textured = True
	paused = False
	expected_divergence = 0.85

	step = 0
	last_canvas = None
	history: list[dict] = []
	history_max_len = 240

	window_name = "Synthetic target acquisition + optical flow divergence debug"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	print("")
	print("Synthetic target acquisition + optical flow divergence debug")
	print("------------------------------------------------------------")
	print("q or Esc  -> quit")
	print("space     -> pause/resume")
	print("n         -> toggle target on/off")
	print("t         -> toggle rich texture / poor texture")
	print("[ and ]   -> decrease/increase expected synthetic divergence")
	print("r         -> reset optical-flow estimator and history")
	print("s         -> save current preview image")
	print("")

	while True:
		timestamp = step * dt

		frame, synthetic_meta = _render_expanding_circle_target(
			width=width,
			height=height,
			timestamp=timestamp,
			with_target=with_target,
			textured=textured,
			expected_divergence=expected_divergence,
			initial_radius_px=34.0,
			cycle_duration_sec=8.0,
		)

		# Final architecture step 1: target acquisition runs first.
		target_debug = target_detector.process_debug(frame, timestamp=timestamp)
		target = target_debug["target"]

		# Final architecture step 2: optical flow receives the target and computes
		# divergence only on the detected target ROI.
		flow = flow_estimator.update(frame, timestamp, target=target)
		flow_debug = flow_estimator.last_debug_data()

		if flow is not None:
			history.append({
				"timestamp": timestamp,
				"valid": bool(getattr(flow, "valid", False)),
				"divergence": float(getattr(flow, "divergence", 0.0)),
				"raw_divergence": float(getattr(flow, "raw_divergence", 0.0)),
				"fit_quality": float(getattr(flow, "fit_quality", 0.0)),
				"area_fraction": float(getattr(target, "area_fraction", 0.0)),
				"fov_saturated": bool(getattr(target, "fov_saturated", False)),
			})
			history = history[-history_max_len:]

		mode_text = (
			f"synthetic pure zoom | expected_div={expected_divergence:.2f} 1/s | "
			f"target={'ON' if with_target else 'OFF'} | "
			f"texture={'RICH' if textured else 'POOR'} | "
			f"paused={'YES' if paused else 'NO'}"
		)

		canvas = _make_combined_canvas(
			target_debug=target_debug,
			flow_debug=flow_debug,
			history=history,
			synthetic_meta=synthetic_meta,
			mode_text=mode_text,
		)

		cv2.imshow(window_name, canvas)
		last_canvas = canvas

		key = cv2.waitKey(40) & 0xFF

		if key in (ord("q"), 27):
			break

		if key == ord(" "):
			paused = not paused

		if key == ord("n"):
			with_target = not with_target
			flow_estimator.reset()
			target_detector.reset()
			history.clear()
			step = 0
			continue

		if key == ord("t"):
			textured = not textured
			flow_estimator.reset()
			target_detector.reset()
			history.clear()
			step = 0
			continue

		if key == ord("["):
			expected_divergence = max(0.10, expected_divergence - 0.10)
			flow_estimator.reset()
			target_detector.reset()
			history.clear()
			step = 0
			continue

		if key == ord("]"):
			expected_divergence = min(3.00, expected_divergence + 0.10)
			flow_estimator.reset()
			target_detector.reset()
			history.clear()
			step = 0
			continue

		if key == ord("r"):
			flow_estimator.reset()
			target_detector.reset()
			history.clear()
			step = 0
			continue

		if key == ord("s") and last_canvas is not None:
			cv2.imwrite("optFlow_targetAcqu_synthetic_debug_preview.png", last_canvas)
			print("Saved: optFlow_targetAcqu_synthetic_debug_preview.png")

		if not paused:
			step += 1

	cv2.destroyAllWindows()


if __name__ == "__main__":
	test()
