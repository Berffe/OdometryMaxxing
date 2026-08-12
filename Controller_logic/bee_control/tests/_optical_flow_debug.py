"""
Standalone optical-flow debug visualization.

WHAT CHANGED AND WHY
--------------------
This harness used to pit Farneback ROI divergence against a box-size
(apparent-growth) estimator. That comparison is settled: dense flow won, the box
estimator never reached the controller, and keeping a decided experiment on
screen costs attention every time the file is opened. Both are gone.

The question this file now answers is the one that is still open:

    given ONE Farneback flow field, which reduction recovers lambda best?

The estimators live in ``_divergence_estimators.py`` so this harness and
``_optFlow_targetAcqu_debug.py`` cannot drift apart. The production 4-parameter
constrained fit is read back from the FlowResult rather than recomputed, so the
panel can never disagree with what the vehicle would have flown.

Because the scene is synthetic, ground truth is known ANALYTICALLY: the frame at
step k is the base scene scaled about the image centre by s(k), so the flow
between consecutive frames is an exact scaling by r = s(k)/s(k-1) and the
lambda the fit should return is (r - 1) / dt.

Run through optical_flow.py, from the repo root:

	python -m bee_control.vision.optical_flow

or directly:

	python -m bee_control.tests._optical_flow_debug

Keyboard:

	q or Esc  -> quit
	m         -> cycle motion mode (zoom / translate / combined)
	r         -> reset estimator and statistics
	s         -> save the current preview
"""

from typing import Optional

import cv2
import numpy as np

from bee_control.core.state import TargetEstimate
from bee_control.vision.optical_flow import OpticalFlowEstimator

from ._divergence_estimators import (
	EstimatorComparison,
	evaluate_estimators,
	make_comparison_plot,
	make_comparison_table,
)


def _motion_parameters(step: int, mode: str):
	"""
	Return the synthetic camera/target motion.

	The same scale and translation are used both to warp the synthetic
	image and to generate the synthetic TargetEstimate. This makes the
	box-size divergence meaningful.
	"""
	if mode == "translate":
		scale = 1.0
		# Amplitude and rate are chosen so the per-frame displacement stays
		# around 2 px. The original 70 px / 0.35 rad-per-step gave ~24 px per
		# frame, which is past what Farneback resolves at this downsample: the
		# fit then reported lambda swinging over several 1/s with fit_quality
		# collapsing to ~0.05. That measured TRACKING FAILURE, not reduction
		# bias, and it made translate mode useless as the zero-lambda test it
		# is meant to be. Raise these again only alongside a Farneback winsize
		# or pyramid-level change.
		dx = 20.0 * np.sin(0.10 * step)
		dy = 14.0 * np.cos(0.08 * step)

	elif mode == "zoom":
		scale = 1.0 + 0.25 * np.sin(0.09 * step)
		dx = 0.0
		dy = 0.0

	else:
		scale = 1.0 + 0.5 * np.sin(0.05 * step)
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

	COORDINATE SCALING -- do not remove
	-----------------------------------
	``flow_px_s`` is the WORKING grid, which is the ROI after downsampling, so
	its indices are NOT full-frame pixels. The ROI rectangle, by contrast, is in
	full-frame pixels. Adding a working-grid index straight to the ROI origin
	therefore squeezes every arrow into the top-left ``scale`` fraction of the
	box -- 50% x 50% once the ROI is large enough for the downsampler to clamp
	at 0.5, which is precisely the near-field regime the arrows matter most in.

	The step is recovered from the two shapes rather than passed in, so the
	renderer stays correct if the downsample policy changes.

	Only the sample POSITIONS need this. The vectors themselves are already
	amplitude-corrected to original px/s by the estimator, so direction and
	length are in full-frame units and must not be rescaled again.
	"""
	vis = frame_bgr.copy()

	grid_height, grid_width = flow_px_s.shape[:2]

	if roi is None:
		x_offset = 0
		y_offset = 0
		x_step = 1.0
		y_step = 1.0
	else:
		x_offset, y_offset, x1, y1 = roi
		cv2.rectangle(vis, (x_offset, y_offset), (x1, y1), (0, 255, 0), 2)

		x_step = (x1 - x_offset) / max(1, grid_width)
		y_step = (y1 - y_offset) / max(1, grid_height)

	roi_height, roi_width = grid_height, grid_width

	u = flow_px_s[:, :, 0]
	v = flow_px_s[:, :, 1]
	magnitude = np.sqrt(u * u + v * v)

	mag_ref = float(np.percentile(magnitude, 95))
	mag_ref = max(mag_ref, min_flow_magnitude)

	# Centre the sample lattice. `range(grid_step // 2, n, grid_step)` leaves
	# whatever does not divide evenly unsampled at the far edge, which pushes
	# the arrow cloud toward the top-left of the ROI -- a smaller version of the
	# coordinate bug above, and just as misleading to look at.
	x_origin = ((roi_width - 1) % grid_step) // 2
	y_origin = ((roi_height - 1) % grid_step) // 2

	for y in range(y_origin, roi_height, grid_step):
		for x in range(x_origin, roi_width, grid_step):
			u_xy = float(flow_px_s[y, x, 0])
			v_xy = float(flow_px_s[y, x, 1])

			mag = (u_xy * u_xy + v_xy * v_xy) ** 0.5

			if mag < min_flow_magnitude:
				continue

			dir_x = u_xy / mag
			dir_y = v_xy / mag

			length = max_arrow_length * min(1.0, mag / mag_ref)
			length = max(min_arrow_length, length)

			x_start = int(round(x_offset + x * x_step))
			y_start = int(round(y_offset + y * y_step))

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


def _make_flow_result_panel(flow_debug: dict, mode: str, truth: float) -> np.ndarray:
	"""The FlowResult as the controller would receive it, plus the scene truth."""
	panel_width = 420
	panel_height = 280

	panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

	flow = flow_debug.get("result", None)
	roi = flow_debug.get("roi", None)
	message = flow_debug.get("message", "")

	lines = ["FlowResult as delivered:", ""]

	if flow is None:
		lines += ["no FlowResult object"]
	else:
		lines += [
			f"valid: {flow.valid}",
			f"mean_flow_x: {flow.mean_flow_x:+.2f} px/s",
			f"mean_flow_y: {flow.mean_flow_y:+.2f} px/s",
			f"divergence (filtered): {flow.divergence:+.4f}",
			f"raw_divergence: {flow.raw_divergence:+.4f}",
			f"fit_quality: {flow.fit_quality:.3f}",
			f"timestamp: {flow.timestamp:.3f} s",
		]

	lines += [
		"",
		f"mode: {mode}",
		f"truth lambda: {truth:+.4f} 1/s" if np.isfinite(truth) else "truth lambda: n/a",
	]

	if roi is not None:
		lines += [f"roi: {roi}"]

	if message:
		lines += ["", f"message: {message}"]

	y = 34
	for line in lines:
		cv2.putText(
			panel,
			line,
			(16, y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.46,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)
		y += 21

	return _label_tile(panel, "4 FlowResult")


def _ground_truth_lambda(step: int, mode: str, dt: float) -> float:
	"""The lambda the fit SHOULD return for the frame pair (step-1, step).

	The frame at step k is the base scene scaled about the image centre by
	s(k), so the flow between consecutive frames is exactly a scaling by
	r = s(k) / s(k-1) about that centre:

	    u(x) = (r - 1) * x / dt

	The constrained model fits ``u = t_x + lambda*x - r_rot*y``, hence the
	value it should recover is ``(r - 1) / dt``. The discrete ratio is used
	rather than the continuous derivative of s so that ground truth matches
	what a one-frame displacement measurement can actually see.

	Pure translation contributes only to t_x/t_y, so translate mode has
	lambda = 0 exactly -- which makes it the bias test: any estimator reporting
	non-zero lambda there is leaking translation into expansion.
	"""
	if step <= 0:
		return float("nan")

	scale_now, _, _ = _motion_parameters(step, mode)
	scale_prev, _, _ = _motion_parameters(step - 1, mode)

	if abs(scale_prev) < 1e-9 or dt <= 0.0:
		return float("nan")

	return (scale_now / scale_prev - 1.0) / dt


def _make_debug_canvas(
	flow_debug: dict,
	comparison: EstimatorComparison,
	target: TargetEstimate,
	mode: str,
	truth: float,
) -> np.ndarray:
	"""Top row: what the estimator saw. Bottom row: what each reduction made of it."""
	current_frame = flow_debug.get("current_frame")
	flow_px_s = flow_debug.get("flow_px_s")
	divergence_field = flow_debug.get("divergence_field")
	roi = flow_debug.get("roi")

	current_with_box = (
		_draw_target_box(current_frame, target)
		if current_frame is not None
		else None
	)

	if current_frame is not None and flow_px_s is not None:
		flow_arrows = _draw_flow_arrows(current_frame, flow_px_s, roi=roi)
	else:
		flow_arrows = current_frame.copy() if current_frame is not None else None

	divergence_heatmap = (
		_divergence_to_heatmap(divergence_field)
		if divergence_field is not None
		else None
	)

	tile_width = 420
	tile_height = 280

	top_tiles = [
		("1 Current frame + target box", _to_bgr_for_display(current_with_box)),
		("2 ROI flow arrows", _to_bgr_for_display(flow_arrows)),
		("3 ROI lambda field", _to_bgr_for_display(divergence_heatmap)),
	]

	row_1 = np.hstack([
		_label_tile(_resize_for_tile(image, tile_width, tile_height), label)
		for label, image in top_tiles
	])

	total_width = row_1.shape[1]

	# The table and the plot carry their own titles, so they are NOT passed
	# through _label_tile -- its black header bar would sit on top of them.
	table_width = total_width // 2
	plot_width = total_width - table_width

	flow = flow_debug.get("result")
	subtitle = (
		f"mode: {mode}   valid: {getattr(flow, 'valid', False)}   "
		f"raw: {getattr(flow, 'raw_divergence', float('nan')):+.4f}   "
		f"filtered: {getattr(flow, 'divergence', float('nan')):+.4f}"
	)

	table = make_comparison_table(
		comparison,
		width=table_width,
		height=tile_height,
		subtitle=subtitle,
	)
	plot = make_comparison_plot(comparison, width=plot_width, height=tile_height)

	row_2 = np.hstack([table, plot])

	canvas = np.vstack([row_1, row_2])

	# Footer is its own strip. Drawing it over the panels clipped the last row
	# of whichever panel it landed on.
	footer = np.zeros((30, canvas.shape[1], 3), dtype=np.uint8)
	cv2.putText(
		footer,
		f"mode: {mode} | m: switch mode (zoom/translate/combined) | "
		f"r: reset | s: save | q/Esc: quit",
		(16, 20),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.52,
		(255, 255, 255),
		1,
		cv2.LINE_AA,
	)

	canvas = np.vstack([canvas, footer])

	return canvas


def test():
	"""Compare every lambda reduction on one synthetic scene with known truth.

	Modes:

	``zoom``       pure expansion about the image centre. The accuracy test.
	``translate``  pure translation, lambda = 0 exactly. The BIAS test: a
	               reduction that leaks translation into expansion shows up
	               here and nowhere else.
	``combined``   expansion plus translation, which is the realistic case and
	               the one where the constrained model earns its keep -- it has
	               explicit translation terms to absorb the lateral motion.
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

	comparison = EstimatorComparison(history_length=240)

	modes = ["zoom", "translate", "combined"]
	mode_index = 0

	step = 0
	last_canvas = None

	window_name = "Divergence reduction comparison"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	print("")
	print("Divergence reduction comparison")
	print("-------------------------------")
	print("One Farneback field per frame, reduced to lambda seven ways.")
	print("Ground truth is analytic; bias and RMS accumulate until reset.")
	print("")
	print("q or Esc  -> quit")
	print("m         -> change synthetic motion mode (zoom/translate/combined)")
	print("r         -> reset estimator and statistics")
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

		flow_result = flow_estimator.update(frame, timestamp, target=target)
		flow_debug = flow_estimator.last_debug_data()

		truth = _ground_truth_lambda(step, mode, dt)

		# Only score frames the estimator itself considered valid; a warm-up
		# frame with no predecessor would otherwise pollute every bias figure.
		if flow_result is not None and flow_result.valid:
			values = evaluate_estimators(flow_estimator, flow_result, flow_debug)
			comparison.update(values, truth, flow_result.fit_quality)

		canvas = _make_debug_canvas(
			flow_debug=flow_debug,
			comparison=comparison,
			target=target,
			mode=mode,
			truth=truth,
		)

		cv2.imshow(window_name, canvas)
		last_canvas = canvas

		key = cv2.waitKey(40) & 0xFF

		if key in (ord("q"), 27):
			break

		if key == ord("m"):
			mode_index = (mode_index + 1) % len(modes)
			flow_estimator.reset()
			comparison.reset()
			step = 0
			continue

		if key == ord("r"):
			flow_estimator.reset()
			comparison.reset()
			step = 0
			continue

		if key == ord("s") and last_canvas is not None:
			cv2.imwrite("divergence_reduction_comparison.png", last_canvas)
			print("Saved: divergence_reduction_comparison.png")

		step += 1

	cv2.destroyAllWindows()

	_print_summary(comparison, modes[mode_index])


def _print_summary(comparison: EstimatorComparison, mode: str):
	"""Console summary, so a headless run still yields the numbers."""
	from ._divergence_estimators import ESTIMATORS

	print("")
	print(f"Reduction comparison summary (mode: {mode})")
	print(f"{'estimator':32} {'bias':>10} {'rms':>10} {'n':>6}")
	print("-" * 62)

	for key, label, _, _ in ESTIMATORS:
		print(
			f"{label:32} {comparison.bias(key):>+10.5f} "
			f"{comparison.rms(key):>10.5f} {comparison.samples(key):>6d}"
		)
	print("")


if __name__ == "__main__":
	test()