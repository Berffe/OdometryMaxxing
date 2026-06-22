"""
Combined target-acquisition + optical-flow debug visualization.

This file tests the final intended architecture:

	1. Generate or receive a camera-like frame.
	2. Run TargetAcquisition first.
	3. Use the detected target box as the ROI for OpticalFlowEstimator.
	4. Display target detection and ROI optical-flow divergence together.

Run directly:

	python optFlow_targetAcqu_debug.py

or from the package root:

	python -m bee_control.optFlow_targetAcqu_debug
"""

from typing import Optional

import cv2
import numpy as np

try:
	from .target_acquisition import TargetAcquisition
	from .optical_flow import OpticalFlowEstimator

	from .target_acquisition_debug import (
		_make_dummy_frame,
		_draw_detection_debug,
		_to_bgr_for_display,
		_resize_for_tile,
		_label_tile,
	)

	from .optical_flow_debug import (
		_draw_flow_arrows,
		_flow_to_color,
		_divergence_to_heatmap,
	)

except ImportError:
	from target_acquisition import TargetAcquisition
	from optical_flow import OpticalFlowEstimator

	from target_acquisition_debug import (
		_make_dummy_frame,
		_draw_detection_debug,
		_to_bgr_for_display,
		_resize_for_tile,
		_label_tile,
	)

	from optical_flow_debug import (
		_draw_flow_arrows,
		_flow_to_color,
		_divergence_to_heatmap,
	)


def _make_target_estimate_panel(target_debug: dict) -> np.ndarray:
	"""
	Text panel for TargetEstimate only.
	"""
	panel_width = 420
	panel_height = 280

	panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

	target = target_debug.get("target", None)

	lines = [
		"TargetEstimate:",
		"",
	]

	if target is None:
		lines += [
			"no target object",
		]
	else:
		lines += [
			f"found: {target.found}",
			f"offset_x: {target.offset_x:+.3f}",
			f"offset_y: {target.offset_y:+.3f}",
			f"confidence: {target.confidence:.2f}",
			f"detection_width: {target.detection_width:.0f} px",
			f"detection_height: {target.detection_height:.0f} px",
			f"timestamp: {target.timestamp:.3f} s",
		]

	y = 42
	for line in lines:
		cv2.putText(
			panel,
			line,
			(24, y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.65,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)
		y += 32

	return _label_tile(panel, "5 TargetEstimate")


def _make_flow_result_panel(flow_debug: dict) -> np.ndarray:
	"""
	Text panel for FlowResult only.
	"""
	panel_width = 420
	panel_height = 280

	panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

	flow = flow_debug.get("result", None)
	roi = flow_debug.get("roi", None)
	message = flow_debug.get("message", "")

	lines = [
		"FlowResult:",
		"",
	]

	if flow is None:
		lines += [
			"no flow object",
		]
	else:
		lines += [
			f"valid: {flow.valid}",
			f"mean_flow_x: {flow.mean_flow_x:+.2f} px/s",
			f"mean_flow_y: {flow.mean_flow_y:+.2f} px/s",
			f"divergence: {flow.divergence:+.4f} 1/s",
			f"raw divergence: {flow_debug.get('raw_divergence', 0.0):+.4f} 1/s",
			f"roi: {roi}",
			f"timestamp: {flow.timestamp:.3f} s",
		]

	if message:
		lines += [
			"",
			f"message: {message}",
		]

	y = 42
	for line in lines:
		cv2.putText(
			panel,
			line,
			(24, y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.62,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)
		y += 30

	return _label_tile(panel, "6 FlowResult")

def _make_combined_canvas(
	target_debug: dict,
	flow_debug: dict,
	mode_text: str,
) -> np.ndarray:
	"""
	Build a 2x3 visualization canvas:

	1. BGR frame
	2. Target detection
	3. ROI flow arrows
	4. ROI divergence heatmap
	5. TargetEstimate
	6. FlowResult
	"""
	frame = target_debug.get("frame")
	contour = target_debug.get("contour")
	target = target_debug.get("target")

	flow_px_s = flow_debug.get("flow_px_s")
	divergence_field = flow_debug.get("divergence_field")
	roi = flow_debug.get("roi")

	target_result = _draw_detection_debug(frame, target, contour)

	if frame is not None and flow_px_s is not None:
		flow_arrows = _draw_flow_arrows(frame, flow_px_s, roi=roi)
	else:
		flow_arrows = frame.copy() if frame is not None else None

	if divergence_field is not None:
		divergence_heatmap = _divergence_to_heatmap(divergence_field)
	else:
		divergence_heatmap = None

	target_panel = _make_target_estimate_panel(target_debug)
	flow_panel = _make_flow_result_panel(flow_debug)

	tiles = [
		("1 BGR frame", _to_bgr_for_display(frame)),
		("2 Target detection", _to_bgr_for_display(target_result)),
		("3 ROI flow arrows", _to_bgr_for_display(flow_arrows)),
		("4 ROI divergence heatmap", _to_bgr_for_display(divergence_heatmap)),
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
	Combined visual test.

	Keyboard controls:

		q or Esc  -> quit
		n         -> toggle target on/off
		d         -> toggle distractors on/off
		c         -> toggle close-range target mode
		s         -> save current preview image
		r         -> reset optical-flow estimator memory
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
	)

	flow_estimator = OpticalFlowEstimator(
		require_target_roi=True,
		roi_margin_fraction=0.25,
		min_roi_size_px=32,
		divergence_smoothing=0.6,
		store_debug=True,
	)

	with_target = True
	with_distractors = True
	close_range_mode = False

	step = 0
	last_canvas = None

	window_name = "Target acquisition + optical flow debug"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	print("")
	print("Target acquisition + optical flow debug")
	print("---------------------------------------")
	print("q or Esc  -> quit")
	print("n         -> toggle target on/off")
	print("d         -> toggle distractors on/off")
	print("c         -> toggle close-range target mode")
	print("r         -> reset optical-flow estimator")
	print("s         -> save current preview image")
	print("")

	while True:
		timestamp = step * dt

		target_radius = 190 if close_range_mode else None

		frame = _make_dummy_frame(
			width=width,
			height=height,
			step=step,
			with_target=with_target,
			with_distractors=with_distractors,
			target_radius=target_radius,
		)

		# ------------------------------------------------------------
		# Final architecture step 1:
		# Target acquisition runs first.
		# ------------------------------------------------------------
		target_debug = target_detector.process_debug(
			frame,
			timestamp=timestamp,
		)

		target = target_debug["target"]

		# ------------------------------------------------------------
		# Final architecture step 2:
		# Optical flow receives the target and computes divergence only
		# on the detected target ROI.
		# ------------------------------------------------------------
		flow_estimator.update(
			frame,
			timestamp,
			target=target,
		)

		flow_debug = flow_estimator.last_debug_data()

		mode_text = (
			f"target={'ON' if with_target else 'OFF'} | "
			f"distractors={'ON' if with_distractors else 'OFF'} | "
			f"close_range={'ON' if close_range_mode else 'OFF'}"
		)

		canvas = _make_combined_canvas(
			target_debug=target_debug,
			flow_debug=flow_debug,
			mode_text=mode_text,
		)

		cv2.imshow(window_name, canvas)
		last_canvas = canvas

		key = cv2.waitKey(40) & 0xFF

		if key in (ord("q"), 27):
			break

		if key == ord("n"):
			with_target = not with_target
			flow_estimator.reset()
			step = 0
			continue

		if key == ord("d"):
			with_distractors = not with_distractors

		if key == ord("c"):
			close_range_mode = not close_range_mode
			flow_estimator.reset()
			step = 0
			continue

		if key == ord("r"):
			flow_estimator.reset()
			step = 0
			continue

		if key == ord("s") and last_canvas is not None:
			cv2.imwrite("optFlow_targetAcqu_debug_preview.png", last_canvas)
			print("Saved: optFlow_targetAcqu_debug_preview.png")

		step += 1

	cv2.destroyAllWindows()


if __name__ == "__main__":
	test()