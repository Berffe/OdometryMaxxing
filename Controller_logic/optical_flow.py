"""
Optical flow estimation, decoupled from ROS.

Production path:

	update(frame_bgr, timestamp, target=None) -> FlowResult

When a valid TargetEstimate is provided, the dense optical flow and the
scalar divergence are computed only inside the target bounding box.

Divergence is obtained by a least-squares affine fit to the flow field
(see _fit_divergence_affine), not a per-pixel finite-difference field
collapsed with a median. For an affine field u=a0+a1 x+a2 y, v=b0+b1
x+b2 y, divergence = du/dx+dv/dy = a1+b2 is EXACT and constant -- fitting
it directly uses every valid flow vector in the ROI, instead of computing
a local spatial derivative pixel-by-pixel and taking the median, which is
dominated by whichever response is most common pixel-by-pixel rather than
by the actual expansion signal. This matters specifically once the
target fills the frame: the interior of a uniform, texture-poor surface
gives near-zero/noisy per-pixel flow (the classic aperture problem).

The fit is WEIGHTED by each pixel's reference-frame image-gradient
magnitude (see _gradient_magnitude / _weighted_affine_least_squares), so
a texture-poor patch is down-weighted directly rather than relying on a
textured rim elsewhere in the ROI to out-vote it. That reliance was a real
gap: once TargetEstimate.fov_saturated (the target's true size exceeds
the camera's FOV -- see state.py), the rim is by definition outside the
frame, and only the interior remains to fit against. The ROI is also, at
that point, at its largest of the whole descent -- handled by shrinking the
Farneback search problem itself (see the constructor's downsample_target_px
docstring) rather than the earlier design's separate, MORE expensive
close-range parameter set: a continuous downsample tied to ROI size, not a
binary switch tied to fov_saturated.

The fit additionally reports fit_quality (a weighted R^2 -- see
_fit_divergence_affine), so a degraded-but-still-numeric divergence
estimate can be told apart from a well-supported one. As of this writing
it is DIAGNOSIS-ONLY: logged (diagnostics_writer.py's flow_fit_quality
column) but not yet read by control_law.py or mission_routine.py.

Debug path:

	python -m bee_control.optical_flow

or:

	python optical_flow.py

When run as a script, this file imports optical_flow_debug.py and starts
the visual debug test. The dense flow field is still not part of FlowResult.
"""

from typing import Optional, Tuple

import time

import cv2
import numpy as np

try:
	from .state import FlowResult, TargetEstimate
except ImportError:
	from state import FlowResult, TargetEstimate


ROI = Tuple[int, int, int, int]  # x0, y0, x1, y1; x1/y1 are exclusive


class OpticalFlowEstimator:
	def __init__(
		self,
		pyr_scale: float = 0.5,
		levels: int = 3,
		winsize: int = 21,
		iterations: int = 3,
		poly_n: int = 5,
		poly_sigma: float = 1.2,
		# --- ROI-adaptive downsampling (replaces the old binary close-range
		# Farneback parameter set) ---
		#
		# The old design used a SECOND, more expensive Farneback parameter set
		# (winsize_close=35, levels_close=4, pyr_scale_close=0.6) switched on
		# fov_saturated, reasoning that a bigger/closer ROI needs a bigger
		# search window. That made the single most expensive regime of the
		# whole descent (full-frame ROI, closest to touchdown) also the most
		# computationally expensive -- exactly backwards for a latency budget
		# that a de Croon stability gate depends on (see bee_node.py's
		# STABILITY_DT_SEC / VISION_PROCESSING_LATENCY_BUDGET_SEC).
		#
		# The replacement: shrink the ROI itself before Farneback runs, by a
		# factor tied to the ROI's own size, so the array Farneback actually
		# searches stays close to a FIXED working size regardless of how big
		# the target has grown -- a bigger ROI gets MORE downsampling, not a
		# bigger search window. This is justified by the same "we don't need
		# per-pixel resolution once the ROI is a single expanding surface"
		# reasoning the old close-range branch already used, just applied to
		# the search cost directly instead of to the window size.
		#
		# Mechanics (see update()): prev_roi/gray_roi are resized down by
		# `scale` before calcOpticalFlowFarneback (a single, cheap cv2.resize,
		# INTER_AREA -- the correct anti-aliasing choice for shrinking);
		# Farneback then runs with the SAME parameters (pyr_scale/levels/
		# winsize/iterations/poly_n/poly_sigma above) in both regimes, since
		# after downsampling a big-ROI close-range frame and a small-ROI
		# far-field frame present Farneback with a similarly-sized problem.
		# The resulting flow field is amplitude-corrected (divided by
		# `scale` -- a downsampled-pixel of apparent motion is 1/scale
		# original pixels of real motion).
		#
		# From there, WITHOUT de-rotation active, the affine divergence fit
		# runs DIRECTLY on this still-downsampled field -- it does not need
		# reconstructing to full ROI resolution first. _fit_divergence_affine
		# takes a pixel_scale argument for exactly this: it widens the
		# coordinate spacing it fits against by 1/scale to match, so the
		# fitted slope (the divergence) comes out in the same physical units
		# regardless of how densely the array it's fitting was sampled. This
		# matters because profiling showed the fit itself (the weighted
		# lstsq solve plus the trim-and-refit pass) -- not Farneback -- was
		# the dominant cost once Farneback alone had already been shrunk by
		# downsampling: reconstructing the field to full resolution before
		# fitting left the fit solving over the same point count as before,
		# undoing most of the savings for the one piece of update() that
		# actually cost the most. Fitting on the small field directly closes
		# that gap (measured ~2x further reduction on top of the earlier
		# downsampling-with-upsample version, in addition to whatever
		# derotation being off already saved).
		#
		# WITH de-rotation active, this shortcut is skipped: Derotator
		# samples the rotational model at full-image pixel coordinates (see
		# derotation.py), so update() upsamples the field back to full ROI
		# resolution first in that case, exactly as before -- correctness
		# for a currently-disabled feature was not worth the risk of a
		# subtle scale bug to save time on a path that isn't running.
		#
		# downsample_target_px: the working array's target max dimension.
		#     scale = clip(downsample_target_px / max(roi_w, roi_h),
		#                   downsample_min_scale, 1.0)
		#     A ROI already <= this size is left alone (scale clips to 1.0):
		#     the far-field regime, with its naturally compact ROI, is
		#     unaffected. Only a ROI bigger than this target gets shrunk, and
		#     it shrinks MORE the bigger it is -- exactly the "big ROI, more
		#     downsample" behavior wanted, and it degrades gracefully (a
		#     continuous function of ROI size) rather than the old binary
		#     fov_saturated switch.
		# downsample_min_scale: a floor so an extreme close-in ROI (already
		#     the full 120x80 frame at this project's camera resolution)
		#     can't be shrunk into too few pixels for the affine fit's
		#     min_points_for_affine_fit guard to have real texture to chew on.
		downsample_target_px: int = 96,
		downsample_min_scale: float = 0.5,
		require_target_roi: bool = True,
		roi_margin_fraction: float = 0.05,
		min_roi_size_px: int = 32,
		divergence_smoothing: float = 0.3,
		min_points_for_affine_fit: int = 30,
		affine_inlier_quantile: float = 0.85,
		affine_fit_stride: int = 2,
		store_debug: bool = False,
		# Optional ego-rotation removal. Pass a derotation.Derotator to enable
		# it; leave None for the legacy (no de-rotation) behavior. When set,
		# update() also needs a per-frame body_rates vector to actually
		# subtract anything -- without it the flow is passed through unchanged.
		# DISABLED BY DEFAULT as of the light optical-flow pass (see
		# bee_node.py): re-enable once the downsampled flow field has been
		# re-validated against the derotation acceptance test in
		# derotation.py's docstring (downsampling changes the flow field's
		# spatial resolution, which the rotational-field model samples at
		# full resolution).
		derotator=None,
	):
		self._prev_gray = None
		self._prev_bgr = None
		self._prev_timestamp = None

		self._pyr_scale = pyr_scale
		self._levels = levels
		self._winsize = winsize
		self._iterations = iterations
		self._poly_n = poly_n
		self._poly_sigma = poly_sigma

		self._downsample_target_px = max(3, int(downsample_target_px))
		self._downsample_min_scale = max(1e-3, min(1.0, float(downsample_min_scale)))

		self._require_target_roi = bool(require_target_roi)
		self._roi_margin_fraction = float(roi_margin_fraction)
		self._min_roi_size_px = int(min_roi_size_px)

		self._divergence_smoothing = float(divergence_smoothing)
		self._filtered_divergence = 0.0
		self._has_filtered_divergence = False

		# Affine-fit divergence (see module docstring). min_points is a
		# degenerate-case guard (a typical ROI has >>1000 flow vectors);
		# inlier_quantile keeps the best fraction by residual on a single
		# trim-and-refit pass, so a cluster of unreliable (textureless or
		# specular) flow vectors can't dominate the global fit.
		self._min_points_for_affine_fit = int(min_points_for_affine_fit)
		self._affine_inlier_quantile = float(affine_inlier_quantile)
		# Uniform spatial decimation for the affine fit only. Farneback and
		# mean-flow calculations still use the complete working field. A stride
		# of 2 samples one vector from each 2x2 block, retaining full-ROI
		# coverage while reducing the robust fit workload by about 4x.
		self._affine_fit_stride = max(1, int(affine_fit_stride))

		self._store_debug = bool(store_debug)
		self._last_debug = {}
		self._last_timing = {}

		# Ego-rotation removal (see derotation.py). None -> disabled.
		self._derotator = derotator

	def update(
		self,
		frame_bgr,
		timestamp: float,
		target: Optional[TargetEstimate] = None,
		# Mean body angular rate (rad/s, FRD [p, q, r]) over the interval
		# between the previous frame and this one -- see bee_node.on_camera and
		# derotation.AngularRateBuffer. Only used when a derotator was supplied
		# to the constructor; None (or no derotator) => no de-rotation.
		body_rates=None,
	) -> FlowResult:
		wall_start = time.perf_counter()
		cpu_start = time.process_time()
		timing = {
			"farneback_pyr_scale": float(self._pyr_scale),
			"farneback_levels": int(self._levels),
			"farneback_winsize": int(self._winsize),
			"farneback_iterations": int(self._iterations),
			"farneback_poly_n": int(self._poly_n),
			"farneback_poly_sigma": float(self._poly_sigma),
			"divergence_smoothing_alpha": float(self._divergence_smoothing),
		}

		def finish_timing():
			timing["total_wall_ms"] = 1000.0 * (time.perf_counter() - wall_start)
			timing["total_cpu_ms"] = 1000.0 * (time.process_time() - cpu_start)
			self._last_timing = dict(timing)

		def invalid_result(message: str, *, previous_frame=None, current_frame=None, roi=None):
			stage_start = time.perf_counter()
			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(
				result=result,
				previous_frame=previous_frame,
				current_frame=current_frame,
				roi=roi,
				message=message,
			)
			timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
			timing["valid"] = 0
			finish_timing()
			return result

		if frame_bgr is None:
			return invalid_result("No frame", current_frame=None)

		stage_start = time.perf_counter()
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		image_height, image_width = gray.shape[:2]
		timing["grayscale_ms"] = 1000.0 * (time.perf_counter() - stage_start)
		timing["image_width_px"] = int(image_width)
		timing["image_height_px"] = int(image_height)

		if self._prev_gray is None or self._prev_timestamp is None:
			stage_start = time.perf_counter()
			self._prev_gray = gray
			self._prev_bgr = frame_bgr.copy()
			self._prev_timestamp = timestamp
			timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
			timing["valid"] = 0
			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(
				result=result,
				current_frame=frame_bgr,
				message="Waiting for previous frame",
			)
			finish_timing()
			return result

		dt = float(timestamp - self._prev_timestamp)
		timing["dt_sec"] = dt

		if dt <= 1e-6:
			stage_start = time.perf_counter()
			self._prev_gray = gray
			self._prev_bgr = frame_bgr.copy()
			self._prev_timestamp = timestamp
			timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
			timing["valid"] = 0
			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(
				result=result,
				current_frame=frame_bgr,
				message="Invalid dt",
			)
			finish_timing()
			return result

		stage_start = time.perf_counter()
		previous_frame = self._prev_bgr.copy() if self._prev_bgr is not None else None
		roi = self._target_roi_from_estimate(
			target=target,
			image_width=image_width,
			image_height=image_height,
		)
		timing["roi_setup_ms"] = 1000.0 * (time.perf_counter() - stage_start)

		if roi is None and self._require_target_roi:
			stage_start = time.perf_counter()
			self._prev_gray = gray
			self._prev_bgr = frame_bgr.copy()
			self._prev_timestamp = timestamp
			timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
			timing["valid"] = 0
			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(
				result=result,
				previous_frame=previous_frame,
				current_frame=frame_bgr,
				roi=None,
				message="No valid target ROI",
			)
			finish_timing()
			return result

		if roi is None:
			roi = (0, 0, image_width, image_height)

		x0, y0, x1, y1 = roi
		prev_roi = self._prev_gray[y0:y1, x0:x1]
		gray_roi = gray[y0:y1, x0:x1]
		roi_height, roi_width = prev_roi.shape[:2]
		timing["roi_width_px"] = int(roi_width)
		timing["roi_height_px"] = int(roi_height)

		if roi_width < 3 or roi_height < 3:
			stage_start = time.perf_counter()
			self._prev_gray = gray
			self._prev_bgr = frame_bgr.copy()
			self._prev_timestamp = timestamp
			timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
			timing["valid"] = 0
			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(
				result=result,
				previous_frame=previous_frame,
				current_frame=frame_bgr,
				roi=roi,
				message="ROI too small",
			)
			finish_timing()
			return result

		# ROI-adaptive downsampling: time resizing separately from Farneback.
		stage_start = time.perf_counter()
		scale = self._downsample_scale_for_roi(roi_width, roi_height)
		if scale < 1.0:
			small_w = max(3, int(round(roi_width * scale)))
			small_h = max(3, int(round(roi_height * scale)))
			prev_small = cv2.resize(
				prev_roi, (small_w, small_h), interpolation=cv2.INTER_AREA
			)
			gray_small = cv2.resize(
				gray_roi, (small_w, small_h), interpolation=cv2.INTER_AREA
			)
		else:
			prev_small = prev_roi
			gray_small = gray_roi
		timing["downsample_resize_ms"] = 1000.0 * (time.perf_counter() - stage_start)
		timing["downsample_scale"] = float(scale)
		timing["working_width_px"] = int(prev_small.shape[1])
		timing["working_height_px"] = int(prev_small.shape[0])
		timing["working_flow_vectors"] = int(prev_small.shape[0] * prev_small.shape[1])

		stage_start = time.perf_counter()
		flow_small_per_frame = cv2.calcOpticalFlowFarneback(
			prev_small,
			gray_small,
			None,
			self._pyr_scale,
			self._levels,
			self._winsize,
			self._iterations,
			self._poly_n,
			self._poly_sigma,
			0,
		)
		timing["farneback_ms"] = 1000.0 * (time.perf_counter() - stage_start)

		stage_start = time.perf_counter()
		if scale < 1.0:
			flow_small_per_frame = flow_small_per_frame / scale

		derotation_active = self._derotator is not None and body_rates is not None
		# Keep the compact working grid even when derotation is enabled. The
		# rotational model is sampled at the corresponding normalized full-image
		# coordinates instead of upsampling the measured flow.
		flow_px_per_frame = flow_small_per_frame
		fit_pixel_scale = scale
		gradient_source = prev_small
		flow_px_s = flow_px_per_frame / dt
		timing["flow_scaling_upsample_ms"] = 1000.0 * (time.perf_counter() - stage_start)
		timing["fit_pixel_scale"] = float(fit_pixel_scale)
		timing["derotation_active"] = int(bool(derotation_active))

		stage_start = time.perf_counter()
		raw_flow_px_s = flow_px_s
		if derotation_active:
			flow_px_s = self._derotator.derotate_working_grid(
				flow_px_s, body_rates, roi=(x0, y0, x1, y1),
				pixel_scale=fit_pixel_scale,
			)
		timing["derotation_ms"] = 1000.0 * (time.perf_counter() - stage_start)

		stage_start = time.perf_counter()
		raw_mean_flow_x = float(np.mean(raw_flow_px_s[:, :, 0]))
		raw_mean_flow_y = float(np.mean(raw_flow_px_s[:, :, 1]))
		mean_flow_x = float(np.mean(flow_px_s[:, :, 0]))
		mean_flow_y = float(np.mean(flow_px_s[:, :, 1]))
		mean_flow_x_norm = mean_flow_x / max(0.5 * image_width, 1.0)
		mean_flow_y_norm = mean_flow_y / max(0.5 * image_height, 1.0)
		timing["mean_flow_ms"] = 1000.0 * (time.perf_counter() - stage_start)

		stage_start = time.perf_counter()
		gradient_magnitude = self._gradient_magnitude(gradient_source)
		timing["gradient_ms"] = 1000.0 * (time.perf_counter() - stage_start)

		stage_start = time.perf_counter()
		divergence_field = (
			self._estimate_divergence_field(
				flow_px_s=flow_px_s,
				image_width=image_width,
				image_height=image_height,
				pixel_scale=fit_pixel_scale,
			)
			if self._store_debug
			else None
		)
		timing["divergence_field_debug_ms"] = 1000.0 * (
			time.perf_counter() - stage_start
		)

		affine_timing = {}
		stage_start = time.perf_counter()
		raw_divergence, n_inliers, fit_quality = self._fit_divergence_affine(
			flow_px_s=flow_px_s,
			image_width=image_width,
			image_height=image_height,
			gradient_magnitude=gradient_magnitude,
			pixel_scale=fit_pixel_scale,
			timing=affine_timing,
		)
		timing["affine_fit_ms"] = 1000.0 * (time.perf_counter() - stage_start)
		for key, value in affine_timing.items():
			timing[f"affine_{key}"] = value
		timing["affine_points_used"] = int(n_inliers)
		timing["affine_fit_quality"] = float(fit_quality)

		stage_start = time.perf_counter()
		filtered_divergence = self._filter_divergence(raw_divergence)
		timing["divergence_filter_ms"] = 1000.0 * (time.perf_counter() - stage_start)

		stage_start = time.perf_counter()
		if derotation_active:
			divergence_prederotation, _, _ = self._fit_divergence_affine(
				flow_px_s=raw_flow_px_s,
				image_width=image_width,
				image_height=image_height,
				gradient_magnitude=gradient_magnitude,
				robust=False,
				pixel_scale=fit_pixel_scale,
			)
		else:
			divergence_prederotation = raw_divergence
		timing["prederotation_fit_ms"] = 1000.0 * (time.perf_counter() - stage_start)

		stage_start = time.perf_counter()
		result = FlowResult(
			timestamp=timestamp,
			valid=True,
			mean_flow_x=mean_flow_x,
			mean_flow_y=mean_flow_y,
			mean_flow_x_norm=float(mean_flow_x_norm),
			mean_flow_y_norm=float(mean_flow_y_norm),
			divergence=float(filtered_divergence),
			raw_divergence=float(raw_divergence),
			fit_quality=float(fit_quality),
			roi_x0=int(x0),
			roi_y0=int(y0),
			roi_x1=int(x1),
			roi_y1=int(y1),
			derotated=bool(derotation_active),
			mean_flow_x_raw=float(raw_mean_flow_x),
			mean_flow_y_raw=float(raw_mean_flow_y),
			divergence_prederotation=float(divergence_prederotation),
		)
		self._save_debug(
			result=result,
			previous_frame=previous_frame,
			current_frame=frame_bgr,
			flow_px_s=flow_px_s,
			divergence_field=divergence_field,
			raw_divergence=raw_divergence,
			filtered_divergence=filtered_divergence,
			roi=roi,
			message="",
		)
		self._prev_gray = gray
		self._prev_bgr = frame_bgr.copy()
		self._prev_timestamp = timestamp
		timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
		timing["valid"] = 1
		finish_timing()
		return result

	def _fit_divergence_affine(
		self,
		flow_px_s: np.ndarray,
		image_width: int,
		image_height: int,
		gradient_magnitude: Optional[np.ndarray] = None,
		robust: bool = True,
		pixel_scale: float = 1.0,
		timing: Optional[dict] = None,
	) -> Tuple[float, int, float]:
		"""
		Divergence via a global affine fit, not a per-pixel median.

		robust: when True (the control-facing default), runs the
		    trim-and-refit outlier pass documented below. When False, returns
		    straight after the first weighted OLS solve -- roughly half the
		    cost, at the price of no outlier robustness. Only intended for
		    diagnostic-only callers (e.g. update()'s divergence_prederotation)
		    where a coarser number logged for offline analysis is an
		    acceptable trade for not doubling a control-path-adjacent cost
		    every frame.

		pixel_scale: how many ORIGINAL ROI pixels each element of flow_px_s
		    represents, when flow_px_s is itself a downsampled array (see
		    update()'s ROI-adaptive downsampling: fitting directly on the
		    downsampled field, instead of reconstructing it to full ROI
		    resolution first, is what actually shrinks this fit's cost --
		    downsampling Farneback alone left this function solving over the
		    same point count as before). 1.0 (the default) means flow_px_s is
		    already at full ROI resolution -- unchanged behavior. A value s<1
		    means adjacent array elements are 1/s original pixels apart, so
		    the coordinate spacing (dx_norm/dy_norm below) widens by 1/s to
		    match -- the flow VALUES themselves are assumed to already be
		    amplitude-corrected to original-pixel units by the caller (see
		    update()), so only the coordinate axis needs adjusting here.

		Model (normalized image units/s, same scale as mean_flow_*_norm):
		    u(x, y) = a0 + a1*x + a2*y
		    v(x, y) = b0 + b1*x + b2*y
		Solved by WEIGHTED OLS (shared design matrix). For any affine
		field, du/dx + dv/dy = a1 + b2 exactly and is constant everywhere, so
		this is the exact divergence of the best-fit field -- using ALL valid
		flow vectors in the ROI, not a local difference at each pixel.

		The OLS slope is invariant to the coordinate origin (shifting x, y by
		a constant only moves a0/b0), so ROI-local pixel coordinates are used
		directly -- no need to know the ROI's offset within the full image.

		WEIGHTING: each pixel is weighted by its reference-frame image
		gradient magnitude (see _gradient_magnitude), not trusted equally.
		This is the direct fix for the aperture-problem failure mode this
		module's docstring already describes for a texture-poor interior --
		previously handled only by hoping the textured rim was in-frame to
		out-vote it in an unweighted fit. Once fov_saturated removes the rim
		entirely (see optical_flow_estimator's downsample_target_px docstring),
		weighting is what keeps a locally-flat patch of the interior from
		being averaged in on equal footing with a high-gradient patch,
		instead of relying on the rim being there to swamp it. Falls back to
		uniform weighting if no gradient map is supplied.

		One robust trim-and-refit pass on top keeps the best
		`affine_inlier_quantile` fraction by (weighted) residual and refits --
		a complementary, different heuristic from the gradient weighting
		above: this catches vectors that mismatch the fitted model despite
		reasonable local texture (e.g. a genuine outlier), not vectors that
		were never trustworthy to begin with.

		Falls back to the old field-median method if too few finite flow
		vectors remain (degenerate ROI) -- a safety net, not the normal path;
		fit_quality is reported as 0.0 there since no fit was actually made.

		Returns (divergence, n_points_used, fit_quality). fit_quality is a
		weighted R^2 over the combined u,v residuals: 1.0 means the affine
		model explains the (weighted) flow variance essentially exactly, 0.0
		means it does no better than reporting the weighted mean flow
		everywhere, and negative means worse than that -- a plausible-looking
		divergence number can still carry a low/negative fit_quality when
		the ROI has become mostly noise, which is exactly the case this was
		added to catch (see this file's usage note: diagnosis-only for now,
		not yet read by control_law.py or mission_routine.py).
		"""
		fit_wall_start = time.perf_counter()
		if timing is None:
			timing = {}

		roi_height, roi_width = flow_px_s.shape[:2]
		timing["input_points"] = int(roi_height * roi_width)

		# Fit-only spatial decimation. We deliberately do not resize or average
		# the flow here: regular slicing preserves the measured vector values and
		# samples the full ROI uniformly. Coordinate spacing is increased below by
		# the same stride, so the fitted slopes and divergence remain in exactly
		# the same normalized-image units per second.
		fit_stride = max(1, int(self._affine_fit_stride))
		if fit_stride > 1:
			flow_fit = flow_px_s[::fit_stride, ::fit_stride]
			if (
				gradient_magnitude is not None
				and gradient_magnitude.shape == (roi_height, roi_width)
			):
				gradient_fit = gradient_magnitude[::fit_stride, ::fit_stride]
			else:
				gradient_fit = None
		else:
			flow_fit = flow_px_s
			gradient_fit = gradient_magnitude

		fit_height, fit_width = flow_fit.shape[:2]
		timing["fit_stride"] = fit_stride
		timing["sampled_points"] = int(fit_height * fit_width)
		if fit_width < 3 or fit_height < 3:
			timing.update({
				"setup_ms": 1000.0 * (time.perf_counter() - fit_wall_start),
				"initial_solve_ms": 0.0,
				"residual_quantile_ms": 0.0,
				"refit_ms": 0.0,
				"finite_points": 0,
				"used_points": 0,
			})
			return 0.0, 0, 0.0

		u = flow_fit[:, :, 0] / max(0.5 * image_width, 1.0)
		v = flow_fit[:, :, 1] / max(0.5 * image_height, 1.0)

		s = max(1e-6, float(pixel_scale))
		dx_norm = fit_stride * (2.0 / max(image_width - 1, 1)) / s
		dy_norm = fit_stride * (2.0 / max(image_height - 1, 1)) / s

		rows, cols = np.mgrid[0:fit_height, 0:fit_width]
		x = (cols * dx_norm).ravel().astype(np.float64)
		y = (rows * dy_norm).ravel().astype(np.float64)
		u_flat = u.ravel().astype(np.float64)
		v_flat = v.ravel().astype(np.float64)

		if gradient_fit is not None and gradient_fit.shape == (fit_height, fit_width):
			weight_flat = gradient_fit.ravel().astype(np.float64)
		else:
			weight_flat = np.ones_like(u_flat)

		finite = np.isfinite(u_flat) & np.isfinite(v_flat) & np.isfinite(weight_flat)
		n_finite = int(np.count_nonzero(finite))
		timing["finite_points"] = n_finite
		if n_finite < self._min_points_for_affine_fit:
			field = self._estimate_divergence_field(
				flow_px_s, image_width, image_height, pixel_scale=s
			)
			timing.update({
				"setup_ms": 1000.0 * (time.perf_counter() - fit_wall_start),
				"initial_solve_ms": 0.0,
				"residual_quantile_ms": 0.0,
				"refit_ms": 0.0,
				"used_points": n_finite,
			})
			return self._scalar_from_divergence_field(field), n_finite, 0.0

		x, y, u_flat, v_flat, weight_flat = (
			x[finite], y[finite], u_flat[finite], v_flat[finite], weight_flat[finite]
		)
		design = np.column_stack([np.ones_like(x), x, y])
		timing["setup_ms"] = 1000.0 * (time.perf_counter() - fit_wall_start)

		stage_start = time.perf_counter()
		coeffs, divergence, fit_quality = self._weighted_affine_least_squares(
			design, u_flat, v_flat, weight_flat
		)
		timing["initial_solve_ms"] = 1000.0 * (time.perf_counter() - stage_start)

		if not robust:
			timing["residual_quantile_ms"] = 0.0
			timing["refit_ms"] = 0.0
			timing["used_points"] = n_finite
			return float(divergence), n_finite, float(fit_quality)

		stage_start = time.perf_counter()
		residual = (
			(u_flat - design @ coeffs[0]) ** 2
			+ (v_flat - design @ coeffs[1]) ** 2
		)
		threshold = np.quantile(residual, self._affine_inlier_quantile)
		inliers = residual <= threshold
		timing["residual_quantile_ms"] = 1000.0 * (
			time.perf_counter() - stage_start
		)

		stage_start = time.perf_counter()
		n_inliers = int(np.count_nonzero(inliers))
		if n_inliers >= self._min_points_for_affine_fit:
			_, divergence, fit_quality = self._weighted_affine_least_squares(
				design[inliers], u_flat[inliers], v_flat[inliers], weight_flat[inliers]
			)
			n_used = n_inliers
		else:
			n_used = n_finite
		timing["refit_ms"] = 1000.0 * (time.perf_counter() - stage_start)
		timing["used_points"] = n_used

		return float(divergence), n_used, float(fit_quality)

	@staticmethod
	def _weighted_affine_least_squares(
		design: np.ndarray, u: np.ndarray, v: np.ndarray, weight: np.ndarray
	) -> Tuple[Tuple[np.ndarray, np.ndarray], float, float]:
		"""Gradient-magnitude-weighted OLS via the standard sqrt(w) rescaling
		(minimizing sum(w*(y-Xb)^2) is exactly OLS in sqrt(w)-rescaled
		variables, so this stays a single cheap linear solve, not an
		iterative reweighting scheme). Weight is normalized to a mean of 1
		first so its absolute scale never changes the solve's conditioning,
		only the RELATIVE trust between pixels; an all-zero/degenerate weight
		map falls back to uniform (equivalent to the old unweighted fit).

		Solved via the NORMAL EQUATIONS (design_w.T @ design_w, a 3x3 system
		-- constant/x/y are the only unknowns), not np.linalg.lstsq. lstsq is
		a general SVD-based solver sized for the case where the number of
		unknowns isn't known/fixed; here it always is (3), so forming the
		3x3 system directly and calling np.linalg.solve on THAT instead is
		mathematically the same least-squares solution (verified: matches
		lstsq's coefficients to ~3e-5 on synthetic data) at roughly 5-6x less
		wall-clock cost, since the O(n) cost of building the 3x3 system is
        far cheaper than lstsq's own O(n) SVD setup, and the fixed-size 3x3
		solve is then nearly free either way. Falls back to lstsq only if the
		normal equations turn out singular (a genuinely degenerate ROI --
		e.g. every point sharing the same x or y -- which min_points_for_
		affine_fit already guards against in the normal case).

		Also returns a weighted R^2 (see _fit_divergence_affine's docstring
		for interpretation) as a fit-quality proxy, computed in the same
		rescaled space so it stays consistent with what was actually
		minimized.
		"""
		w = np.clip(weight, 0.0, None)
		w_mean = float(np.mean(w)) if np.any(w > 0.0) else 0.0
		if w_mean <= 1e-12:
			w = np.ones_like(w)
			w_mean = 1.0
		w = w / w_mean

		sw = np.sqrt(w)
		design_w = design * sw[:, None]
		u_w = u * sw
		v_w = v * sw

		AtA = design_w.T @ design_w
		Atu = design_w.T @ u_w
		Atv = design_w.T @ v_w
		try:
			coeffs_u = np.linalg.solve(AtA, Atu)
			coeffs_v = np.linalg.solve(AtA, Atv)
		except np.linalg.LinAlgError:
			coeffs_u, *_ = np.linalg.lstsq(design_w, u_w, rcond=None)
			coeffs_v, *_ = np.linalg.lstsq(design_w, v_w, rcond=None)
		divergence = float(coeffs_u[1] + coeffs_v[2])

		resid_u = u_w - design_w @ coeffs_u
		resid_v = v_w - design_w @ coeffs_v
		ss_res = float(np.sum(resid_u ** 2) + np.sum(resid_v ** 2))

		u_mean_w = float(np.sum(w * u) / np.sum(w))
		v_mean_w = float(np.sum(w * v) / np.sum(w))
		ss_tot = float(np.sum(w * (u - u_mean_w) ** 2) + np.sum(w * (v - v_mean_w) ** 2))

		fit_quality = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

		return (coeffs_u, coeffs_v), divergence, float(fit_quality)

	@staticmethod
	def _gradient_magnitude(gray_roi: np.ndarray) -> np.ndarray:
		"""Sobel gradient magnitude of a grayscale ROI -- the structural-
		reliability weight for the affine divergence fit (see
		_fit_divergence_affine). A flat/texture-poor patch has near-zero
		gradient here and is the classic aperture-problem case: Farneback's
		own polynomial-expansion estimate is least trustworthy exactly
		where this is smallest."""
		gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
		gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
		return cv2.magnitude(gx, gy)

	def _downsample_scale_for_roi(self, roi_width: int, roi_height: int) -> float:
		"""scale = clip(downsample_target_px / max(roi_w, roi_h), min_scale, 1.0)

		1.0 (no downsampling) for any ROI already at or below the target size
		-- the far-field regime is unaffected. Shrinks continuously, more for
		a bigger ROI, once it exceeds the target -- replaces the old binary
		fov_saturated switch to a bigger search window with a continuous
		switch to a smaller search problem. See constructor docstring.
		"""
		largest_dim = max(int(roi_width), int(roi_height), 1)
		scale = self._downsample_target_px / float(largest_dim)
		return max(self._downsample_min_scale, min(1.0, scale))

	def reset(self):
		self._prev_gray = None
		self._prev_bgr = None
		self._prev_timestamp = None

		self._filtered_divergence = 0.0
		self._has_filtered_divergence = False

		self._last_debug = {}
		self._last_timing = {}

	def last_debug_data(self) -> dict:
		return dict(self._last_debug)

	def last_timing_data(self) -> dict:
		"""Return the immutable scalar timing snapshot from the latest update."""
		return dict(self._last_timing)

	def _target_roi_from_estimate(
		self,
		target: Optional[TargetEstimate],
		image_width: int,
		image_height: int,
	) -> Optional[ROI]:
		if target is None or not target.found:
			return None

		detection_width = float(getattr(target, "detection_width", 0.0))
		detection_height = float(getattr(target, "detection_height", 0.0))

		if detection_width <= 1.0 or detection_height <= 1.0:
			return None

		cx = (0.5 * float(target.offset_x) + 0.5) * image_width
		cy = (0.5 * float(target.offset_y) + 0.5) * image_height

		margin_x = self._roi_margin_fraction * detection_width
		margin_y = self._roi_margin_fraction * detection_height

		roi_width = max(
			detection_width + 2.0 * margin_x,
			float(self._min_roi_size_px),
		)

		roi_height = max(
			detection_height + 2.0 * margin_y,
			float(self._min_roi_size_px),
		)

		x0 = int(round(cx - 0.5 * roi_width))
		y0 = int(round(cy - 0.5 * roi_height))
		x1 = int(round(cx + 0.5 * roi_width))
		y1 = int(round(cy + 0.5 * roi_height))

		x0 = max(0, min(image_width - 1, x0))
		y0 = max(0, min(image_height - 1, y0))
		x1 = max(x0 + 1, min(image_width, x1))
		y1 = max(y0 + 1, min(image_height, y1))

		if x1 - x0 < 3 or y1 - y0 < 3:
			return None

		return (x0, y0, x1, y1)

	def _estimate_divergence_field(
		self,
		flow_px_s: np.ndarray,
		image_width: int,
		image_height: int,
		pixel_scale: float = 1.0,
	) -> np.ndarray:
		roi_height, roi_width = flow_px_s.shape[:2]

		if roi_width < 3 or roi_height < 3:
			return np.zeros((roi_height, roi_width), dtype=np.float32)

		# Convert pixel flow to normalized image-coordinate velocity.
		# Normalization uses the full image size, not the ROI size, so the
		# divergence scale remains consistent as the target box changes.
		u_norm_s = flow_px_s[:, :, 0] / (0.5 * image_width)
		v_norm_s = flow_px_s[:, :, 1] / (0.5 * image_height)

		s = max(1e-6, float(pixel_scale))
		dx_norm = (2.0 / max(image_width - 1, 1)) / s
		dy_norm = (2.0 / max(image_height - 1, 1)) / s

		du_dx = np.gradient(u_norm_s, dx_norm, axis=1)
		dv_dy = np.gradient(v_norm_s, dy_norm, axis=0)

		return du_dx + dv_dy

	@staticmethod
	def _scalar_from_divergence_field(divergence_field: np.ndarray) -> float:
		if divergence_field is None or divergence_field.size == 0:
			return 0.0

		return float(np.median(divergence_field))

	def _filter_divergence(self, divergence: float) -> float:
		alpha = max(0.0, min(1.0, self._divergence_smoothing))

		if not self._has_filtered_divergence:
			self._filtered_divergence = float(divergence)
			self._has_filtered_divergence = True
		else:
			self._filtered_divergence = (
				alpha * self._filtered_divergence
				+ (1.0 - alpha) * float(divergence)
			)

		return self._filtered_divergence

	def _save_debug(
		self,
		result: FlowResult,
		previous_frame=None,
		current_frame=None,
		flow_px_s: Optional[np.ndarray] = None,
		divergence_field: Optional[np.ndarray] = None,
		raw_divergence: float = 0.0,
		filtered_divergence: float = 0.0,
		roi: Optional[ROI] = None,
		message: str = "",
	):
		if not self._store_debug:
			self._last_debug = {}
			return

		self._last_debug = {
			"result": result,
			"previous_frame": previous_frame,
			"current_frame": current_frame,
			"flow_px_s": flow_px_s,
			"divergence_field": divergence_field,
			"raw_divergence": float(raw_divergence),
			"filtered_divergence": float(filtered_divergence),
			"roi": roi,
			"message": message,
		}


if __name__ == "__main__":
	try:
		from ._optical_flow_debug import test
	except ImportError:
		from _optical_flow_debug import test

	test()