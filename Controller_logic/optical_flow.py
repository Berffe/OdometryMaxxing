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
frame, and only the interior remains to fit against. A separate,
close-range Farneback parameter set (winsize_close/levels_close/
pyr_scale_close) engages on the same fov_saturated signal, since that
regime also has the descent's largest per-frame pixel displacement.

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
		# --- Close-range Farneback parameters (target.fov_saturated) ---
		# The far-field params above are tuned for a small-to-moderate target:
		# modest per-frame pixel displacement, a compact ROI. Once
		# fov_saturated (the target's true size exceeds the camera's FOV --
		# see state.py's TargetEstimate docstring), the ROI is pinned at the
		# full frame, physical closing rate is at its highest of the whole
		# descent (larger per-frame pixel displacement), and there is no
		# longer a textured rim to anchor the divergence fit (see this
		# module's docstring) -- only whatever texture is in the interior.
		# A larger winsize averages over more of that texture per estimate
		# (more robust to a locally-flat patch, at the cost of spatial
		# resolution we don't need once the ROI is a single expanding
		# surface); more levels + a larger pyr_scale (finer-grained pyramid
		# steps) track the larger displacement more robustly. Binary switch
		# on fov_saturated rather than a continuous ramp on area_fraction:
		# it's already a validated, zero-extra-cost signal (see the
		# fov_saturation_vs_divergence.png diagnostic), and this regime is
		# entered once, close to touchdown, not revisited repeatedly --
		# revisit as a smooth ramp only if the step itself turns out to
		# leave a visible mark on the divergence trace.
		winsize_close: int = 35,
		levels_close: int = 4,
		pyr_scale_close: float = 0.6,
		require_target_roi: bool = True,
		roi_margin_fraction: float = 0.05,
		min_roi_size_px: int = 32,
		divergence_smoothing: float = 0.7,
		min_points_for_affine_fit: int = 30,
		affine_inlier_quantile: float = 0.85,
		store_debug: bool = False,
		# Optional ego-rotation removal. Pass a derotation.Derotator to enable
		# it; leave None for the legacy (no de-rotation) behavior. When set,
		# update() also needs a per-frame body_rates vector to actually
		# subtract anything -- without it the flow is passed through unchanged.
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

		self._winsize_close = int(winsize_close)
		self._levels_close = int(levels_close)
		self._pyr_scale_close = float(pyr_scale_close)

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

		self._store_debug = bool(store_debug)
		self._last_debug = {}

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
		if frame_bgr is None:
			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(result=result, current_frame=None)
			return result

		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		image_height, image_width = gray.shape[:2]

		if self._prev_gray is None or self._prev_timestamp is None:
			self._prev_gray = gray
			self._prev_bgr = frame_bgr.copy()
			self._prev_timestamp = timestamp

			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(
				result=result,
				current_frame=frame_bgr,
				message="Waiting for previous frame",
			)

			return result

		dt = float(timestamp - self._prev_timestamp)

		if dt <= 1e-6:
			self._prev_gray = gray
			self._prev_bgr = frame_bgr.copy()
			self._prev_timestamp = timestamp

			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(
				result=result,
				current_frame=frame_bgr,
				message="Invalid dt",
			)

			return result

		previous_frame = self._prev_bgr.copy() if self._prev_bgr is not None else None

		roi = self._target_roi_from_estimate(
			target=target,
			image_width=image_width,
			image_height=image_height,
		)

		if roi is None and self._require_target_roi:
			self._prev_gray = gray
			self._prev_bgr = frame_bgr.copy()
			self._prev_timestamp = timestamp

			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(
				result=result,
				previous_frame=previous_frame,
				current_frame=frame_bgr,
				roi=None,
				message="No valid target ROI",
			)

			return result

		if roi is None:
			roi = (0, 0, image_width, image_height)

		x0, y0, x1, y1 = roi

		prev_roi = self._prev_gray[y0:y1, x0:x1]
		gray_roi = gray[y0:y1, x0:x1]

		if prev_roi.shape[1] < 3 or prev_roi.shape[0] < 3:
			self._prev_gray = gray
			self._prev_bgr = frame_bgr.copy()
			self._prev_timestamp = timestamp

			result = FlowResult(timestamp=timestamp, valid=False)
			self._save_debug(
				result=result,
				previous_frame=previous_frame,
				current_frame=frame_bgr,
				roi=roi,
				message="ROI too small",
			)

			return result

		# Close-range regime: see the constructor's winsize_close/levels_close/
		# pyr_scale_close docstring. Read directly off the TargetEstimate
		# already passed in -- no new signal needed.
		close_range = bool(target is not None and getattr(target, "fov_saturated", False))
		pyr_scale = self._pyr_scale_close if close_range else self._pyr_scale
		levels = self._levels_close if close_range else self._levels
		winsize = self._winsize_close if close_range else self._winsize

		flow_px_per_frame = cv2.calcOpticalFlowFarneback(
			prev_roi,
			gray_roi,
			None,
			pyr_scale,
			levels,
			winsize,
			self._iterations,
			self._poly_n,
			self._poly_sigma,
			0,
		)

		flow_px_s = flow_px_per_frame / dt

		# --- Ego-rotation removal (see derotation.py) ----------------------
		# Subtract the predicted rotational flow field (depth-independent, a
		# pure function of body rate + camera geometry) BEFORE mean flow and the
		# affine divergence fit, so both the [offset, flow_norm] state and the
		# divergence see translation-only flow.
		#
		# raw_flow_px_s keeps a reference to the PRE-de-rotation field for the
		# raw-vs-corrected diagnostics below; derotate() returns a NEW array, so
		# the reference stays intact. When de-rotation is inactive the two names
		# alias the same array and raw == corrected everywhere (no double work).
		raw_flow_px_s = flow_px_s
		derotation_active = self._derotator is not None and body_rates is not None
		if derotation_active:
			flow_px_s = self._derotator.derotate(
				flow_px_s, body_rates, roi=(x0, y0, x1, y1)
			)

		raw_mean_flow_x = float(np.mean(raw_flow_px_s[:, :, 0]))
		raw_mean_flow_y = float(np.mean(raw_flow_px_s[:, :, 1]))

		mean_flow_x = float(np.mean(flow_px_s[:, :, 0]))
		mean_flow_y = float(np.mean(flow_px_s[:, :, 1]))

		# Same mean flow, but in the normalized image-coordinate system
		# used by TargetEstimate.offset_x/offset_y. This is the preferred
		# velocity-like state for roll/pitch control and identification.
		mean_flow_x_norm = mean_flow_x / max(0.5 * image_width, 1.0)
		mean_flow_y_norm = mean_flow_y / max(0.5 * image_height, 1.0)

		# Reference-frame gradient magnitude, used to weight the affine
		# divergence fit toward pixels with real local structure (see
		# _fit_divergence_affine / this module's docstring): a flat,
		# texture-poor patch gives near-random Farneback output and should
		# be trusted less than a high-gradient patch, not averaged in
		# equally. Computed on prev_roi (the frame flow vectors are anchored
		# to) so it costs one Sobel pass, not two.
		gradient_magnitude = self._gradient_magnitude(prev_roi)

		# Debug-only: a per-pixel finite-difference field, useful to *look at*
		# (e.g. to see whether signal is rim-only vs whole-field). The
		# production scalar below is fit independently and more robustly --
		# see the module docstring.
		divergence_field = self._estimate_divergence_field(
			flow_px_s=flow_px_s,
			image_width=image_width,
			image_height=image_height,
		)

		raw_divergence, n_inliers, fit_quality = self._fit_divergence_affine(
			flow_px_s=flow_px_s,
			image_width=image_width,
			image_height=image_height,
			gradient_magnitude=gradient_magnitude,
		)
		filtered_divergence = self._filter_divergence(raw_divergence)

		# Divergence on the PRE-de-rotation field, diagnostics-only: the gap
		# between this and raw_divergence is how much ego-rotation was biasing
		# the divergence estimate. Recomputed only when de-rotation actually ran
		# (otherwise identical to raw_divergence). Same gradient weighting -- the
		# weight map is image-derived and de-rotation-independent.
		if derotation_active:
			divergence_prederotation, _, _ = self._fit_divergence_affine(
				flow_px_s=raw_flow_px_s,
				image_width=image_width,
				image_height=image_height,
				gradient_magnitude=gradient_magnitude,
			)
		else:
			divergence_prederotation = raw_divergence

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

		return result

	def _fit_divergence_affine(
		self,
		flow_px_s: np.ndarray,
		image_width: int,
		image_height: int,
		gradient_magnitude: Optional[np.ndarray] = None,
	) -> Tuple[float, int, float]:
		"""
		Divergence via a global affine fit, not a per-pixel median.

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
		entirely (see optical_flow_estimator's winsize_close docstring),
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
		roi_height, roi_width = flow_px_s.shape[:2]
		if roi_width < 3 or roi_height < 3:
			return 0.0, 0, 0.0

		u = flow_px_s[:, :, 0] / max(0.5 * image_width, 1.0)
		v = flow_px_s[:, :, 1] / max(0.5 * image_height, 1.0)

		dx_norm = 2.0 / max(image_width - 1, 1)
		dy_norm = 2.0 / max(image_height - 1, 1)

		rows, cols = np.mgrid[0:roi_height, 0:roi_width]
		x = (cols * dx_norm).ravel().astype(np.float64)
		y = (rows * dy_norm).ravel().astype(np.float64)
		u_flat = u.ravel().astype(np.float64)
		v_flat = v.ravel().astype(np.float64)

		if gradient_magnitude is not None and gradient_magnitude.shape == (roi_height, roi_width):
			weight_flat = gradient_magnitude.ravel().astype(np.float64)
		else:
			weight_flat = np.ones_like(u_flat)

		finite = np.isfinite(u_flat) & np.isfinite(v_flat) & np.isfinite(weight_flat)
		n_finite = int(np.count_nonzero(finite))
		if n_finite < self._min_points_for_affine_fit:
			field = self._estimate_divergence_field(flow_px_s, image_width, image_height)
			return self._scalar_from_divergence_field(field), n_finite, 0.0

		x, y, u_flat, v_flat, weight_flat = (
			x[finite], y[finite], u_flat[finite], v_flat[finite], weight_flat[finite]
		)
		design = np.column_stack([np.ones_like(x), x, y])

		coeffs, divergence, fit_quality = self._weighted_affine_least_squares(
			design, u_flat, v_flat, weight_flat
		)

		residual = (u_flat - design @ coeffs[0]) ** 2 + (v_flat - design @ coeffs[1]) ** 2
		threshold = np.quantile(residual, self._affine_inlier_quantile)
		inliers = residual <= threshold

		if np.count_nonzero(inliers) >= self._min_points_for_affine_fit:
			_, divergence, fit_quality = self._weighted_affine_least_squares(
				design[inliers], u_flat[inliers], v_flat[inliers], weight_flat[inliers]
			)
			n_used = int(np.count_nonzero(inliers))
		else:
			n_used = n_finite

		return float(divergence), n_used, float(fit_quality)

	@staticmethod
	def _weighted_affine_least_squares(
		design: np.ndarray, u: np.ndarray, v: np.ndarray, weight: np.ndarray
	) -> Tuple[Tuple[np.ndarray, np.ndarray], float, float]:
		"""Gradient-magnitude-weighted OLS via the standard sqrt(w) rescaling
		(minimizing sum(w*(y-Xb)^2) is exactly OLS in sqrt(w)-rescaled
		variables, so this stays a single cheap linear solve, not an
		iterative reweighting scheme). Weight is normalized to a mean of 1
		first so its absolute scale never changes lstsq's conditioning, only
		the RELATIVE trust between pixels; an all-zero/degenerate weight map
		falls back to uniform (equivalent to the old unweighted fit).

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

	def reset(self):
		self._prev_gray = None
		self._prev_bgr = None
		self._prev_timestamp = None

		self._filtered_divergence = 0.0
		self._has_filtered_divergence = False

		self._last_debug = {}

	def last_debug_data(self) -> dict:
		return dict(self._last_debug)

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
	) -> np.ndarray:
		roi_height, roi_width = flow_px_s.shape[:2]

		if roi_width < 3 or roi_height < 3:
			return np.zeros((roi_height, roi_width), dtype=np.float32)

		# Convert pixel flow to normalized image-coordinate velocity.
		# Normalization uses the full image size, not the ROI size, so the
		# divergence scale remains consistent as the target box changes.
		u_norm_s = flow_px_s[:, :, 0] / (0.5 * image_width)
		v_norm_s = flow_px_s[:, :, 1] / (0.5 * image_height)

		dx_norm = 2.0 / max(image_width - 1, 1)
		dy_norm = 2.0 / max(image_height - 1, 1)

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