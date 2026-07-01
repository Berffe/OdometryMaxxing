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
it directly uses every valid flow vector in the ROI (weighted by how
well it's explained by one global radial-expansion trend), instead of
computing a local spatial derivative pixel-by-pixel and taking the
median, which is dominated by whichever response is most common pixel-
by-pixel rather than by the actual expansion signal. This matters
specifically once the target fills the frame: the interior of a
uniform, texture-poor surface gives near-zero/noisy per-pixel flow (the
classic aperture problem), so a median across mostly-interior pixels
washes the real signal out even though the textured rim still carries
it; the affine fit instead lets that rim data pull the whole-field
trend toward the correct value.

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
		require_target_roi: bool = True,
		roi_margin_fraction: float = 0.05,
		min_roi_size_px: int = 32,
		divergence_smoothing: float = 0.7,
		min_points_for_affine_fit: int = 30,
		affine_inlier_quantile: float = 0.85,
		store_debug: bool = False,
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

	def update(
		self,
		frame_bgr,
		timestamp: float,
		target: Optional[TargetEstimate] = None,
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

		flow_px_per_frame = cv2.calcOpticalFlowFarneback(
			prev_roi,
			gray_roi,
			None,
			self._pyr_scale,
			self._levels,
			self._winsize,
			self._iterations,
			self._poly_n,
			self._poly_sigma,
			0,
		)

		flow_px_s = flow_px_per_frame / dt

		mean_flow_x = float(np.mean(flow_px_s[:, :, 0]))
		mean_flow_y = float(np.mean(flow_px_s[:, :, 1]))

		# Same mean flow, but in the normalized image-coordinate system
		# used by TargetEstimate.offset_x/offset_y. This is the preferred
		# velocity-like state for roll/pitch control and identification.
		mean_flow_x_norm = mean_flow_x / max(0.5 * image_width, 1.0)
		mean_flow_y_norm = mean_flow_y / max(0.5 * image_height, 1.0)

		# Debug-only: a per-pixel finite-difference field, useful to *look at*
		# (e.g. to see whether signal is rim-only vs whole-field). The
		# production scalar below is fit independently and more robustly --
		# see the module docstring.
		divergence_field = self._estimate_divergence_field(
			flow_px_s=flow_px_s,
			image_width=image_width,
			image_height=image_height,
		)

		raw_divergence, n_inliers = self._fit_divergence_affine(
			flow_px_s=flow_px_s,
			image_width=image_width,
			image_height=image_height,
		)
		filtered_divergence = self._filter_divergence(raw_divergence)

		result = FlowResult(
			timestamp=timestamp,
			valid=True,
			mean_flow_x=mean_flow_x,
			mean_flow_y=mean_flow_y,
			mean_flow_x_norm=float(mean_flow_x_norm),
			mean_flow_y_norm=float(mean_flow_y_norm),
			divergence=float(filtered_divergence),
			raw_divergence=float(raw_divergence),
			roi_x0=int(x0),
			roi_y0=int(y0),
			roi_x1=int(x1),
			roi_y1=int(y1),
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
	) -> Tuple[float, int]:
		"""
		Divergence via a global affine fit, not a per-pixel median.

		Model (normalized image units/s, same scale as mean_flow_*_norm):
		    u(x, y) = a0 + a1*x + a2*y
		    v(x, y) = b0 + b1*x + b2*y
		Solved independently by OLS (shared design matrix). For any affine
		field, du/dx + dv/dy = a1 + b2 exactly and is constant everywhere, so
		this is the exact divergence of the best-fit field -- using ALL valid
		flow vectors in the ROI, not a local difference at each pixel.

		The OLS slope is invariant to the coordinate origin (shifting x, y by
		a constant only moves a0/b0), so ROI-local pixel coordinates are used
		directly -- no need to know the ROI's offset within the full image.

		One robust trim-and-refit pass keeps the best `affine_inlier_quantile`
		fraction by residual and refits, so a cluster of unreliable vectors
		(e.g. a textureless patch returning near-random flow) can't dominate
		the fit the way it would dominate a per-pixel median.

		Falls back to the old field-median method if too few finite flow
		vectors remain (degenerate ROI) -- a safety net, not the normal path.
		"""
		roi_height, roi_width = flow_px_s.shape[:2]
		if roi_width < 3 or roi_height < 3:
			return 0.0, 0

		u = flow_px_s[:, :, 0] / max(0.5 * image_width, 1.0)
		v = flow_px_s[:, :, 1] / max(0.5 * image_height, 1.0)

		dx_norm = 2.0 / max(image_width - 1, 1)
		dy_norm = 2.0 / max(image_height - 1, 1)

		rows, cols = np.mgrid[0:roi_height, 0:roi_width]
		x = (cols * dx_norm).ravel().astype(np.float64)
		y = (rows * dy_norm).ravel().astype(np.float64)
		u_flat = u.ravel().astype(np.float64)
		v_flat = v.ravel().astype(np.float64)

		finite = np.isfinite(u_flat) & np.isfinite(v_flat)
		n_finite = int(np.count_nonzero(finite))
		if n_finite < self._min_points_for_affine_fit:
			field = self._estimate_divergence_field(flow_px_s, image_width, image_height)
			return self._scalar_from_divergence_field(field), n_finite

		x, y, u_flat, v_flat = x[finite], y[finite], u_flat[finite], v_flat[finite]
		design = np.column_stack([np.ones_like(x), x, y])

		coeffs, divergence = self._affine_least_squares(design, u_flat, v_flat)

		residual = (u_flat - design @ coeffs[0]) ** 2 + (v_flat - design @ coeffs[1]) ** 2
		threshold = np.quantile(residual, self._affine_inlier_quantile)
		inliers = residual <= threshold

		if np.count_nonzero(inliers) >= self._min_points_for_affine_fit:
			_, divergence = self._affine_least_squares(
				design[inliers], u_flat[inliers], v_flat[inliers]
			)
			n_used = int(np.count_nonzero(inliers))
		else:
			n_used = n_finite

		return float(divergence), n_used

	@staticmethod
	def _affine_least_squares(
		design: np.ndarray, u: np.ndarray, v: np.ndarray
	) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
		"""OLS-solve u, v against `design`=[1, x, y]; return (coeffs, a1+b2)."""
		coeffs_u, *_ = np.linalg.lstsq(design, u, rcond=None)
		coeffs_v, *_ = np.linalg.lstsq(design, v, rcond=None)
		divergence = float(coeffs_u[1] + coeffs_v[2])
		return (coeffs_u, coeffs_v), divergence

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