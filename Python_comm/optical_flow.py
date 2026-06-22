"""
Optical flow estimation, decoupled from ROS.

bee_node.py calls update() from on_camera() on every incoming frame.
The result is stored as the node's "latest_flow" and picked up by the
slower control timer — see bee_node.py.
"""

import cv2
import numpy as np

from .state import FlowResult


class OpticalFlowEstimator:
	"""
	Dense optical-flow estimator.

	This first implementation uses Farneback dense optical flow. It returns:

	- flow_field: HxWx2 array of optical-flow velocity in pixels/second.
	- mean_flow_x: mean horizontal flow in pixels/second.
	- mean_flow_y: mean vertical flow in pixels/second.
	- divergence: mean normalized optical-flow divergence in 1/second.

	Sign convention for divergence:
		positive divergence -> image expansion -> approaching the scene/target
		negative divergence -> image contraction -> moving away from scene/target
	"""

	def __init__(
		self,
		pyr_scale: float = 0.5,
		levels: int = 3,
		winsize: int = 21,
		iterations: int = 3,
		poly_n: int = 5,
		poly_sigma: float = 1.2,
		use_central_roi: bool = True,
		central_roi_fraction: float = 0.70,
		divergence_smoothing: float = 0.6,
	):
		self._prev_gray = None
		self._prev_timestamp = None

		self._pyr_scale = pyr_scale
		self._levels = levels
		self._winsize = winsize
		self._iterations = iterations
		self._poly_n = poly_n
		self._poly_sigma = poly_sigma

		self._use_central_roi = use_central_roi
		self._central_roi_fraction = float(central_roi_fraction)
		self._divergence_smoothing = float(divergence_smoothing)
		self._filtered_divergence = 0.0
		self._has_filtered_divergence = False

	def update(self, frame_bgr, timestamp: float) -> FlowResult:
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

		if self._prev_gray is None or self._prev_timestamp is None:
			self._prev_gray = gray
			self._prev_timestamp = timestamp
			return FlowResult(timestamp=timestamp, valid=False)

		dt = float(timestamp - self._prev_timestamp)
		if dt <= 1e-6:
			self._prev_gray = gray
			self._prev_timestamp = timestamp
			return FlowResult(timestamp=timestamp, valid=False)

		flow_px_per_frame = cv2.calcOpticalFlowFarneback(
			self._prev_gray,
			gray,
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
		divergence = self._estimate_divergence(flow_px_s)
		divergence = self._filter_divergence(divergence)

		self._prev_gray = gray
		self._prev_timestamp = timestamp

		return FlowResult(
			timestamp=timestamp,
			valid=True,
			mean_flow_x=mean_flow_x,
			mean_flow_y=mean_flow_y,
			divergence=float(divergence),
		)

	def _estimate_divergence(self, flow_px_s: np.ndarray) -> float:
		"""
		Estimate image-plane divergence from dense optical flow.

		The dense flow is first converted from pixels/second to normalized
		image-coordinate velocity/second. Then:

			div = du_norm/dx_norm + dv_norm/dy_norm

		The result has units 1/second.
		"""
		height, width = flow_px_s.shape[:2]
		if width < 3 or height < 3:
			return 0.0

		# Normalized image coordinates are approximately in [-1, +1].
		# u_norm_s and v_norm_s are normalized-coordinate velocities.
		u_norm_s = flow_px_s[:, :, 0] / (0.5 * width)
		v_norm_s = flow_px_s[:, :, 1] / (0.5 * height)

		dx_norm = 2.0 / max(width - 1, 1)
		dy_norm = 2.0 / max(height - 1, 1)

		du_dx = np.gradient(u_norm_s, dx_norm, axis=1)
		dv_dy = np.gradient(v_norm_s, dy_norm, axis=0)

		divergence_field = du_dx + dv_dy

		if self._use_central_roi:
			divergence_field = self._central_roi(divergence_field)

		# Median is more robust than mean against local noisy flow vectors.
		return float(np.median(divergence_field))

	def _central_roi(self, array: np.ndarray) -> np.ndarray:
		height, width = array.shape[:2]
		fraction = max(0.05, min(1.0, self._central_roi_fraction))

		roi_w = int(fraction * width)
		roi_h = int(fraction * height)

		x0 = max(0, (width - roi_w) // 2)
		y0 = max(0, (height - roi_h) // 2)
		x1 = min(width, x0 + roi_w)
		y1 = min(height, y0 + roi_h)

		return array[y0:y1, x0:x1]

	def _filter_divergence(self, divergence: float) -> float:
		"""
		First-order smoothing to reduce thrust chatter.
		"""
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
