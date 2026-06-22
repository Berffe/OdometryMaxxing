"""
Optical flow estimation, decoupled from ROS.

bee_node.py calls update() from on_camera() on every incoming frame.
The result is stored as the node's "latest_flow" and picked up by the
(slower) control timer — see bee_node.py.
"""

import cv2

from .state import FlowResult


class OpticalFlowEstimator:
	"""
	Placeholder. update() currently just converts the frame to grayscale,
	stores it, and returns an invalid FlowResult.

	TODO: replace the body of update() with the actual flow algorithm,
	e.g. cv2.calcOpticalFlowFarneback(self._prev_gray, gray, ...) for a
	dense field, or a divergence-based estimator if you only need a
	scalar closing-rate for landing.
	"""

	def __init__(self):
		self._prev_gray = None

	def update(self, frame_bgr, timestamp: float) -> FlowResult:
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

		result = FlowResult(timestamp=timestamp, valid=False)

		# TODO: compute flow between self._prev_gray and gray, fill in
		# result.flow_field / result.mean_flow_x / result.mean_flow_y,
		# and set result.valid = True once self._prev_gray is not None.

		self._prev_gray = gray
		return result
