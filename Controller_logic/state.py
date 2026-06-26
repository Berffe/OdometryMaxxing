"""
Shared, ROS-free data containers passed between bee_node.py and the algorithm
modules, so the vision/control code can be imported and tested without rclpy.
"""

from dataclasses import dataclass


@dataclass
class VehicleState:
	"""PX4 local-position estimate (NED), for diagnostics/handoff only.

	NED sign convention: z grows more negative when climbing, vz > 0 = descending.
	Not used by the control law after visual handoff.
	"""

	timestamp: float = 0.0
	x: float = 0.0
	y: float = 0.0
	z: float = 0.0
	vx: float = 0.0
	vy: float = 0.0
	vz: float = 0.0
	yaw: float = 0.0  # VehicleLocalPosition.heading [rad]


@dataclass
class FlowResult:
	"""Output of OpticalFlowEstimator.update().

	Control inputs (used by control_law.py):
	    mean_flow_x_norm / mean_flow_y_norm : normalized image velocity [1/s], in
	        the same convention as TargetEstimate.offset_x/y. Closed-loop control
	        uses these because the identified models have state [offset, flow_norm].
	    divergence : filtered divergence [1/s], used by the thrust loop.

	Debug / log-only:
	    mean_flow_x / mean_flow_y : same velocity in px/s, kept for camera
	        debugging and logging. Relation: *_norm = *_px_s / (0.5 * image_dim),
	        so the normalized field is the px/s field rescaled (recoverable).
	    raw_divergence : unfiltered divergence; logged separately because the
	        filter adds phase lag that matters for identification.
	    roi_* : ROI the flow was computed in (target box), x1/y1 exclusive.
	"""

	timestamp: float = 0.0
	valid: bool = False

	mean_flow_x_norm: float = 0.0
	mean_flow_y_norm: float = 0.0
	divergence: float = 0.0

	mean_flow_x: float = 0.0
	mean_flow_y: float = 0.0
	raw_divergence: float = 0.0

	roi_x0: int = -1
	roi_y0: int = -1
	roi_x1: int = -1
	roi_y1: int = -1


@dataclass
class TargetEstimate:
	"""Output of TargetAcquisition.update() -- what the controller tracks.

	offset_x/y : centroid offset from image center, normalized to [-1, 1].
	area_fraction : detection area / frame area, in [0, 1]; the scheduling
	    variable for the identified visual models.
	fov_saturated : the detection's bounding box touches all four image
	    borders -- the true target exceeds the camera's field of view, not
	    just fills it. cv2.boundingRect is mechanically capped at the image
	    array's own dimensions, so area_fraction/detection_width/height stop
	    tracking true range entirely once this is True (confirmed: identical
	    output across a 3x range of true target size). found stays True --
	    a real contour is still selected -- but the geometry it reports is a
	    frame-size artifact, not a measurement. Consumers that use these
	    fields as a scheduling or identification variable should treat rows
	    with fov_saturated=True as uninformative, not as a steady operating
	    point.
	"""

	timestamp: float = 0.0
	found: bool = False
	offset_x: float = 0.0
	offset_y: float = 0.0
	detection_width: float = 0.0
	detection_height: float = 0.0
	confidence: float = 0.0
	area_fraction: float = 0.0
	fov_saturated: bool = False


@dataclass
class PlatformState:
	"""
	Known/commanded landing-platform motion, for diagnostics and relative-
	motion analysis only -- the control law never sees this (see
	control_law.py: visual-only by design, divergence/offset are its only
	inputs). Reconstructed analytically by platform_motion.py from the
	OscillatingPlatformController SDF plugin's own amplitude/frequency/phase,
	since the plugin does not publish its pose to ROS.

	Same x/y/z, vx/vy/vz shape as VehicleState, but in the SDF world's own
	frame/units -- see platform_motion.py's module docstring for the NED-vs-
	ENU sign/axis caveat before comparing directly against VehicleState.
	"""

	timestamp: float = 0.0
	x: float = 0.0
	y: float = 0.0
	z: float = 0.0
	vx: float = 0.0
	vy: float = 0.0
	vz: float = 0.0


@dataclass
class AttitudeSetpoint:
	"""Desired attitude/thrust consumed by the MAVSDK/PX4 backend."""

	timestamp: float = 0.0
	roll: float = 0.0
	pitch: float = 0.0
	yaw: float = 0.0
	thrust: float = 0.0  # normalized collective thrust, [0, 1]