"""
Shared data containers passed between bee_node.py and the algorithm modules.

These classes intentionally stay ROS-free so the vision and control code can be
imported and unit-tested without rclpy.
"""

from dataclasses import dataclass


@dataclass
class VehicleState:
	"""Latest PX4 local-position estimate used for diagnostics and handoff checks.

	PX4 local position is NED: z becomes more negative when climbing, and
	vz > 0 means descending.
	"""

	timestamp: float = 0.0
	x: float = 0.0
	y: float = 0.0
	z: float = 0.0
	vx: float = 0.0
	vy: float = 0.0
	vz: float = 0.0
	yaw: float = 0.0  # VehicleLocalPosition.heading, radians


@dataclass
class FlowResult:
	"""Output of OpticalFlowEstimator.update().

	mean_flow_x/mean_flow_y are in px/s and are useful for camera debugging.
	mean_flow_x_norm/mean_flow_y_norm are normalized image velocities in the
	same coordinate convention as TargetEstimate.offset_x/y. The closed-loop
	controller uses the normalized values because the identified roll/pitch
	models were fitted with state = [offset, normalized_flow].

	divergence is filtered; raw_divergence is logged separately because filtering
	adds phase lag and can be useful for identification/debugging.
	"""

	timestamp: float = 0.0
	valid: bool = False
	mean_flow_x: float = 0.0
	mean_flow_y: float = 0.0
	mean_flow_x_norm: float = 0.0
	mean_flow_y_norm: float = 0.0
	divergence: float = 0.0
	raw_divergence: float = 0.0
	roi_x0: int = -1
	roi_y0: int = -1
	roi_x1: int = -1
	roi_y1: int = -1


@dataclass
class TargetEstimate:
	"""Output of TargetAcquisition.update() — what the controller should track."""

	timestamp: float = 0.0
	found: bool = False
	offset_x: float = 0.0
	offset_y: float = 0.0
	detection_width: float = 0.0
	detection_height: float = 0.0
	confidence: float = 0.0
	# Detection area as a fraction of the full frame area, in [0, 1]. Used as
	# the gain-scheduling variable for the identified visual models.
	area_fraction: float = 0.0


@dataclass
class AttitudeSetpoint:
	"""Desired attitude/thrust command consumed by the MAVSDK/PX4 backend."""

	timestamp: float = 0.0
	roll: float = 0.0
	pitch: float = 0.0
	yaw: float = 0.0
	thrust: float = 0.0  # normalized collective thrust, [0, 1]
