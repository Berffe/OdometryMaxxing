"""
Shared data containers passed between bee_node.py and the algorithm
modules (optical_flow.py, target_acquisition.py, control_law.py,
px4_interface.py).

Keeping these as plain dataclasses (no ROS message types) means every
module downstream of bee_node.py can be imported and unit-tested without
rclpy being initialized.
"""

from dataclasses import dataclass


@dataclass
class VehicleState:
	"""Latest PX4 state estimates used only for diagnostics/identification.

	Position and velocity come from VehicleLocalPosition. roll/pitch and
	attitude_yaw come from VehicleAttitude or VehicleOdometry when one of those topics is available.
	PX4 local position uses NED convention, so vz > 0 means descending.
	"""

	# Local-position timestamp, normalized by DiagnosticsWriter when logged.
	timestamp: float = 0.0
	x: float = 0.0
	y: float = 0.0
	z: float = 0.0
	vx: float = 0.0
	vy: float = 0.0
	vz: float = 0.0
	yaw: float = 0.0  # VehicleLocalPosition.heading, radians

	# Attitude timestamp and Euler angles from VehicleAttitude.q.
	attitude_timestamp: float = 0.0
	roll: float = 0.0
	pitch: float = 0.0
	attitude_yaw: float = 0.0
	attitude_source: str = ""  # "vehicle_attitude" or "vehicle_odometry"


@dataclass
class FlowResult:
	"""
	Output of OpticalFlowEstimator.update().

	mean_flow_x/mean_flow_y are kept in px/s for easy camera debugging.
	mean_flow_x_norm/mean_flow_y_norm are the same image velocity in
	normalized image coordinates per second, matching TargetEstimate's
	offset_x/offset_y convention in [-1, 1]. Use the normalized values for
	control and identification whenever possible.

	divergence is the filtered scalar used by the controller; raw_divergence
	is logged separately for identification because filtering adds phase lag.

	The ROI fields record where the optical-flow calculation was performed.
	They are useful when a bad fit is actually caused by a changing or
	clipped target box rather than by dynamics.
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
	# Detection area as a fraction of the full frame area, in [0, 1].
	# Lets downstream consumers (control_law.py) tell a normally-sized
	# detection apart from one that fills most/all of the frame, which is
	# the expected case in the final seconds of a landing approach and
	# should not be treated like an outlier.
	area_fraction: float = 0.0


@dataclass
class AttitudeSetpoint:
	"""
	Output of ControlLaw.compute(). Consumed by PX4Interface, which turns
	this into a VehicleAttitudeSetpoint message.
	"""

	timestamp: float = 0.0
	roll: float = 0.0
	pitch: float = 0.0
	yaw: float = 0.0
	thrust: float = 0.0  # normalized collective thrust, [0, 1]
