"""
Shared data containers passed between bee_node.py and the algorithm
modules (optical_flow.py, target_acquisition.py, control_law.py,
px4_interface.py).

Keeping these as plain dataclasses (no ROS message types) means every
module downstream of bee_node.py can be imported and unit-tested
without rclpy being initialized.
"""

from dataclasses import dataclass


@dataclass
class VehicleState:
	"""Latest PX4 local-position estimate, as last received by on_local_position()."""

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
	"""
	Output of OpticalFlowEstimator.update().

	flow_field is left untyped (None for now) since its shape depends on
	the algorithm you pick later (e.g. a dense HxWx2 numpy array for
	Farneback, or just a handful of scalars for a sparse/divergence-based
	estimator). mean_flow is a convenience summary most controllers will
	want regardless of the underlying algorithm.
	"""

	timestamp: float = 0.0
	valid: bool = False
	mean_flow_x: float = 0.0
	mean_flow_y: float = 0.0
	divergence: float = 0.0


@dataclass
class TargetEstimate:
	"""Output of TargetAcquisition.update() — what the controller should track."""

	timestamp: float = 0.0
	found: bool = False
	offset_x: float = 0.0
	offset_y: float = 0.0
	detection_width : float = 0.0 ; 
	detection_height : float = 0.0
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
	