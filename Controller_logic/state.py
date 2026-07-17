"""ROS-free data exchanged by the BEE_LAND controller modules.

Architecture boundary
---------------------
The live controller is visual-only after handoff. This module therefore
contains only:

* visual measurements produced from camera frames;
* the attitude/thrust command formed from those measurements; and
* the minimal contact latch allowed to cross from Gazebo truth into the
terminal touchdown path.

Simulation position, velocity, height, platform motion, forces, and other gold-
standard values deliberately do NOT appear here. They belong to the independent
truth log and are merged with controller diagnostics only in ``analyse_log.py``.
"""

from dataclasses import dataclass


@dataclass
class FlowResult:
	"""Output of ``OpticalFlowEstimator.update()``.

	``timestamp`` is the source camera timestamp in Gazebo simulation seconds.
	It is the time base used for optical-flow units, control ``dt`` and mission
	progression. It must never be replaced by wall receipt time.

	Control-facing fields
	---------------------
	``mean_flow_x_norm`` / ``mean_flow_y_norm``
		Normalized image velocity [1/s], in the same image convention as the
		target offsets.
	``divergence``
		Filtered divergence [1/s] used by the vertical controller.

	Remaining fields are diagnostics-only and preserve the current optical-flow
	and optional de-rotation interfaces.
	"""

	timestamp: float = 0.0
	valid: bool = False

	mean_flow_x_norm: float = 0.0
	mean_flow_y_norm: float = 0.0
	divergence: float = 0.0

	mean_flow_x: float = 0.0
	mean_flow_y: float = 0.0
	raw_divergence: float = 0.0
	fit_quality: float = 0.0

	roi_x0: int = -1
	roi_y0: int = -1
	roi_x1: int = -1
	roi_y1: int = -1

	derotated: bool = False
	mean_flow_x_raw: float = 0.0
	mean_flow_y_raw: float = 0.0
	divergence_prederotation: float = 0.0


@dataclass
class TargetEstimate:
	"""Output of ``TargetAcquisition.update()``.

	``timestamp`` is the source camera timestamp in Gazebo simulation seconds.

	``offset_x`` and ``offset_y`` are centroid offsets normalized to [-1, 1].
	``area_fraction`` and the detection dimensions stop carrying range
	information once ``fov_saturated`` becomes true; the visual controller may
	continue using offsets and divergence, but consumers must not interpret the
	saturated box geometry as a physical size measurement.
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
class AttitudeSetpoint:
	"""Visual command consumed by the PX4 publication adapter.

	``timestamp`` is the camera/flow simulation timestamp that produced the
	command. It is provenance only. ``PX4Interface`` independently stamps the
	outgoing uXRCE-DDS messages with a fresh system-wall timestamp.
	"""

	timestamp: float = 0.0
	roll: float = 0.0
	pitch: float = 0.0
	yaw: float = 0.0
	thrust: float = 0.0  # normalized collective thrust, [0, 1]


@dataclass
class ContactState:
	"""Minimal Gazebo-truth subset allowed into the live controller.

	The truth plugin publishes a much richer atomic packet, but ``bee_node``
	extracts only this terminal-event subset. No position, velocity, distance,
	divergence truth or force measurement may enter the controller state.

	``sim_timestamp`` is the Gazebo physics timestamp of the truth packet.
	Receipt wall/monotonic times are transport diagnostics owned by
	``bee_node`` / ``DiagnosticsWriter`` and are intentionally not stored here.
	"""

	valid: bool = False
	sequence: int = -1
	sim_timestamp: float = 0.0
	left_contact: bool = False
	right_contact: bool = False
	any_contact: bool = False
	confirmed: bool = False
