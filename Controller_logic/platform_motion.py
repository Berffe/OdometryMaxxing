"""
Platform-relative motion helpers, for diagnostics only -- NOT used by the
control law (see control_law.py: visual-only by design).

calibration_node.py's primary source for the platform's state is now EXACT:
a PosePublisher plugin on platform_link (see bee_platform.sdf) publishes its
true world pose, bridged into PlatformState via finite-differenced velocity
(see calibration_node.py's on_platform_pose). relative_motion() below
converts that (or any other source of PlatformState) into vehicle-relative
motion for logging -- this is the part every caller wants.

AxisOscillation/PlatformMotionModel reconstruct the COMMANDED motion
analytically instead, from the SDF's amplitude/frequency/phase. With live
pose now available this is a SECONDARY, validation-only tool: compare the
analytical prediction against the live-logged platform_*/relative_* columns
to confirm the plugin is actually doing what the SDF says (units, startup
transients, etc.), not a substitute for the real measurement. Keep its
parameters in sync with the SDF by hand if you use it, the same way the
A<amp>F<freq> filename convention already tracks them per test.

TWO UNVERIFIED ASSUMPTIONS IN THE ANALYTICAL MODEL ONLY -- check these
against the plugin's source before trusting AxisOscillation's numbers (the
live pose path above has neither problem, since it reads the true state
directly rather than predicting it):

1. Units of "frequency". This module assumes frequency_hz is in Hz, giving
   pos(t) = amplitude*sin(2*pi*frequency_hz*t + phase). If the plugin instead
   treats its "frequency" parameter as a raw angular frequency in rad/s, the
   true period is 2*pi/frequency_hz longer than this module assumes (e.g. a
   configured value of 0.2 would be a ~31s period, not the ~5s this module
   would compute). A0a3F0f2.csv (amplitude=0.3, frequency=0.2, presumed Hz)
   showed NO spectral content near the Hz-assumption's 0.2 Hz/5s anywhere in
   vz/altitude/divergence -- the dominant response was a ~0.043 Hz/~23s mode,
   absent from an equivalent stationary-platform run, so it is real and
   platform-related, just not a 1:1 echo of a 5s input. That is fully
   consistent with a slow, underdamped thrust loop (not yet damped the way
   roll/pitch were -- see control_law.py) ringing at its OWN natural
   frequency when excited by a disturbance faster than the loop can track
   cycle-by-cycle, so it is not, by itself, proof the rad/s interpretation is
   right either. Resolve this from the plugin's source if you have it;
   fit_axis_models.py's platform_tracking_report() runs the same spectral
   check this module's docstring describes on any new log, so each future
   test re-validates (or flags a mismatch with) whichever convention you set.

2. NED vs ENU. VehicleState.z/vz are PX4 NED (z down, vz>0=descending).
   bee_platform.sdf's world is ENU (z up). This module's AxisOscillation.z
   follows the SDF's own up-positive convention; relative_motion() converts
   to a NED-down-positive "closing rate" by ADDING (not subtracting) the
   converted terms -- see its docstring for the physical derivation. Verify
   the sign once: relative_vz should correlate POSITIVELY with measured
   divergence. Horizontal axes have an additional, unverified NED<->ENU axis
   swap (NED x=North=ENU y, NED y=East=ENU x) that x/y oscillation tests will
   need to confirm -- both are 0 in every test so far, so this is untested.

Phase/timing caveat: t is whatever timestamp basis the caller passes in
(this module is ROS-free and has no clock of its own). The plugin's t=0 is
Gazebo's sim-time origin, which is NOT calibration_node.py's t_sec=0 (that
starts at the first diagnostics write, well after world launch and PX4 boot)
-- there is a real, likely nonzero, but roughly CONSTANT offset between the
two for a given launch sequence. time_offset_sec absorbs it; calibrate it
once empirically (compare this model's predicted phase against the logged
signal's, e.g. via platform_tracking_report()) and it should hold for later
runs using the same world-launch sequence.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

try:
	from .state import PlatformState, VehicleState
except ImportError:
	from state import PlatformState, VehicleState


# Flip to -1.0 if relative_vz comes out anti-correlated with measured
# divergence -- see assumption 2 above.
PLATFORM_NED_SIGN = 1.0

# Untested (x_amplitude/y_amplitude are 0 in every run so far): NED's x=North
# corresponds to ENU's y, and NED's y=East corresponds to ENU's x. Set True
# once a lateral oscillation test shows relative_x/y tracking the WRONG
# platform axis (e.g. relative_x correlates with offset_y instead of offset_x).
PLATFORM_XY_AXIS_SWAP = False


@dataclass
class AxisOscillation:
	"""One axis of the SDF plugin's motion: amplitude*sin(2*pi*f*t + phase)."""

	amplitude_m: float = 0.0
	frequency_hz: float = 0.0
	phase_rad: float = 0.0

	def position(self, t: float) -> float:
		return self.amplitude_m * math.sin(2.0 * math.pi * self.frequency_hz * t + self.phase_rad)

	def velocity(self, t: float) -> float:
		omega = 2.0 * math.pi * self.frequency_hz
		return self.amplitude_m * omega * math.cos(omega * t + self.phase_rad)


class PlatformMotionModel:
	"""
	Reconstructs PlatformState(t) from the SDF's per-axis amplitude/
	frequency/phase. Keep these in sync with bee_platform.sdf's
	OscillatingPlatformController block for the CURRENT test -- see the
	module docstring for the unverified unit/sign assumptions involved.
	"""

	def __init__(
		self,
		x: AxisOscillation = AxisOscillation(),
		y: AxisOscillation = AxisOscillation(),
		z: AxisOscillation = AxisOscillation(),
		time_offset_sec: float = 0.0,
	):
		self._x, self._y, self._z = x, y, z
		self._time_offset_sec = float(time_offset_sec)

	def state_at(self, t: float) -> PlatformState:
		t_eff = t + self._time_offset_sec
		return PlatformState(
			timestamp=t,
			x=self._x.position(t_eff), y=self._y.position(t_eff), z=self._z.position(t_eff),
			vx=self._x.velocity(t_eff), vy=self._y.velocity(t_eff), vz=self._z.velocity(t_eff),
		)


def relative_motion(
	vehicle: Optional[VehicleState], platform: Optional[PlatformState]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
	"""
	(relative_x, relative_y, relative_z, relative_vx, relative_vy, relative_vz)
	in PX4's NED convention -- relative_vz is exactly d(relative_z)/dt by
	construction, and is a "closing rate": positive means vehicle and
	platform are approaching each other, matching divergence's own sign
	convention directly. (None,)*6 if either input is missing.

	z derivation (x/y follow the same logic once their own NED<->ENU axis
	mapping is confirmed -- see module docstring assumption 2):
	    relative_z  = vehicle.z + PLATFORM_NED_SIGN*platform.z
	    relative_vz = vehicle.vz + PLATFORM_NED_SIGN*platform.vz  (= d(relative_z)/dt)
	platform.z is ENU (up-positive); vehicle.z is NED (down-positive). Vehicle
	descending (vehicle.vz>0) and platform RISING (platform.vz>0, moving up
	toward the vehicle from below) both shrink the physical gap between them,
	so their NED-converted contributions to closing rate ADD, not subtract.
	"""
	if vehicle is None or platform is None:
		return (None, None, None, None, None, None)

	platform_x = platform.y if PLATFORM_XY_AXIS_SWAP else platform.x
	platform_y = platform.x if PLATFORM_XY_AXIS_SWAP else platform.y
	platform_vx = platform.vy if PLATFORM_XY_AXIS_SWAP else platform.vx
	platform_vy = platform.vx if PLATFORM_XY_AXIS_SWAP else platform.vy

	rel_x = vehicle.x - platform_x
	rel_y = vehicle.y - platform_y
	rel_vx = vehicle.vx - platform_vx
	rel_vy = vehicle.vy - platform_vy

	rel_z = vehicle.z + PLATFORM_NED_SIGN * platform.z
	rel_vz = vehicle.vz + PLATFORM_NED_SIGN * platform.vz

	return rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz