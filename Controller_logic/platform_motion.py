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

# bee_platform.sdf's platform_link carries its collision/visual_body as a
# cylinder (radius=1.0, length=0.2) with NO offset <pose> -- so it is
# centered exactly on platform_link's own origin, which is the pose
# OscillatingPlatformController publishes as platform.z. That means
# platform.z is the disc's GEOMETRIC CENTER, not its top face -- the actual
# landing surface the drone's legs make contact with sits HALF THE
# CYLINDER'S LENGTH above that center, a fixed +0.1m in the platform's own
# (ENU, up-positive) frame that was previously silently missing from every
# relative_z/relative_vz computation below.
#
# This was found by comparing bee_x500/x500_base's actual leg geometry
# (base_link_collision_3/_4: unrotated 0.25x0.015x0.015 boxes centered at
# z=-0.2195 relative to base_link, i.e. a true skid-bottom clearance of
# 0.227m -- MORE than MissionRoutine's leg_clearance_m=0.20 default, not
# less) against logged touchdown relative_z_m, which read far below what
# either number alone would predict. A center-vs-surface offset on the
# platform side, missing this whole time, is a clean, geometry-verified
# partial (very possibly not sole) explanation: relative_z_m has been
# reporting distance-to-the-disc's-CENTER, not distance-to-the-surface the
# legs actually reach, understating the true clearance by this amount at
# every height, not just at touchdown.
#
# Applied to the RAW platform.z (ENU) value BEFORE the PLATFORM_NED_SIGN
# conversion below, not after: this offset is a fact about the platform's
# own z-axis geometry (its reported origin sits 0.1m below its true top
# surface, in ITS OWN up-positive frame), so it belongs in the same frame
# platform.z is already in, before any NED-conversion sign logic is
# applied -- exactly like correcting a mismeasured platform.z at the
# source, not a separate correction bolted onto the combined NED result.
PLATFORM_TOP_SURFACE_OFFSET_M = 0.2 / 2.0  # bee_platform.sdf <length>0.2</length>, halved

# The DRONE side of the same story. relative_z_m is only "coherent" -- reads the
# gap the pilot cares about -- if it measures the drone's LOWEST point (its feet /
# belly) to the platform's TOP surface. The platform offset above lifts the
# reference from disc-centre to disc-top; this one lowers it from base_link to the
# skids, so together they make relative_z = 0 mean "feet exactly on the surface"
# (and relative_z > 0 mean the feet have gone THROUGH it, i.e. penetration).
#
# Value from bee_x500/x500_base (model_base.sdf): base_link_collision_3/_4 are the
# landing-skid cross-bars, 0.25x0.015x0.015 boxes centred at z=-0.2195 relative to
# base_link, so their underside -- the true first-contact point -- sits
#   0.2195 + 0.015/2 = 0.227 m
# below base_link. This is the same 0.227 m the module docstring already cites as
# the skid-bottom clearance; it was just never subtracted from relative_z.
#
# NOT to be confused with LEG_CLEARANCE_M in bee_node (0.182 m): that is a
# CAMERA-referenced height (camera above feet = 0.227 - 0.045), used by the
# feasibility gate because the control's height sense is the camera's divergence.
# This constant is base_link -> feet, used to place the diagnostic relative_z at
# the belly. Different references, different jobs; keep them distinct.
DRONE_BELLY_OFFSET_M = 0.227  # base_link -> skid underside, model_base.sdf

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
	    relative_z  = vehicle.z + PLATFORM_NED_SIGN*(platform.z + PLATFORM_TOP_SURFACE_OFFSET_M)
	    relative_vz = vehicle.vz + PLATFORM_NED_SIGN*platform.vz  (= d(relative_z)/dt;
	        the offset is a constant, so it drops out of the derivative -- vz is
	        unaffected)
	platform.z is ENU (up-positive); vehicle.z is NED (down-positive). Vehicle
	descending (vehicle.vz>0) and platform RISING (platform.vz>0, moving up
	toward the vehicle from below) both shrink the physical gap between them,
	so their NED-converted contributions to closing rate ADD, not subtract.

	PLATFORM_TOP_SURFACE_OFFSET_M (see module docstring) shifts platform.z
	from the disc's reported CENTER to its actual top (landing) surface
	before combining -- without it, relative_z measures distance to the
	platform's geometric center, ~0.1m below where the legs actually reach,
	understating true clearance at every height, not just at touchdown.
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

	# rel_z measures the drone's FEET to the platform's TOP surface, in the same
	# NED closing-convention as rel_vz (negative while the feet are still above the
	# surface, 0 at feet-on-surface, positive = penetration).
	#
	# Two geometric shifts, in the frame each belongs to:
	#   + PLATFORM_TOP_SURFACE_OFFSET_M : platform.z is the disc CENTRE (ENU,
	#       up-positive); its top surface is half a thickness higher. Added inside
	#       the platform term, before the NED sign, exactly as before.
	#   + DRONE_BELLY_OFFSET_M          : vehicle.z is base_link (NED, down-positive);
	#       the feet are 0.227 m BELOW it, i.e. 0.227 m closer to the surface. In the
	#       down-positive NED frame "lower" is "more positive", so the feet's z is
	#       vehicle.z + DRONE_BELLY_OFFSET_M. This shrinks the clearance magnitude,
	#       which is correct -- the skids reach the deck before base_link would.
	# Both are constants, so rel_vz (the derivative) is unchanged.
	rel_z = (
		(vehicle.z + DRONE_BELLY_OFFSET_M)
		+ PLATFORM_NED_SIGN * (platform.z + PLATFORM_TOP_SURFACE_OFFSET_M)
	)
	rel_vz = vehicle.vz + PLATFORM_NED_SIGN * platform.vz

	return rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz