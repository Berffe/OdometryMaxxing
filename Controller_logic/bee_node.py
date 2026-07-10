import contextlib
import math
import multiprocessing as mp
import os
import queue
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (
	QoSProfile,
	QoSReliabilityPolicy,
	QoSDurabilityPolicy,
	QoSHistoryPolicy,
)

try:
	from mavsdk import System
except ImportError:
	System = None

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from px4_msgs.msg import (
	VehicleLocalPosition,
	VehicleAttitude,
	VehicleStatus,
	VehicleAngularVelocity,
)
from cv_bridge import CvBridge, CvBridgeError

from .state import VehicleState, PlatformState, AttitudeSetpoint, TargetEstimate
# v2.0: TargetAcquisition + OpticalFlowEstimator are no longer constructed or
# called here -- they run in a separate process (see vision_worker.py). bee_node
# ships frames to it and drains results back, so it only needs the worker entry
# point. CameraGeometry/Derotator/AngularRateBuffer stay: on_camera still
# computes the per-interval mean body rate and ships it (the derotator itself is
# built but currently unused, same as before -- de-rotation, when re-enabled,
# moves into the worker).
from .derotation import CameraGeometry, Derotator, AngularRateBuffer
from .control_law import ControlLaw
from .mission_routine import MissionRoutine, INFEASIBLE as MISSION_INFEASIBLE
from .diagnostics_writer import DiagnosticsWriter
from .px4_interface import PX4Interface
from .mavsdk_worker import MavsdkWorker
from .clock import TimeManager
from .vision_worker import run_vision_worker


# Control computation is intentionally faster than the PX4 publication stream:
# - control timer: low-latency polling for NEW vision samples
# - PX4 setpoint timer: fixed, predictable offboard publication cadence
CONTROL_PERIOD_SEC = 0.01
MISSION_PERIOD_SEC = 0.05
PX4_SETPOINT_PERIOD_SEC = 0.05
COMPUTE_CONTROL_ONLY_ON_NEW_VISION = True
PX4_OFFBOARD_SWITCH_SETTLE_SEC = 0.5

# Depth of the bee_node -> vision_worker frame queue (v2.0). Kept shallow on
# purpose: under load, on_camera drops the OLDEST queued frame (see
# _ship_frame_to_vision) so the worker always resumes on a FRESH frame instead
# of draining a stale backlog -- for a divergence-based landing loop, freshness
# beats processing every frame. 2 gives a little jitter slack without letting
# latency build up in the queue.
VISION_INPUT_QUEUE_MAX = 2

# ============================================================================
# WARNING -- READ BEFORE TUNING descent_divergence_setpoint / initial_thrust_gain
# ============================================================================
# bee_node.py constructs MissionRoutine with EXPLICIT keyword arguments below
# (search "MissionRoutine("), which means the DEFAULT values written into
# mission_routine.py's own constructor signature are DEAD CODE for any real
# flight -- they are unconditionally overridden every time bee_node.py runs.
# Those constructor defaults only matter if MissionRoutine is constructed
# STANDALONE (e.g. mission_routine.py's own __main__ smoke test), never here.
#
# This is a real trap: editing mission_routine.py's default value (e.g.
# changing "initial_thrust_gain: float = 0.833" to "= 6.5" in its constructor
# signature) has ZERO effect on what actually flies through bee_node.py. The
# constants below -- DESCENT_DIVERGENCE_SETPOINT and INITIAL_THRUST_GAIN --
# are the ONLY place that governs real flight behavior. This exact mismatch
# already caused one very weak-gain, high-descent-rate run (mission_thrust_
# gain_k logged as 0.3332 = the OLD center-thrust-gain-scale mechanism (0.4)
# times the OLD default 0.833, not the intended 6.5) that looked like a
# control regression but was actually just a stale/desynced constant. (That
# reduced-CENTER-thrust-gain mechanism has since been removed entirely --
# CENTER's vertical loop now always runs at the full exploration gain -- but
# the lesson about dead-code defaults below still applies to whatever
# constants ARE still live.) If you want a value to take effect,
# change it HERE, not in mission_routine.py's signature. If you also update
# mission_routine.py's defaults (recommended, so a standalone construction/
# the smoke test reflects the same tuning), that is in ADDITION to changing
# the constants here, never instead of it.
# ============================================================================

# --- Moving-platform landing: probe -> gate -> scheduled-gain descent ---
# (see mission_routine.py's module docstring for the full derivation).
# DESCENT_DIVERGENCE_SETPOINT is both the D* commanded during descent AND the
# omega* in the Herisse floor k_min=peak_accel/D* -- the same value plays both
# roles by construction, so changing it changes descent speed AND the
# feasibility gate together; faster descent (larger D*) makes k_min SMALLER
# (and so h_crit smaller / more likely feasible), at the cost of less time to
# react to vision dropouts. 0.15 1/s is a starting point, not tuned.
# *** SOURCE OF TRUTH for the flown value -- see the WARNING above. ***
DESCENT_DIVERGENCE_SETPOINT = 0.30
# Ramp the commanded D* linearly from 0 to DESCENT_DIVERGENCE_SETPOINT over
# this many seconds at descent entry, instead of stepping it instantly. Fixes
# the thrust/vz transient measured at PROBE->DESCEND (a real step in the
# control_law error term, not in k(t) which was already smooth). 0 recovers
# the old instant-step behavior. See mission_routine.py's constructor
# docstring for why this does not also affect k(t)'s decay rate.
D_STAR_RAMP_IN_SEC = 5.0
# CENTER's vertical (thrust) loop runs at the full exploration gain
# (k_explore) throughout -- the earlier reduced-CENTER-thrust-gain mechanism
# (which decoupled the vertical loop from CENTER's heavy lateral banking) has
# been removed; it is not used. See mission_routine.py's constructor
# docstring near center_to_probe_lateral_ramp_sec for the current reasoning.
#
# Time to smoothly restore the LATERAL gains (CENTER -> PROBE steady-state)
# once CENTER hands off (same raised-cosine shape as D_STAR_RAMP_IN_SEC, same
# reasoning: avoid a step right at a handoff we've already found transients
# matter at). The probe's own peak_accel/min-duration measurement does not
# start until this completes. 0 recovers an instant restore. Thrust is NOT
# part of this ramp (see above).
CENTER_TO_PROBE_LATERAL_RAMP_SEC = 0.1
# LATERAL gain during CENTER, mission-PHASE-based (fixed for the whole phase)
# rather than instantaneous-|offset|-based (control_law's large_offset_gain_
# scale, which restores full gain every time the vehicle swings near center
# mid-oscillation -- including repeatedly, while still early in CENTER). This
# is the originally-requested design: small gain to approach/center gently,
# THEN increase once genuinely converged, THEN start following the
# disturbance -- which an offset-magnitude blend alone cannot express, since
# it cannot distinguish "still centering, momentarily near zero" from
# "actually converged". Ramps to the PROBE steady-state scales
# (PROBE_LATERAL_P_SCALE/PROBE_LATERAL_D_SCALE) over
# CENTER_TO_PROBE_LATERAL_RAMP_SEC.
#
# TWO INDEPENDENT scales, not one -- a single shared scale cannot reproduce
# an earlier validated (kp, kd) pair, since kp and kd were boosted by
# DIFFERENT factors when tuned for platform-oscillation tracking (kp x2, kd x
# sqrt(2) -- see control_law.py's gain history comments). Verified directly: a
# single scale=0.5 landed kp exactly on the earlier validated baseline
# (0.44*0.5=0.22) but left kd under-damped relative to that SAME baseline
# (0.155*0.5=0.078 vs the validated 0.11) -- CENTER was actually LESS damped
# than the historical "before horizontal oscillations" baseline it was meant
# to recover, not more. These values EXACTLY reverse the two historical
# factors: 1/2 reverses "kp x2"; 1/sqrt(2) reverses "kd x sqrt(2)" -- together
# reproducing the original validated (roll_kp=0.22, roll_kd=0.11,
# pitch_kp=0.15, pitch_kd=0.07) gains to within <0.4% (rounding in the stored
# decimal constants, not a modeling error). 1.0/1.0 disables this (CENTER at
# full lateral gain, old behavior).
CENTER_LATERAL_P_SCALE = 1
CENTER_LATERAL_D_SCALE = 1
# STEADY PROBE lateral P scale (kd left at 1.0 -- raising kd was already A/B
# tested and found to make tracking WORSE: it operates on optical flow, a
# real noisy/lagged signal, not a clean derivative).
#
# Motivated by a cross-spectral measurement of the closed loop at full gain
# against the platform's own oscillation: amplitude ratio 2.229 (residual
# EXCEEDS the platform's own motion -- not simple under-tracking, which would
# show ratio->1 at ~180 deg phase, the original pre-kp-boost signature) and
# phase -98.9 deg (close to -90, the textbook signature of a lightly-damped
# 2nd-order system driven AT its own resonant frequency). The platform's
# ~0.055 Hz oscillation is landing almost exactly on the closed loop's
# resonant peak -- the worst possible frequency for an underdamped loop.
# kp sets natural frequency; raising it further pushes the resonant peak AWAY
# from the platform's low frequency. Unlike kd, this isn't undermined by
# derivative noise, since kp acts on offset (position), a clean signal. 1.5
# is a MODERATE first step (kp_eff: roll 0.44->0.66, pitch 0.30->0.45), not a
# confident final answer -- re-measure the same amplitude-ratio/phase
# diagnostic on the next log. 1.0 disables this (PROBE unchanged).
PROBE_LATERAL_P_SCALE = 1
PROBE_LATERAL_D_SCALE = 1
# How long to hold the D*=0 probe before computing peak_accel/k_min/h_crit.
# No periodicity assumption is needed (unlike the dropped mode-estimator
# design) -- this only needs to be long enough to see the platform swing
# through a representative excursion. 15s is a generous starting guess with no
# real-platform validation yet; tighten or extend once logged against an
# actual oscillating deck.
PROBE_MIN_DURATION_SEC = 10.0

# dt fed into the de Croon feasibility gate -- see mission_routine.py's
# stability_dt_sec constructor docstring for the base reasoning (why this must
# be a hardware-real, RTF=1 quantity, not a wall-clock-measured one). This
# composes THREE terms, each independently real and RTF-independent, and each
# missing from the original single-term (1/30 s) estimate:
#
# 1. CAMERA_FRAME_PERIOD_SEC: what the camera delivers at RTF=1 -- unchanged
#    reasoning from before.
# 2. PX4_SETPOINT_PERIOD_SEC (defined above): the ROS WALL-clock timer that
#    actually publishes to PX4. This is a genuine hardware period (a fixed
#    ROS timer, not gated to the vision/sim clock -- confirmed against logs:
#    timing_px4_publish_period_wall_sec tracks 0.05s regardless of
#    timing_sim_rtf_estimate), so no RTF correction is needed for it, unlike
#    timing_control_period_wall_sec (which IS inflated by RTF and must never
#    feed this estimate). It matters because a fresh vision-driven correction
#    can only reach the actuator at the rate of WHICHEVER TIMER IS SLOWER: if
#    the camera produces a new estimate faster than PX4_SETPOINT_PERIOD_SEC,
#    intermediate corrections are simply never sent (the publish timer only
#    ever picks up the latest). At 30 Hz camera / 20 Hz publish, the publish
#    period (0.05s) is the binding one, not the camera's own 1/30s -- taking
#    max() of the two is therefore the physically correct choice, not a
#    guess; a system with a faster publish timer than its camera would
#    instead be camera-bound and max() would correctly fall back to that.
# 3. VISION_PROCESSING_LATENCY_BUDGET_SEC: real, wall-clock CPU time between
#    a frame arriving and a corrected command being READY to publish (target
#    acquisition + optical flow + the divergence fit). This is a SEPARATE
#    term from the two periods above -- it's not "how often can a fresh
#    correction go out", it's "how long after the sensor sample is one ready
#    at all" -- so it is ADDED, not chosen via max(). Confirmed
#    RTF-independent from logs, and as of the on_camera per-stage timing
#    breakdown (timing_stage_*_ms below), confirmed to be dominated by
#    optical_flow.update() specifically -- not Farneback itself, but the
#    divergence affine fit's np.linalg.lstsq calls (two fits x a
#    trim-and-refit pass each = 4 solves/frame with derotation on). Set from
#    a conservative (not mean) reading of timing_stage_optical_flow_ms /
#    timing_camera_cb_duration_ms -- the same "a single spike only makes the
#    gate MORE conservative, which is the safe direction" logic
#    PlatformProbe.result() already applies to peak_accel -- because an
#    underestimate here silently reopens the exact gap this whole correction
#    exists to close. Re-measure and update whenever the optical-flow
#    pipeline's cost changes.
CAMERA_FRAME_PERIOD_SEC = 1.0 / 30.0
VISION_PROCESSING_LATENCY_BUDGET_SEC = 0.01  # conservative p95-ish reading of
                                              # timing_camera_cb_duration_ms;
                                              # re-measure after pipeline changes.
STABILITY_DT_SEC = (
	max(CAMERA_FRAME_PERIOD_SEC, PX4_SETPOINT_PERIOD_SEC)
	+ VISION_PROCESSING_LATENCY_BUDGET_SEC
)

# --- Belly-camera intrinsics, mirrored from model.sdf's bee_camera sensor -----
# model.sdf: <horizontal_fov>1.3962634</horizontal_fov> (= 80 deg),
#            <image><width>120</width><height>80</height></image>.
# The SDF stores FOV + width, NOT focal length; the pinhole relation
# f_px = (width/2)/tan(hfov/2) recovers it (~71.5 px here) -- done inside
# CameraGeometry.from_horizontal_fov so the value stays tied to these numbers.
# This bridge publishes no CameraInfo topic, so these are the single source of
# truth: if you change the sensor in model.sdf, change these to match.
CAMERA_HFOV_RAD = 1.3962634
CAMERA_WIDTH_PX = 120
CAMERA_HEIGHT_PX = 80

# LOG-ONLY wall-clock-duration hint for mission_routine's sim-time timers
# (probe_min_duration_sec, center_dwell_sec, center_timeout_sec,
# d_star_ramp_in_sec -- see that module's CLOCK note). Every one of those is
# measured on sim time by design; under a simulator real-time-factor (RTF) < 1
# they take proportionally longer in wall-clock terms. This constant is NOT
# read by any control-relevant code path -- it exists purely so the startup
# log can print an honest wall-time estimate instead of leaving the operator
# to discover it empirically (as "15 seconds" quietly becoming ~75). Measure
# your own RTF with analyse_log.py's estimate_rtf() on a recent log, or from
# the sim_wall_offset_sec drift, and update this if your machine/scene load
# changes meaningfully -- a stale value only misleads the printed estimate,
# it cannot affect flight behavior.
EXPECTED_SIM_RTF = 0.2

# Vehicle's own ground/landing-gear clearance, in meters -- the feasibility
# gate's threshold for h_crit. PLACEHOLDER: set this from the actual airframe
# geometry before flying for real; 0.20 m is not derived from anything here.
LEG_CLEARANCE_M = 0.20

# FIRST-BRINGUP KNOB. True -> the mission runs the D*=0 probe, computes and logs
# k_min / h_crit / feasibility, then HOLDS hover indefinitely without ever
# descending or aborting. Use this to validate the hover loop and the probe /
# bounds in isolation before trusting the descent. Set False only once the
# probe numbers look right on a real log.
HOVER_PROBE_ONLY = False

# Hand-tuned initial/exploration thrust gain "k" (m/s) for the vertical loop --
# set like the lateral PD gains, NOT derived from takeoff height + de Croon's
# ceiling. The ATTITUDE_HOLD height reference is ground-relative while the
# platform sits ~2 m up, so the height-derived gain was wrong; this decouples
# it. This is the gain the hover/probe runs at and the value the descent
# schedule decays from. 0.833 was the original validated-hover value; raised
# to 6.5 per subsequent tuning (see e.g. the platform-tracking-gain and
# vertical-thrust-during-CENTER discussions) -- re-validate hover stability
# if reverting toward a much smaller value.
# *** SOURCE OF TRUTH for the flown value -- see the WARNING above. ***
INITIAL_THRUST_GAIN = 6.5

SHOW_CAMERA = False
VERBOSE_STREAM_LOGS = False

# Start the attempt already airborne. 5 m corresponds to the cleanest far-range
# calibration operating point (area_fraction around 0.066 in the last batch).
TAKEOFF_ALTITUDE_M = 5.0
EKF2_SETTLE_TIME = 5.0
MAVSDK_SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
MAVSDK_PORT_TO_FREE = 14540
PX4_HOLD_CURRENT_YAW = True

MAVSDK_CONNECT_TIMEOUT_SEC = 15.0
MAVSDK_HEALTH_TIMEOUT_SEC = 30.0
MAVSDK_TAKEOFF_ALTITUDE_TIMEOUT_SEC = 130.0
OFFBOARD_PRESTREAM_SEC = 2.0

# PX4 VehicleStatus enums (px4_msgs). Verify against your px4_msgs version with:
#   ros2 interface show px4_msgs/msg/VehicleStatus
PX4_NAV_STATE_OFFBOARD = 14
PX4_ARMING_STATE_ARMED = 2
# How long to keep re-commanding offboard while waiting for VehicleStatus to
# confirm it, before giving up. PX4 will only accept the switch once it has seen
# a few OffboardControlMode heartbeats, so a single command can be missed.
PX4_OFFBOARD_CONFIRM_TIMEOUT_SEC = 5.0
PX4_OFFBOARD_REENGAGE_INTERVAL_SEC = 0.5

_PX4_NAV_STATE_NAMES = {
	0: "MANUAL", 1: "ALTCTL", 2: "POSCTL", 3: "AUTO_MISSION", 4: "AUTO_LOITER",
	5: "AUTO_RTL", 10: "ACRO", 12: "DESCEND", 13: "TERMINATION",
	14: "OFFBOARD", 15: "STAB", 17: "AUTO_TAKEOFF", 18: "AUTO_LAND",
	20: "AUTO_PRECLAND", 21: "ORBIT",
}
_PX4_ARMING_STATE_NAMES = {1: "DISARMED", 2: "ARMED"}


def _nav_state_name(v):
	return f"{_PX4_NAV_STATE_NAMES.get(v, 'UNKNOWN')}({v})"


def _arming_state_name(v):
	return f"{_PX4_ARMING_STATE_NAMES.get(v, 'UNKNOWN')}({v})"
OFFBOARD_START_TIMEOUT_SEC = 5.0

# Runtime safety for the first closed-loop tests. After the ROS 2 offboard handoff,
# nominal commands are visual-only. If the visual target/flow is lost, keep
# streaming a neutral visual-hover setpoint instead of using PX4 velocity/altitude
# feedback. PX4 local state remains diagnostics-only after handoff.
LOST_TARGET_TIMEOUT_SEC = 2.0
ENABLE_INERTIAL_SAFETY_ABORTS = False
SAFETY_VZ_LIMIT = 1.0
SAFETY_LATERAL_VELOCITY_LIMIT = 2.0

# Bridged (ros_gz_bridge) ROS2 topic for the platform's exact world pose.
# Published directly by OscillatingPlatformController itself (see
# MovingPlatformController.cpp's publishPose) as a plain gz.msgs.Pose, on
# its own dedicated single-entity topic -- not via gz-sim's generic pose
# broadcasting, which two earlier approaches both confirmed unreliable here:
# a PosePublisher SDF plugin only emitted a one-shot static snapshot for
# this <static>true</static> model, and SceneBroadcaster's pose/info (a
# Pose_V of every entity) bridges through ros_gz_bridge into
# tf2_msgs/msg/TFMessage with every entity's name left empty -- confirmed
# directly via this node's own "Entity names seen" log -- so there was no
# way to pick this entity back out on the ROS side. Publishing our own
# topic sidesteps both problems entirely, back to the simple message type:
#   ros2 run ros_gz_bridge parameter_bridge \
#       /platform/pose@geometry_msgs/msg/Pose@gz.msgs.Pose
# Same topic/bridge as calibration_node.py -- keep these in sync.
# Diagnostics-only, same as vehicle_state: the control law never sees this
# (see control_law.py's module docstring). Set PLATFORM_POSE_TOPIC to None
# to disable (e.g. a stationary-platform run); diagnostics rows just get
# empty platform_*/relative_* fields either way.
PLATFORM_POSE_TOPIC = "/platform/pose"

# uXRCE-DDS exposes VehicleStatus under a version-suffixed name that varies by
# PX4 build. This build publishes "/fmu/out/vehicle_status_v4" (confirmed: it
# echoes nav_state, the others had no publisher). We subscribe to every known
# candidate (same callback); whichever the bridge actually publishes delivers,
# the rest stay silently empty. NOTE: subscribing to a name makes it appear in
# `ros2 topic list` even with no publisher, so to find the REAL one, list topics
# with the node STOPPED, or check which one echoes:
#   ros2 topic echo /fmu/out/vehicle_status_v4 --field nav_state --once
VEHICLE_STATUS_TOPICS = (
	"/fmu/out/vehicle_status_v4",
	"/fmu/out/vehicle_status_v1",
	"/fmu/out/vehicle_status",
)

# Real pose telemetry is noisy/jittery sample-to-sample; smooth the finite-
# differenced velocity the same way OpticalFlowEstimator smooths divergence
# (see optical_flow.py's module docstring for the same underlying argument).
PLATFORM_VELOCITY_SMOOTHING = 0.5

# Touchdown bridge. The Gazebo side is published by TouchPlugin in
# bee_platform.sdf. Bridge it with:
#   ros2 run ros_gz_bridge parameter_bridge \
#       /bee_platform/touched@std_msgs/msg/Bool@gz.msgs.Boolean
TOUCHDOWN_TOPIC = "/bee_platform/touched"
TOUCHDOWN_STATUS_TOPIC = "/bee_land/touchdown"
ENABLE_TOUCHDOWN_MOTOR_STOP = True
# SITL fallback: if PX4 refuses a normal disarm because its internal land
# detector does not recognize the moving-platform touchdown yet, kill() stops
# the simulated motors. Keep this False for real hardware.
ENABLE_TOUCHDOWN_KILL_FALLBACK = True

PHASE_WAITING_FOR_STREAMS = "waiting_for_streams"
PHASE_MAVSDK_TAKEOFF = "mavsdk_takeoff"
PHASE_PRESTREAM = "prestream_offboard"
PHASE_WAIT_OFFBOARD = "wait_offboard"
PHASE_CLOSED_LOOP = "closed_loop"
PHASE_LANDED = "landed"
PHASE_ABORTED = "aborted"


@contextlib.contextmanager
def suppress_stderr_fd(enabled: bool = True):
	"""Temporarily suppress native stderr spam from OpenCV/Qt while keeping video."""
	if not enabled:
		yield
		return
	try:
		stderr_fd = 2
		saved_fd = os.dup(stderr_fd)
		with open(os.devnull, "w") as devnull:
			os.dup2(devnull.fileno(), stderr_fd)
			yield
	finally:
		try:
			os.dup2(saved_fd, stderr_fd)
			os.close(saved_fd)
		except Exception:
			pass


class BeeLandNode(Node):
	def __init__(self):
		super().__init__("bee_land_node")

		# Single source of "now" for the whole node. Created first so every
		# subsystem (px4_interface included) shares one definition of each
		# clock family -- see clock.py.
		self.time = TimeManager(self)
		self._node_start_time = self.time.wall_sec()
		self.bridge = CvBridge()
		self._vehicle_state = VehicleState()

		# Platform pose (dedicated bridge -> on_platform_pose): exact
		# world-frame position each message, finite-differenced into a
		# smoothed velocity (see PLATFORM_VELOCITY_SMOOTHING). None until the
		# first message arrives, or forever if PLATFORM_POSE_TOPIC is None --
		# diagnostics rows just get empty platform_*/relative_* fields.
		self._platform_state = None
		self._prev_platform_pose_t = None
		self._prev_platform_pose_xyz = None
		self._platform_velocity_filtered = (0.0, 0.0, 0.0)
		self._has_filtered_platform_velocity = False
		self._platform_pose_count = 0
		self._platform_pose_stall_logged = False

		self._latest_flow = None
		self._latest_frame = None
		self._latest_target = TargetEstimate()

		self.control_law = ControlLaw()
		self._latest_setpoint = AttitudeSetpoint(thrust=self.control_law.hover_thrust)

		# Probe -> gate -> scheduled-gain descent for the moving-platform
		# landing -- see mission_routine.py's module docstring. Parameterizes
		# control_law.compute() each tick (divergence_setpoint,
		# thrust_gain_override) without forming any commands itself;
		# control_law remains the sole, visual-only command former.
		self.mission = MissionRoutine(
			hover_thrust=self.control_law.hover_thrust,
			control_period_sec=CONTROL_PERIOD_SEC,
			descent_divergence_setpoint=DESCENT_DIVERGENCE_SETPOINT,
			probe_min_duration_sec=PROBE_MIN_DURATION_SEC,
			leg_clearance_m=LEG_CLEARANCE_M,
			probe_only=HOVER_PROBE_ONLY,
			initial_thrust_gain=INITIAL_THRUST_GAIN,
			d_star_ramp_in_sec=D_STAR_RAMP_IN_SEC,
			center_to_probe_lateral_ramp_sec=CENTER_TO_PROBE_LATERAL_RAMP_SEC,
			center_lateral_p_scale=CENTER_LATERAL_P_SCALE,
			center_lateral_d_scale=CENTER_LATERAL_D_SCALE,
			probe_lateral_p_scale=PROBE_LATERAL_P_SCALE,
			probe_lateral_d_scale=PROBE_LATERAL_D_SCALE,
			stability_dt_sec=STABILITY_DT_SEC,
		)
		self._mission_infeasible_logged = False
		self._last_mission_log_time = 0.0
		self._latest_mission_control = None

		self._have_local_position = False
		self._have_vehicle_attitude = False
		self._vehicle_attitude_count = 0
		self._image_count = 0
		self._last_image_log_time = 0.0
		self._last_position_log_time = 0.0
		self._last_attitude_log_time = 0.0

		self._mission_phase = PHASE_WAITING_FOR_STREAMS
		self._phase_start_time = self.time.wall_sec()
		self._streams_ready_logged = False
		self._closed_loop_logged = False
		self._lost_target_since = None

		# Visual/control time bookkeeping. Target acquisition, optical flow, and
		# control_law.compute() must use one clock family. The source of truth is
		# the camera Image.header.stamp when ros_gz_bridge provides it. PX4 time is
		# kept only as a fallback for missing image stamps and as diagnostics.
		# Do NOT compare image/Gazebo timestamps and PX4 timestamps by absolute
		# value: in this setup they can live in different epochs. Only deltas inside
		# one clock family are meaningful.
		self._prev_control_flow_timestamp = None
		# CONTROL_PERIOD_SEC can be faster than the camera. In that case the
		# 100 Hz timer should act as a low-latency poller for fresh vision, not
		# as a fake 100 Hz visual controller repeatedly integrating/filtering the
		# same optical-flow sample. This stamp records the last flow sample that
		# actually produced a new command.
		self._last_controlled_flow_timestamp = None

		# Wall-clock latency diagnostics. These do not feed the controller; they
		# only tell us whether delay comes from vision processing, control polling,
		# the fixed PX4 publication cadence, or simulator real-time factor.
		self._latest_camera_cb_start_wall = None
		self._latest_camera_cb_end_wall = None
		self._latest_camera_cb_duration_ms = None
		# Per-stage breakdown of on_camera, added to find which stage actually
		# owns camera_cb_duration_ms's ~40-200ms wall-clock cost -- isolated
		# benchmarking of Farneback (~2.5ms at this project's 120x80
		# resolution) and target_acquisition's masks/contours (<1ms combined)
		# accounts for only a small fraction of that, and disabling
		# SHOW_CAMERA changed the total by ~4% (noise), ruling out imshow as
		# the dominant cost too. These are diagnostics-only -- like
		# camera_cb_duration_ms itself, never read by the controller -- and
		# exist purely to find the real cost before trusting any wall-clock
		# number as bee_node.py's VISION_PROCESSING_LATENCY_BUDGET_SEC.
		self._latest_stage_bridge_ms = None
		self._latest_stage_rotate_ms = None
		self._latest_stage_show_camera_ms = None
		self._latest_stage_body_rate_ms = None
		# v2.0: target_acquisition/optical_flow no longer run in on_camera, so
		# these two PARENT-process stage slots are structurally empty now (they
		# stay None). The worker's own measurements of those two calls are
		# reported separately below (worker_*), NOT back through these fields --
		# that back-fill was the v2.0-first-test bug where a 16ms "sub-stage"
		# appeared under a 0.38ms on_camera parent.
		self._latest_stage_target_acquisition_ms = None
		self._latest_stage_optical_flow_ms = None
		# v2.0 out-of-process vision instrumentation. worker_* are the worker's
		# own perf_counter timings of the two calls (durations, comparable across
		# processes). frame_to_available/frame_to_command/result_period are the
		# real cross-boundary latencies the v1.0-era stage timers never captured,
		# all measured in THIS process's wall clock (see _drain_vision_results /
		# on_control_timer).
		self._latest_worker_target_acquisition_ms = None
		self._latest_worker_optical_flow_ms = None
		self._latest_frame_to_available_ms = None
		self._latest_frame_to_command_ms = None
		self._latest_vision_result_period_ms = None
		# Frame-arrival wall time of the flow currently in _latest_flow, carried
		# across the process boundary so on_control_timer can close the
		# frame->command measurement when this sample actually produces a command.
		self._latest_flow_frame_wall = None
		self._prev_vision_result_frame_wall = None
		self._last_control_compute_start_wall = None
		self._last_control_compute_end_wall = None
		self._last_control_compute_duration_ms = None
		self._last_control_period_wall_sec = None
		self._last_control_dt_vision_sec = None
		self._latest_setpoint_compute_wall = None
		self._latest_setpoint_flow_timestamp = None
		self._prev_px4_publish_wall = None
		self._last_px4_publish_wall = None
		self._last_px4_publish_period_wall_sec = None
		self._last_px4_command_age_ms = None
		self._last_px4_flow_age_ms = None
		self._px4_publish_count = 0

		self._control_dt_fallback_logged = False
		self._image_stamp_fallback_logged = False

		# MAVSDK subsystem (takeoff + terminal motor-stop), extracted into its own
		# thread/event-loop module. The node reads worker.takeoff_done/error and
		# calls worker.request_motor_stop()/request_stop(); the worker calls back
		# via on_pre_motor_stop to latch our outgoing setpoint to a zero-thrust
		# hold just before disarm.
		self.mavsdk = MavsdkWorker(
			logger=self.get_logger(),
			on_pre_motor_stop=self._latch_zero_thrust_hold,
			system_address=MAVSDK_SYSTEM_ADDRESS,
			port_to_free=MAVSDK_PORT_TO_FREE,
			takeoff_altitude_m=TAKEOFF_ALTITUDE_M,
			connect_timeout_sec=MAVSDK_CONNECT_TIMEOUT_SEC,
			health_timeout_sec=MAVSDK_HEALTH_TIMEOUT_SEC,
			takeoff_altitude_timeout_sec=MAVSDK_TAKEOFF_ALTITUDE_TIMEOUT_SEC,
			ekf2_settle_time_sec=EKF2_SETTLE_TIME,
			enable_kill_fallback=ENABLE_TOUCHDOWN_KILL_FALLBACK,
		)
		self._px4_offboard_start_requested = False
		self._px4_offboard_started = False
		self._px4_offboard_error = None
		self._px4_offboard_request_time = None
		# Latest PX4 VehicleStatus, for offboard-handoff confirmation + logging.
		# NAV_STATE_OFFBOARD is 14 in PX4; ARMING_STATE_ARMED is 2. None until
		# the first VehicleStatus arrives.
		self._px4_nav_state = None
		self._px4_arming_state = None
		self._px4_failsafe = None
		self._px4_offboard_confirmed = False
		self._px4_offboard_reengage_count = 0
		self._last_nav_state_logged = None

		# Touchdown is a mission-level terminal event, not a visual-control input.
		# The contact signal comes from Gazebo/TouchPlugin through ros_gz_bridge.
		self._touchdown_detected = False
		self._touchdown_time = None
		self._touchdown_message_count = 0

		# Optical-flow de-rotation (see derotation.py). CameraGeometry derives
		# the focal length from the SDF's horizontal FOV + width; the default
		# body->optical rotation already folds in the nadir mount and the
		# cv2.ROTATE_180 that on_camera applies -- but its SIGNS must be
		# validated against a pure-rotation segment (see derotation.py's banner)
		# before the de-rotated flow is trusted for control.
		camera_geometry = CameraGeometry.from_horizontal_fov(
			CAMERA_HFOV_RAD, CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX
		)
		self._derotator = Derotator(camera_geometry)
		# Body-rate history, averaged over each inter-frame interval in
		# on_camera. Kept here (not in VehicleState) so de-rotation uses the
		# interval MEAN rate, not a single latest sample.
		self._rate_buffer = AngularRateBuffer()
		self._prev_camera_stamp = None

		# DE-ROTATION DISABLED FOR NOW. self._derotator/_rate_buffer are still
		# built above because on_camera still computes and ships the per-interval
		# mean body rate regardless. The flow estimator that would consume the
		# derotator now lives in vision_worker (built there with derotator=None,
		# matching this disabled state); to re-enable, construct the Derotator in
		# vision_worker and pass it in -- body_rates already arrive on the queue,
		# so nothing here changes.

		# --- Out-of-process vision pipeline (v2.0) ---------------------------
		# The two heavy vision stages (TargetAcquisition + OpticalFlowEstimator)
		# used to run INLINE in on_camera, on the single ROS executor thread,
		# starving the 20 Hz PX4 setpoint publisher for the 40-200 ms each frame
		# took. They now run in vision_worker.run_vision_worker in a SEPARATE
		# process: on_camera only ships (frame, timestamp, body_rates) and
		# returns immediately, and on_control_timer drains (target, flow) results
		# back. This frees the executor to keep publishing setpoints on cadence
		# while the worker crunches in parallel on another core. See
		# vision_worker.py for the loop itself.
		self._vision_dropped_frames = 0
		self._vision_worker_dead_logged = False
		self._start_vision_worker()

		self.diagnostics = DiagnosticsWriter(output_dir="logs", filename=None, flush_every_row=True)
		self.get_logger().info(f"Diagnostics CSV: {self.diagnostics.filepath}")

		px4_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)
		self.px4_interface = PX4Interface(self, px4_qos, time_manager=self.time)
		camera_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.VOLATILE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)
		touchdown_status_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.RELIABLE,
			durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)

		self.create_subscription(
			VehicleLocalPosition,
			"/fmu/out/vehicle_local_position_v1",
			self.on_local_position,
			px4_qos,
		)
		self.create_subscription(
			VehicleAttitude,
			"/fmu/out/vehicle_attitude",
			self.on_vehicle_attitude,
			px4_qos,
		)
		# Body-frame angular rate (FRD, rad/s) for optical-flow de-rotation.
		# This topic must be in your uXRCE-DDS dds_topics.yaml; like the
		# vehicle_status discovery note above, subscribing does NOT make PX4
		# publish it. If it's absent, add it to dds_topics.yaml, or swap this
		# for VehicleOdometry and read msg.angular_velocity (same FRD rates) --
		# the buffer/derotator downstream are identical either way.
		self.create_subscription(
			VehicleAngularVelocity,
			"/fmu/out/vehicle_angular_velocity",
			self.on_vehicle_angular_velocity,
			px4_qos,
		)
		# VehicleStatus carries nav_state (is PX4 ACTUALLY in offboard?),
		# arming_state, and failsafe. Without this the offboard handoff is
		# open-loop: the node commands the mode switch and assumes it worked.
		# Subscribing lets us CONFIRM offboard engaged and detect if PX4 later
		# drops it (the classic cause of "commands ignored, vehicle holds level
		# and sinks"). Subscribe to every candidate topic name (see
		# VEHICLE_STATUS_TOPICS) so PX4 version differences don't silently leave
		# us blind.
		for status_topic in VEHICLE_STATUS_TOPICS:
			self.create_subscription(
				VehicleStatus,
				status_topic,
				self.on_vehicle_status,
				px4_qos,
			)
		self.create_subscription(Image, "/bee_x500/camera/image", self.on_camera, camera_qos)

		if PLATFORM_POSE_TOPIC:
			self.create_subscription(
				Pose,
				PLATFORM_POSE_TOPIC,
				self.on_platform_pose,
				camera_qos,
			)
			self.get_logger().info(
				f"Platform pose tracking enabled: listening on {PLATFORM_POSE_TOPIC}. "
				"If platform_*/relative_* diagnostics columns stay empty, the bridge "
				"(ros_gz_bridge) for this topic likely isn't running, or the topic name "
				"doesn't match what Gazebo actually publishes -- see this node's warning "
				"after a few seconds with no messages, and PLATFORM_POSE_TOPIC's comment "
				"for how to check both."
			)
		else:
			self.get_logger().info("Platform pose tracking disabled (PLATFORM_POSE_TOPIC is None).")

		self._touchdown_status_pub = self.create_publisher(
			Bool,
			TOUCHDOWN_STATUS_TOPIC,
			touchdown_status_qos,
		)
		self.create_subscription(
			Bool,
			TOUCHDOWN_TOPIC,
			self.on_touchdown,
			camera_qos,
		)
		self._publish_touchdown_status(False)
		self.get_logger().info(
			f"Touchdown detection enabled: listening on {TOUCHDOWN_TOPIC}. "
			f"Latched status is republished on {TOUCHDOWN_STATUS_TOPIC}."
		)

		self.create_timer(MISSION_PERIOD_SEC, self.on_mission_timer)
		self.create_timer(CONTROL_PERIOD_SEC, self.on_control_timer)
		self.create_timer(PX4_SETPOINT_PERIOD_SEC, self.on_px4_setpoint_timer)

		if SHOW_CAMERA:
			with suppress_stderr_fd(True):
				cv2.namedWindow("Bee Land - Camera", cv2.WINDOW_NORMAL)

		self.get_logger().info("bee_land_node started.")
		self.get_logger().info("Waiting for required streams: local_position and camera.")

	# ------------------------------------------------------------------ vision
	# Out-of-process vision pipeline plumbing (v2.0). on_camera ships frames in;
	# on_control_timer drains results out. The ROS executor is single-threaded,
	# so on_camera and on_control_timer never run concurrently -- the only
	# cross-boundary handoff is the two multiprocessing.Queues, which are
	# process-safe, so no locking is needed around _latest_target/_latest_flow.

	def _start_vision_worker(self):
		"""Spawn the vision worker process and its two queues, once at startup.

		Uses the 'spawn' start method deliberately, NOT the Linux-default
		'fork': the child must not inherit this node's rclpy/DDS threads and
		locks (forking a process with live background threads is a classic
		source of deadlocks). spawn gives the worker a clean interpreter that
		imports only vision_worker's algorithm dependencies -- never rclpy.
		"""
		ctx = mp.get_context("spawn")
		# Shallow input queue + drop-oldest in _ship_frame_to_vision: under load
		# we would rather the worker resume on a fresh frame than drain a stale
		# backlog.
		self._vision_in_q = ctx.Queue(maxsize=VISION_INPUT_QUEUE_MAX)
		# Output queue is emptied every control tick (100 Hz) while the worker
		# produces at <=30 Hz, so it never backs up; left unbounded so the
		# worker never blocks on put().
		self._vision_out_q = ctx.Queue()
		self._vision_worker = ctx.Process(
			target=run_vision_worker,
			args=(self._vision_in_q, self._vision_out_q),
			name="bee_vision_worker",
			daemon=True,
		)
		self._vision_worker.start()
		self.get_logger().info(
			f"Vision worker started (pid={self._vision_worker.pid}, "
			"start_method=spawn). target_acquisition + optical_flow now run "
			"out-of-process; on_camera no longer blocks the control/setpoint "
			"timers on vision."
		)

	def _ship_frame_to_vision(self, frame, timestamp, body_rates, frame_wall):
		"""Hand one camera frame to the vision worker. Non-blocking.

		If the worker is momentarily behind and the shallow input queue is full,
		discard the OLDEST queued frame and enqueue this one, so the worker
		always resumes on the freshest frame rather than a stale backlog.
		Skipping a frame is safe for the divergence loop: OpticalFlowEstimator
		works in px/s off each frame's own timestamp, so a skipped frame simply
		widens the baseline for the next one.

		frame_wall is this process's wall clock at frame arrival; it rides along
		and is echoed back in the VisionResult so _drain_vision_results can
		measure the true frame->available round trip without any cross-process
		clock comparison.
		"""
		payload = (frame, timestamp, body_rates, frame_wall)
		try:
			self._vision_in_q.put_nowait(payload)
		except queue.Full:
			try:
				self._vision_in_q.get_nowait()  # drop the stale frame
				self._vision_dropped_frames += 1
			except queue.Empty:
				pass
			try:
				self._vision_in_q.put_nowait(payload)
			except queue.Full:
				# Worker refilled it between our get and put -- skip this frame.
				self._vision_dropped_frames += 1

	def _drain_vision_results(self):
		"""Pull the freshest (target, flow) the worker has produced and publish
		it into _latest_target/_latest_flow for the controller.

		Called at the top of on_control_timer (100 Hz), far faster than the
		<=30 Hz the worker can produce, so the output queue is emptied every
		tick and the controller always acts on the newest result.

		INSTRUMENTATION (v2.0): this is where the real cross-boundary numbers are
		measured. frame_to_available_ms is the full round trip -- frame arrival
		in on_camera, IPC send, queue wait, worker compute, IPC return, until
		drained here -- computed purely in THIS process's wall clock via the
		echoed frame_wall (no cross-process clock comparison). The worker's own
		per-call timings land in worker_* fields, NOT in on_camera's stage
		fields; on_camera no longer runs those two stages, so folding worker
		numbers back through its stage schema is exactly the arithmetic-
		impossibility bug the first v2.0 test surfaced.
		"""
		newest = None
		while True:
			try:
				newest = self._vision_out_q.get_nowait()
			except queue.Empty:
				break

		if newest is not None:
			drain_wall = self.time.wall_sec()
			self._latest_target = newest.target
			self._latest_flow = newest.flow

			# Worker-side compute cost (honest, separate from on_camera stages).
			self._latest_worker_target_acquisition_ms = newest.target_acquisition_ms
			self._latest_worker_optical_flow_ms = newest.optical_flow_ms

			# True frame->available round trip (parent wall clock throughout).
			if newest.frame_wall is not None:
				self._latest_frame_to_available_ms = 1000.0 * (drain_wall - newest.frame_wall)
			# Effective processed-frame period: gap between the arrival times of
			# consecutive results the worker actually returned. If this grows
			# well past the camera period, the worker is not keeping up (frames
			# dropped at the shallow input queue) and _latest_flow is going
			# stale -- the second thing the first test could not see.
			if (
				self._prev_vision_result_frame_wall is not None
				and newest.frame_wall is not None
			):
				self._latest_vision_result_period_ms = 1000.0 * (
					newest.frame_wall - self._prev_vision_result_frame_wall
				)
			self._prev_vision_result_frame_wall = newest.frame_wall
			# Held so on_control_timer can close the frame->command measurement
			# when THIS sample actually drives a new command.
			self._latest_flow_frame_wall = newest.frame_wall

		# Liveness: if the worker process has died, results silently stop and the
		# controller would coast on a stale sample until LOST_TARGET_TIMEOUT_SEC
		# aborts to a neutral hover. Surface it loudly, exactly once.
		if (
			not self._vision_worker_dead_logged
			and self._vision_worker is not None
			and not self._vision_worker.is_alive()
		):
			self._vision_worker_dead_logged = True
			self.get_logger().error(
				"Vision worker process is no longer alive; target/flow updates "
				"have stopped. The controller will hit LOST_TARGET_TIMEOUT and "
				"hold a neutral visual hover. Check the worker's stderr for the "
				"cause (e.g. an unpicklable TargetEstimate/FlowResult field)."
			)

	def shutdown_vision_worker(self):
		"""Stop the vision worker cleanly. Safe to call more than once.

		Sends the stop sentinel, lets the worker finish any queued frames, joins
		with a timeout, and terminates only as a last resort. Called from main()'s
		finally clause alongside the other subsystem teardown.
		"""
		worker = getattr(self, "_vision_worker", None)
		if worker is None:
			return
		try:
			if worker.is_alive():
				try:
					# Blocking put with a short timeout so the sentinel is
					# actually delivered even if a frame is queued ahead of it.
					self._vision_in_q.put(None, timeout=0.5)
				except Exception:
					pass
				worker.join(timeout=2.0)
			if worker.is_alive():
				self.get_logger().warning(
					"Vision worker did not exit on request; terminating it."
				)
				worker.terminate()
				worker.join(timeout=1.0)
		except Exception as exc:
			self.get_logger().warning(
				f"Error while shutting down vision worker: {repr(exc)}"
			)
		finally:
			self._vision_worker = None

	def on_camera(self, msg: Image):
		self._image_count += 1
		now = self.time.wall_sec()
		self._latest_camera_cb_start_wall = now
		if VERBOSE_STREAM_LOGS and now - self._last_image_log_time >= 1.0:
			self._last_image_log_time = now
			self.get_logger().info(f"image #{self._image_count}: {msg.width}x{msg.height}, encoding={msg.encoding}")

		try:
			src = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		except CvBridgeError as exc:
			self.get_logger().error(f"cv_bridge conversion failed: {exc}")
			return

		t_bridge = self.time.wall_sec()

		# Keep camera orientation independent of whether the debug window is open.
		frame = cv2.rotate(src, cv2.ROTATE_180)
		t_rotate = self.time.wall_sec()

		if SHOW_CAMERA:
			with suppress_stderr_fd(True):
				cv2.imshow("Bee Land - Camera", frame)
				cv2.waitKey(1)
		t_show = self.time.wall_sec()

		stamp = self._image_timestamp_sec(msg)
		self.time.observe_sim_timestamp(stamp)

		# Mean body rate over THIS inter-frame interval, for de-rotation. dt is a
		# SIM-family difference (stamp - previous stamp); the buffer then walks
		# back by that duration through its own PX4-family stamps. Neither step
		# crosses clock families, so clock.py's diagnostic offsets never touch
		# this control path. body_rates stays None on the first frame or before
		# the rate stream is up, which makes optical_flow take its legacy
		# no-de-rotation path.
		body_rates = None
		if self._prev_camera_stamp is not None:
			dt_cam = stamp - self._prev_camera_stamp
			omega_mean, _n, ok = self._rate_buffer.mean_recent(dt_cam)
			if ok:
				body_rates = omega_mean
		self._prev_camera_stamp = stamp
		t_body_rate = self.time.wall_sec()

		# v2.0: hand the frame to the out-of-process vision pipeline instead of
		# running target_acquisition/optical_flow inline. This returns almost
		# immediately (a queue put), so the single ROS executor thread is no
		# longer held for the 40-200 ms the two vision stages take -- that work
		# now happens in vision_worker, in parallel. The (target, flow) result
		# lands back in _latest_target/_latest_flow when on_control_timer calls
		# _drain_vision_results. _latest_frame is set here (not from the result)
		# so the "have we ever received a frame" gates (_ready_to_start and the
		# top of on_control_timer) fire as soon as the first frame arrives, just
		# as before.
		self._latest_frame = frame
		self._ship_frame_to_vision(frame, stamp, body_rates, now)

		self._latest_camera_cb_end_wall = self.time.wall_sec()
		self._latest_camera_cb_duration_ms = (
			1000.0 * (self._latest_camera_cb_end_wall - self._latest_camera_cb_start_wall)
			if self._latest_camera_cb_start_wall is not None else None
		)

		# Per-stage breakdown -- diagnostics-only. bridge/rotate/show/body_rate
		# still happen HERE and are timed here; the target_acquisition and
		# optical_flow stages now run in the worker and are filled into
		# _latest_stage_target_acquisition_ms / _latest_stage_optical_flow_ms
		# from the worker's own measurements when _drain_vision_results runs. So
		# these on_camera stages NO LONGER sum to camera_cb_duration_ms (which is
		# now just the light ship-the-frame path, and should be small) -- that is
		# the whole point of the v2.0 split.
		self._latest_stage_bridge_ms = 1000.0 * (t_bridge - now)
		self._latest_stage_rotate_ms = 1000.0 * (t_rotate - t_bridge)
		self._latest_stage_show_camera_ms = 1000.0 * (t_show - t_rotate)
		self._latest_stage_body_rate_ms = 1000.0 * (t_body_rate - t_show)

	def on_platform_pose(self, msg: Pose):
		"""
		Exact platform world pose, published directly by
		OscillatingPlatformController on its own dedicated topic (see
		PLATFORM_POSE_TOPIC and MovingPlatformController.cpp's publishPose) --
		no entity matching needed, since every message on this topic IS the
		platform, by construction. Position is exact; Pose carries no
		velocity, so velocity is finite-differenced against the previous
		message using this callback's own receipt time (same time.time()
		pattern as on_camera/on_local_position), then smoothed -- raw
		frame-to-frame differencing of real, slightly-jittery pose telemetry
		amplifies noise the same way it would for optical flow (see
		optical_flow.py's module docstring for the same argument). Stored in
		the SDF world's own ENU convention; platform_motion.relative_motion()
		handles the NED conversion when this is logged alongside
		vehicle_state. Diagnostics-only -- never read by control_law.
		"""
		now = self.time.wall_sec()
		x, y, z = msg.position.x, msg.position.y, msg.position.z

		self._platform_pose_count += 1
		if self._platform_pose_count == 1:
			self.get_logger().info(
				f"First platform pose received on {PLATFORM_POSE_TOPIC}: "
				f"x={x:.3f} y={y:.3f} z={z:.3f} (SDF world/ENU). "
				"Platform tracking is live."
			)

		if self._prev_platform_pose_t is not None:
			dt = now - self._prev_platform_pose_t
			if dt > 1e-3:
				px, py, pz = self._prev_platform_pose_xyz
				raw_v = ((x - px) / dt, (y - py) / dt, (z - pz) / dt)

				alpha = PLATFORM_VELOCITY_SMOOTHING
				if not self._has_filtered_platform_velocity:
					self._platform_velocity_filtered = raw_v
					self._has_filtered_platform_velocity = True
				else:
					fv = self._platform_velocity_filtered
					self._platform_velocity_filtered = tuple(
						alpha * fv[i] + (1.0 - alpha) * raw_v[i] for i in range(3)
					)

		self._prev_platform_pose_t = now
		self._prev_platform_pose_xyz = (x, y, z)

		vx, vy, vz = self._platform_velocity_filtered
		self._platform_state = PlatformState(
			timestamp=now, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
		)

	def on_touchdown(self, msg: Bool):
		"""Gazebo/TouchPlugin contact event bridged from /bee_platform/touched.

		The event is latched: after the first True sample, the mission is considered
		landed even if the contact signal later drops because the platform moves or
		the vehicle bounces. TouchPlugin's <time> parameter in the SDF already filters
		out single-frame grazes before this callback receives True.
		"""
		self._touchdown_message_count += 1
		if not bool(msg.data):
			return

		if self._touchdown_detected:
			return

		self._touchdown_detected = True
		self._touchdown_time = self.time.wall_sec()
		self.get_logger().warning(
			"Gazebo touchdown detected: platform contact is stable."
		)
		self._publish_touchdown_status(True)

		if self._mission_phase == PHASE_CLOSED_LOOP:
			self._enter_landed_phase("touchdown contact event")

	def on_vehicle_attitude(self, msg: VehicleAttitude):
		"""PX4 attitude telemetry for diagnostics.

		VehicleLocalPosition gives position, velocity, and heading, but not roll/pitch.
		This callback merges roll/pitch/yaw from /fmu/out/vehicle_attitude into the
		shared VehicleState object without feeding it to the visual control law.
		"""
		now = self.time.wall_sec()
		self.time.observe_px4_timestamp(msg.timestamp)
		self._have_vehicle_attitude = True
		self._vehicle_attitude_count += 1

		try:
			roll, pitch, yaw = self._quat_wxyz_to_euler(msg.q)
		except Exception as exc:
			self.get_logger().warning(f"Could not decode vehicle attitude quaternion: {repr(exc)}")
			return

		# Mutate only the attitude fields so the latest local-position values remain
		# intact. on_local_position() below preserves these fields when it rebuilds
		# VehicleState from a new local-position message.
		self._vehicle_state.attitude_timestamp = msg.timestamp / 1e6
		self._vehicle_state.roll = roll
		self._vehicle_state.pitch = pitch
		self._vehicle_state.attitude_yaw = yaw
		self._vehicle_state.attitude_source = "vehicle_attitude"

		if VERBOSE_STREAM_LOGS and now - self._last_attitude_log_time >= 1.0:
			self._last_attitude_log_time = now
			self.get_logger().info(
				f"vehicle attitude: roll={roll:.3f} rad, pitch={pitch:.3f} rad, yaw={yaw:.3f} rad"
			)

	def on_vehicle_angular_velocity(self, msg: VehicleAngularVelocity):
		"""Body-frame angular rate (FRD, rad/s) for optical-flow de-rotation.

		Buffered rather than merged into VehicleState, so on_camera can average
		it over each camera inter-frame interval (see AngularRateBuffer's CLOCK
		NOTE). The sample is tagged with the message's own PX4 stamp in seconds
		(msg.timestamp / 1e6) -- same family/handling as on_vehicle_attitude's
		attitude_timestamp -- and observed for offset diagnostics like the other
		/fmu/out streams. Never feeds the control law directly; the de-rotated
		flow does.
		"""
		self.time.observe_px4_timestamp(msg.timestamp)
		self._rate_buffer.add(msg.timestamp / 1e6, msg.xyz)

	def on_vehicle_status(self, msg: VehicleStatus):
		"""PX4 VehicleStatus: the authority on whether we are ACTUALLY in
		offboard. Tracks nav_state/arming_state/failsafe, logs every nav_state
		transition, and warns loudly if offboard is lost after being achieved
		(the classic 'commands ignored, vehicle holds level and sinks' failure).
		"""
		self._px4_nav_state = int(getattr(msg, "nav_state", -1))
		self._px4_arming_state = int(getattr(msg, "arming_state", -1))
		self._px4_failsafe = bool(getattr(msg, "failsafe", False))

		in_offboard = self._px4_nav_state == PX4_NAV_STATE_OFFBOARD

		if self._px4_nav_state != self._last_nav_state_logged:
			self._last_nav_state_logged = self._px4_nav_state
			self.get_logger().info(
				f"PX4 nav_state -> {_nav_state_name(self._px4_nav_state)} "
				f"(arming={_arming_state_name(self._px4_arming_state)}, "
				f"failsafe={self._px4_failsafe})"
			)

		if in_offboard:
			if not self._px4_offboard_confirmed:
				self._px4_offboard_confirmed = True
				self.get_logger().info("PX4 OFFBOARD confirmed active by VehicleStatus.")
		else:
			# Lost/never-entered offboard while we believe we are flying it.
			if self._px4_offboard_confirmed and self._mission_phase == PHASE_CLOSED_LOOP:
				self.get_logger().error(
					"PX4 DROPPED OFFBOARD during closed-loop control "
					f"(now {_nav_state_name(self._px4_nav_state)}, "
					f"failsafe={self._px4_failsafe}). Attitude/thrust setpoints are "
					"no longer being applied -- this is the 'commands ignored' failure."
				)
				self._px4_offboard_confirmed = False

	def on_local_position(self, msg: VehicleLocalPosition):
		now = self.time.wall_sec()
		self.time.observe_px4_timestamp(msg.timestamp)
		self._have_local_position = True
		if VERBOSE_STREAM_LOGS and now - self._last_position_log_time >= 1.0:
			self._last_position_log_time = now
			self.get_logger().info(f"local position: x={msg.x:.2f} m, y={msg.y:.2f} m, z={msg.z:.2f} m")

		previous = self._vehicle_state
		self._vehicle_state = VehicleState(
			timestamp=now,
			x=msg.x,
			y=msg.y,
			z=msg.z,
			vx=msg.vx,
			vy=msg.vy,
			vz=msg.vz,
			yaw=msg.heading,
			px4_timestamp_sec=msg.timestamp / 1e6,
			attitude_timestamp=getattr(previous, "attitude_timestamp", 0.0),
			roll=getattr(previous, "roll", 0.0),
			pitch=getattr(previous, "pitch", 0.0),
			attitude_yaw=getattr(previous, "attitude_yaw", 0.0),
			attitude_source=getattr(previous, "attitude_source", ""),
		)

	def _log_mission_timer_wall_estimates(self):
		"""LOG-ONLY: print the approximate WALL-clock duration of mission_routine's
		sim-time timers at EXPECTED_SIM_RTF, so 'probe_min_duration_sec=15.0'
		does not silently become ~75 real seconds with no warning. Does not
		affect flight behavior -- see EXPECTED_SIM_RTF's docstring."""
		rtf = max(1e-3, float(EXPECTED_SIM_RTF))
		self.get_logger().info(
			f"Mission sim-time timers at EXPECTED_SIM_RTF={rtf:.2f} (update this "
			"constant if your measured RTF differs -- see analyse_log.py's "
			f"estimate_rtf()): probe_min_duration={PROBE_MIN_DURATION_SEC:.1f}s "
			f"sim (~{PROBE_MIN_DURATION_SEC/rtf:.0f}s wall), "
			f"d_star_ramp_in={D_STAR_RAMP_IN_SEC:.1f}s sim "
			f"(~{D_STAR_RAMP_IN_SEC/rtf:.0f}s wall)."
		)

	def on_mission_timer(self):
		now = self.time.wall_sec()

		if (
			PLATFORM_POSE_TOPIC
			and self._platform_pose_count == 0
			and not self._platform_pose_stall_logged
			and now - self._node_start_time >= 10.0
		):
			self._platform_pose_stall_logged = True
			self.get_logger().warning(
				f"No platform pose received on {PLATFORM_POSE_TOPIC} after "
				f"{now - self._node_start_time:.0f}s. diagnostics will log empty "
				"platform_*/relative_* fields until this is fixed. Check, in order: "
				f"(1) `gz topic -l` shows {PLATFORM_POSE_TOPIC} and `gz topic -e -t "
				"<that topic>` shows live data, not just a topic name that exists -- if "
				"not, OscillatingPlatformController's new publisher may need the plugin "
				"rebuilt/reinstalled, or the .so may be stale; "
				"(2) the ros_gz_bridge process for this topic is actually running; "
				f"(3) `ros2 topic info {PLATFORM_POSE_TOPIC} -v` WHILE this node is "
				"still running (not after stopping it) shows this node's own name as "
				"a subscriber, not just ros_gz_bridge's internal pub/sub pair."
			)

		if self._mission_phase == PHASE_LANDED:
			# Keep the terminal state latched and keep publishing an explicit zero-
			# thrust command until the MAVSDK takeoff worker confirms motor stop or the node
			# is shut down by the user.
			self._latest_setpoint = self._landed_zero_thrust_setpoint()
			self._publish_touchdown_status(True)
			return

		if self._mission_phase == PHASE_ABORTED:
			# Keep streaming a safe inertial hold setpoint instead of simply
			# stopping ROS 2 PX4 offboard. Stopping the stream can trigger a PX4
			# offboard failsafe while the vehicle still has velocity.
			self._latest_setpoint = self._neutral_visual_hold_setpoint()
			return

		if self._mission_phase == PHASE_WAITING_FOR_STREAMS:
			if self._ready_to_start():
				self._enter_phase(PHASE_MAVSDK_TAKEOFF)
			return

		if self._mission_phase == PHASE_MAVSDK_TAKEOFF:
			self.mavsdk.start()
			if self.mavsdk.takeoff_error is not None:
				self._abort(f"MAVSDK takeoff failed: {self.mavsdk.takeoff_error}")
				return
			if self.mavsdk.takeoff_done:
				self.get_logger().info("MAVSDK takeoff complete. Starting ROS 2 PX4 offboard prestream.")
				self._enter_phase(PHASE_PRESTREAM)
			return

		if self._mission_phase == PHASE_PRESTREAM:
			# Let the direct ROS 2 PX4 publisher stream a stable hover setpoint before switching to offboard.
			self._latest_setpoint = AttitudeSetpoint(
				timestamp=getattr(self._latest_target, "timestamp", 0.0),
				roll=0.0,
				pitch=0.0,
				yaw=0.0,
				thrust=self.control_law.hover_thrust,
			)
			if now - self._phase_start_time >= OFFBOARD_PRESTREAM_SEC:
				self.get_logger().info("Requesting PX4 offboard mode through ROS 2/uXRCE-DDS.")
				self.px4_interface.engage_offboard_mode()
				self._px4_offboard_start_requested = True
				self._px4_offboard_request_time = now
				self._enter_phase(PHASE_WAIT_OFFBOARD)
			return

		if self._mission_phase == PHASE_WAIT_OFFBOARD:
			if self._px4_offboard_error is not None:
				self._abort(f"PX4 offboard start failed: {self._px4_offboard_error}")
				return

			waited = (
				now - self._px4_offboard_request_time
				if self._px4_offboard_request_time is not None else 0.0
			)

			# Preferred path: proceed only once VehicleStatus CONFIRMS offboard.
			if self._px4_nav_state == PX4_NAV_STATE_OFFBOARD:
				self._px4_offboard_started = True
				self.control_law.reset_visual_integrators()
				self._last_controlled_flow_timestamp = None
				self.mission.start(
					t=float(getattr(self._latest_flow, "timestamp", now)),
					start_height_m=TAKEOFF_ALTITUDE_M-2,
				)
				self._log_mission_timer_wall_estimates()
				self.get_logger().info(
					"ROS 2 PX4 attitude offboard stream is active and CONFIRMED in "
					f"offboard. Closed-loop visual controller is now active. "
					f"Mission probe starting (h0={TAKEOFF_ALTITUDE_M-2:.2f} m seed)."
				)
				self._enter_phase(PHASE_CLOSED_LOOP)
				return

			# Not yet confirmed: keep re-commanding offboard. PX4 only accepts the
			# switch after it has seen a few heartbeats, so the first command can
			# be too early; re-issuing every interval is harmless and robust.
			if waited - self._px4_offboard_reengage_count * PX4_OFFBOARD_REENGAGE_INTERVAL_SEC \
					>= PX4_OFFBOARD_REENGAGE_INTERVAL_SEC:
				self._px4_offboard_reengage_count += 1
				self.px4_interface.engage_offboard_mode()

			if waited >= PX4_OFFBOARD_CONFIRM_TIMEOUT_SEC:
				if self._px4_nav_state is None:
					# Never received VehicleStatus -- topic name/bridge mismatch.
					# Fall back to the old optimistic behavior so we don't hard-
					# block, but warn loudly: offboard is UNCONFIRMED.
					self.get_logger().warning(
						"No VehicleStatus received after "
						f"{PX4_OFFBOARD_CONFIRM_TIMEOUT_SEC:.1f}s -- cannot confirm "
						"offboard. Check the vehicle_status topic name/bridge (see "
						"on_vehicle_status). Proceeding UNCONFIRMED; if the vehicle "
						"holds level and ignores commands, offboard did not engage."
					)
					self._px4_offboard_started = True
					self.control_law.reset_visual_integrators()
					self._last_controlled_flow_timestamp = None
					self.mission.start(
						t=float(getattr(self._latest_flow, "timestamp", now)),
						start_height_m=TAKEOFF_ALTITUDE_M,
					)
					self._log_mission_timer_wall_estimates()
					self._enter_phase(PHASE_CLOSED_LOOP)
					return
				# We DO see status, but it never became offboard -> PX4 rejected it.
				self._abort(
					"PX4 rejected offboard: nav_state stuck at "
					f"{_nav_state_name(self._px4_nav_state)} after "
					f"{PX4_OFFBOARD_CONFIRM_TIMEOUT_SEC:.1f}s "
					f"(arming={_arming_state_name(self._px4_arming_state)}, "
					f"failsafe={self._px4_failsafe}). Common causes: setpoint stream "
					"gap/timestamp-clock mismatch (see clock.py), not armed, or a "
					"failsafe blocking the mode switch."
				)
				return
			return

		if self._mission_phase == PHASE_CLOSED_LOOP:
			if self._touchdown_detected:
				self._enter_landed_phase("latched touchdown flag")
				return

			if not self._closed_loop_logged:
				self._closed_loop_logged = True
				self.get_logger().info("Closed-loop visual landing/hover attempt running.")
			return

	def on_px4_setpoint_timer(self):
		"""Publish the latest attitude/thrust setpoint directly to PX4.

		This is the deliberately fixed-cadence zero-order-hold output of the
		visual controller. The controller may poll for fresh vision faster than
		this timer, but this timer is the single publication cadence used for
		stabilization analysis and for PX4 offboard keepalive.
		"""
		if self._mission_phase not in (
			PHASE_PRESTREAM,
			PHASE_WAIT_OFFBOARD,
			PHASE_CLOSED_LOOP,
			PHASE_LANDED,
			PHASE_ABORTED,
		):
			return

		# If an early abort happens before the ROS 2 offboard handoff, do not
		# inject external setpoints into PX4 while it is still in its previous mode.
		if self._mission_phase == PHASE_ABORTED and not self._px4_offboard_started:
			return

		sp = self._latest_setpoint
		yaw_rad = sp.yaw
		if PX4_HOLD_CURRENT_YAW and self._vehicle_state.timestamp > 0.0:
			yaw_rad = self._vehicle_state.yaw

		try:
			# One wall-clock instant for the whole cycle, shared by the
			# heartbeat and the setpoint so PX4 sees them as one coherent pair.
			tx_us = self.time.px4_tx_timestamp_us()
			publish_wall = tx_us / 1e6

			self.px4_interface.publish_heartbeat(tx_us)
			self.px4_interface.publish_attitude_setpoint(
				sp.roll,
				sp.pitch,
				yaw_rad,
				self._clamp(sp.thrust, 0.0, 1.0),
				timestamp_us=tx_us,
			)

			if self._prev_px4_publish_wall is not None:
				self._last_px4_publish_period_wall_sec = publish_wall - self._prev_px4_publish_wall
			self._prev_px4_publish_wall = publish_wall
			self._last_px4_publish_wall = publish_wall
			self._px4_publish_count += 1

			if self._latest_setpoint_compute_wall is not None:
				self._last_px4_command_age_ms = 1000.0 * (
					publish_wall - self._latest_setpoint_compute_wall
				)

			flow_wall = self.time.sim_to_wall_sec(getattr(self._latest_flow, "timestamp", None))
			self._last_px4_flow_age_ms = (
				1000.0 * (publish_wall - flow_wall) if flow_wall is not None else None
			)
		except Exception as exc:
			self._px4_offboard_error = repr(exc)
			self.get_logger().error(f"PX4 direct setpoint publication failed: {repr(exc)}")

	def on_control_timer(self):
		now = self.time.wall_sec()

		# v2.0: pick up whatever the vision worker has produced since the last
		# tick before doing anything else -- this is what advances
		# _latest_target/_latest_flow now that on_camera only ships frames.
		# Drained unconditionally (every phase) so the output queue never backs
		# up, and BEFORE the have-a-frame gate below so _latest_flow can leave
		# None once the first result arrives.
		self._drain_vision_results()

		if self._latest_flow is None or self._latest_frame is None:
			return

		if self._mission_phase == PHASE_LANDED:
			self._latest_setpoint = self._landed_zero_thrust_setpoint()
			self._publish_touchdown_status(True)
			self._write_diagnostics_row()
			return

		if self._mission_phase == PHASE_ABORTED:
			self._latest_setpoint = self._neutral_visual_hold_setpoint()
			self._write_diagnostics_row()
			return

		if self._mission_phase != PHASE_CLOSED_LOOP:
			return

		if self._touchdown_detected:
			self._enter_landed_phase("latched touchdown flag")
			self._write_diagnostics_row()
			return

		# Optional diagnostic-only safety aborts. Disabled by default because the
		# project constraint is that PX4 state must not participate in the visual
		# control logic after handoff. When disabled, these states are logged only.
		if ENABLE_INERTIAL_SAFETY_ABORTS:
			if abs(self._vehicle_state.vz) > SAFETY_VZ_LIMIT:
				self._abort(f"vertical velocity safety limit exceeded: vz={self._vehicle_state.vz:.3f} m/s")
				self._latest_setpoint = self._neutral_visual_hold_setpoint()
				self._write_diagnostics_row()
				return

			if (
				abs(self._vehicle_state.vx) > SAFETY_LATERAL_VELOCITY_LIMIT
				or abs(self._vehicle_state.vy) > SAFETY_LATERAL_VELOCITY_LIMIT
			):
				self._abort(
					"lateral velocity safety limit exceeded: "
					f"vx={self._vehicle_state.vx:.3f} m/s, vy={self._vehicle_state.vy:.3f} m/s"
				)
				self._latest_setpoint = self._neutral_visual_hold_setpoint()
				self._write_diagnostics_row()
				return

		target_ok = bool(getattr(self._latest_target, "found", False))
		flow_ok = bool(getattr(self._latest_flow, "valid", False))

		if not (target_ok and flow_ok):
			if self._lost_target_since is None:
				self._lost_target_since = now
			elif now - self._lost_target_since >= LOST_TARGET_TIMEOUT_SEC:
				self._abort(
					f"target/flow lost for >= {LOST_TARGET_TIMEOUT_SEC:.1f}s "
					f"(target_found={target_ok}, flow_valid={flow_ok})"
				)
				self._latest_setpoint = self._neutral_visual_hold_setpoint()
				self._write_diagnostics_row()
				return
		else:
			self._lost_target_since = None

		if COMPUTE_CONTROL_ONLY_ON_NEW_VISION:
			flow_stamp = float(getattr(self._latest_flow, "timestamp", 0.0))
			if flow_stamp <= 0.0:
				return
			if (
				self._last_controlled_flow_timestamp is not None
				and flow_stamp <= self._last_controlled_flow_timestamp + 1e-9
			):
				# The 100 Hz control timer has caught the same camera/flow sample
				# again. Leave _latest_setpoint untouched; the PX4 setpoint timer
				# will keep publishing it at its deliberately fixed 20 Hz cadence.
				return
			self._last_controlled_flow_timestamp = flow_stamp

		control_compute_start_wall = self.time.wall_sec()
		if self._last_control_compute_start_wall is not None:
			self._last_control_period_wall_sec = (
				control_compute_start_wall - self._last_control_compute_start_wall
			)
		self._last_control_compute_start_wall = control_compute_start_wall

		# v2.0 headline latency: frame arrival (on_camera) -> this command,
		# spanning the whole out-of-process vision round trip PLUS the wait for a
		# control tick to pick the result up. Measured only on the ticks that
		# actually consume a NEW vision sample (this point is past the
		# COMPUTE_CONTROL_ONLY_ON_NEW_VISION gate), in this process's wall clock.
		# This is the number that decides whether v2.0 helped the loop -- compare
		# it against v1.0's blocking on_camera duration (~11-18ms).
		if self._latest_flow_frame_wall is not None:
			self._latest_frame_to_command_ms = 1000.0 * (
				control_compute_start_wall - self._latest_flow_frame_wall
			)

		# Mission routine: probe -> gate -> scheduled-gain descent. Feed it
		# last cycle's ACTUAL commanded thrust (the efference copy of what has
		# really been acting on the vehicle since the previous tick) before
		# that field gets overwritten below by this tick's new setpoint.
		t_vision = float(getattr(self._latest_flow, "timestamp", now))
		dt = self._control_dt_sec()
		self._last_control_dt_vision_sec = dt
		previous_thrust = float(getattr(self._latest_setpoint, "thrust", self.control_law.hover_thrust))

		previous_substate = self.mission.substate
		tgt = self._latest_target
		mc = self.mission.update(
			t_vision, dt, previous_thrust,
			offset_x=float(getattr(tgt, "offset_x", 0.0)),
			offset_y=float(getattr(tgt, "offset_y", 0.0)),
			target_found=bool(getattr(tgt, "found", False)),
		)
		self._latest_mission_control = mc

		if mc.substate != previous_substate:
			self.get_logger().info(f"Mission substate: {previous_substate} -> {mc.substate}")
		if mc.info.get("event") == "center_done":
			# CENTER runs at full lateral authority while the target is often
			# far off-center: large banking + noisy divergence (the same
			# tilt-contamination that motivated resetting the platform probe
			# here -- see mission_routine._do_center). Clear ONLY the divergence
			# integral so that transient does not carry a bias into the
			# following probe/hover. Deliberately NOT reset_visual_integrators():
			# that also rebases the command-shaping filter to (0, 0, hover),
			# which would step the commanded attitude/thrust at the handoff
			# instead of continuing smoothly from wherever centering left off.
			self.control_law.reset_divergence_integral()
			self.get_logger().info(
				"CENTER done -> cleared divergence integral before probe (filter state kept continuous)."
			)
		if now - self._last_mission_log_time >= 2.0:
			self._last_mission_log_time = now
			self.get_logger().info(self.mission.status_line())

		if mc.substate == MISSION_INFEASIBLE:
			if not self._mission_infeasible_logged:
				self._mission_infeasible_logged = True
				self.get_logger().error(
					"Mission gate: INFEASIBLE -- "
					f"h_crit={self.mission.gate.h_crit:.2f} m exceeds leg clearance "
					f"({LEG_CLEARANCE_M:.2f} m) at the current control rate "
					f"({CONTROL_PERIOD_SEC:.2f} s period). No constant gain can both "
					"reject this platform's motion and stay below the de Croon "
					"self-induced-oscillation ceiling down to touchdown. Aborting "
					"descent; holding visual hover."
				)
			self._abort("mission gate infeasible: platform motion exceeds achievable gain band")
			self._latest_setpoint = self._neutral_visual_hold_setpoint()
			self._write_diagnostics_row()
			return

		self._latest_setpoint = self.control_law.compute(
			self._latest_target,
			self._latest_flow,
			dt,
			divergence_setpoint=mc.divergence_setpoint,
			thrust_gain_override=mc.thrust_gain_override,
			lateral_p_scale=mc.lateral_p_scale,
			lateral_d_scale=mc.lateral_d_scale,
			enable_integral=mc.enable_integral,
		)
		self._latest_setpoint_compute_wall = control_compute_start_wall
		self._latest_setpoint_flow_timestamp = getattr(self._latest_flow, "timestamp", None)
		self._last_control_compute_end_wall = self.time.wall_sec()
		self._last_control_compute_duration_ms = (
			1000.0 * (self._last_control_compute_end_wall - control_compute_start_wall)
		)
		self._write_diagnostics_row()

	def _control_dt_sec(self) -> float:
		"""Return the control step in the same clock as optical flow.

		OpticalFlowEstimator converts px/frame into px/s using FlowResult.timestamp,
		which comes from the camera image timestamp. Therefore the control law's
		integral and slew-rate terms must use the delta between consecutive visual
		timestamps, not a PX4 timestamp and not wall-clock time. This keeps the
		units of flow/divergence [1/s] and controller dt [s] consistent.
		"""
		current = float(getattr(self._latest_flow, "timestamp", 0.0))
		previous = self._prev_control_flow_timestamp
		self._prev_control_flow_timestamp = current

		if current <= 0.0 or previous is None:
			return CONTROL_PERIOD_SEC

		delta = current - previous

		# Same frame / paused sim / duplicate image timestamp: do almost nothing,
		# but keep control_law.compute() numerically happy.
		if 0.0 <= delta <= 1e-6:
			return 1e-3

		# Reject true timestamp glitches without mixing in PX4/wall clocks.
		if not (0.0 < delta <= 10.0 * CONTROL_PERIOD_SEC):
			if not self._control_dt_fallback_logged:
				self._control_dt_fallback_logged = True
				self.get_logger().warning(
					f"Implausible visual timestamp dt ({delta:.4f}s, nominal "
					f"{CONTROL_PERIOD_SEC:.4f}s). Falling back to the fixed "
					"control period for this tick."
				)
			return CONTROL_PERIOD_SEC

		return delta

	def _image_timestamp_sec(self, msg: Image) -> float:
		"""Return the timestamp used by target acquisition and optical flow.

		Preferred source: sensor_msgs/Image.header.stamp, normally filled by
		ros_gz_bridge from Gazebo simulation time. If it is missing/zero, fall back
		to PX4's simulated timestamp if available. Wall-clock is only a last-resort
		startup fallback. The returned value is only compared to previous image
		timestamps, never to PX4/wall timestamps by absolute value.
		"""
		stamp = getattr(getattr(msg, "header", None), "stamp", None)
		stamp_sec = self._ros_stamp_to_sec(stamp)
		if stamp_sec > 0.0:
			return stamp_sec

		px4_time = float(getattr(self._vehicle_state, "px4_timestamp_sec", 0.0))
		if px4_time > 0.0:
			if not self._image_stamp_fallback_logged:
				self._image_stamp_fallback_logged = True
				self.get_logger().warning(
					"Camera Image.header.stamp is zero; using PX4 timestamp as the "
					"vision timestamp. Prefer fixing the camera bridge so images carry "
					"Gazebo sim time."
				)
			return px4_time

		if not self._image_stamp_fallback_logged:
			self._image_stamp_fallback_logged = True
			self.get_logger().warning(
				"Camera Image.header.stamp and PX4 timestamp are unavailable; "
				"temporarily using wall-clock for vision timestamps."
			)
		return self.time.wall_sec()

	@staticmethod
	def _quat_wxyz_to_euler(q):
		"""Convert a PX4 Hamilton quaternion [w, x, y, z] to roll/pitch/yaw [rad]."""
		if len(q) != 4:
			raise ValueError(f"expected 4 quaternion components, got {len(q)}")
		w, x, y, z = [float(value) for value in q]

		# Normalize defensively. PX4 should already publish a unit quaternion, but
		# normalization avoids occasional startup/transport numerical weirdness.
		norm = math.sqrt(w * w + x * x + y * y + z * z)
		if norm <= 1e-12:
			raise ValueError("zero-norm vehicle attitude quaternion")
		w, x, y, z = w / norm, x / norm, y / norm, z / norm

		sinr_cosp = 2.0 * (w * x + y * z)
		cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
		roll = math.atan2(sinr_cosp, cosr_cosp)

		sinp = 2.0 * (w * y - z * x)
		if abs(sinp) >= 1.0:
			pitch = math.copysign(math.pi / 2.0, sinp)
		else:
			pitch = math.asin(sinp)

		siny_cosp = 2.0 * (w * z + x * y)
		cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
		yaw = math.atan2(siny_cosp, cosy_cosp)

		return roll, pitch, yaw

	@staticmethod
	def _ros_stamp_to_sec(stamp) -> float:
		if stamp is None:
			return 0.0
		try:
			return float(stamp.sec) + 1e-9 * float(stamp.nanosec)
		except AttributeError:
			return 0.0

	def _write_diagnostics_row(self):
		self.diagnostics.write(
			wall_timestamp=self.time.wall_sec(),
			target=self._latest_target,
			flow=self._latest_flow,
			setpoint=self._latest_setpoint,
			vehicle_state=self._vehicle_state,
			platform_state=self._platform_state,
			px4_wall_offset_sec=self.time.px4_wall_offset_sec(),
			sim_wall_offset_sec=self.time.sim_wall_offset_sec(),
			mission=self._mission_telemetry(),
			px4_nav_state=self._px4_nav_state,
			px4_arming_state=self._px4_arming_state,
			px4_failsafe=self._px4_failsafe,
			divergence_integral=self.control_law.divergence_integral,
			timing=self._timing_telemetry(),
		)

	def _timing_telemetry(self):
		"""Timing/latency diagnostics only -- never feeds the controller."""
		flow_wall = self.time.sim_to_wall_sec(getattr(self._latest_flow, "timestamp", None))
		control_wall = self._last_control_compute_start_wall
		flow_age_at_control_ms = (
			1000.0 * (control_wall - flow_wall)
			if control_wall is not None and flow_wall is not None else None
		)
		return {
			"camera_cb_start_wall_sec": self._latest_camera_cb_start_wall,
			"camera_cb_end_wall_sec": self._latest_camera_cb_end_wall,
			"camera_cb_duration_ms": self._latest_camera_cb_duration_ms,
			"stage_bridge_ms": self._latest_stage_bridge_ms,
			"stage_rotate_ms": self._latest_stage_rotate_ms,
			"stage_show_camera_ms": self._latest_stage_show_camera_ms,
			"stage_body_rate_ms": self._latest_stage_body_rate_ms,
			# v2.0: these two on_camera sub-stages are structurally empty now
			# (target_acquisition/optical_flow moved out of process). Kept as
			# columns so the on_camera decomposition schema is unchanged; the
			# real per-call cost is in worker_* below.
			"stage_target_acquisition_ms": self._latest_stage_target_acquisition_ms,
			"stage_optical_flow_ms": self._latest_stage_optical_flow_ms,
			# v2.0 out-of-process vision measurements. worker_* = the worker's own
			# perf_counter cost of each call (compute moved, not reduced -- expect
			# ~v1.0 numbers here). frame_to_available/frame_to_command = the true
			# cross-boundary latencies the v1.0-era stage timers never captured;
			# frame_to_command is the one to compare against v1.0's blocking
			# on_camera (~11-18ms). vision_result_period_ms >> camera period means
			# the worker is falling behind; vision_dropped_frames counts frames
			# shed at the shallow input queue under load.
			"worker_target_acquisition_ms": self._latest_worker_target_acquisition_ms,
			"worker_optical_flow_ms": self._latest_worker_optical_flow_ms,
			"frame_to_available_wall_ms": self._latest_frame_to_available_ms,
			"frame_to_command_wall_ms": self._latest_frame_to_command_ms,
			"vision_result_period_wall_ms": self._latest_vision_result_period_ms,
			"vision_dropped_frames": self._vision_dropped_frames,
			"control_compute_start_wall_sec": self._last_control_compute_start_wall,
			"control_compute_end_wall_sec": self._last_control_compute_end_wall,
			"control_compute_duration_ms": self._last_control_compute_duration_ms,
			"control_period_wall_sec": self._last_control_period_wall_sec,
			"control_dt_vision_sec": self._last_control_dt_vision_sec,
			"flow_age_at_control_wall_ms": flow_age_at_control_ms,
			"px4_publish_wall_sec": self._last_px4_publish_wall,
			"px4_publish_period_wall_sec": self._last_px4_publish_period_wall_sec,
			"command_age_at_px4_publish_ms": self._last_px4_command_age_ms,
			"flow_age_at_px4_publish_wall_ms": self._last_px4_flow_age_ms,
			"px4_publish_count": self._px4_publish_count,
			"sim_rtf_estimate": self.time.sim_rtf_estimate(),
			"px4_rtf_estimate": self.time.px4_rtf_estimate(),
		}

	def _mission_telemetry(self):
		"""Flatten the latest MissionControl + mission bounds into a plain dict
		for the diagnostics CSV. None before closed-loop (mission not started),
		so those rows leave the mission_* columns blank."""
		mc = self._latest_mission_control
		if mc is None:
			return None
		info = getattr(mc, "info", {}) or {}
		gate = self.mission.gate
		return {
			"substate": mc.substate,
			"divergence_setpoint": mc.divergence_setpoint,
			"thrust_gain_k": mc.thrust_gain_override,
			"lateral_p_scale": mc.lateral_p_scale,
			"lateral_d_scale": mc.lateral_d_scale,
			"k_min": gate.k_min,
			"k_explore": gate.k_explore,
			"h_crit": gate.h_crit,
			# h_pred only exists during descent; blank otherwise.
			"h_pred": info.get("h_pred"),
			"peak_accel": self.mission.probe_result.peak_accel,
			"feasible": gate.feasible,
		}

	def _ready_to_start(self) -> bool:
		if not self._have_local_position:
			return False
		if self._latest_frame is None:
			return False
		if not self._streams_ready_logged:
			self._streams_ready_logged = True
			self.get_logger().info("Required streams are available; starting automatic climb.")
		return True

	def _publish_touchdown_status(self, value: bool):
		msg = Bool()
		msg.data = bool(value)
		self._touchdown_status_pub.publish(msg)

	def _latch_zero_thrust_hold(self):
		"""Invoked by MavsdkWorker (on the worker thread) just before disarm, so
		our outgoing setpoint is a zero-thrust hold and nothing fights the stop.
		A single attribute assignment -- safe to call cross-thread."""
		self._latest_setpoint = self._landed_zero_thrust_setpoint()

	def _landed_zero_thrust_setpoint(self) -> AttitudeSetpoint:
		"""Terminal setpoint after confirmed touchdown.

		This is deliberately separate from _neutral_visual_hold_setpoint(): abort keeps
		hover thrust, while a successful landing commands zero thrust until PX4/MAVSDK
		accepts disarm or kill.
		"""
		return AttitudeSetpoint(
			timestamp=getattr(self._latest_target, "timestamp", 0.0),
			roll=0.0,
			pitch=0.0,
			yaw=0.0,
			thrust=0.0,
		)

	def _enter_landed_phase(self, reason: str):
		if self._mission_phase == PHASE_LANDED:
			return

		self.get_logger().warning(f"LANDING COMPLETE: {reason}. Entering landed phase.")
		self._latest_setpoint = self._landed_zero_thrust_setpoint()
		self._publish_touchdown_status(True)
		if ENABLE_TOUCHDOWN_MOTOR_STOP:
			self.mavsdk.request_motor_stop()
		else:
			self.get_logger().warning(
				"Touchdown motor stop disabled; landed phase will only stream zero thrust."
			)
		self._enter_phase(PHASE_LANDED)

	def _neutral_visual_hold_setpoint(self) -> AttitudeSetpoint:
		"""Neutral visual-hover setpoint used after abort/target loss.

		This deliberately does NOT use PX4 local position or velocity. After
		handoff, PX4 state is diagnostics-only; this fallback simply keeps the
		ROS 2 PX4 offboard stream alive with zero roll/pitch and nominal hover thrust
		until the user stops the node.
		"""
		return AttitudeSetpoint(
			timestamp=getattr(self._latest_target, "timestamp", 0.0),
			roll=0.0,
			pitch=0.0,
			yaw=0.0,
			thrust=self.control_law.hover_thrust,
		)


	def _enter_phase(self, phase: str):
		if phase != self._mission_phase:
			self.get_logger().info(f"Mission phase: {self._mission_phase} -> {phase}")
		self._mission_phase = phase
		self._phase_start_time = self.time.wall_sec()

	def _abort(self, reason: str):
		if self._mission_phase != PHASE_ABORTED:
			self.get_logger().error(f"ABORTING bee_land_node: {reason}")
		self._mission_phase = PHASE_ABORTED
		# If ROS 2 offboard was never started, there is no stream to maintain.
		# Once offboard is active, keep publishing _neutral_visual_hold_setpoint()
		# until the user stops the node.
		if not self._px4_offboard_started:
			self.mavsdk.request_stop()

	@staticmethod
	def _clamp(value: float, lower: float, upper: float) -> float:
		return max(lower, min(upper, float(value)))


def main(args=None):
	rclpy.init(args=args)
	node = BeeLandNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		# Best-effort teardown. On a Ctrl+C shutdown some subsystems may already
		# be partway down -- notably the MAVSDK asyncio loop, whose request_stop()
		# can raise "Event loop is closed". Isolate each step so a late error
		# neither dumps a traceback after a clean landing nor skips the rest of
		# teardown (previously request_stop() ran first, so that RuntimeError
		# aborted shutdown_vision_worker/diagnostics.close/destroy_node entirely).
		for teardown in (
			node.mavsdk.request_stop,
			node.shutdown_vision_worker,
			node.diagnostics.close,
			node.destroy_node,
		):
			try:
				teardown()
			except Exception:
				pass
		if SHOW_CAMERA:
			try:
				with suppress_stderr_fd(True):
					cv2.destroyAllWindows()
			except Exception:
				pass
		try:
			if rclpy.ok():
				rclpy.shutdown()
		except Exception:
			pass


if __name__ == "__main__":
	main()