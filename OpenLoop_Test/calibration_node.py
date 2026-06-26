"""
Integrated takeoff + open-loop calibration node.

This node replaces the previous two-process workflow:

    MAVSDK takeoff.py  ->  ros2 calibration_node

with one launch process:

    wait for local position/camera
    -> perform normal PX4/MAVSDK action takeoff
    -> pre-stream attitude setpoints
    -> switch PX4 to attitude OFFBOARD and verify actual roll response
    -> settle vertical velocity
    -> run the open-loop calibration sequence

The important point is that one node owns the whole sequence. PX4 first uses
its normal action-takeoff path, then the same MAVSDK connection switches to
attitude offboard and streams the calibration setpoints. This avoids the
previous ROS VehicleAttitudeSetpoint handoff failure.

Only the final open-loop calibration phase is written as identification data.
Takeoff and settle phases are intentionally not logged to the fit CSV.

Safety/damping, after a real run exposed three compounding gaps in this:

  1. The damper's sign was backwards (subtracting vz's correction instead
     of adding it) -- not a magnitude problem, active positive-feedback
     instability: descending reduced thrust further, which accelerated
     the descent further. Fixed in VerticalVelocityDamper.
  2. Damping (and the safety-bounds check) didn't start until PHASE_VZ_SETTLE
     -- everything from PHASE_PRESTREAM through PHASE_ALTITUDE_SETTLE held a
     bare, uncorrected HOVER_THRUST with zero protection. Both now start as
     soon as offboard thrust commands do (see _damped_hover_setpoint and the
     centralized check at the top of on_mission_timer).
  3. Settling only checked vz, never altitude -- a vehicle resting on the
     ground/platform also reads vz≈0, since the surface's normal force, not
     a real hover equilibrium, is what stopped it. VerticalSettler now also
     requires being within ALTITUDE_SETTLE_TOLERANCE_M of the altitude
     reached right after takeoff, captured once as soon as takeoff
     completes and held fixed through every phase after that (DAMPER_KZ is
     the term that actively pulls back toward it, not just toward zero
     velocity).

None of this is a substitute for HOVER_THRUST actually being close to
correct -- it bounds how much an imperfect guess can cost, it doesn't
remove the value of a better guess.

A fourth, separate gap: roll/pitch=0 zeroes lateral *acceleration*, not
velocity, same structural issue as the vertical case one axis over. A
real run showed target_offset_x starting well off-center and drifting
further -- OFFBOARD_PROBE deliberately commands a real roll to verify
attitude response, and nothing corrected the resulting lateral velocity
afterward. LateralVelocityDamper addresses this, active during every
pre-test phase and the inter-axis "settle" gaps within the open-loop
sequence, but deliberately NEVER during any axis' own active test or
during OFFBOARD_PROBE itself -- both would reintroduce exactly the
cause/effect entanglement this whole setup exists to avoid. Its sign
convention is derived from the same ZYX Euler convention
_euler_to_quaternion/_quaternion_to_euler use, not assumed, and checked
numerically before trusting it -- see LateralVelocityDamper's docstring.
"""

import asyncio
import math
import os
import signal
import subprocess
import threading
import time

import cv2
import rclpy
from rclpy.node import Node

try:
	from mavsdk import System
	from mavsdk.offboard import Attitude as MavsdkAttitude, OffboardError
except ImportError:
	System = None
	MavsdkAttitude = None
	OffboardError = Exception
from rclpy.qos import (
	QoSProfile,
	QoSReliabilityPolicy,
	QoSDurabilityPolicy,
	QoSHistoryPolicy,
)

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

try:
	from px4_msgs.msg import VehicleOdometry
except ImportError:
	VehicleOdometry = None

try:
	from px4_msgs.msg import VehicleStatus
except ImportError:
	VehicleStatus = None

try:
	from px4_msgs.msg import VehicleCommandAck
except ImportError:
	VehicleCommandAck = None

from cv_bridge import CvBridge, CvBridgeError

from .state import VehicleState, PlatformState, AttitudeSetpoint, TargetEstimate
from .optical_flow import OpticalFlowEstimator
from .target_acquisition import TargetAcquisition
from .px4_interface import PX4Interface
from .diagnostics_writer import DiagnosticsWriter
from .calibration_sequence import (
	build_calibration_sequence,
	VerticalSettler,
	VerticalVelocityDamper,
	LateralVelocityDamper,
	exceeds_safety_bounds,
)


HEARTBEAT_PERIOD_SEC = 0.1
MISSION_PERIOD_SEC = 0.1
CALIBRATION_LOG_PERIOD_SEC = 0.5  # keep fit_axis_models.py --dt default at 0.5 s

# Keep the terminal readable during long tests. Turn this on only when
# debugging raw streams. Phase changes, aborts, takeoff/offboard milestones,
# and calibration milestones are still always logged.
VERBOSE_STREAM_LOGS = False

# OpenCV windows are useful during vision debugging, but cv2/Qt can spam font
# warnings in WSL. Keep it off by default for calibration runs.
SHOW_CAMERA = True

# Takeoff strategy. The previous pure attitude-thrust offboard takeoff failed
# because PX4 did not confirm OFFBOARD+ARMED, so the node was commanding
# thrust into a mode that was not actually accepting it. Use the same PX4
# action-takeoff path that already worked in takeoff.py, but launch it from
# this node and only switch to attitude-offboard after the vehicle is airborne.
USE_MAVSDK_TAKEOFF = True

# Use MAVSDK for the attitude-offboard handoff too. The previous ROS
# VehicleAttitudeSetpoint handoff did take off, but the roll probe showed PX4
# was not accepting the attitude setpoints. MAVSDK's offboard plugin switches
# the mode and streams the same roll/pitch/thrust setpoints through the path
# PX4 already accepted for takeoff.
CONTROL_BACKEND = "mavsdk_offboard"  # "mavsdk_offboard" or "ros_attitude"
# A real run with "udp://" connected successfully (confirmed: "MAVSDK:
# connected." appeared right on schedule) -- so the earlier caution about
# udp:// vs udpin:// behaving differently wasn't warranted, they're the
# same thing here. MAVSDK's own runtime warning ("Connection using udp://
# is deprecated, please use udpin:// or udpout://") is the actual reason
# to use this form going forward.
MAVSDK_SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
MAVSDK_PORT_TO_FREE = 14540
MAVSDK_OFFBOARD_PERIOD_SEC = 0.05
MAVSDK_OFFBOARD_START_TIMEOUT_SEC = 5.0
MAVSDK_HOLD_CURRENT_YAW = True
# Set the takeoff height from the node via the Action plugin
# (set_takeoff_altitude), NOT the raw parameter plugin -- param.set_param_float
# is the call that dropped the mavsdk_server socket on recent PX4 builds, and a
# server crash there aborts the whole mission. Set False to skip it entirely and
# rely on PX4's current MIS_TAKEOFF_ALT (e.g. `param set MIS_TAKEOFF_ALT 5` once
# in the pxh console); takeoff() uses whatever PX4 already has.
MAVSDK_SET_TAKEOFF_ALT = True
EKF2_SETTLE_TIME = 5.0

# Explicit timeouts for every "wait for X" step in the MAVSDK handshake.
# Before this, none of connection/health/altitude-reached had any timeout
# at all: if any single one of them never resolved (wrong system address,
# SITL not up yet, a health flag that never goes true), the node would
# sit in PHASE_MAVSDK_TAKEOFF silently forever -- no abort, no error, no
# CSV rows. That produces exactly an empty diagnostics file with nothing
# to explain why. Each step now fails loudly and specifically instead.
MAVSDK_CONNECT_TIMEOUT_SEC = 15.0
MAVSDK_HEALTH_TIMEOUT_SEC = 30.0
# Raised again alongside TAKEOFF_ALTITUDE_M going from 3.0 to 6.0: the
# observed climb rate isn't constant, it decelerates approaching the
# target (PX4's own position control slowing down on approach), so
# doubling the altitude needs more than double the time. Extrapolating
# the measured post-ramp rate (~0.09 m/s, after PX4's ~15s initial
# ramp) to the new, larger remaining distance gives a rough estimate of
# ~80s -- likely optimistic, since that deceleration should bite earlier
# and harder approaching a taller target. 130s gives real margin instead
# of repeating "estimated just enough, then wasn't quite."
MAVSDK_TAKEOFF_ALTITUDE_TIMEOUT_SEC = 130.0
# Backstop covering the whole PHASE_MAVSDK_TAKEOFF mission phase, in case
# something other than the three waits above hangs (e.g. arm()/takeoff()
# themselves). Should rarely fire if the three timeouts above are doing
# their job; it's there so NOTHING in this phase can hang forever. Kept
# comfortably above connect+health+settle+the altitude wait combined.
MAVSDK_TAKEOFF_PHASE_TIMEOUT_SEC = 200.0

# Integrated takeoff. PX4 local position is NED, so climbing means z becomes
# more negative. The target z is captured as z_start - TAKEOFF_ALTITUDE_M when
# the climb phase begins.
#
# Raised from 3.0: a real run reached 3.0m and measured area_fraction=0.979
# right there -- the platform fills ~98% of the frame at that height, which
# is exactly what ABORT_AREA_FRACTION_MAX=0.97 exists to catch, and it
# tripped on the very first tick of open-loop testing before any test
# command was even issued. area_fraction scales roughly as 1/altitude^2
# for a fixed real-world target size, so 6.0 = 3.0 * sqrt(0.979/target),
# solved for a target of ~0.25 (comfortable margin below the abort bound,
# leaving room for the open-loop test's own excursions on top of it). If
# 6.0 still isn't enough clearance, re-solve the same way using whatever
# area_fraction this value actually produces.
TAKEOFF_ALTITUDE_M = 5
OFFBOARD_PRESTREAM_SEC = 2.0
OFFBOARD_CONFIRM_TIMEOUT_SEC = 5.0
TAKEOFF_ALTITUDE_TOL_M = 0.15
TAKEOFF_VZ_TOL_M_S = 0.08
TAKEOFF_STABLE_SEC = 1.5
TAKEOFF_TIMEOUT_SEC = 30.0
TAKEOFF_ALT_KP = 0.10
TAKEOFF_VZ_KD = 0.10
TAKEOFF_THRUST_MIN = 0.38
TAKEOFF_THRUST_MAX = 0.62
REQUIRE_CAMERA_BEFORE_ARM = True
REQUIRE_ATTITUDE_BEFORE_CALIBRATION = True

# After MAVSDK takeoff, verify that PX4 really accepted attitude-offboard
# before starting calibration. This directly catches the previous failure mode:
# command_roll changed, but vehicle_roll stayed essentially flat.
OFFBOARD_PROBE_ROLL_RAD = 0.04
OFFBOARD_PROBE_DURATION_SEC = 1.2
OFFBOARD_PROBE_MIN_ROLL_RESPONSE_RAD = 0.010
OFFBOARD_REQUEST_REPEAT_SEC = 1.0

# A real run showed the probe's last tick commanding roll=OFFBOARD_PROBE_ROLL_RAD
# and the very next tick (one MISSION_PERIOD_SEC later) commanding roll=0.0 --
# an instant step de-rotation -- immediately followed by a safety-bound abort
# on vz, 0.2s into the next phase. A sudden attitude change like that can
# produce a real vertical transient on its own; ramping the roll back to
# level over a couple of seconds instead of stepping it removes that as a
# cause rather than just hoping the safety margin covers it.
PROBE_RECOVERY_SEC = 2.0

# PX4 bridge topic names differ slightly across versions. Subscribing to
# both common attitude-topic names is harmless; whichever one exists will
# feed on_vehicle_attitude().
VEHICLE_ATTITUDE_TOPICS = (
	"/fmu/out/vehicle_attitude",
	"/fmu/out/vehicle_attitude_v1",
)

# Some PX4 DDS topic sets do not expose VehicleAttitude directly, but do
# expose VehicleOdometry, which also carries q. Use it as an attitude
# fallback so the diagnostics can still log actual roll/pitch.
VEHICLE_ODOMETRY_TOPICS = (
	"/fmu/out/vehicle_odometry",
	"/fmu/out/vehicle_odometry_v1",
)

VEHICLE_STATUS_TOPICS = (
	"/fmu/out/vehicle_status",
	"/fmu/out/vehicle_status_v1",
)

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
#   ros2 run ros_gz_bridge parameter_bridge \\
#       /platform/pose@geometry_msgs/msg/Pose@gz.msgs.Pose
# Set PLATFORM_POSE_TOPIC to None to disable platform-state logging
# entirely (e.g. for a stationary-platform run, or before the bridge is set
# up) -- diagnostics rows are written with empty platform_*/relative_*
# fields either way.
PLATFORM_POSE_TOPIC = "/platform/pose"

# Real pose telemetry is noisy/jittery sample-to-sample; smooth the finite-
# differenced velocity the same way OpticalFlowEstimator smooths divergence,
# so relative_vz_m_s isn't dominated by differentiation noise (see
# optical_flow.py's module docstring for the same underlying argument).
PLATFORM_VELOCITY_SMOOTHING = 0.5

VEHICLE_COMMAND_ACK_TOPICS = (
	"/fmu/out/vehicle_command_ack",
	"/fmu/out/vehicle_command_ack_v1",
)

# Vehicle trim. Keep this matched to ControlLaw's hover_thrust — the
# thrust step train is defined as a deviation from it, and the settle
# phase / continuous roll-pitch damping below both use it as the
# baseline they damp around.
#
# Now backed by two independent, converging estimates rather than a
# single guess: ~0.74 from matching an earlier abort's timing against a
# simulated hover_thrust gap, and ~0.7325 from directly regressing this
# run's actual (vz, command_thrust) pairs against a simple thrust-to-vz
# plant model. Splitting the two: 0.73. Validated against the FULL
# uncertainty in both that regression's mass_gain estimate (it was noisy,
# R^2=0.45, so treat 4.68 as a rough number, not a precise one) and the
# remaining hover gap: across mass_gain in [3, 11] and true hover in
# [0.70, 0.76], this value converges to ~0 altitude error well within
# the 45s settle budget using the EXISTING kp/ki/kz/kiz gains, unchanged.
# (Tried increasing kiz directly first to speed up a 45s settle timeout
# this value should now make largely moot -- that's the wrong fix here:
# at the actual (much weaker than previously assumed) mass_gain, a
# larger kiz drove the simulated altitude into growing oscillation
# instead of faster convergence. Closing the gap itself is the safer
# lever; the gains didn't need to move.)
HOVER_THRUST = 0.73

# Step-train shape. Keep the roll/pitch amplitudes comfortably inside
# ControlLaw's roll_limit/pitch_limit, and the thrust amplitude
# comfortably inside thrust_min/thrust_max, so the identified model
# stays valid over the range the real controller will actually command.
# Amplitude raised from 0.04 previously; with the real fitted b this
# weak, ±0.04 rad wasn't exciting the system enough to reliably
# separate b from noise.
ROLL_TEST_AMPLITUDE_RAD = 0.04
PITCH_TEST_AMPLITUDE_RAD = 0.04
# Reduced from 0.04: with the damper off during the thrust test, one +step is
# ~ (amp/hover)*g*hold m/s of climb, so 0.04 over 1.5s built ~0.8 m/s/step and
# accumulated past the vz abort. 0.03 over 1.0s (below) roughly halves that.
# The velocity-gated recenter is what actually bounds it; this just keeps the
# nominal oscillation gentle so the recenter fires rarely.
THRUST_TEST_AMPLITUDE = 0.03

# Hold duration, per axis (not shared — see build_calibration_sequence's
# docstring). Roll/pitch raised from 2.0s to 6.0s: for a roughly
# double-integrator response (tilt -> acceleration -> velocity ->
# position), position scales with hold_sec^2, so this is a much bigger
# lever on signal strength than amplitude is, and amplitude is already
# near its ceiling. Thrust's hold deliberately stays short — it already
# commands real altitude excursions, and lengthening it directly widens
# the area_fraction range swept within one file (a separate problem;
# see fit_axis_models.py's wide-range warning).
ROLL_TEST_HOLD_SEC = 1.0
PITCH_TEST_HOLD_SEC = 1.0
THRUST_TEST_HOLD_SEC = 1.0

ROLL_TEST_REPEATS = 8
PITCH_TEST_REPEATS = 8
# Halved from 8: a real run showed vz climbing monotonically from ~0 to
# 0.47 m/s over one uninterrupted ~30s thrust test -- the vertical
# damper is fully disabled during thrust's own test (it's what's being
# identified), so a residual hover_thrust error has nothing checking it
# for the whole duration. THRUST_RESET_SEC below is the other half of
# the fix: bounding how far that can drift between repeats rather than
# requiring hover_thrust to be guessed exactly right. Fewer, periodically
# reset repeats means less time for the same residual error to
# compound between corrections.
THRUST_TEST_REPEATS = 4
# Brief damped return-to-trim between thrust repeats (not within one --
# see build_axis_step_train's docstring). Labeled "settle", not
# "thrust", so the vertical damper activates for it and
# fit_axis_models.py excludes it from the thrust fit automatically, the
# same way it already excludes the inter-axis gaps.
#
# 4.0s, not a round number picked for convenience -- swept reset
# duration from 2-10s in simulation and found a real, sharp band
# (5.5-7s) where it's actively worse than no reset at all: the damper
# is mildly underdamped at these short timescales (its gains were tuned
# for the much longer vertical settle, not a few-second touch-up), so a
# duration landing mid-oscillation amplifies drift instead of removing
# it. 4.0s lands in a genuinely good window (small, non-growing residual
# vz at the end of every reset, not just the first), validated across a
# range of plausible mass_gain and residual-bias combinations -- not a
# perfect guarantee for every case (a couple of the more extreme
# combinations tested still showed some growth), but a clear, validated
# improvement over the uninterrupted version. If a future run still
# shows meaningful drift during thrust testing, the more robust fix
# would be a runtime "wait until actually settled" check between
# repeats (like VerticalSettler) instead of this fixed duration.
THRUST_RESET_SEC = 6.0
# Brief damped "settle"-labelled gaps between roll/pitch repeats too, so the
# lateral damper can periodically re-center during a long (repeats * 4 *
# hold_sec) train -- the damper is off during the active steps, so without
# this a slow lateral drift can walk the target out of frame mid-test. The
# gaps are excluded from the fit exactly like the thrust resets. 0.0 disables.
ROLL_RESET_SEC = 2.0
PITCH_RESET_SEC = 2.0
TEST_SETTLE_SEC = 2.0
# Now running all three axes back to back in one CSV: roll's own
# identification has been clean and significant for a while, the
# vertical settle is fast and reliable now, and the lateral damper
# below addresses the main remaining risk (drifting out of frame
# before/between axes) for a single-axis run becoming a real problem
# over a ~3x longer one.
TEST_AXES = ("roll", "pitch", "thrust")

# Defensive clamps applied to whatever the sequence produces, mirroring
# ControlLaw's own limits, in case the amplitudes above are ever set
# too large by mistake.
ROLL_LIMIT_RAD = 0.10
PITCH_LIMIT_RAD = 0.10
THRUST_MIN = 0.35
# Headroom above HOVER_THRUST=0.73 for kp/ki/kz/kiz to actually have
# authority -- a ceiling sitting at or below the true hover value means
# the damper saturates with zero corrective authority left, regardless
# of how good the gains are.
THRUST_MAX = 0.80

# Pre-test settle phase, and the continuous damping during roll/pitch
# testing that follows it, share ONE VerticalVelocityDamper instance
# (constructed in __init__) — the integral term is exactly the part
# that should carry over across that boundary rather than resetting,
# since whatever disturbance it's correcting for doesn't reset either.
#
# Commanding exactly HOVER_THRUST zeroes vertical *acceleration*, not
# velocity, so a residual vz at t=0 would otherwise persist. That's
# what kp/ki are for. kz is newer: a real run showed the damper settle
# "successfully" (|vz|<threshold) while the vehicle had actually
# descended onto the platform — vz reads near zero whether the vehicle
# is genuinely hovering or just resting on a surface, and a sign bug
# (now also fixed) made this an active runaway, not just a slow drift.
# kz adds a term that pulls back toward the altitude reached right after
# takeoff, not just toward zero velocity, so a `hover_thrust` guess that's
# meaningfully wrong gets corrected before it turns into a landing.
#
# integral_limit was previously sized for a small residual bias (~0.01
# m/s of vz) and is nowhere near big enough to correct a substantially
# wrong hover_thrust guess: ki*integral_limit caps the integral term's
# entire contribution, and the old 0.05 limit caps it at 0.001 -- a
# rounding error, not a real correction. Widened so it can actually do
# something. ALTITUDE_SETTLE_TOLERANCE_M is the other half of the kz
# fix: settling now requires being near the target altitude, not just
# near zero velocity, so "resting on the platform" can no longer pass
# as "hovering". SETTLE_TIMEOUT_SEC raised to give a genuine recovery
# (climbing back from a real deviation, not just damping noise) enough
# time to actually finish instead of timing out mid-recovery.
SETTLE_VZ_THRESHOLD = 0.05
SETTLE_MIN_DURATION_SEC = 1.0
SETTLE_TIMEOUT_SEC = 45.0
ALTITUDE_SETTLE_TOLERANCE_M = 1.0
DAMPER_KP = 0.08
DAMPER_KI = 0.02
DAMPER_KZ = 0.10
DAMPER_KIZ = 0.02
DAMPER_INTEGRAL_LIMIT = 0.15
# A real 45+ second settle timeout (vz=-0.057, just past threshold) traced
# back to a measured ~1.18m steady-state altitude error -- kz alone is
# P-only on position and reaches an equilibrium against a constant
# hover_thrust error instead of closing it, the same limitation ki was
# added to fix for velocity. Simulating the same gap against kiz=0.02
# drives the error to within ALTITUDE_SETTLE_TOLERANCE_M by ~t=10s and to
# ~0 by t=20-30s, instead of never closing it. integral_z_limit is sized
# in meters*seconds (not thrust units): 10.0 bounds windup without being
# so tight it caps kiz's contribution below a plausible hover_thrust gap,
# the same mistake the original (too-tight) DAMPER_INTEGRAL_LIMIT made.
DAMPER_INTEGRAL_Z_LIMIT = 10.0

# Lateral (roll/pitch) damping against vx/vy drift -- active ONLY during
# the pre-test handoff phases and the inter-axis "settle" gaps within
# the open-loop sequence, NEVER during any axis' own active test (that
# would reintroduce exactly the cause/effect entanglement this whole
# setup exists to avoid). Same motivation as the vertical damper, one
# axis over: a real run showed target_offset_x starting around 0.35-0.37
# and drifting further, because roll=pitch=0 zeroes lateral
# *acceleration* (not velocity) and nothing corrected the velocity that
# OFFBOARD_PROBE's deliberate roll excursion leaves behind. Gains are
# derived from physics (lateral accel ~= g*angle for small angles, a
# known constant, unlike the vertical thrust-to-accel relationship which
# had to be estimated from data) rather than guessed -- see
# LateralVelocityDamper's docstring for the sign derivation and its
# direct numerical verification. roll/pitch limits here are deliberately
# small relative to ROLL_LIMIT_RAD/PITCH_LIMIT_RAD: this should gently
# hold position, not aggressively chase it.
LATERAL_DAMPER_KP = 0.10
LATERAL_DAMPER_KI = 0.03
LATERAL_DAMPER_ROLL_LIMIT = 0.05
LATERAL_DAMPER_PITCH_LIMIT = 0.05
LATERAL_DAMPER_INTEGRAL_LIMIT = 0.3

# Hard safety bounds. Checked on EVERY tick from the moment offboard
# thrust commands start being sent (PHASE_PRESTREAM) through the rest of
# the mission -- previously this was only checked inside the open-loop
# test phase, which meant the entire takeoff-handoff and settle sequence
# had zero protection. That gap is exactly how a real run produced a
# "safety bound exceeded" abort 0.09s into open-loop testing with an
# empty CSV: area_fraction was already past the limit from whatever
# happened during the unprotected phases before it, and nothing could
# have caught it earlier even if it had been far worse. Tripping any
# bound now halts at HOVER_THRUST permanently (reposition and restart
# the node) rather than trying to recover automatically.
# Hard emergency net. Raised from 1.0: the thrust ID test inherently produces
# vertical velocity, so 1.0 m/s aborted legitimate runs. This is now a genuine
# runaway threshold; normal thrust testing is held far inside it by the
# velocity-gated recenter below (THRUST_TEST_VZ_SOFT_LIMIT).
ABORT_VZ_LIMIT = 1.5
ABORT_AREA_FRACTION_MAX = 0.97
# New alongside the lateral damper -- if it (or anything else) ever
# drove a runaway instead of gently correcting drift, this catches it
# the same way ABORT_VZ_LIMIT catches a vertical runaway.
ABORT_LATERAL_VELOCITY_LIMIT = 2.0

# If target/flow stays lost this long DURING the open-loop test phase
# (not the settle phase, which doesn't need vision), there is no more
# useful data left to collect for whichever axis is currently under
# test — abort instead of running to completion on dead air and only
# discovering "not enough valid samples" at analysis time. This is
# exactly what a thrust phase that drifts the target out of detection
# range produces: roll/pitch can still come back with real data, while
# thrust ends up with nothing, and the run has no way to tell you that
# happened until fit_axis_models.py runs much later.
LOST_TARGET_ABORT_SEC = 5.0

# Velocity-gated thrust recenter. The thrust axis runs open-loop with the
# vertical damper off, so its steps drive a real climb whose size depends on
# hover_thrust error and altitude -- a fixed amplitude can't be safe across all
# of them. Instead, bound it by actual vz: when |vz| exceeds the soft limit
# during a thrust step, pause the sequence clock and let the damper bring vz
# back below the resume threshold (logged as "settle", excluded from the fit),
# then resume the steps where they left off. This is a safety interrupt, not
# feedback on the identified signal, so the thrust model stays open-loop while
# the test self-limits regardless of climb value. Hysteresis (soft > resume)
# avoids chattering in and out. If a recenter can't settle within the timeout
# the run is genuinely diverging -> abort and tag the file.
THRUST_TEST_VZ_SOFT_LIMIT = 0.6
THRUST_TEST_VZ_RESUME = 0.2
THRUST_TEST_RECENTER_TIMEOUT_SEC = 8.0

# Operating-point drift guard. The fit treats one run as one operating point
# (one area_fraction). If area_fraction wanders more than this from its value
# at the first valid open-loop sample, the single-operating-point assumption is
# already broken and the rest of the run is wasted -- abort and mark the file
# (op_point_drift) instead of collecting a wide-sweep run the fitter will only
# reject later. The thrust axis deliberately moves area_fraction a little
# (short holds), so keep this comfortably above that expected excursion.
OPEN_LOOP_AREA_FRACTION_DRIFT_MAX = 0.12

PHASE_WAITING_FOR_STREAMS = "waiting_for_streams"
PHASE_MAVSDK_TAKEOFF = "mavsdk_takeoff"
PHASE_PRESTREAM = "prestream_offboard"
PHASE_WAIT_OFFBOARD = "wait_offboard"
PHASE_CLIMB = "climb"
PHASE_OFFBOARD_PROBE = "offboard_probe"
PHASE_PROBE_RECOVERY = "probe_recovery"
PHASE_ALTITUDE_SETTLE = "altitude_settle"
PHASE_VZ_SETTLE = "vz_settle"
PHASE_OPEN_LOOP = "open_loop_test"
PHASE_FINISHED = "finished"
PHASE_ABORTED = "aborted"


class CalibrationNode(Node):
	def __init__(self):
		super().__init__("bee_calibration_node")

		self._node_start_time = time.time()
		self.bridge = CvBridge()

		self._last_position_log_time = 0.0
		self._position_log_period_sec = 1.0

		self._image_count = 0
		self._last_image_log_time = 0.0
		self._image_log_period_sec = 1.0

		self._attitude_message_count = 0
		self._last_attitude_status_log_time = 0.0
		self._attitude_status_log_period_sec = 2.0

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
		self._latest_setpoint = AttitudeSetpoint(
			roll=0.0, pitch=0.0, yaw=0.0, thrust=HOVER_THRUST
		)

		self._mission_phase = PHASE_WAITING_FOR_STREAMS
		self._phase_start_time = time.time()
		self._offboard_request_time = None
		self._takeoff_start_z = None
		self._takeoff_target_z = None
		self._takeoff_stable_since = None
		self._last_calibration_log_time = None
		self._have_local_position = False
		self._vehicle_status_seen = False
		self._nav_state = None
		self._arming_state = None
		self._last_command_ack = None
		self._last_offboard_request_retry = None
		self._mavsdk_thread = None
		self._mavsdk_takeoff_started = False
		self._mavsdk_takeoff_done = False
		self._mavsdk_takeoff_error = None
		self._mavsdk_offboard_start_requested = False
		self._mavsdk_offboard_started = False
		self._mavsdk_offboard_error = None
		self._mavsdk_stop_requested = False
		self._seen_ack_messages = set()
		self._streams_ready_logged = False
		self._probe_baseline_roll = None
		self._probe_max_roll_delta = 0.0

		self._test_start_time = None
		self._sequence_finished_logged = False

		self._damper = VerticalVelocityDamper(
			hover_thrust=HOVER_THRUST,
			kp=DAMPER_KP,
			ki=DAMPER_KI,
			kz=DAMPER_KZ,
			kiz=DAMPER_KIZ,
			thrust_min=THRUST_MIN,
			thrust_max=THRUST_MAX,
			integral_limit=DAMPER_INTEGRAL_LIMIT,
			integral_z_limit=DAMPER_INTEGRAL_Z_LIMIT,
		)
		self._lateral_damper = LateralVelocityDamper(
			kp=LATERAL_DAMPER_KP,
			ki=LATERAL_DAMPER_KI,
			roll_limit=LATERAL_DAMPER_ROLL_LIMIT,
			pitch_limit=LATERAL_DAMPER_PITCH_LIMIT,
			integral_limit=LATERAL_DAMPER_INTEGRAL_LIMIT,
		)
		self._settler = VerticalSettler(
			self._damper,
			vz_threshold=SETTLE_VZ_THRESHOLD,
			min_duration_sec=SETTLE_MIN_DURATION_SEC,
			timeout_sec=SETTLE_TIMEOUT_SEC,
			altitude_tolerance_m=ALTITUDE_SETTLE_TOLERANCE_M,
		)
		self._settle_logged = False
		self._aborted = False
		self._lost_since = None
		self._open_loop_af_ref = None  # area_fraction at the first valid open-loop sample
		# Velocity-gated thrust recenter state (see THRUST_TEST_VZ_SOFT_LIMIT).
		self._thrust_recenter_active = False
		self._thrust_recenter_since = None
		self._test_paused_sec = 0.0       # wall time spent paused; frozen out of `elapsed`
		self._last_open_loop_now = None

		self.optical_flow = OpticalFlowEstimator()
		self.target_acquisition = TargetAcquisition()

		self.sequence = build_calibration_sequence(
			hover_thrust=HOVER_THRUST,
			roll_amplitude=ROLL_TEST_AMPLITUDE_RAD,
			pitch_amplitude=PITCH_TEST_AMPLITUDE_RAD,
			thrust_amplitude=THRUST_TEST_AMPLITUDE,
			roll_hold_sec=ROLL_TEST_HOLD_SEC,
			pitch_hold_sec=PITCH_TEST_HOLD_SEC,
			thrust_hold_sec=THRUST_TEST_HOLD_SEC,
			roll_repeats=ROLL_TEST_REPEATS,
			pitch_repeats=PITCH_TEST_REPEATS,
			thrust_repeats=THRUST_TEST_REPEATS,
			roll_reset_sec=ROLL_RESET_SEC,
			pitch_reset_sec=PITCH_RESET_SEC,
			thrust_reset_sec=THRUST_RESET_SEC,
			settle_sec=TEST_SETTLE_SEC,
			axes=TEST_AXES,
		)

		date_str = time.strftime("%Y%m%d_%H%M%S")
		self.diagnostics = DiagnosticsWriter(
			output_dir="logs",
			filename=f"calibration_{date_str}.csv",
			flush_every_row=True,
		)

		self.get_logger().info(
			f"Calibration diagnostics CSV: {self.diagnostics.filepath}"
		)
		self.get_logger().info(
			f"Test sequence duration: {self.sequence.total_duration:.1f} s "
			f"(axes: {TEST_AXES})"
		)

		px4_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)

		camera_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			durability=QoSDurabilityPolicy.VOLATILE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=5,
		)

		self.create_subscription(
			VehicleLocalPosition,
			"/fmu/out/vehicle_local_position_v1",
			self.on_local_position,
			px4_qos,
		)

		for attitude_topic in VEHICLE_ATTITUDE_TOPICS:
			self.create_subscription(
				VehicleAttitude,
				attitude_topic,
				self.on_vehicle_attitude,
				px4_qos,
			)

		if VehicleOdometry is not None:
			for odometry_topic in VEHICLE_ODOMETRY_TOPICS:
				self.create_subscription(
					VehicleOdometry,
					odometry_topic,
					self.on_vehicle_odometry,
					px4_qos,
				)

		if VehicleStatus is not None:
			for status_topic in VEHICLE_STATUS_TOPICS:
				self.create_subscription(
					VehicleStatus,
					status_topic,
					self.on_vehicle_status,
					px4_qos,
				)

		if VehicleCommandAck is not None:
			for ack_topic in VEHICLE_COMMAND_ACK_TOPICS:
				self.create_subscription(
					VehicleCommandAck,
					ack_topic,
					self.on_vehicle_command_ack,
					px4_qos,
				)

		self.create_subscription(
			Image,
			"/bee_x500/camera/image",
			self.on_camera,
			camera_qos,
		)

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

		self.px4 = PX4Interface(self, px4_qos)

		self.create_timer(HEARTBEAT_PERIOD_SEC, self.on_heartbeat_timer)
		self.create_timer(MISSION_PERIOD_SEC, self.on_mission_timer)

		if SHOW_CAMERA:
			cv2.namedWindow("Bee Calibration - Camera", cv2.WINDOW_NORMAL)

		self.get_logger().info("bee_calibration_node started.")
		self.get_logger().info(
			"Waiting for required streams: local_position, camera, and attitude/odometry."
		)

	def on_camera(self, msg: Image):
		self._image_count += 1

		now = time.time()

		if VERBOSE_STREAM_LOGS and now - self._last_image_log_time >= self._image_log_period_sec:
			self._last_image_log_time = now

			self.get_logger().info(
				f"image #{self._image_count}: "
				f"{msg.width}x{msg.height}, encoding={msg.encoding}"
			)

		try:
			src = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		except CvBridgeError as exc:
			self.get_logger().error(f"cv_bridge conversion failed: {exc}")
			return

		# Keep the camera orientation used by the rest of the vision pipeline
		# independent of whether the debug window is enabled.
		frame = cv2.rotate(src, cv2.ROTATE_180)

		if SHOW_CAMERA:
			cv2.imshow("Bee Calibration - Camera", frame)
			cv2.waitKey(1)

		stamp = time.time()

		# Vision pipeline still runs as normal: this is what we're
		# measuring the response of. Only the control law is skipped.
		target = self.target_acquisition.update(frame, timestamp=stamp)
		flow = self.optical_flow.update(frame, stamp, target=target)

		self._latest_frame = frame
		self._latest_target = target
		self._latest_flow = flow

	def on_platform_pose(self, msg: Pose):
		"""
		Exact platform world pose, published directly by
		OscillatingPlatformController on its own dedicated topic (see
		PLATFORM_POSE_TOPIC and MovingPlatformController.cpp's publishPose) --
		no entity matching needed, since every message on this topic IS the
		platform, by construction. Position is exact; Pose carries no
		velocity, so velocity is finite-differenced against the previous
		message using THIS callback's own receipt time (same time.time()
		pattern as on_camera/on_local_position), then smoothed -- raw
		frame-to-frame differencing of real, slightly-jittery pose telemetry
		amplifies noise the same way it would for optical flow (see
		optical_flow.py's module docstring for the general argument). Stored
		directly in the SDF world's own ENU convention; platform_motion.
		relative_motion() handles the NED conversion when this is logged
		alongside vehicle_state.
		"""
		now = time.time()
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

	def on_local_position(self, msg: VehicleLocalPosition):
		now = time.time()
		self._have_local_position = True

		if VERBOSE_STREAM_LOGS and now - self._last_position_log_time >= self._position_log_period_sec:
			self._last_position_log_time = now

			self.get_logger().info(
				f"local position: "
				f"x={msg.x:.2f} m, y={msg.y:.2f} m, z={msg.z:.2f} m"
			)

		# Preserve the latest attitude fields, which arrive on a separate PX4
		# topic. The CSV then contains both local-position state and actual
		# roll/pitch/yaw at the most recent attitude callback.
		self._vehicle_state = VehicleState(
			timestamp=now,
			x=msg.x,
			y=msg.y,
			z=msg.z,
			vx=msg.vx,
			vy=msg.vy,
			vz=msg.vz,
			yaw=msg.heading,
			attitude_timestamp=self._vehicle_state.attitude_timestamp,
			roll=self._vehicle_state.roll,
			pitch=self._vehicle_state.pitch,
			attitude_yaw=self._vehicle_state.attitude_yaw,
			attitude_source=self._vehicle_state.attitude_source,
		)

	def on_vehicle_attitude(self, msg: VehicleAttitude):
		self._update_attitude_from_quaternion(msg.q, source="vehicle_attitude")

	def on_vehicle_odometry(self, msg):
		# VehicleOdometry.q uses the same PX4 quaternion convention [w, x, y, z].
		self._update_attitude_from_quaternion(msg.q, source="vehicle_odometry")

	def _update_attitude_from_quaternion(self, q, source: str):
		now = time.time()

		try:
			roll, pitch, yaw = self._quaternion_to_euler(q)
		except Exception as exc:
			self.get_logger().warning(f"Ignoring bad attitude quaternion from {source}: {exc}")
			return

		if not all(math.isfinite(v) for v in (roll, pitch, yaw)):
			self.get_logger().warning(f"Ignoring non-finite attitude quaternion from {source}: {q}")
			return

		self._vehicle_state.attitude_timestamp = now
		self._vehicle_state.roll = roll
		self._vehicle_state.pitch = pitch
		self._vehicle_state.attitude_yaw = yaw
		self._vehicle_state.attitude_source = source
		self._attitude_message_count += 1

		if VERBOSE_STREAM_LOGS and now - self._last_attitude_status_log_time >= self._attitude_status_log_period_sec:
			self._last_attitude_status_log_time = now
			self.get_logger().info(
				f"actual attitude from {source}: "
				f"roll={roll:+.3f}, pitch={pitch:+.3f}, yaw={yaw:+.3f}"
			)

	def on_vehicle_status(self, msg):
		self._vehicle_status_seen = True
		self._nav_state = getattr(msg, "nav_state", None)
		self._arming_state = getattr(msg, "arming_state", None)

	def on_vehicle_command_ack(self, msg):
		command = getattr(msg, "command", None)
		result = getattr(msg, "result", None)
		self._last_command_ack = (command, result, time.time())

		# MAVSDK repeatedly requests PX4 messages during telemetry setup and PX4
		# answers those request-message commands with acks. They are not useful
		# for calibration and made the terminal unreadable.
		if not VERBOSE_STREAM_LOGS:
			quiet_mavsdk_request_commands = {511, 512, 520}
			if command in quiet_mavsdk_request_commands:
				return
			# Avoid repeating identical accepted acks forever.
			key = (command, result)
			if key in self._seen_ack_messages:
				return
			self._seen_ack_messages.add(key)

		log_fn = self.get_logger().info if result == 0 else self.get_logger().warning
		log_fn(f"PX4 command ack: command={command}, result={result}")

	def on_heartbeat_timer(self):
		# When CONTROL_BACKEND is mavsdk_offboard, MAVSDK owns the attitude
		# offboard stream. Keep the ROS publisher path available only as an
		# explicit fallback/debug backend.
		if CONTROL_BACKEND != "ros_attitude":
			return

		# Do not stream attitude-offboard while the MAVSDK action takeoff owns PX4.
		# Start streaming only in the explicit offboard handoff/prestream phases.
		if not self._should_stream_offboard():
			return

		self.px4.publish_heartbeat()
		self.px4.publish_attitude_setpoint(
			self._latest_setpoint.roll,
			self._latest_setpoint.pitch,
			self._latest_setpoint.yaw,
			self._latest_setpoint.thrust,
		)

	def on_mission_timer(self):
		now = time.time()

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

		# Checked first, before any phase-specific logic, and covers every
		# phase from PHASE_PRESTREAM onward -- previously this only ran
		# inside the open-loop test phase, leaving the entire takeoff
		# handoff and settle sequence with zero protection. See the
		# constants block above for what this gap actually produced.
		if not self._aborted and self._should_stream_offboard() and self._mission_phase not in (
			PHASE_FINISHED, PHASE_ABORTED,
		):
			vz = self._vehicle_state.vz
			vx = self._vehicle_state.vx
			vy = self._vehicle_state.vy
			area_fraction = float(getattr(self._latest_target, "area_fraction", 0.0))
			if exceeds_safety_bounds(
				vz, area_fraction, ABORT_VZ_LIMIT, ABORT_AREA_FRACTION_MAX,
				vx=vx, vy=vy, lateral_velocity_limit=ABORT_LATERAL_VELOCITY_LIMIT,
			):
				self._abort(
					f"safety bound exceeded during phase={self._mission_phase}: "
					f"vz={vz:.3f} m/s, vx={vx:.3f} m/s, vy={vy:.3f} m/s, area_fraction={area_fraction:.3f}"
				)
				return

		if self._mission_phase == PHASE_ABORTED:
			self._hold_trim()
			return

		if self._mission_phase == PHASE_WAITING_FOR_STREAMS:
			self._hold_trim()
			if self._ready_to_start_prestream():
				if USE_MAVSDK_TAKEOFF:
					self._enter_phase(PHASE_MAVSDK_TAKEOFF)
				else:
					# No altitude target yet -- the vehicle is still on the
					# ground here; one gets set once PHASE_CLIMB actually
					# reaches its target altitude, same as the MAVSDK path
					# sets one once ITS takeoff reaches altitude. Setting
					# one now would have the damper trying to hold ground
					# level against the climb that's about to happen.
					self._enter_phase(PHASE_PRESTREAM)
			return

		if self._mission_phase == PHASE_MAVSDK_TAKEOFF:
			self._hold_trim()
			self._ensure_mavsdk_takeoff_started()
			if self._mavsdk_takeoff_error is not None:
				self._abort(f"MAVSDK takeoff failed: {self._mavsdk_takeoff_error}")
				return
			if now - self._phase_start_time >= MAVSDK_TAKEOFF_PHASE_TIMEOUT_SEC:
				self._abort(
					f"MAVSDK takeoff phase exceeded {MAVSDK_TAKEOFF_PHASE_TIMEOUT_SEC:.0f}s "
					f"without completing or reporting an error. This previously produced an "
					f"empty diagnostics CSV with no explanation -- if you see this, one of the "
					f"per-step MAVSDK timeouts didn't catch whatever's actually stuck."
				)
				return
			if self._mavsdk_takeoff_done:
				self.get_logger().info("MAVSDK takeoff complete. Starting attitude-offboard prestream.")
				# Captured here, at the altitude PX4's own (already-working)
				# takeoff just reached -- this is the reference our own
				# thrust commands need to hold onto from here on, since
				# they're about to take over control authority.
				self._damper.set_altitude_target(self._vehicle_state.z)
				self.get_logger().info(
					f"Altitude-hold target set to current z={self._vehicle_state.z:.3f} m."
				)
				self._enter_phase(PHASE_PRESTREAM)
			return

		if self._mission_phase == PHASE_PRESTREAM:
			self._latest_setpoint = self._damped_hover_setpoint(now)
			if now - self._phase_start_time >= OFFBOARD_PRESTREAM_SEC:
				self._request_offboard_and_arm(now, reason="initial handoff")
				self._enter_phase(PHASE_WAIT_OFFBOARD)
			return

		if self._mission_phase == PHASE_WAIT_OFFBOARD:
			self._latest_setpoint = self._damped_hover_setpoint(now)
			if self._mavsdk_offboard_error is not None:
				self._abort(f"MAVSDK offboard start failed: {self._mavsdk_offboard_error}")
				return
			if (
				self._last_offboard_request_retry is None
				or now - self._last_offboard_request_retry >= OFFBOARD_REQUEST_REPEAT_SEC
			):
				self._request_offboard_and_arm(now, reason="retry")

			if self._offboard_and_armed():
				self.get_logger().info("PX4 reports OFFBOARD+ARMED. Verifying actual attitude response.")
				self._start_offboard_probe()
				return

			if self._offboard_confirm_timed_out(now):
				if CONTROL_BACKEND == "mavsdk_offboard":
					self._abort(
						"MAVSDK offboard did not start before timeout. Check MAVSDK offboard "
						"errors above and verify no other process owns the MAVLink port."
					)
					return
				if self._vehicle_status_seen:
					self._abort(
						"PX4 vehicle_status was received but did not become OFFBOARD+ARMED "
						f"(nav_state={self._nav_state}, arming_state={self._arming_state})"
					)
					return
				self.get_logger().warning(
					"No PX4 vehicle_status received. Continuing to a small attitude-response "
					"probe; calibration will abort if actual roll does not move."
				)
				self._start_offboard_probe()
			return

		if self._mission_phase == PHASE_OFFBOARD_PROBE:
			self._latest_setpoint = self._damped_hover_setpoint(
				now, roll=OFFBOARD_PROBE_ROLL_RAD, apply_lateral_damping=False
			)
			if self._probe_baseline_roll is not None:
				delta = abs(self._vehicle_state.roll - self._probe_baseline_roll)
				self._probe_max_roll_delta = max(self._probe_max_roll_delta, delta)
			if now - self._phase_start_time >= OFFBOARD_PROBE_DURATION_SEC:
				if self._probe_max_roll_delta < OFFBOARD_PROBE_MIN_ROLL_RESPONSE_RAD:
					self._abort(
						"attitude-offboard probe failed: commanded "
						f"roll={OFFBOARD_PROBE_ROLL_RAD:.3f} rad, but actual roll changed only "
						f"{self._probe_max_roll_delta:.4f} rad. PX4 is not accepting "
						"VehicleAttitudeSetpoint commands."
					)
					return
				self.get_logger().info(
					f"Attitude-offboard probe passed: actual roll changed "
					f"{self._probe_max_roll_delta:.4f} rad."
				)
				if USE_MAVSDK_TAKEOFF:
					self._enter_phase(PHASE_PROBE_RECOVERY)
				else:
					self._takeoff_start_z = self._vehicle_state.z
					self._takeoff_target_z = self._takeoff_start_z - TAKEOFF_ALTITUDE_M
					self._takeoff_stable_since = None
					self._enter_phase(PHASE_CLIMB)
					self.get_logger().info(
						f"Takeoff target: start_z={self._takeoff_start_z:.3f} m, "
						f"target_z={self._takeoff_target_z:.3f} m "
						f"({TAKEOFF_ALTITUDE_M:.2f} m above start, NED z)."
					)
			return

		if self._mission_phase == PHASE_PROBE_RECOVERY:
			# Ramp roll from the probe's value back to level over
			# PROBE_RECOVERY_SEC, instead of stepping it in one tick --
			# see PROBE_RECOVERY_SEC's definition for why.
			elapsed_in_phase = now - self._phase_start_time
			ramp_fraction = max(0.0, 1.0 - elapsed_in_phase / PROBE_RECOVERY_SEC)
			ramped_roll = OFFBOARD_PROBE_ROLL_RAD * ramp_fraction
			self._latest_setpoint = self._damped_hover_setpoint(now, roll=ramped_roll)
			if elapsed_in_phase >= PROBE_RECOVERY_SEC:
				self._enter_phase(PHASE_ALTITUDE_SETTLE)
			return

		if self._mission_phase == PHASE_CLIMB:
			self._latest_setpoint = AttitudeSetpoint(
				timestamp=self._latest_target.timestamp,
				roll=0.0,
				pitch=0.0,
				yaw=0.0,
				thrust=self._takeoff_thrust(),
			)

			if now - self._phase_start_time > TAKEOFF_TIMEOUT_SEC:
				self._abort(
					f"takeoff timeout after {TAKEOFF_TIMEOUT_SEC:.1f}s; "
					f"z={self._vehicle_state.z:.3f}, target_z={self._takeoff_target_z:.3f}"
				)
				return

			if self._at_takeoff_altitude():
				if self._takeoff_stable_since is None:
					self._takeoff_stable_since = now
				elif now - self._takeoff_stable_since >= TAKEOFF_STABLE_SEC:
					self.get_logger().info("Takeoff altitude reached and stable. Settling before calibration.")
					self._damper.set_altitude_target(self._vehicle_state.z)
					self._enter_phase(PHASE_ALTITUDE_SETTLE)
			else:
				self._takeoff_stable_since = None
			return

		if self._mission_phase == PHASE_ALTITUDE_SETTLE:
			self._latest_setpoint = self._damped_hover_setpoint(now)
			if now - self._phase_start_time >= TEST_SETTLE_SEC:
				self._enter_phase(PHASE_VZ_SETTLE)
			return

		if self._mission_phase == PHASE_VZ_SETTLE:
			if REQUIRE_ATTITUDE_BEFORE_CALIBRATION and self._vehicle_state.attitude_timestamp <= 0.0:
				self._hold_trim()
				self._log_waiting_for_attitude(now)
				return

			vz = self._vehicle_state.vz
			z = self._vehicle_state.z
			thrust = self._settler.step(now, vz, z=z)
			lateral_roll, lateral_pitch = self._lateral_damper.step(
				now, self._vehicle_state.vx, self._vehicle_state.vy, yaw=self._vehicle_state.yaw
			)
			self._latest_setpoint = AttitudeSetpoint(
				timestamp=self._latest_target.timestamp,
				roll=lateral_roll,
				pitch=lateral_pitch,
				yaw=0.0,
				thrust=thrust,
			)

			if self._settler.is_settled:
				if self._settler.timed_out:
					self.diagnostics.set_run_status("settle_timeout")
					self.get_logger().warning(
						f"Vertical-velocity settle timed out after {SETTLE_TIMEOUT_SEC}s "
						f"(vz={vz:.3f} m/s); starting calibration anyway (file tagged settle_timeout)."
					)
				else:
					self.get_logger().info(
						f"Vertical velocity settled (|vz|<{SETTLE_VZ_THRESHOLD} m/s)."
					)
				self._enter_phase(PHASE_OPEN_LOOP)
				self._test_start_time = now
				self._last_calibration_log_time = None
				self._thrust_recenter_active = False
				self._thrust_recenter_since = None
				self._test_paused_sec = 0.0
				self._last_open_loop_now = None
				self.get_logger().info("Open-loop calibration sequence started.")
			return

		if self._mission_phase == PHASE_OPEN_LOOP:
			self._run_open_loop_calibration(now)
			return

		if self._mission_phase == PHASE_FINISHED:
			self._hold_trim()
			return

	def _run_open_loop_calibration(self, now: float):
		if self._latest_flow is None or self._latest_frame is None:
			self._hold_trim()
			return

		# Safety bounds are checked centrally at the top of on_mission_timer
		# now, covering every phase from PRESTREAM onward -- not just here.
		vz = self._vehicle_state.vz
		z = self._vehicle_state.z

		target_ok = bool(self._latest_target.found)
		flow_ok = bool(self._latest_flow.valid)

		if not (target_ok and flow_ok):
			if self._lost_since is None:
				self._lost_since = now
			elif now - self._lost_since >= LOST_TARGET_ABORT_SEC:
				self._abort(
					f"target/flow lost for >= {LOST_TARGET_ABORT_SEC}s during calibration "
					f"(target_found={target_ok}, flow_valid={flow_ok})",
					status="lost_target",
				)
				return
		else:
			self._lost_since = None

			# Operating-point drift guard: capture area_fraction at the first
			# valid sample, then abort if it wanders too far. Keeps each run a
			# single operating point instead of a wide sweep the fit will reject.
			area_fraction = float(getattr(self._latest_target, "area_fraction", 0.0))
			if self._open_loop_af_ref is None:
				self._open_loop_af_ref = area_fraction
			elif abs(area_fraction - self._open_loop_af_ref) > OPEN_LOOP_AREA_FRACTION_DRIFT_MAX:
				self._abort(
					f"operating point drifted: area_fraction={area_fraction:.3f} is "
					f">{OPEN_LOOP_AREA_FRACTION_DRIFT_MAX:.2f} from its test-start value "
					f"{self._open_loop_af_ref:.3f}; the run is no longer a single operating point",
					status="op_point_drift",
				)
				return

		# Elapsed test time, with any paused (recenter) wall time frozen out so
		# the sequence resumes exactly where it left off.
		tick_dt = (now - self._last_open_loop_now) if self._last_open_loop_now is not None else 0.0
		self._last_open_loop_now = now

		elapsed = now - self._test_start_time - self._test_paused_sec
		current_axis = self.sequence.axis_at(elapsed)

		# Velocity-gated recenter (thrust axis only -- it's the one running with
		# the damper off). Enter when |vz| leaves the soft band, stay until it is
		# back under the resume threshold, abort if it can't settle in time. While
		# recentering we freeze `elapsed` and relabel the tick "settle", so the
		# damper takes thrust, the lateral damper re-centers, and these rows are
		# excluded from the fit -- the identified steps stay open-loop.
		if current_axis == "thrust":
			if self._thrust_recenter_active:
				if abs(vz) <= THRUST_TEST_VZ_RESUME:
					self._thrust_recenter_active = False
					self._thrust_recenter_since = None
				elif now - self._thrust_recenter_since >= THRUST_TEST_RECENTER_TIMEOUT_SEC:
					self._abort(
						f"thrust recenter could not bring |vz| under {THRUST_TEST_VZ_RESUME} m/s "
						f"within {THRUST_TEST_RECENTER_TIMEOUT_SEC:.0f}s (vz={vz:.3f}); the vertical "
						f"axis is diverging",
						status="thrust_runaway",
					)
					return
			elif abs(vz) >= THRUST_TEST_VZ_SOFT_LIMIT:
				self._thrust_recenter_active = True
				self._thrust_recenter_since = now
				self.get_logger().info(
					f"Thrust recenter: |vz|={abs(vz):.2f} >= {THRUST_TEST_VZ_SOFT_LIMIT} m/s; "
					f"pausing steps and damping back to < {THRUST_TEST_VZ_RESUME} m/s."
				)

			if self._thrust_recenter_active:
				self._test_paused_sec += tick_dt
				elapsed = now - self._test_start_time - self._test_paused_sec
				current_axis = "settle"

		roll, pitch, thrust = self.sequence.command_at(elapsed)

		if current_axis != "thrust":
			thrust = self._damper.step(now, vz, z=z)

		# Always step the lateral damper to keep its internal timing
		# continuous (same reasoning as the vertical damper during the
		# probe), but only apply its output during the inter-axis
		# "settle" gaps -- never during any axis' own active test, which
		# must stay genuinely open-loop for identification.
		lateral_roll, lateral_pitch = self._lateral_damper.step(
			now, self._vehicle_state.vx, self._vehicle_state.vy, yaw=self._vehicle_state.yaw
		)
		if current_axis == "settle":
			roll += lateral_roll
			pitch += lateral_pitch

		roll = self._clamp(roll, -ROLL_LIMIT_RAD, ROLL_LIMIT_RAD)
		pitch = self._clamp(pitch, -PITCH_LIMIT_RAD, PITCH_LIMIT_RAD)
		thrust = self._clamp(thrust, THRUST_MIN, THRUST_MAX)

		self._latest_setpoint = AttitudeSetpoint(
			timestamp=self._latest_target.timestamp,
			roll=roll,
			pitch=pitch,
			yaw=0.0,
			thrust=thrust,
		)

		should_log = (
			self._last_calibration_log_time is None
			or now - self._last_calibration_log_time >= CALIBRATION_LOG_PERIOD_SEC
		)
		if should_log:
			self._last_calibration_log_time = now
			self.diagnostics.write(
				wall_timestamp=time.time(),
				target=self._latest_target,
				flow=self._latest_flow,
				setpoint=self._latest_setpoint,
				vehicle_state=self._vehicle_state,
				calibration_axis=current_axis,
				platform_state=self._platform_state,
			)

		if self.sequence.is_finished(elapsed) and not self._sequence_finished_logged:
			self._sequence_finished_logged = True
			self._enter_phase(PHASE_FINISHED)
			self.get_logger().info(
				"Test sequence finished; holding at trim. "
				"Stop the node and run fit_axis_models.py on the CSV above."
			)

	def _should_stream_offboard(self) -> bool:
		return self._mission_phase in (
			PHASE_PRESTREAM,
			PHASE_WAIT_OFFBOARD,
			PHASE_CLIMB,
			PHASE_OFFBOARD_PROBE,
			PHASE_PROBE_RECOVERY,
			PHASE_ALTITUDE_SETTLE,
			PHASE_VZ_SETTLE,
			PHASE_OPEN_LOOP,
			PHASE_FINISHED,
			PHASE_ABORTED,
		)

	def _request_offboard_and_arm(self, now: float, reason: str):
		if CONTROL_BACKEND == "mavsdk_offboard":
			if not self._mavsdk_offboard_start_requested:
				self.get_logger().info(
					f"Requesting MAVSDK attitude offboard start ({reason})."
				)
			self._mavsdk_offboard_start_requested = True
		else:
			self.get_logger().info(f"Requesting PX4 OFFBOARD mode and arming ({reason}).")
			self.px4.engage_offboard_mode()
			self.px4.arm()

		if self._offboard_request_time is None:
			self._offboard_request_time = now
		self._last_offboard_request_retry = now

	def _start_offboard_probe(self):
		if REQUIRE_ATTITUDE_BEFORE_CALIBRATION and self._vehicle_state.attitude_timestamp <= 0.0:
			self._abort("cannot verify attitude offboard because no actual attitude has been received")
			return
		self._probe_baseline_roll = self._vehicle_state.roll
		self._probe_max_roll_delta = 0.0
		self._enter_phase(PHASE_OFFBOARD_PROBE)

	def _ensure_mavsdk_takeoff_started(self):
		if self._mavsdk_takeoff_started:
			return
		if System is None or MavsdkAttitude is None:
			self._mavsdk_takeoff_error = "mavsdk/offboard is not installed in this Python environment"
			return
		self._mavsdk_takeoff_started = True
		self.get_logger().info(
			f"Starting MAVSDK takeoff to {TAKEOFF_ALTITUDE_M:.2f} m from inside calibration node."
		)
		self._mavsdk_thread = threading.Thread(
			target=self._run_mavsdk_worker_thread,
			name="mavsdk_takeoff_and_offboard",
			daemon=True,
		)
		self._mavsdk_thread.start()

	@staticmethod
	async def _wait_for_condition(
		async_iterable,
		condition,
		timeout: float,
		label: str,
		progress_fn=None,
		progress_interval: float = 5.0,
	):
		"""
		Consume async_iterable until condition(item) is True, or raise a
		clear, labeled TimeoutError after `timeout` seconds. Every
		"async for ... : break" wait in the MAVSDK handshake used to have
		no timeout at all -- one stuck step (wrong system address, SITL
		not up yet, a health flag that never goes true) meant the node
		sat in PHASE_MAVSDK_TAKEOFF forever, with no error and an empty
		diagnostics CSV as the only symptom. The timeout fixes that --
		but a timeout firing only tells you a step failed, not what was
		actually happening leading up to it (e.g. "stuck at 0m the whole
		time" vs "climbing, just slower than expected" look identical
		from the outside otherwise). progress_fn, if given, is called
		with (item, elapsed_sec) every progress_interval seconds while
		still waiting, so a future timeout comes with that context
		instead of just "it failed".
		"""
		async def _inner():
			loop = asyncio.get_event_loop()
			start = loop.time()
			last_progress = start

			async for item in async_iterable:
				if condition(item):
					return item

				if progress_fn is not None:
					now = loop.time()
					if now - last_progress >= progress_interval:
						last_progress = now
						progress_fn(item, now - start)

		try:
			return await asyncio.wait_for(_inner(), timeout=timeout)
		except asyncio.TimeoutError:
			raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {label}")

	def _run_mavsdk_worker_thread(self):
		try:
			asyncio.run(self._mavsdk_worker_async())
		except Exception as exc:
			# If takeoff was not complete yet, report this as a takeoff error;
			# otherwise report it as an offboard-stream error.
			if not self._mavsdk_takeoff_done:
				self._mavsdk_takeoff_error = repr(exc)
			else:
				self._mavsdk_offboard_error = repr(exc)

	async def _mavsdk_worker_async(self):
		self._free_mavsdk_port(MAVSDK_PORT_TO_FREE)
		drone = System()
		await drone.connect(system_address=MAVSDK_SYSTEM_ADDRESS)

		self.get_logger().info("MAVSDK: waiting for drone connection...")
		await self._wait_for_condition(
			drone.core.connection_state(),
			lambda state: state.is_connected,
			MAVSDK_CONNECT_TIMEOUT_SEC,
			"MAVSDK connection",
		)
		self.get_logger().info("MAVSDK: connected.")

		self.get_logger().info("MAVSDK: waiting for global/home/local position estimates...")
		await self._wait_for_condition(
			drone.telemetry.health(),
			lambda health: (
				health.is_global_position_ok
				and health.is_home_position_ok
				and health.is_local_position_ok
			),
			MAVSDK_HEALTH_TIMEOUT_SEC,
			"global/home/local position health",
		)
		self.get_logger().info("MAVSDK: all position estimates OK.")

		await asyncio.sleep(EKF2_SETTLE_TIME)

		home_position = await self._wait_for_condition(
			drone.telemetry.position(),
			lambda position: True,
			MAVSDK_CONNECT_TIMEOUT_SEC,
			"an initial position reading",
		)
		home_baro_offset = home_position.relative_altitude_m
		self.get_logger().info(f"MAVSDK: home altitude offset {home_baro_offset:.2f} m.")

		if MAVSDK_SET_TAKEOFF_ALT:
			# Action plugin setter, not param.set_param_float: the raw parameter
			# protocol is the call that dropped the mavsdk_server socket. Best
			# effort -- if it fails without killing the server, fall back to PX4's
			# current MIS_TAKEOFF_ALT rather than aborting the mission.
			self.get_logger().info(f"MAVSDK: setting takeoff altitude={TAKEOFF_ALTITUDE_M:.2f} m.")
			try:
				await drone.action.set_takeoff_altitude(float(TAKEOFF_ALTITUDE_M))
				await asyncio.sleep(0.5)
			except Exception as exc:
				self.get_logger().warning(
					f"MAVSDK: could not set takeoff altitude ({exc!r}); using PX4's current "
					f"MIS_TAKEOFF_ALT. Set it once in the pxh console if needed: "
					f"param set MIS_TAKEOFF_ALT {TAKEOFF_ALTITUDE_M:.1f}"
				)

		self.get_logger().info("MAVSDK: arming.")
		await drone.action.arm()
		self.get_logger().info("MAVSDK: takeoff command.")
		await drone.action.takeoff()

		def _log_takeoff_progress(position, elapsed):
			current_alt = position.relative_altitude_m - home_baro_offset
			self.get_logger().info(
				f"MAVSDK: still climbing after {elapsed:.0f}s -- "
				f"current relative altitude={current_alt:.2f} m, "
				f"target={TAKEOFF_ALTITUDE_M - 0.20:.2f} m"
			)

		await self._wait_for_condition(
			drone.telemetry.position(),
			lambda position: (position.relative_altitude_m - home_baro_offset) >= TAKEOFF_ALTITUDE_M - 0.20,
			MAVSDK_TAKEOFF_ALTITUDE_TIMEOUT_SEC,
			"takeoff altitude reached",
			progress_fn=_log_takeoff_progress,
			progress_interval=5.0,
		)
		self.get_logger().info("MAVSDK: reached takeoff altitude; hovering.")

		self._mavsdk_takeoff_done = True
		self.get_logger().info("MAVSDK: takeoff complete; waiting for attitude-offboard request.")

		while not self._mavsdk_offboard_start_requested and not self._mavsdk_stop_requested:
			await asyncio.sleep(0.05)

		if self._mavsdk_stop_requested:
			return

		# MAVSDK requires at least one setpoint before offboard.start(). Send a
		# short trim stream first, then start offboard, then keep streaming the
		# node's current calibration setpoint.
		for _ in range(10):
			await self._send_mavsdk_attitude_setpoint(drone)
			await asyncio.sleep(MAVSDK_OFFBOARD_PERIOD_SEC)

		try:
			await drone.offboard.start()
		except OffboardError as exc:
			self._mavsdk_offboard_error = repr(exc)
			return

		self._mavsdk_offboard_started = True
		self.get_logger().info("MAVSDK: attitude offboard started.")

		while not self._mavsdk_stop_requested:
			await self._send_mavsdk_attitude_setpoint(drone)
			await asyncio.sleep(MAVSDK_OFFBOARD_PERIOD_SEC)

	async def _send_mavsdk_attitude_setpoint(self, drone):
		sp = self._latest_setpoint
		yaw_rad = sp.yaw
		if MAVSDK_HOLD_CURRENT_YAW and self._vehicle_state.attitude_timestamp > 0.0:
			yaw_rad = self._vehicle_state.attitude_yaw

		await drone.offboard.set_attitude(
			MavsdkAttitude(
				math.degrees(sp.roll),
				math.degrees(sp.pitch),
				math.degrees(yaw_rad),
				self._clamp(sp.thrust, 0.0, 1.0),
			)
		)

	@staticmethod
	def _free_mavsdk_port(port: int):
		try:
			result = subprocess.run(
				["lsof", "-t", f"-i:UDP:{port}"],
				capture_output=True,
				text=True,
				check=False,
			)
		except FileNotFoundError:
			return

		for pid in result.stdout.strip().split():
			try:
				os.kill(int(pid), signal.SIGKILL)
			except ProcessLookupError:
				pass


	def _ready_to_start_prestream(self) -> bool:
		if not self._have_local_position:
			return False
		if REQUIRE_CAMERA_BEFORE_ARM and self._latest_frame is None:
			return False
		if not self._streams_ready_logged:
			self._streams_ready_logged = True
			self.get_logger().info("Required streams are available; starting mission sequence.")
		return True

	def _takeoff_thrust(self) -> float:
		if self._takeoff_target_z is None:
			return HOVER_THRUST

		# PX4 local position is NED. If current z is above target_z numerically
		# (less negative / lower altitude), error is positive and thrust should rise.
		z_error = self._vehicle_state.z - self._takeoff_target_z
		vz = self._vehicle_state.vz  # NED: positive means descending -> add thrust.
		thrust = HOVER_THRUST + TAKEOFF_ALT_KP * z_error + TAKEOFF_VZ_KD * vz
		return self._clamp(thrust, TAKEOFF_THRUST_MIN, TAKEOFF_THRUST_MAX)

	def _at_takeoff_altitude(self) -> bool:
		if self._takeoff_target_z is None:
			return False
		z_error = self._vehicle_state.z - self._takeoff_target_z
		return abs(z_error) <= TAKEOFF_ALTITUDE_TOL_M and abs(self._vehicle_state.vz) <= TAKEOFF_VZ_TOL_M_S

	def _offboard_and_armed(self) -> bool:
		if CONTROL_BACKEND == "mavsdk_offboard":
			return self._mavsdk_offboard_started

		if VehicleStatus is None or not self._vehicle_status_seen:
			return False

		offboard_value = getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14)
		armed_value = getattr(VehicleStatus, "ARMING_STATE_ARMED", 2)
		return self._nav_state == offboard_value and self._arming_state == armed_value

	def _offboard_confirm_timed_out(self, now: float) -> bool:
		return (
			self._offboard_request_time is not None
			and now - self._offboard_request_time >= OFFBOARD_CONFIRM_TIMEOUT_SEC
		)

	def _hold_trim(self):
		self._latest_setpoint = AttitudeSetpoint(
			timestamp=getattr(self._latest_target, "timestamp", 0.0),
			roll=0.0,
			pitch=0.0,
			yaw=0.0,
			thrust=HOVER_THRUST,
		)

	def _damped_hover_setpoint(
		self, now: float, roll: float = 0.0, pitch: float = 0.0, apply_lateral_damping: bool = True
	) -> AttitudeSetpoint:
		"""
		Build a setpoint with thrust from the shared VerticalVelocityDamper
		(velocity + altitude-hold) rather than a bare HOVER_THRUST, and
		roll/pitch nudged by the LateralVelocityDamper unless explicitly
		disabled. Used throughout the takeoff handoff (PRESTREAM onward)
		instead of _hold_trim(), so a hover_thrust guess that's
		meaningfully wrong gets corrected starting from the moment our
		own thrust commands take over control authority -- not several
		seconds later, by which point real altitude (or lateral position)
		can already have drifted.

		apply_lateral_damping=False is for the attitude-offboard probe
		specifically: that phase needs a clean, known roll value to
		measure the actual response to, and mixing in a lateral
		correction would contaminate that measurement. The lateral
		damper's own step() still runs either way (just below), so its
		internal timing stays continuous across the probe instead of
		producing a misleadingly large dt jump immediately after it.
		"""
		thrust = self._damper.step(now, self._vehicle_state.vz, z=self._vehicle_state.z)
		lateral_roll, lateral_pitch = self._lateral_damper.step(
			now, self._vehicle_state.vx, self._vehicle_state.vy, yaw=self._vehicle_state.yaw
		)

		if apply_lateral_damping:
			roll = roll + lateral_roll
			pitch = pitch + lateral_pitch

		return AttitudeSetpoint(
			timestamp=getattr(self._latest_target, "timestamp", 0.0),
			roll=roll,
			pitch=pitch,
			yaw=0.0,
			thrust=thrust,
		)

	def _enter_phase(self, phase: str):
		if phase != self._mission_phase:
			self.get_logger().info(f"Mission phase: {self._mission_phase} -> {phase}")
		self._mission_phase = phase
		self._phase_start_time = time.time()

	def _abort(self, reason: str, status: str = "aborted"):
		if not self._aborted:
			self._aborted = True
			self.diagnostics.set_run_status(status)
			self.get_logger().error(f"ABORTING calibration mission: {reason}. Holding HOVER_THRUST.")
		self._enter_phase(PHASE_ABORTED)
		self._hold_trim()

	def _log_waiting_for_attitude(self, now: float):
		if VERBOSE_STREAM_LOGS and now - self._last_attitude_status_log_time >= self._attitude_status_log_period_sec:
			self._last_attitude_status_log_time = now
			self.get_logger().warning(
				"Waiting for actual attitude before calibration. Check that "
				"/fmu/out/vehicle_attitude or /fmu/out/vehicle_odometry is bridged."
			)

	@staticmethod
	def _quaternion_to_euler(q):
		"""Return roll, pitch, yaw [rad] from PX4 VehicleAttitude.q = [w, x, y, z]."""
		qw, qx, qy, qz = [float(v) for v in q]

		sinr_cosp = 2.0 * (qw * qx + qy * qz)
		cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
		roll = math.atan2(sinr_cosp, cosr_cosp)

		sinp = 2.0 * (qw * qy - qz * qx)
		sinp = max(-1.0, min(1.0, sinp))
		pitch = math.asin(sinp)

		siny_cosp = 2.0 * (qw * qz + qx * qy)
		cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
		yaw = math.atan2(siny_cosp, cosy_cosp)

		return roll, pitch, yaw

	@staticmethod
	def _clamp(value: float, lower: float, upper: float) -> float:
		return max(lower, min(upper, float(value)))


def main(args=None):
	rclpy.init(args=args)

	node = CalibrationNode()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node._mavsdk_stop_requested = True
		node.diagnostics.close()
		node.destroy_node()

		if SHOW_CAMERA:
			cv2.destroyAllWindows()
		rclpy.shutdown()


if __name__ == "__main__":
	main()