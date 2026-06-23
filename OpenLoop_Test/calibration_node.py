"""
Open-loop calibration node.

This is a sibling to bee_node.py, not a replacement: same camera and
PX4 plumbing, same vision pipeline, same diagnostics CSV — but
ControlLaw is not used. Instead, a StepSequence (calibration_sequence.py)
drives roll, pitch, and thrust through a known step train, one axis at
a time, while the other axes are held at trim. That's what makes the
resulting log usable for identifying control_law.py's per-axis discrete
model (e[k+1] = a*e[k] + b*u[k]): the command has to be independent of
the measured state for the regression to be valid, and it isn't, if the
closed-loop controller is the thing producing it.

Two layers protect against vertical drift, for two different reasons:

	1. Before the open-loop sequence starts, VerticalSettler (see
	   calibration_sequence.py) damps out any residual vertical
	   velocity. Commanding exactly HOVER_THRUST zeroes acceleration,
	   not velocity — any vz left over from arming or mode-switching
	   would otherwise persist through the whole test.
	2. During roll/pitch testing (and the inter-axis settle gaps),
	   thrust stays under the *same* continuous damping rather than
	   trusting a bare HOVER_THRUST constant for the full run.

Both layers now share one VerticalVelocityDamper instance with PI (not
just P) control. Real data is why: a proportional-only damper computed
exactly as designed (formula matched the logged thrust to floating-
point precision) while vz still sat at a persistent nonzero mean in
every phase, growing over the run's duration — that's not a tuning
problem, proportional action structurally cannot cancel a constant
disturbance, it can only reach an equilibrium against it. The integral
term does what proportional action can't, and is shared continuously
across the settle phase and roll/pitch testing on purpose, since
whatever disturbance it's correcting for doesn't reset at phase
boundaries either. Thrust testing itself stays genuinely open-loop
throughout — that's the one phase where closed-loop feedback would
re-create the cause/effect entanglement this whole node exists to
avoid.

Neither layer is a substitute for keeping HOVER_THRUST accurate — they
bound the damage, they don't remove the need to update it (see below).

A hard safety envelope is also checked every tick, in every phase: if
|vz| or area_fraction exceeds ABORT_VZ_LIMIT / ABORT_AREA_FRACTION_MAX,
the node halts at HOVER_THRUST permanently rather than trying to
recover automatically — reposition in Gazebo and restart the node.
Separately, during the test sequence itself (not the settle phase,
which doesn't need vision), losing target/flow for LOST_TARGET_ABORT_SEC
also aborts — there's no point running the rest of a step train against
a target that isn't there, and silently finishing with zero usable
samples for the current axis is worse than stopping early and saying so.

How to use this:

	1. In Gazebo, position the vehicle at the altitude/area_fraction
	   operating point you want to identify (this node does not manage
	   altitude — see the safety note below on why).
	2. Run this node instead of bee_node.py. It will settle first (a few
	   seconds, usually), then run the test sequence (TEST_SETTLE_SEC +
	   one settle gap and step train per axis in TEST_AXES, each train
	   being TEST_REPEATS * 4 * <that axis' hold_sec> seconds; printed
	   at startup).
	3. Stop the node once it logs that the sequence is finished, and
	   feed the resulting CSV to fit_axis_models.py.
	4. Reposition to the next operating point and repeat.

Safety note: the thrust step train commands real deviations from
hover_thrust, so the vehicle will actually climb/descend during that
part of the sequence — make sure there's enough clearance below (and
above) wherever you run it in Gazebo. Roll/pitch hold each step much
longer than thrust does now (ROLL_TEST_HOLD_SEC/PITCH_TEST_HOLD_SEC —
raised because position from a held tilt scales with hold_sec^2 for a
roughly double-integrator response, a much bigger lever on signal
strength than amplitude, which is already near ROLL_LIMIT_RAD/
PITCH_LIMIT_RAD). The abort bounds above are the backstop if any of
this still goes further than expected.

Run:

	ros2 run bee_control calibration_node

or directly:

	python -m bee_control.calibration_node
"""

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

from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleLocalPosition
from cv_bridge import CvBridge, CvBridgeError

from .state import VehicleState, AttitudeSetpoint, TargetEstimate
from .optical_flow import OpticalFlowEstimator
from .target_acquisition import TargetAcquisition
from .px4_interface import PX4Interface
from .diagnostics_writer import DiagnosticsWriter
from .calibration_sequence import (
	build_calibration_sequence,
	VerticalSettler,
	VerticalVelocityDamper,
	exceeds_safety_bounds,
)


HEARTBEAT_PERIOD_SEC = 0.1
TEST_PERIOD_SEC = 0.5  # must match the dt control_law.py's local models assume
ARM_AFTER_HEARTBEATS = 10
SHOW_CAMERA = True

# Vehicle trim. Keep this matched to ControlLaw's hover_thrust — the
# thrust step train is defined as a deviation from it, and the settle
# phase / continuous roll-pitch damping below both use it as the
# baseline they damp around. THIS WAS LEFT AT THE OLD 0.45 DEFAULT FOR
# SEVERAL ROUNDS OF CALIBRATION AFTER 0.4412 was identified — that
# mismatch is exactly what produces a slow, continuous vz drift for the
# entire run (not just a leftover-velocity blip at t=0): holding thrust
# 0.009 above true hover commands a small constant upward acceleration
# the whole time. Keep this in sync with whatever fit_axis_models.py's
# thrust intercept most recently implied.
HOVER_THRUST = 0.4412

# Step-train shape. Keep the roll/pitch amplitudes comfortably inside
# ControlLaw's roll_limit/pitch_limit, and the thrust amplitude
# comfortably inside thrust_min/thrust_max, so the identified model
# stays valid over the range the real controller will actually command.
# Amplitude raised from 0.04 previously; with the real fitted b this
# weak, ±0.04 rad wasn't exciting the system enough to reliably
# separate b from noise.
ROLL_TEST_AMPLITUDE_RAD = 0.08
PITCH_TEST_AMPLITUDE_RAD = 0.08
THRUST_TEST_AMPLITUDE = 0.05

# Hold duration, per axis (not shared — see build_calibration_sequence's
# docstring). Roll/pitch raised from 2.0s to 6.0s: for a roughly
# double-integrator response (tilt -> acceleration -> velocity ->
# position), position scales with hold_sec^2, so this is a much bigger
# lever on signal strength than amplitude is, and amplitude is already
# near its ceiling. Thrust's hold deliberately stays short — it already
# commands real altitude excursions, and lengthening it directly widens
# the area_fraction range swept within one file (a separate problem;
# see fit_axis_models.py's wide-range warning).
ROLL_TEST_HOLD_SEC = 6.0
PITCH_TEST_HOLD_SEC = 6.0
THRUST_TEST_HOLD_SEC = 2.0

TEST_REPEATS = 3
TEST_SETTLE_SEC = 2.0
TEST_AXES = ("roll", "pitch", "thrust")

# Defensive clamps applied to whatever the sequence produces, mirroring
# ControlLaw's own limits, in case the amplitudes above are ever set
# too large by mistake.
ROLL_LIMIT_RAD = 0.10
PITCH_LIMIT_RAD = 0.10
THRUST_MIN = 0.35
THRUST_MAX = 0.65

# Pre-test settle phase, and the continuous damping during roll/pitch
# testing that follows it, share ONE VerticalVelocityDamper instance
# (constructed in __init__) — the integral term is exactly the part
# that should carry over across that boundary rather than resetting,
# since whatever disturbance it's correcting for doesn't reset either.
#
# Commanding exactly HOVER_THRUST zeroes vertical *acceleration*, not
# velocity, so a residual vz at t=0 would otherwise persist through the
# whole sequence. That's what the settle phase is for. But proportional
# damping alone (kp only) reaches an equilibrium against any constant
# disturbance instead of cancelling it — confirmed directly from a real
# run: the P-only formula was computing exactly as designed, while vz
# still sat at a persistent nonzero mean in every phase, growing over
# the run's duration. ki adds the integral action that actually drives
# a steady disturbance's effect on vz to zero over time. integral_limit
# bounds windup (a long, large transient — e.g. early in the settle
# phase — accumulating a correction so large it overshoots once the
# real disturbance is gone).
SETTLE_VZ_THRESHOLD = 0.05
SETTLE_MIN_DURATION_SEC = 1.0
SETTLE_TIMEOUT_SEC = 15.0
DAMPER_KP = 0.08
DAMPER_KI = 0.02
DAMPER_INTEGRAL_LIMIT = 0.05

# Hard safety bounds, checked every tick regardless of phase. Tripping
# either one halts at HOVER_THRUST permanently (reposition and restart
# the node) rather than trying to recover automatically.
ABORT_VZ_LIMIT = 1.0
ABORT_AREA_FRACTION_MAX = 0.97

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


class CalibrationNode(Node):
	def __init__(self):
		super().__init__("bee_calibration_node")

		self.bridge = CvBridge()

		self._last_position_log_time = 0.0
		self._position_log_period_sec = 1.0

		self._image_count = 0
		self._last_image_log_time = 0.0
		self._image_log_period_sec = 1.0

		self._vehicle_state = VehicleState()

		self._latest_flow = None
		self._latest_frame = None
		self._latest_target = TargetEstimate()
		self._latest_setpoint = AttitudeSetpoint(
			roll=0.0, pitch=0.0, yaw=0.0, thrust=HOVER_THRUST
		)

		self._heartbeat_count = 0
		self._offboard_engaged = False

		self._test_start_time = None
		self._sequence_finished_logged = False

		self._damper = VerticalVelocityDamper(
			hover_thrust=HOVER_THRUST,
			kp=DAMPER_KP,
			ki=DAMPER_KI,
			thrust_min=THRUST_MIN,
			thrust_max=THRUST_MAX,
			integral_limit=DAMPER_INTEGRAL_LIMIT,
		)
		self._settler = VerticalSettler(
			self._damper,
			vz_threshold=SETTLE_VZ_THRESHOLD,
			min_duration_sec=SETTLE_MIN_DURATION_SEC,
			timeout_sec=SETTLE_TIMEOUT_SEC,
		)
		self._settle_logged = False
		self._aborted = False
		self._lost_since = None

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
			repeats=TEST_REPEATS,
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

		self.create_subscription(
			Image,
			"/bee_x500/camera/image",
			self.on_camera,
			camera_qos,
		)

		self.px4 = PX4Interface(self, px4_qos)

		self.create_timer(HEARTBEAT_PERIOD_SEC, self.on_heartbeat_timer)
		self.create_timer(TEST_PERIOD_SEC, self.on_test_timer)

		if SHOW_CAMERA:
			cv2.namedWindow("Bee Calibration - Camera", cv2.WINDOW_NORMAL)

		self.get_logger().info("bee_calibration_node started.")
		self.get_logger().info("Waiting for PX4 local position on /fmu/out/vehicle_local_position_v1")
		self.get_logger().info("Waiting for camera images on /bee_x500/camera/image")

	def on_camera(self, msg: Image):
		self._image_count += 1

		now = time.time()

		if now - self._last_image_log_time >= self._image_log_period_sec:
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

		if SHOW_CAMERA:
			frame = cv2.rotate(src, cv2.ROTATE_180)
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

	def on_local_position(self, msg: VehicleLocalPosition):
		now = time.time()

		if now - self._last_position_log_time >= self._position_log_period_sec:
			self._last_position_log_time = now

			self.get_logger().info(
				f"local position: "
				f"x={msg.x:.2f} m, y={msg.y:.2f} m, z={msg.z:.2f} m"
			)

		self._vehicle_state = VehicleState(
			timestamp=now,
			x=msg.x,
			y=msg.y,
			z=msg.z,
			vx=msg.vx,
			vy=msg.vy,
			vz=msg.vz,
			yaw=msg.heading,
		)

	def on_heartbeat_timer(self):
		self.px4.publish_heartbeat()

		self.px4.publish_attitude_setpoint(
			self._latest_setpoint.roll,
			self._latest_setpoint.pitch,
			self._latest_setpoint.yaw,
			self._latest_setpoint.thrust,
		)

		self._heartbeat_count += 1

		if not self._offboard_engaged and self._heartbeat_count == ARM_AFTER_HEARTBEATS:
			self.px4.arm()
			self.px4.engage_offboard_mode()
			self._offboard_engaged = True

	def on_test_timer(self):
		if self._latest_flow is None or self._latest_frame is None:
			return

		now = time.time()
		vz = self._vehicle_state.vz
		area_fraction = float(getattr(self._latest_target, "area_fraction", 0.0))

		# --- Safety envelope: checked every tick, every phase, first. ---
		if self._aborted or exceeds_safety_bounds(
			vz, area_fraction, ABORT_VZ_LIMIT, ABORT_AREA_FRACTION_MAX
		):
			if not self._aborted:
				self._aborted = True
				self.get_logger().error(
					f"ABORTING calibration: vz={vz:.3f} m/s, area_fraction={area_fraction:.3f} "
					f"exceeded safety bounds (limits: |vz|<{ABORT_VZ_LIMIT}, "
					f"area_fraction<{ABORT_AREA_FRACTION_MAX}). Holding HOVER_THRUST. "
					f"Reposition in Gazebo and restart this node to try again."
				)

			self._latest_setpoint = AttitudeSetpoint(
				timestamp=self._latest_target.timestamp,
				roll=0.0, pitch=0.0, yaw=0.0, thrust=HOVER_THRUST,
			)
			return  # not logged -- an abort-hold sample is not test data

		# --- Phase 1: settle residual vertical velocity before starting. ---
		# Commanding exactly HOVER_THRUST zeroes acceleration, not velocity
		# — any vz left over from arming/mode-switching would otherwise
		# persist through the whole open-loop sequence. See
		# VerticalSettler in calibration_sequence.py.
		if not self._settler.is_settled:
			if not self._settle_logged:
				self._settle_logged = True
				self.get_logger().info(
					f"Settling residual vertical velocity (vz={vz:.3f} m/s) "
					f"before starting the test sequence..."
				)

			thrust = self._settler.step(now, vz)
			self._latest_setpoint = AttitudeSetpoint(
				timestamp=self._latest_target.timestamp,
				roll=0.0, pitch=0.0, yaw=0.0, thrust=thrust,
			)

			if self._settler.is_settled:
				if self._settler.timed_out:
					self.get_logger().warning(
						f"Settle phase timed out after {SETTLE_TIMEOUT_SEC}s (vz={vz:.3f} m/s); "
						f"starting the test sequence anyway. If this happens often, increase "
						f"SETTLE_TIMEOUT_SEC or DAMPER_KP/DAMPER_KI."
					)
				else:
					self.get_logger().info(
						f"Settled (|vz|<{SETTLE_VZ_THRESHOLD} m/s for {SETTLE_MIN_DURATION_SEC}s). "
						f"Starting test sequence."
					)

			return  # settle-phase samples are closed-loop -- never logged as test data

		# --- Phase 2: the actual open-loop test sequence. ---
		if self._test_start_time is None:
			self._test_start_time = now
			self.get_logger().info("Test sequence started.")

		target_ok = bool(self._latest_target.found)
		flow_ok = bool(self._latest_flow.valid)

		if not (target_ok and flow_ok):
			if self._lost_since is None:
				self._lost_since = now
			elif now - self._lost_since >= LOST_TARGET_ABORT_SEC:
				self._aborted = True
				self.get_logger().error(
					f"ABORTING calibration: target/flow lost for >= {LOST_TARGET_ABORT_SEC}s "
					f"during the test sequence (target_found={target_ok}, flow_valid={flow_ok}). "
					f"No more useful data will be collected this way -- holding HOVER_THRUST. "
					f"Reposition in Gazebo and restart this node to try again."
				)
				self._latest_setpoint = AttitudeSetpoint(
					timestamp=self._latest_target.timestamp,
					roll=0.0, pitch=0.0, yaw=0.0, thrust=HOVER_THRUST,
				)
				return
		else:
			self._lost_since = None

		elapsed = now - self._test_start_time

		roll, pitch, thrust = self.sequence.command_at(elapsed)
		current_axis = self.sequence.axis_at(elapsed)

		if current_axis != "thrust":
			# Roll/pitch (and the inter-axis settle gaps) don't want to
			# rely on HOVER_THRUST being exactly right for the whole
			# duration of the run -- actively damp vz instead, continuing
			# the SAME damper (and its accumulated integral) the settle
			# phase was using. Thrust itself must stay genuinely
			# open-loop here: that's the one axis where closed-loop
			# feedback would re-create the same cause/effect entanglement
			# this whole node exists to avoid.
			thrust = self._damper.step(now, vz)

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

		self.diagnostics.write(
			wall_timestamp=time.time(),
			target=self._latest_target,
			flow=self._latest_flow,
			setpoint=self._latest_setpoint,
			vehicle_state=self._vehicle_state,
			calibration_axis=current_axis,
		)

		if self.sequence.is_finished(elapsed) and not self._sequence_finished_logged:
			self._sequence_finished_logged = True
			self.get_logger().info(
				"Test sequence finished; holding at trim. "
				"Stop the node and run fit_axis_models.py on the CSV above."
			)

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
		node.diagnostics.close()
		node.destroy_node()

		if SHOW_CAMERA:
			cv2.destroyAllWindows()
		rclpy.shutdown()


if __name__ == "__main__":
	main()