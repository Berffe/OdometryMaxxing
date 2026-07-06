"""
Optical-flow de-rotation: remove the ego-rotation component of the measured
flow field using PX4 body angular rates, leaving (ideally) only the
translational flow that carries the divergence / offset landing signal.

ROS-free by design (numpy + stdlib only), same as optical_flow.py / state.py,
so it can be unit-tested without rclpy and imported from either. Nothing here
touches clock.py or state.py.

WHY GYRO-ONLY IS SUFFICIENT
---------------------------
The motion field at an image point splits into a TRANSLATIONAL part (depends on
camera translation AND scene depth -- this is the looming/divergence signal we
WANT to keep) and a ROTATIONAL part that depends ONLY on the camera's angular
velocity and the image geometry (focal length + pixel position), NOT on depth
or translation. So the rotational part is fully predictable from angular rate
alone and can be subtracted with no depth and no accelerometer information --
the bio-inspired "haltere" pathway: rotation sensing, not gravity sensing.

MODEL (Longuet-Higgins & Prazdny motion field, rotational term only)
--------------------------------------------------------------------
Pixel coords measured from the principal point, x right / y down / z into the
scene, focal length f in pixels, camera angular velocity Omega=(wx,wy,wz) in
the SAME optical frame:

    u_rot(x,y) = (wx*x*y)/f - wy*(f + x^2/f) + wz*y
    v_rot(x,y) =  wx*(f + y^2/f) - (wy*x*y)/f - wz*x

Both are px/s -- the same units as OpticalFlowEstimator's flow_px_s -- so
de-rotation is a straight per-pixel subtraction of this field BEFORE mean-flow
and the affine divergence fit run.

Note the rotational field is itself linear-ish in (x,y): the wz terms are
exactly linear, and the wx,wy terms add a quadratic that is near-linear across a
small ROI. So it contaminates the affine divergence SLOPE (a1+b2), not just the
mean flow -- which is exactly why de-rotating helps the divergence-based descent
and not only the lateral (offset) loop. The contamination is worst for an
off-centre ROI: a nadir-centred ROI sees mostly the near-uniform wx,wy shift,
but the moment the target box drifts off the principal point the linear terms
bite.

FRAMES AND SIGNS -- *** VALIDATE EMPIRICALLY BEFORE TRUSTING THE DEFAULT ***
---------------------------------------------------------------------------
Omega is obtained from the body-frame PX4 rate (FRD: x fwd, y right, z down, the
convention of VehicleAngularVelocity.xyz) via a fixed 3x3 R_body_to_optical that
must encode, in order:
  1. FRD -> base_link's FLU convention,
  2. the camera MOUNT (model.sdf: bee_camera_link is pitched +90 deg -> nadir),
  3. Gazebo's camera-looks-along-local-+X convention (not the OpenCV/ROS
     optical Z-forward convention this module's equations assume), and
  4. the cv2.ROTATE_180 that bee_node.on_camera applies to EVERY frame BEFORE
     optical flow sees it -- a 180 deg in-plane rotation about the optical axis.

See DEFAULT_R_BODY_TO_OPTICAL's comment for the derivation and its revision
history. The first-pass guess (a simple per-axis sign flip, diag(-1,-1,+1))
was wrong in a way worth remembering: on real flight data, trying the opposite
diagonal (diag(+1,+1,+1)) just mirrored the sign of the induced error at the
same magnitude instead of shrinking it -- proof the mistake was an AXIS SWAP
(which body rate feeds which optical component), not a polarity error. No
diagonal matrix can fix a swap, which is why the search moved to re-deriving
the transform from the mount geometry instead of trying more sign
permutations. Acceptance test: take a segment with strong yaw/pitch RATE but
near-zero translation (a hover wobble suffices) and confirm the de-rotated
mean flow AND divergence collapse toward zero -- and confirm variance and
|correlation with attitude rate| both DECREASE relative to raw, not increase.
An increase either way means R is still wrong somewhere in this chain, not
just a sign to flip.
"""

import collections

import numpy as np


# Maps body FRD rate (p,q,r) -> optical-frame Omega for a nadir camera whose
# frames are additionally rotated 180 deg in software (see docstring #3). This
# is diag(-1,-1,+1): optical axes taken coincident with body FRD for a straight-
# down camera (optical z = body down), then x,y negated for the 180 deg image
# rotation; yaw-about-optical-axis (r) is unchanged by an in-plane rotation.
# *** Treat as a hypothesis to be validated, not ground truth. ***
# Maps body FRD rate (p,q,r) -> optical-frame Omega for the belly camera.
#
# REVISION HISTORY (kept because the reasoning matters more than the number):
# the original guess here was diag(-1,-1,+1), assuming a simple per-axis sign
# flip was enough for a nadir camera + the software 180 deg image rotation.
# Flight data proved that wrong in a specific way: flipping the WHOLE diagonal
# (trying diag(+1,+1,+1) next) just mirrored the sign of the induced error
# without shrinking it (variance still rose, |correlation with attitude rate|
# stayed ~0.4-0.7 either way). A same-magnitude, sign-mirrored failure under a
# global sign flip is the signature of a wrong AXIS mapping, not a wrong sign --
# no diagonal matrix can ever fix that.
#
# Re-deriving from model.sdf's actual mount chain (rather than guessing again):
#   1. FRD (p,q,r) -> base_link's own FLU convention: (p, -q, -r)
#      [FLU: x same, y = -y_FRD (left vs right), z = -z_FRD (up vs down)]
#   2. base_link -> bee_camera_link, via the joint's <pose ...> pitch=+1.5708
#      (model.sdf): rotating the FLU vector through R=Ry(90deg) and taking the
#      link-local components (R^T v) gives (r, -q, p) in the link's own
#      (X=look-direction, Y=left, Z=up) axes.
#   3. Gazebo's <camera> sensor looks along its link's local +X (not the
#      OpenCV/ROS optical convention of +Z) -- remap link-local (X=look,
#      Y=left, Z=up) into the OpenCV-style optical frame this module's flow
#      equations assume (Xo=right, Yo=down, Zo=look-direction):
#      Zo=Xg, Xo=-Yg, Yo=-Zg  ->  (q, -p, r).
#   4. The software cv2.ROTATE_180 (bee_node.on_camera) flips both image axes
#      about the optical axis: Xo'=-Xo, Yo'=-Yo, Zo'=Zo -> (-q, p, r).
#
# Net: wx=-q (pitch rate), wy=+p (roll rate), wz=+r (yaw rate) -- an axis SWAP
# (roll feeds wy, pitch feeds wx), not a per-axis sign flip. This matches flight
# data cleanly: raw (untouched) mean_flow_x correlates strongly with ROLL rate
# and not pitch, raw mean_flow_y correlates strongly with PITCH rate and not
# roll, both with the sign this matrix predicts. That cross-check (derivation
# vs. independent empirical evidence) is why this is a real fix candidate and
# not a third guess -- but it is still validated against real closed-loop
# flight, where translation and rotation are correlated with each other, not
# the isolated hover-yaw/pitch-wobble test recommended below. Confirm on such a
# segment (near-zero translation, deliberate rate oscillation) before fully
# trusting the derotated flow in the control loop.
DEFAULT_R_BODY_TO_OPTICAL = np.array(
	[[0.0, -1.0, 0.0],
	 [1.0, 0.0, 0.0],
	 [0.0, 0.0, 1.0]],
	dtype=np.float64,
)


class CameraGeometry:
	"""Pinhole geometry + body->optical extrinsic for the belly camera.

	focal_px is in PIXELS. Prefer from_horizontal_fov() so the value stays tied
	to model.sdf's <horizontal_fov> rather than being hand-copied.
	"""

	def __init__(
		self,
		focal_px,
		width,
		height,
		cx=None,
		cy=None,
		R_body_to_optical=None,
		sign_convention: float = 1.0,
	):
		self.focal_px = float(focal_px)
		self.width = int(width)
		self.height = int(height)
		# Principal point at the image centre in 0..N-1 index coordinates, the
		# same convention the flow grid is built on.
		self.cx = float(cx) if cx is not None else 0.5 * (self.width - 1)
		self.cy = float(cy) if cy is not None else 0.5 * (self.height - 1)
		self.R = (
			np.asarray(R_body_to_optical, dtype=np.float64)
			if R_body_to_optical is not None
			else DEFAULT_R_BODY_TO_OPTICAL.copy()
		)
		if self.R.shape != (3, 3):
			raise ValueError("R_body_to_optical must be 3x3")
		# A single global flip knob, handy during sign validation without
		# touching R. Leave at +1.0 in normal use.
		self.sign_convention = float(sign_convention)

	@classmethod
	def from_horizontal_fov(cls, horizontal_fov_rad, width, height, **kwargs):
		"""f_px = (width/2) / tan(hfov/2) -- the exact focal length implied by a
		Gazebo/SDF camera's <horizontal_fov> and pixel <width>. Square pixels
		(fx == fy) are assumed, as the SDF specifies only a single FOV."""
		f = (0.5 * float(width)) / float(np.tan(0.5 * float(horizontal_fov_rad)))
		return cls(f, width, height, **kwargs)


class Derotator:
	"""Predicts and subtracts the rotational optical-flow field over a ROI."""

	def __init__(self, geometry: CameraGeometry):
		self.geo = geometry
		self._last_rotational_mean = (0.0, 0.0)
		self._last_optical_rates = (0.0, 0.0, 0.0)

	def optical_rates(self, omega_body):
		"""Body FRD rate -> optical-frame (wx, wy, wz)."""
		w = self.geo.R @ np.asarray(omega_body, dtype=np.float64).reshape(3)
		w = self.geo.sign_convention * w
		self._last_optical_rates = (float(w[0]), float(w[1]), float(w[2]))
		return w

	def rotational_flow(self, omega_body, x0, y0, roi_width, roi_height):
		"""Per-pixel rotational flow (u_rot, v_rot) in px/s over the ROI.

		x0, y0 are the ROI's top-left corner in FULL-image pixels, so each ROI
		pixel is placed at its true offset from the principal point (the
		rotational field depends on absolute image position, not ROI-local).
		"""
		wx, wy, wz = self.optical_rates(omega_body)
		f = self.geo.focal_px

		cols = np.arange(roi_width, dtype=np.float64)
		rows = np.arange(roi_height, dtype=np.float64)
		x = (x0 + cols) - self.geo.cx           # (roi_width,)
		y = (y0 + rows) - self.geo.cy           # (roi_height,)
		xg, yg = np.meshgrid(x, y)              # (roi_height, roi_width)

		u_rot = (wx * xg * yg) / f - wy * (f + xg * xg / f) + wz * yg
		v_rot = wx * (f + yg * yg / f) - (wy * xg * yg) / f - wz * xg
		return u_rot, v_rot

	def derotate(self, flow_px_s, omega_body, roi):
		"""Subtract the predicted rotational field from a measured px/s flow
		field over `roi` = (x0, y0, x1, y1). Returns a NEW array; the input is
		left untouched. If omega_body is None (or the field is None) the flow is
		returned unchanged -- the legacy no-de-rotation path.
		"""
		if flow_px_s is None or omega_body is None:
			return flow_px_s

		x0, y0, _x1, _y1 = roi
		roi_height, roi_width = flow_px_s.shape[:2]
		u_rot, v_rot = self.rotational_flow(
			omega_body, x0, y0, roi_width, roi_height
		)
		self._last_rotational_mean = (
			float(np.mean(u_rot)),
			float(np.mean(v_rot)),
		)

		out = flow_px_s.copy()
		out[:, :, 0] -= u_rot
		out[:, :, 1] -= v_rot
		return out

	def last_rotational_mean(self):
		"""(mean u_rot, mean v_rot) px/s subtracted on the last derotate() call
		-- a cheap magnitude to log against the raw mean flow when validating."""
		return self._last_rotational_mean

	def last_optical_rates(self):
		return self._last_optical_rates


class AngularRateBuffer:
	"""Short ring buffer of (t_sec, omega_xyz) body-rate samples, averaged over a
	requested recent duration.

	CLOCK NOTE (see clock.py). Samples are tagged with the rate message's OWN
	PX4 stamp in seconds (msg.timestamp / 1e6), which advances at the lockstep
	sim RATE -- the same rate as the camera Image sim stamp. mean_recent()
	selects its window by DURATION only: the caller passes the camera
	inter-frame dt (a SIM-family difference, valid subtraction) and this buffer
	walks back by that many seconds of its OWN PX4-family stamps (also a valid
	within-family difference). No WALL<->SIM<->PX4 absolute offset ever enters
	this path, so clock.py's rule that the measured diagnostic offsets must not
	feed control is respected by construction. The single assumption is that the
	rate and camera streams are delivered roughly in step (true over one
	uXRCE-DDS bridge); any residual latency skew merely shifts which handful of
	smooth samples get averaged, and is exactly what the pure-rotation
	validation in Derotator's docstring checks.
	"""

	def __init__(self, maxlen: int = 256):
		self._t = collections.deque(maxlen=maxlen)
		self._w = collections.deque(maxlen=maxlen)

	def add(self, t_sec, omega_xyz):
		"""Append one sample. No-op on a non-positive/garbage stamp so an
		un-seeded (zero) PX4 timestamp cannot poison the window."""
		try:
			t = float(t_sec)
			w = (
				float(omega_xyz[0]),
				float(omega_xyz[1]),
				float(omega_xyz[2]),
			)
		except (TypeError, ValueError, IndexError):
			return
		if t <= 0.0 or not np.all(np.isfinite(w)):
			return
		self._t.append(t)
		self._w.append(w)

	def mean_recent(self, duration_sec):
		"""Mean body rate over the most recent `duration_sec` of buffered
		samples. Returns (omega_mean [np.ndarray shape (3,)], n_used, valid).
		valid is False only when the buffer is empty (rate stream not up yet).
		A non-positive/None duration falls back to the single latest sample.
		"""
		n = len(self._t)
		if n == 0:
			return None, 0, False

		if duration_sec is None or float(duration_sec) <= 0.0:
			return np.asarray(self._w[-1], dtype=np.float64), 1, True

		t_latest = self._t[-1]
		lo = t_latest - float(duration_sec)
		acc = np.zeros(3, dtype=np.float64)
		k = 0
		for t, w in zip(reversed(self._t), reversed(self._w)):
			if t < lo:
				break
			acc += w
			k += 1

		if k == 0:
			# Interval shorter than the gap between samples -> newest sample.
			return np.asarray(self._w[-1], dtype=np.float64), 1, True
		return acc / k, k, True

	def __len__(self):
		return len(self._t)