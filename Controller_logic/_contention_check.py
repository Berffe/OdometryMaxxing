"""
Standalone host-contention check for optical_flow.update() -- no ROS, no
Gazebo dependency, just cv2/numpy (same requirement as optical_flow.py
itself).

WHY THIS EXISTS: the sandbox profiling in this conversation showed
optical_flow.update() costing ~3.5ms/frame on a clean CPU. The actual flight
log (valid.csv) showed a median of 10.3ms, p95 of 51ms, max of 89ms -- a
3-25x gap. Every stage inside update() was already accounted for by
profiling in the clean environment, so the gap is a strong candidate for
host CPU contention (this project's SITL runs with sim_rtf_estimate~0.22,
meaning Gazebo itself is only managing 22% of real-time speed -- the host is
visibly under heavy load), NOT unaccounted algorithmic cost.

HOW TO USE:
  1. Run this with Gazebo/SITL NOT running:
         python3 contention_check.py --n 500
     This gives a "clean" baseline on this exact machine.
  2. Run it again WHILE a normal SITL mission is actively flying (same
     machine, same time):
         python3 contention_check.py --n 500
     If the numbers come back close to the clean baseline, contention isn't
     the story and the remaining cost is genuinely algorithmic -- worth
     continuing to optimize optical_flow.py directly. If they blow up
     similarly to the real flight log's p95/max, that confirms host
     contention is the dominant remaining factor, and the higher-leverage
     fix is freeing up CPU for the vision pipeline (dedicated core/thread
     priority, a lighter Gazebo physics/render config for testing, or simply
     trusting that real target hardware -- with no Gazebo at all -- will
     perform much closer to the clean baseline than to these SITL numbers).

This benchmarks the REAL code path (imports optical_flow.py directly), not a
reimplementation, so the result is directly comparable to timing_stage_
optical_flow_ms in the diagnostics CSV.

SYNTHETIC vs REAL FRAMES. By default this uses a synthetic frame pair
(make_synthetic_pair). That isolates host/CPU effects but leaves one
hypothesis untested: the synthetic pair (clean speckle + radial gradient,
one clean affine warp) may simply be an EASIER correspondence problem for
Farneback than real platform imagery + the real oscillation/control motion
field -- which would produce the sandbox-vs-flight gap with nothing to do
with contention. To settle that, pass --frames-dir pointing at the .npz
pairs captured by bee_node.py's CAPTURE_FRAMES_FOR_BENCH hook (default dump
location /tmp/bee_frame_capture). The benchmark then times update() on those
REAL captured pairs instead of the synthetic one, cycling through them. This
is the ONE clean flag that switches modes -- with no --frames-dir the script
behaves exactly as before, so the synthetic-baseline runs already done stay
directly comparable.

  Real-frame use, fully offline (SITL NOT running):
      python3 _contention_check.py --frames-dir /tmp/bee_frame_capture --n 500
  If real frames are ALSO fast in isolation here -> the flight-time cost is
  contention (or in-process/GIL scheduling), not the image content or the
  algorithm. If real frames are slow even offline -> it's genuinely harder
  content, and THAT is what any Farneback tuning must be validated against.
"""

import argparse
import glob
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")  # run this from the directory containing optical_flow.py/state.py

from optical_flow import OpticalFlowEstimator
from state import TargetEstimate


def make_synthetic_pair(width=120, height=80, seed=0):
	"""A more realistic synthetic pair than a simple np.roll: textured
	speckle background plus a soft radial gradient (closer to the platform's
	actual visual complexity than uniform noise), with a small combined
	rotation+translation between frames so Farneback has real correspondence
	work to do, not a trivial global shift."""
	rng = np.random.default_rng(seed)
	base = (rng.random((height, width)) * 255).astype(np.uint8)
	base = cv2.GaussianBlur(base, (5, 5), 0)
	yy, xx = np.mgrid[0:height, 0:width]
	cx, cy = width / 2.0, height / 2.0
	radial = (((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5)
	radial = (radial / radial.max() * 40).astype(np.uint8)
	base = cv2.add(base, radial)
	base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

	M = cv2.getRotationMatrix2D((cx, cy), 1.5, 1.02)  # small rotate + zoom
	frame1 = cv2.warpAffine(base_bgr, M, (width, height), flags=cv2.INTER_LINEAR)
	return base_bgr, frame1


def load_real_pairs(frames_dir):
	"""Load every frame_pair_*.npz that bee_node.py's CAPTURE_FRAMES_FOR_BENCH
	hook wrote into frames_dir, as a list of (prev_frame, frame, target)
	tuples ready to feed straight into OpticalFlowEstimator.update() -- the
	same real (prev, current) pairs and TargetEstimate the live pipeline
	actually consumed. Sorted by filename so the natural far-field ->
	fov_saturated progression of the capture run is preserved.

	Returns [] if the directory is empty/missing, so the caller can fall back
	to synthetic with a clear message rather than crashing.
	"""
	paths = sorted(glob.glob(os.path.join(frames_dir, "frame_pair_*.npz")))
	pairs = []
	for path in paths:
		data = np.load(path, allow_pickle=False)
		target = TargetEstimate(
			found=bool(data["target_found"]),
			offset_x=float(data["target_offset_x"]),
			offset_y=float(data["target_offset_y"]),
			detection_width=float(data["target_detection_width"]),
			detection_height=float(data["target_detection_height"]),
			area_fraction=float(data["target_area_fraction"]),
			fov_saturated=bool(data["target_fov_saturated"]),
		)
		pairs.append((data["prev_frame"], data["frame"], target))
	return pairs


def _summarize(durations_ms, mode_label):
	durations_ms.sort()
	n = len(durations_ms)
	p95 = durations_ms[int(0.95 * n)]
	print(f"mode:   {mode_label}")
	print(f"n={n}")
	print(f"mean:   {statistics.mean(durations_ms):.3f} ms")
	print(f"median: {statistics.median(durations_ms):.3f} ms")
	print(f"p95:    {p95:.3f} ms")
	print(f"max:    {max(durations_ms):.3f} ms")
	print()
	print("Compare against valid.csv's timing_stage_optical_flow_ms:")
	print("  median=10.3ms  p95=51.1ms  max=89.3ms")
	print("Close to THIS run's numbers while SITL is idle -> algorithmic, keep optimizing.")
	print("Close to THIS run's numbers only while SITL is ACTIVELY flying -> host contention.")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--n", type=int, default=500, help="number of update() calls to time")
	parser.add_argument("--width", type=int, default=120)
	parser.add_argument("--height", type=int, default=80)
	parser.add_argument(
		"--frames-dir", type=str, default=None,
		help="Directory of real frame_pair_*.npz captures (from bee_node.py's "
		     "CAPTURE_FRAMES_FOR_BENCH hook). If omitted, uses synthetic frames "
		     "-- the ONE flag that switches modes; leaving it off preserves the "
		     "original synthetic-baseline behavior exactly.",
	)
	args = parser.parse_args()

	# --- REAL-FRAME MODE (only when --frames-dir is given) ----------------
	if args.frames_dir is not None:
		pairs = load_real_pairs(args.frames_dir)
		if not pairs:
			print(
				f"No frame_pair_*.npz found in {args.frames_dir!r}. Capture some "
				f"first (set CAPTURE_FRAMES_FOR_BENCH=True in bee_node.py and fly), "
				f"or drop --frames-dir to run the synthetic baseline instead.",
				file=sys.stderr,
			)
			sys.exit(1)

		# Warm-up + timing both cycle through the real pairs. Each update()
		# uses that pair's OWN prev/current, so consecutive calls are seeded
		# per-pair (est.update stores prev internally, but we always pass the
		# captured prev alongside the captured current via two calls: seed
		# then time, so the timed call sees exactly the captured transition).
		est = OpticalFlowEstimator(require_target_roi=False, derotator=None, store_debug=False)

		durations_ms = []
		for i in range(args.n):
			prev_frame, frame, target = pairs[i % len(pairs)]
			# Seed prev (untimed) then time the real captured transition, so
			# the measured call reproduces exactly the (prev -> current) pair
			# the live pipeline saw -- not an accidental cross-pair diff.
			est.update(prev_frame, timestamp=2.0 * i, target=target)
			t0 = time.perf_counter()
			est.update(frame, timestamp=2.0 * i + 0.033, target=target)
			durations_ms.append((time.perf_counter() - t0) * 1000.0)

		_summarize(durations_ms, f"REAL frames from {args.frames_dir} ({len(pairs)} unique pairs)")
		return

	# --- SYNTHETIC MODE (default, unchanged) ------------------------------
	frame0, frame1 = make_synthetic_pair(args.width, args.height)
	target = TargetEstimate(
		found=True, offset_x=0.0, offset_y=0.0,
		detection_width=args.width, detection_height=args.height,
		fov_saturated=True,  # the worst-case, largest-ROI regime
	)

	est = OpticalFlowEstimator(require_target_roi=False, derotator=None, store_debug=False)
	est.update(frame0, timestamp=0.0, target=target)  # seed prev frame, untimed

	durations_ms = []
	for i in range(args.n):
		t0 = time.perf_counter()
		est.update(frame1, timestamp=0.033 * (i + 1), target=target)
		durations_ms.append((time.perf_counter() - t0) * 1000.0)

	_summarize(durations_ms, "SYNTHETIC frame pair")


if __name__ == "__main__":
	main()