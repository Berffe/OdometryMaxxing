"""
vision_worker.py -- out-of-process vision pipeline for bee_node (v2.0).

WHY THIS EXISTS
---------------
Through v1.x, BeeLandNode.on_camera ran the two heavy vision stages --
TargetAcquisition.update() then OpticalFlowEstimator.update() -- INLINE, on the
single ROS 2 executor thread. main() drives the node with rclpy.spin(), i.e. a
SINGLE-threaded executor, so every camera frame blocked the SAME thread that
also has to service on_control_timer (100 Hz) and on_px4_setpoint_timer
(20 Hz). A 40-200 ms vision frame therefore starved the PX4 setpoint publisher
for 40-200 ms at a time -- gaps in the offboard stream are exactly the lag this
refactor removes.

v2.0 moves ONLY the two algorithm calls into this separate process. The
algorithm modules (target_acquisition.py, optical_flow.py) are UNCHANGED; what
changes is who calls them. bee_node ships (frame, timestamp, body_rates) here
over a multiprocessing.Queue and drains VisionResult(...) back on its control
timer, so the executor thread stays free to publish setpoints on cadence while
this process crunches in parallel on another core (its own GIL).

The two calls run in the SAME ORDER as before -- target acquisition first (its
TargetEstimate is an input to the flow update), optical flow second -- so the
numerical pipeline is byte-for-byte the one bee_node ran inline; only the
thread/process it runs on has changed.

DELIBERATELY NOT A ROS NODE
---------------------------
This module must never import rclpy or touch DDS -- that is the whole point of
it being a plain multiprocessing.Process (and why bee_node spawns it with the
'spawn' start method: a clean interpreter that inherits none of the parent's
rclpy/DDS threads or locks). Keep the imports here limited to the algorithm
modules and their dependencies so a spawned child stays light and DDS-free.

PICKLING
--------
multiprocessing.Queue pickles everything crossing the boundary:
  in : (frame: np.ndarray, timestamp: float, body_rates: tuple|np.ndarray|None,
        frame_wall: float, ship_perf: float)
  out: VisionResult(timestamp, target: TargetEstimate, flow: FlowResult, ...)
frame_wall is the parent's wall clock at frame arrival (echoed back untouched);
ship_perf is the parent's perf_counter just before put() (used only to measure
the inbound IPC leg -- see VisionResult below).
frame is a small BGR uint8 array (~28 KB at 120x80) so per-frame IPC cost is
negligible at 30 Hz. TargetEstimate and FlowResult must stay picklable (plain
dataclasses / namedtuples of scalars + small arrays -- no open cv2 handles,
file objects, or locks). If either ever gains a non-picklable field, that is
the first thing to check when results stop arriving.
"""

import signal
import sys
import time
from collections import namedtuple

from .optical_flow import OpticalFlowEstimator
from .target_acquisition import TargetAcquisition


# What the worker hands back for each processed frame. timestamp echoes the
# input frame's vision timestamp so bee_node can correlate; target/flow are the
# unchanged outputs of the two algorithm calls. The two *_ms fields are the
# worker's own measurement of each stage's cost -- they replace the inline
# stage timings on_camera used to record, so bee_node's diagnostics keep the
# same target_acquisition/optical_flow timing columns (see bee_node's
# _drain_vision_results). Defined at module top level so it stays picklable.
#
# frame_wall is bee_node's PARENT-process wall clock at the moment the frame
# entered on_camera, carried through this worker untouched and echoed back so
# bee_node can measure the true frame->available round trip (send + queue wait
# + compute + return) entirely in its OWN clock -- the worker never interprets
# it, so no cross-process clock comparison is involved. target_acquisition_ms /
# optical_flow_ms are the worker's own perf_counter measurement of each call
# (durations, so comparable across processes); they are the HONEST replacement
# for the on_camera stage timers, which no longer run these two stages.
# ipc_in_ms / done_perf added in the A-instrumentation pass to split the
# frame->available round trip into its real legs instead of guessing where the
# ~9-18ms beyond compute goes:
#   ipc_in_ms : send + inbound-queue-wait + unpickle, measured as
#               (worker perf_counter right after in_q.get()) - ship_perf, where
#               ship_perf is the parent's perf_counter taken just before put().
#   done_perf : the worker's perf_counter the instant compute finished, so the
#               parent's drain thread can measure the OUTBOUND leg (out_q
#               transit + wait-until-drained + unpickle) as drain_perf - done_perf.
# Both rely on perf_counter being the same monotonic clock across processes,
# which holds on Linux (CLOCK_MONOTONIC is system-wide); the two legs are
# durations either way, so no absolute-epoch comparison is assumed beyond that.
VisionResult = namedtuple(
	"VisionResult",
	[
		"timestamp",
		"target",
		"flow",
		"target_acquisition_ms",
		"optical_flow_ms",
		"frame_wall",
		"ipc_in_ms",
		"done_perf",
	],
)

# Sentinel pushed onto the input queue by bee_node.shutdown_vision_worker() to
# ask the loop to exit. A real payload is always a 4-tuple, never None.
STOP = None


def run_vision_worker(in_q, out_q):
	"""Long-lived vision loop. Entry point for the multiprocessing.Process.

	Owns one TargetAcquisition and one OpticalFlowEstimator -- constructed
	EXACTLY as bee_node used to construct them (default TargetAcquisition;
	OpticalFlowEstimator with derotator=None, matching the currently-disabled
	de-rotation path). They are long-lived so OpticalFlowEstimator keeps its
	previous-frame state across calls, just as it did living on the node.

	NOTE on re-enabling de-rotation: today bee_node computes body_rates and
	ships them, but the flow estimator is built with derotator=None so they are
	unused. When the downsampled Farneback + de-rotation path is re-validated,
	build the Derotator HERE (from the same CameraGeometry) and pass it in --
	the body_rates already arrive on the queue, so nothing on bee_node's side
	needs to change.
	"""
	# Ctrl+C is delivered to this child too (same process group). We do NOT want
	# it to raise KeyboardInterrupt out of the blocking in_q.get() and dump a
	# traceback on an otherwise clean shutdown -- the parent already stops us
	# cleanly, either by putting the STOP sentinel on the queue
	# (bee_node.shutdown_vision_worker) or by terminate(). So ignore SIGINT here
	# and let the parent drive shutdown. SIGTERM (what terminate() sends) is left
	# alone and still stops the process.
	signal.signal(signal.SIGINT, signal.SIG_IGN)

	target_acquisition = TargetAcquisition()
	optical_flow = OpticalFlowEstimator(derotator=None)

	while True:
		item = in_q.get()  # blocks; costs nothing while idle
		# perf_counter the instant get() returns -- captures inbound-queue-wait
		# AND unpickle (unpickle happens inside get()), before any other work.
		recv_perf = time.perf_counter()
		if item is STOP:
			break

		try:
			frame, timestamp, body_rates, frame_wall, ship_perf = item
			ipc_in_ms = 1000.0 * (recv_perf - ship_perf)

			t0 = time.perf_counter()
			target = target_acquisition.update(frame, timestamp=timestamp)
			t1 = time.perf_counter()
			flow = optical_flow.update(
				frame, timestamp, target=target, body_rates=body_rates
			)
			t2 = time.perf_counter()

			out_q.put(VisionResult(
				timestamp=timestamp,
				target=target,
				flow=flow,
				target_acquisition_ms=1000.0 * (t1 - t0),
				optical_flow_ms=1000.0 * (t2 - t1),
				frame_wall=frame_wall,
				ipc_in_ms=ipc_in_ms,
				done_perf=t2,
			))
		except Exception as exc:
			# A single malformed frame must not kill the pipeline. Skip it and
			# keep going; if results stop ENTIRELY, bee_node notices the dead
			# process (is_alive) and its LOST_TARGET_TIMEOUT still protects the
			# flight by holding a neutral visual hover. No ROS logger here (by
			# design -- see module docstring), so report on stderr.
			sys.stderr.write(f"[vision_worker] dropped a frame: {exc!r}\n")
			sys.stderr.flush()