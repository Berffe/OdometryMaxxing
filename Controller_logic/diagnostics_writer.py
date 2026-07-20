"""Controller-event and Gazebo-truth CSV logging.

The controller CSV records what the visual controller knew and commanded.
The truth CSV records every atomic Gazebo truth packet without reconstruction.
The two files share a run id and are merged later by ``analyse_log.py`` on
Gazebo SIM time.
"""
from __future__ import annotations

import csv
import queue
import threading
import time
from pathlib import Path
from typing import Mapping, Optional

from .truth_layout import TRUTH_FIELDS


class _AsyncCsvSink:
	"""Small non-blocking CSV sink used for the dense truth stream."""

	def __init__(self, path: Path, fieldnames, *, queue_size: int = 2048,
				flush_every_rows: int = 100):
		self.path = str(path)
		self._fieldnames = list(fieldnames)
		self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
		self._flush_every_rows = max(1, int(flush_every_rows))
		self._stop = object()
		self.dropped_rows = 0
		self._thread = threading.Thread(
			target=self._run, name=f"csv:{path.name}", daemon=True)
		self._thread.start()

	def submit(self, row: Mapping):
		try:
			self._queue.put_nowait(dict(row))
		except queue.Full:
			# Truth is dense. Dropping a row under pathological disk pressure is
			# safer than blocking the ROS executor / controller.
			self.dropped_rows += 1

	def close(self):
		try:
			self._queue.put(self._stop, timeout=1.0)
		except queue.Full:
			pass
		self._thread.join(timeout=3.0)

	def _run(self):
		Path(self.path).parent.mkdir(parents=True, exist_ok=True)
		with open(self.path, "w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
			writer.writeheader()
			count = 0
			while True:
				item = self._queue.get()
				if item is self._stop:
					break
				writer.writerow({k: item.get(k, "") for k in self._fieldnames})
				count += 1
				if count % self._flush_every_rows == 0:
					handle.flush()
			handle.flush()


class DiagnosticsWriter:
	CONTROLLER_SCHEMA_VERSION = "4.6-controller"
	TRUTH_LOG_SCHEMA_VERSION = "1.0-truth-log"

	def __init__(self, output_dir="logs", filename=None, *,
				controller_flush_every_rows: int = 25,
				truth_queue_size: int = 2048):
		root = Path(output_dir)
		root.mkdir(parents=True, exist_ok=True)
		run_id = time.strftime("%Y%m%d_%H%M%S")
		filename = filename or f"bee_controller_{run_id}.csv"
		controller_path = root / filename
		if filename.startswith("bee_controller_"):
			truth_name = "bee_truth_" + filename[len("bee_controller_"):]
		else:
			truth_name = f"bee_truth_{run_id}.csv"
		truth_path = root / truth_name

		self.filepath = str(controller_path)
		self.truth_filepath = str(truth_path)
		self._start_wall = time.time()
		self._start_mono = time.monotonic()
		self._controller_flush_every_rows = max(1, int(controller_flush_every_rows))
		self._controller_row_count = 0
		self._controller_file = open(
			self.filepath, "w", newline="", encoding="utf-8")
		self._controller_writer = csv.DictWriter(
			self._controller_file, fieldnames=self._fieldnames())
		self._controller_writer.writeheader()
		self._truth_sink = _AsyncCsvSink(
			truth_path,
			[
				"truth_log_schema_version",
				"truth_receipt_wall_timestamp_sec",
				"truth_receipt_monotonic_timestamp_sec",
			] + list(TRUTH_FIELDS),
			queue_size=truth_queue_size,
			flush_every_rows=100,
		)

	@property
	def truth_dropped_rows(self) -> int:
		return int(self._truth_sink.dropped_rows)

	def write_truth(self, truth: Mapping, *, receipt_wall_sec: float,
					receipt_monotonic_sec: float):
		row = {
			"truth_log_schema_version": self.TRUTH_LOG_SCHEMA_VERSION,
			"truth_receipt_wall_timestamp_sec": float(receipt_wall_sec),
			"truth_receipt_monotonic_timestamp_sec": float(receipt_monotonic_sec),
		}
		row.update(truth)
		self._truth_sink.submit(row)

	def write(self, *, target=None, flow=None, setpoint=None, mission=None,
			timing=None, contact=None, publish=None, event="", event_detail="",
			divergence_integral=None, vision_sequence=None,
			controller_phase="", px4_status: Optional[Mapping] = None):
		wall, mono = time.time(), time.monotonic()
		row = {k: "" for k in self._fieldnames()}
		row.update({
			"diagnostics_schema_version": self.CONTROLLER_SCHEMA_VERSION,
			"log_wall_timestamp_sec": wall,
			"log_monotonic_timestamp_sec": mono,
			"log_elapsed_wall_sec": wall - self._start_wall,
			"log_elapsed_monotonic_sec": mono - self._start_mono,
			"event": event,
			"event_detail": event_detail,
			"controller_phase": controller_phase,
			"vision_sequence": self._value(vision_sequence),
		})
		if px4_status:
			row.update({
				"px4_nav_state": self._value(px4_status.get("nav_state")),
				"px4_arming_state": self._value(px4_status.get("arming_state")),
				"px4_failsafe": self._bool_value(px4_status.get("failsafe")),
				"px4_offboard_confirmed": self._bool_value(
					px4_status.get("offboard_confirmed")),
			})
		if target is not None:
			row.update({
				"vision_sim_timestamp_sec": self._value(target.timestamp),
				"target_found": int(bool(target.found)),
				"target_offset_x": target.offset_x,
				"target_offset_y": target.offset_y,
				"target_detection_width_px": target.detection_width,
				"target_detection_height_px": target.detection_height,
				"target_confidence": target.confidence,
				"target_area_fraction": target.area_fraction,
				"target_fov_saturated": int(bool(target.fov_saturated)),
			})
		if flow is not None:
			row.update({
				"flow_sim_timestamp_sec": self._value(flow.timestamp),
				"flow_valid": int(bool(flow.valid)),
				"flow_mean_x_norm_s": flow.mean_flow_x_norm,
				"flow_mean_y_norm_s": flow.mean_flow_y_norm,
				"flow_mean_x_px_s": flow.mean_flow_x,
				"flow_mean_y_px_s": flow.mean_flow_y,
				"flow_divergence_1_s": flow.divergence,
				"flow_raw_divergence_1_s": flow.raw_divergence,
				"flow_fit_quality": flow.fit_quality,
				"flow_derotated": int(bool(flow.derotated)),
				"flow_mean_x_raw_px_s": flow.mean_flow_x_raw,
				"flow_mean_y_raw_px_s": flow.mean_flow_y_raw,
				"flow_divergence_prederotation_1_s": flow.divergence_prederotation,
				"flow_roi_x0": flow.roi_x0, "flow_roi_y0": flow.roi_y0,
				"flow_roi_x1": flow.roi_x1, "flow_roi_y1": flow.roi_y1,
			})
		if setpoint is not None:
			row.update({
				"command_source_sim_timestamp_sec": self._value(setpoint.timestamp),
				"command_roll_rad": setpoint.roll,
				"command_pitch_rad": setpoint.pitch,
				"command_yaw_rad": setpoint.yaw,
				"command_thrust": setpoint.thrust,
				"command_thrust_integral": self._value(divergence_integral),
			})
		if mission:
			for key, value in mission.items():
				col = f"mission_{key}"
				if col in row:
					row[col] = self._value(value)
		if timing:
			for key, value in timing.items():
				col = f"timing_{key}"
				if col in row:
					row[col] = self._value(value)
		if contact is not None:
			row.update({
				"contact_valid": int(bool(contact.valid)),
				"contact_truth_sequence": contact.sequence,
				"contact_truth_sim_timestamp_sec": self._value(contact.sim_timestamp),
				"contact_left": int(bool(contact.left_contact)),
				"contact_right": int(bool(contact.right_contact)),
				"contact_any": int(bool(contact.any_contact)),
				"contact_confirmed": int(bool(contact.confirmed)),
			})
		if publish is not None:
			row.update({
				"px4_publish_sequence": publish.sequence,
				"px4_publish_wall_timestamp_sec": publish.wall_timestamp_sec,
				"px4_publish_monotonic_timestamp_sec": publish.monotonic_timestamp_sec,
			})
		self._controller_writer.writerow(row)
		self._controller_row_count += 1
		if self._controller_row_count % self._controller_flush_every_rows == 0:
			self._controller_file.flush()

	def close(self):
		self._truth_sink.close()
		if not self._controller_file.closed:
			self._controller_file.flush()
			self._controller_file.close()

	@staticmethod
	def _value(value):
		return "" if value is None else value

	@staticmethod
	def _bool_value(value):
		return "" if value is None else int(bool(value))

	@classmethod
	def _fieldnames(cls):
		base = [
			"diagnostics_schema_version", "log_wall_timestamp_sec",
			"log_monotonic_timestamp_sec", "log_elapsed_wall_sec",
			"log_elapsed_monotonic_sec", "event", "event_detail",
			"controller_phase", "px4_nav_state", "px4_arming_state",
			"px4_failsafe", "px4_offboard_confirmed", "vision_sequence",
			"vision_sim_timestamp_sec", "target_found", "target_offset_x",
			"target_offset_y", "target_detection_width_px", "target_detection_height_px",
			"target_confidence", "target_area_fraction", "target_fov_saturated",
			"flow_sim_timestamp_sec", "flow_valid", "flow_mean_x_norm_s",
			"flow_mean_y_norm_s", "flow_mean_x_px_s", "flow_mean_y_px_s",
			"flow_divergence_1_s", "flow_raw_divergence_1_s", "flow_fit_quality",
			"flow_derotated", "flow_mean_x_raw_px_s", "flow_mean_y_raw_px_s",
			"flow_divergence_prederotation_1_s", "flow_roi_x0", "flow_roi_y0",
			"flow_roi_x1", "flow_roi_y1", "command_source_sim_timestamp_sec",
			"command_roll_rad", "command_pitch_rad", "command_yaw_rad", "command_thrust",
			"command_thrust_integral", "contact_valid", "contact_truth_sequence",
			"contact_truth_sim_timestamp_sec", "contact_left", "contact_right",
			"contact_any", "contact_confirmed", "px4_publish_sequence",
			"px4_publish_wall_timestamp_sec", "px4_publish_monotonic_timestamp_sec",
		]
		mission = [
			"substate", "divergence_setpoint_1_s", "thrust_gain_k",
			"lateral_p_scale", "lateral_d_scale", "enable_integral", "peak_accel_m_s2",
			"k_min", "k_explore", "k_probe", "k_floor", "k_ceiling_leg",
			"h_crit_m", "h_pred_m", "feasible", "probe_phase",
			"probe_accel_m_s2", "probe_mean_accel_m_s2",
			"probe_residual_accel_m_s2", "probe_percentile_accel_m_s2",
		]
		timing = [
			"camera_receipt_wall_timestamp_sec",
			"camera_receipt_monotonic_timestamp_sec",
			"camera_callback_ms", "camera_prequeue_ms",
			"vision_worker_target_acquisition_ms",
			"vision_worker_optical_flow_ms",
			"vision_ipc_in_ms", "vision_ipc_out_ms",
			"vision_transport_total_ms",
			"frame_to_result_ms", "frame_to_command_ms",
			"control_compute_ms", "control_dt_sim_sec",
			"vision_dropped_frames",

			# OpticalFlowEstimator.update() internal stage timings.
			"optical_flow_total_wall_ms", "optical_flow_total_cpu_ms",
			"optical_flow_grayscale_ms", "optical_flow_roi_setup_ms",
			"optical_flow_downsample_resize_ms", "optical_flow_farneback_ms",
			"optical_flow_flow_scaling_upsample_ms",
			"optical_flow_derotation_ms", "optical_flow_mean_flow_ms",
			"optical_flow_gradient_ms",
			"optical_flow_divergence_field_debug_ms",
			"optical_flow_affine_fit_ms",
			"optical_flow_affine_setup_ms",
			"optical_flow_affine_initial_solve_ms",
			"optical_flow_affine_residual_quantile_ms",
			"optical_flow_affine_refit_ms",
			"optical_flow_prederotation_fit_ms",
			"optical_flow_divergence_filter_ms",
			"optical_flow_result_and_state_ms",

			# Per-frame operating point and algorithm configuration.
			"optical_flow_valid", "optical_flow_dt_sec",
			"optical_flow_image_width_px", "optical_flow_image_height_px",
			"optical_flow_roi_width_px", "optical_flow_roi_height_px",
			"optical_flow_working_width_px",
			"optical_flow_working_height_px",
			"optical_flow_working_flow_vectors",
			"optical_flow_downsample_scale",
			"optical_flow_fit_pixel_scale",
			"optical_flow_derotation_active",
			"optical_flow_affine_input_points",
			"optical_flow_affine_sampled_points",
			"optical_flow_affine_fit_stride",
			"optical_flow_affine_finite_points",
			"optical_flow_affine_used_points",
			"optical_flow_affine_points_used",
			"optical_flow_affine_fit_quality",
			"optical_flow_farneback_pyr_scale",
			"optical_flow_farneback_levels",
			"optical_flow_farneback_winsize",
			"optical_flow_farneback_iterations",
			"optical_flow_farneback_poly_n",
			"optical_flow_farneback_poly_sigma",
			"optical_flow_divergence_smoothing_alpha",

			"motor_stop_request_to_pickup_ms", "motor_stop_total_ms",
		]
		return base + [f"mission_{x}" for x in mission] + [f"timing_{x}" for x in timing]
