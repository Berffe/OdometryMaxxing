"""Small event-oriented controller logger.

The controller log contains only what the controller knew and did. Gazebo truth
is recorded separately and merged by analyse_log using SIM time.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path


class DiagnosticsWriter:
	SCHEMA_VERSION = "4.0-controller"

	def __init__(self, output_dir="logs", filename=None, flush_every_row=True):
		root = Path(output_dir)
		root.mkdir(parents=True, exist_ok=True)
		filename = filename or time.strftime("bee_controller_%Y%m%d_%H%M%S.csv")
		self.filepath = str(root / filename)
		self._start_wall = time.time()
		self._start_mono = time.monotonic()
		self._flush = bool(flush_every_row)
		self._file = open(self.filepath, "w", newline="", encoding="utf-8")
		self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
		self._writer.writeheader()

	def write(self, *, target=None, flow=None, setpoint=None, mission=None,
			timing=None, contact=None, publish=None, event="", event_detail="",
			divergence_integral=None, vision_sequence=None):
		wall, mono = time.time(), time.monotonic()
		row = {k: "" for k in self._fieldnames()}
		row.update({
			"diagnostics_schema_version": self.SCHEMA_VERSION,
			"log_wall_timestamp_sec": wall,
			"log_monotonic_timestamp_sec": mono,
			"log_elapsed_wall_sec": wall - self._start_wall,
			"log_elapsed_monotonic_sec": mono - self._start_mono,
			"event": event,
			"event_detail": event_detail,
			"vision_sequence": self._value(vision_sequence),
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
		self._writer.writerow(row)
		if self._flush:
			self._file.flush()

	def close(self):
		if not self._file.closed:
			self._file.flush()
			self._file.close()

	@staticmethod
	def _value(value):
		return "" if value is None else value

	@classmethod
	def _fieldnames(cls):
		base = [
			"diagnostics_schema_version", "log_wall_timestamp_sec",
			"log_monotonic_timestamp_sec", "log_elapsed_wall_sec",
			"log_elapsed_monotonic_sec", "event", "event_detail", "vision_sequence",
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
			"camera_receipt_wall_timestamp_sec", "camera_receipt_monotonic_timestamp_sec",
			"camera_callback_ms", "vision_worker_target_acquisition_ms",
			"vision_worker_optical_flow_ms", "frame_to_result_ms", "frame_to_command_ms",
			"control_compute_ms", "control_dt_sim_sec", "vision_dropped_frames",
			"motor_stop_request_to_pickup_ms", "motor_stop_total_ms",
		]
		return base + [f"mission_{x}" for x in mission] + [f"timing_{x}" for x in timing]
