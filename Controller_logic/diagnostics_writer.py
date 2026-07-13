"""CSV diagnostics writer for BEE_LAND runs.

Important timestamp convention
------------------------------
``wall_timestamp`` is the only wall-clock reference used to build the relative
``t_sec`` column. All other ``*_timestamp_sec`` fields are written as raw values
from their own sources:

- target / flow / command timestamps: vision timestamp used by bee_node.py,
  normally Gazebo image simulation time.
- vehicle_timestamp_sec / vehicle_px4_timestamp_sec: whatever VehicleState
  carries, normally wall receipt time plus PX4's own timestamp depending on the
  active state.py convention.
- platform timestamp is not written because /platform/pose is an unstamped Pose.

The writer deliberately does not subtract wall-clock origin from target/flow/PX4
fields. Mixing those epochs caused the huge negative timestamp columns seen in
recent logs.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Optional

try:
	from .state import AttitudeSetpoint, FlowResult, PlatformState, TargetEstimate, VehicleState
except ImportError:  # Allows standalone import/tests from the controller folder.
	from state import AttitudeSetpoint, FlowResult, PlatformState, TargetEstimate, VehicleState


class DiagnosticsWriter:
	def __init__(
		self,
		output_dir: str = "logs",
		filename: Optional[str] = None,
		flush_every_row: bool = False,
	):
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)

		if filename is None:
			filename = time.strftime("bee_diagnostics_%Y%m%d_%H%M%S.csv")

		self.filepath = str(self.output_dir / filename)
		self._flush_every_row = bool(flush_every_row)
		self._start_wall_timestamp: Optional[float] = None

		self._file = open(self.filepath, "w", newline="", encoding="utf-8")
		self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
		self._writer.writeheader()
		if self._flush_every_row:
			self._file.flush()

	def write(
		self,
		wall_timestamp: float,
		target: Optional[TargetEstimate],
		flow: Optional[FlowResult],
		setpoint: Optional[AttitudeSetpoint],
		vehicle_state: Optional[VehicleState],
		platform_state: Optional[PlatformState] = None,
		calibration_axis: str = "",
		px4_wall_offset_sec: Optional[float] = None,
		sim_wall_offset_sec: Optional[float] = None,
		mission: Optional[dict] = None,
		px4_nav_state: Optional[int] = None,
		px4_arming_state: Optional[int] = None,
		px4_failsafe: Optional[bool] = None,
		divergence_integral: Optional[float] = None,
		timing: Optional[dict] = None,
		**_: Any,
	):
		wall_timestamp = float(wall_timestamp)
		if self._start_wall_timestamp is None:
			self._start_wall_timestamp = wall_timestamp

		row = {name: "" for name in self._fieldnames()}
		row["t_sec"] = wall_timestamp - self._start_wall_timestamp
		row["wall_timestamp"] = wall_timestamp

		if target is not None:
			row.update({
				"target_timestamp_sec": self._raw_ts(getattr(target, "timestamp", 0.0)),
				"target_found": self._bool_int(getattr(target, "found", False)),
				"target_confidence": self._num(getattr(target, "confidence", 0.0)),
				"target_offset_x": self._num(getattr(target, "offset_x", 0.0)),
				"target_offset_y": self._num(getattr(target, "offset_y", 0.0)),
				"target_detection_width_px": self._num(getattr(target, "detection_width", 0.0)),
				"target_detection_height_px": self._num(getattr(target, "detection_height", 0.0)),
				"target_area_fraction": self._num(getattr(target, "area_fraction", 0.0)),
				"target_fov_saturated": self._bool_int(getattr(target, "fov_saturated", False)),
			})

		if flow is not None:
			row.update({
				"flow_timestamp_sec": self._raw_ts(getattr(flow, "timestamp", 0.0)),
				"flow_valid": self._bool_int(getattr(flow, "valid", False)),
				"flow_mean_x_px_s": self._num(getattr(flow, "mean_flow_x", 0.0)),
				"flow_mean_y_px_s": self._num(getattr(flow, "mean_flow_y", 0.0)),
				"flow_mean_x_norm_s": self._num(getattr(flow, "mean_flow_x_norm", 0.0)),
				"flow_mean_y_norm_s": self._num(getattr(flow, "mean_flow_y_norm", 0.0)),
				"flow_divergence_1_s": self._num(getattr(flow, "divergence", 0.0)),
				"flow_raw_divergence_1_s": self._num(getattr(flow, "raw_divergence", 0.0)),
				"flow_fit_quality": self._num(getattr(flow, "fit_quality", 0.0)),
				"flow_derotated": self._bool_int(getattr(flow, "derotated", False)),
				"flow_mean_x_raw_px_s": self._num(getattr(flow, "mean_flow_x_raw", 0.0)),
				"flow_mean_y_raw_px_s": self._num(getattr(flow, "mean_flow_y_raw", 0.0)),
				"flow_divergence_prederotation_1_s": self._num(getattr(flow, "divergence_prederotation", 0.0)),
				"flow_roi_x0": self._int_or_blank(getattr(flow, "roi_x0", -1)),
				"flow_roi_y0": self._int_or_blank(getattr(flow, "roi_y0", -1)),
				"flow_roi_x1": self._int_or_blank(getattr(flow, "roi_x1", -1)),
				"flow_roi_y1": self._int_or_blank(getattr(flow, "roi_y1", -1)),
			})

		if setpoint is not None:
			row.update({
				"command_timestamp_sec": self._raw_ts(getattr(setpoint, "timestamp", 0.0)),
				"command_roll_rad": self._num(getattr(setpoint, "roll", 0.0)),
				"command_pitch_rad": self._num(getattr(setpoint, "pitch", 0.0)),
				"command_yaw_rad": self._num(getattr(setpoint, "yaw", 0.0)),
				"command_thrust": self._num(getattr(setpoint, "thrust", 0.0)),
			})

		# ControlLaw's internal divergence integral (thrust_integral_gain_const *
		# this = the integral contribution to thrust). Logged to directly confirm
		# or refute windup/saturation during transients (e.g. a D* step) instead
		# of inferring it from the thrust trace alone.
		if divergence_integral is not None:
			row["command_thrust_integral"] = self._num(divergence_integral)

		if vehicle_state is not None:
			row.update({
				"vehicle_timestamp_sec": self._raw_ts(getattr(vehicle_state, "timestamp", 0.0)),
				"vehicle_px4_timestamp_sec": self._raw_ts(getattr(vehicle_state, "px4_timestamp_sec", 0.0)),
				"vehicle_x_m": self._num(getattr(vehicle_state, "x", 0.0)),
				"vehicle_y_m": self._num(getattr(vehicle_state, "y", 0.0)),
				"vehicle_z_m": self._num(getattr(vehicle_state, "z", 0.0)),
				"vehicle_vx_m_s": self._num(getattr(vehicle_state, "vx", 0.0)),
				"vehicle_vy_m_s": self._num(getattr(vehicle_state, "vy", 0.0)),
				"vehicle_vz_m_s": self._num(getattr(vehicle_state, "vz", 0.0)),
				"vehicle_yaw_rad": self._num(getattr(vehicle_state, "yaw", 0.0)),
				"vehicle_attitude_timestamp_sec": self._raw_ts(getattr(vehicle_state, "attitude_timestamp", 0.0)),
				"vehicle_roll_rad": self._num(getattr(vehicle_state, "roll", 0.0)),
				"vehicle_pitch_rad": self._num(getattr(vehicle_state, "pitch", 0.0)),
				"vehicle_attitude_yaw_rad": self._num(getattr(vehicle_state, "attitude_yaw", 0.0)),
				"vehicle_attitude_source": getattr(vehicle_state, "attitude_source", "") or "",
			})

		if platform_state is not None:
			row.update({
				"platform_x_m": self._num(getattr(platform_state, "x", 0.0)),
				"platform_y_m": self._num(getattr(platform_state, "y", 0.0)),
				"platform_z_m": self._num(getattr(platform_state, "z", 0.0)),
				"platform_vx_m_s": self._num(getattr(platform_state, "vx", 0.0)),
				"platform_vy_m_s": self._num(getattr(platform_state, "vy", 0.0)),
				"platform_vz_m_s": self._num(getattr(platform_state, "vz", 0.0)),
			})

			if vehicle_state is not None:
				# VehicleState is PX4 local NED: z grows negative upward. PlatformState is
				# Gazebo world ENU. Therefore relative height in the existing logs is
				# vehicle_z_NED + platform_z_ENU, and similarly for vz.
				row.update({
					"relative_x_m": self._num(getattr(vehicle_state, "x", 0.0) - getattr(platform_state, "x", 0.0)),
					"relative_y_m": self._num(getattr(vehicle_state, "y", 0.0) - getattr(platform_state, "y", 0.0)),
					"relative_z_m": self._num(getattr(vehicle_state, "z", 0.0) + getattr(platform_state, "z", 0.0)),
					"relative_vx_m_s": self._num(getattr(vehicle_state, "vx", 0.0) - getattr(platform_state, "vx", 0.0)),
					"relative_vy_m_s": self._num(getattr(vehicle_state, "vy", 0.0) - getattr(platform_state, "vy", 0.0)),
					"relative_vz_m_s": self._num(getattr(vehicle_state, "vz", 0.0) + getattr(platform_state, "vz", 0.0)),
				})

		row["calibration_axis"] = calibration_axis or ""

		# Mission routine telemetry (probe -> gate -> scheduled-gain descent).
		# All blank on pre-closed-loop rows where mission is None. These are the
		# fields needed to tune/diagnose the Herisse/de Croon bounds mechanism:
		# the scheduled thrust gain k(t) and lateral scale, the probe-derived
		# bounds (k_min/k_explore/h_crit), the feasibility verdict, the probe's
		# measured peak platform acceleration, and the OPEN-LOOP predicted height
		# h_pred (compare against relative_z to see prediction drift).
		if mission is not None:
			row.update({
				"mission_substate": mission.get("substate", "") or "",
				"mission_divergence_setpoint_1_s": self._num(mission.get("divergence_setpoint")),
				"mission_thrust_gain_k": self._num(mission.get("thrust_gain_k")),
				"mission_lateral_p_scale": self._num(mission.get("lateral_p_scale")),
				"mission_lateral_d_scale": self._num(mission.get("lateral_d_scale")),
				"mission_k_min": self._num(mission.get("k_min")),
				"mission_k_explore": self._num(mission.get("k_explore")),
				"mission_h_crit_m": self._num(mission.get("h_crit")),
				"mission_h_pred_m": self._num(mission.get("h_pred")),
				"mission_peak_accel_m_s2": self._num(mission.get("peak_accel")),
				"mission_feasible": self._bool_int(mission.get("feasible", False)),
			})

			# The gain WINDOW the descent rides. k(t) now decays toward k_floor =
			# max(k_min, ceiling_margin * k_ceiling_leg) -- the de Croon ceiling at
			# leg height -- NOT toward k_min. k_over_ceiling_leg is the headline
			# number for that change: it used to sit near 0.065 for the whole
			# descent and should now approach ceiling_margin.
			row.update({
				"mission_k_ceiling_leg": self._num(mission.get("k_ceiling_leg")),
				"mission_k_target": self._num(mission.get("k_target")),
				"mission_k_floor": self._num(mission.get("k_floor")),
				# k_probe: the gain held flat through FINAL_PROBE and the value the
				# descent starts from. k_explore is FAR-FIELD only -- above the de
				# Croon ceiling in the near field, where probing at it would feed
				# self-induced oscillation into peak_accel.
				"mission_k_probe": self._num(mission.get("k_probe")),
				"mission_k_descend_start": self._num(mission.get("k_descend_start")),
				"mission_near_field_height_m": self._num(mission.get("near_field_height_m")),
				"mission_ceiling_margin": self._num(mission.get("ceiling_margin")),
				"mission_k_over_ceiling_leg": self._num(mission.get("k_over_ceiling_leg")),
			})

			# PLATFORM PROBE, per step. mission_peak_accel_m_s2 (above) is the
			# leaky-max envelope the feasibility gate consumes (k_min = peak/D*);
			# the columns below are the quantities it is BUILT FROM, logged every
			# step so the envelope can be judged rather than trusted:
			#   *_accel_m_s2          : commanded vertical accel from the thrust cmd
			#   *_mean_accel_m_s2     : the EMA bias removed (hover trim + the slow
			#                           contribution of the D*>0 approach descent)
			#   *_residual_accel_m_s2 : |accel - mean|, the de-biased signal
			#   *_percentile_accel_m_s2: the rolling-window percentile the peak chases
			# The envelope itself keeps its established name, mission_peak_accel_m_s2
			# (unchanged, so existing analyses keep working).
			# probe_phase is far/near (which time constants are live); the retune at
			# the far->near handoff is where the estimate is expected to be revised,
			# and mission_probe_peak_accel_at_handoff_m_s2 freezes its pre-handoff
			# value so the revision is visible in one subtraction.
			row.update({
				"mission_probe_active": self._bool_int(mission.get("probe_active", False)),
				"mission_probe_phase": mission.get("probe_phase", "") or "",
				"mission_probe_accel_m_s2": self._num(mission.get("probe_accel")),
				"mission_probe_mean_accel_m_s2": self._num(mission.get("probe_mean_accel")),
				"mission_probe_residual_accel_m_s2": self._num(mission.get("probe_residual_accel")),
				"mission_probe_percentile_accel_m_s2": self._num(mission.get("probe_percentile_accel")),
				"mission_probe_peak_accel_at_handoff_m_s2": self._num(mission.get("probe_peak_accel_at_handoff")),
				"mission_probe_peak_decay_tau_sec": self._num(mission.get("probe_peak_decay_tau_sec")),
				"mission_probe_elapsed_sec": self._num(mission.get("probe_elapsed_sec")),
				"mission_probe_total_elapsed_sec": self._num(mission.get("probe_total_elapsed_sec")),
			})

		# Timing / latency telemetry. These columns separate the three relevant
		# questions: (1) how old was the vision sample when the command was
		# computed, (2) how regular is the deliberately fixed PX4 publication
		# cadence, and (3) what real-time factor did the simulator/PX4 clock run at.
		if timing is not None:
			row.update({
				"timing_camera_cb_start_wall_sec": self._num(timing.get("camera_cb_start_wall_sec")),
				"timing_camera_cb_end_wall_sec": self._num(timing.get("camera_cb_end_wall_sec")),
				"timing_camera_cb_duration_ms": self._num(timing.get("camera_cb_duration_ms")),
				"timing_stage_bridge_ms": self._num(timing.get("stage_bridge_ms")),
				"timing_stage_rotate_ms": self._num(timing.get("stage_rotate_ms")),
				"timing_stage_show_camera_ms": self._num(timing.get("stage_show_camera_ms")),
				"timing_stage_body_rate_ms": self._num(timing.get("stage_body_rate_ms")),
				"timing_stage_target_acquisition_ms": self._num(timing.get("stage_target_acquisition_ms")),
				"timing_stage_optical_flow_ms": self._num(timing.get("stage_optical_flow_ms")),
				# v2.0 out-of-process vision. The two stage_* columns above are
				# blank now (those calls left on_camera); worker_* below are the
				# worker's own perf_counter cost of the same two calls, and the
				# frame_to_* / result_period / dropped columns are the real
				# cross-process latencies (see bee_node._drain_vision_results and
				# on_control_timer). frame_to_command is the number to compare
				# against v1.0's blocking on_camera duration (~11-18ms).
				"timing_worker_target_acquisition_ms": self._num(timing.get("worker_target_acquisition_ms")),
				"timing_worker_optical_flow_ms": self._num(timing.get("worker_optical_flow_ms")),
				"timing_ipc_in_ms": self._num(timing.get("ipc_in_ms")),
				"timing_ipc_out_ms": self._num(timing.get("ipc_out_ms")),
				"timing_frame_to_available_wall_ms": self._num(timing.get("frame_to_available_wall_ms")),
				"timing_frame_to_command_wall_ms": self._num(timing.get("frame_to_command_wall_ms")),
				"timing_vision_result_period_wall_ms": self._num(timing.get("vision_result_period_wall_ms")),
				"timing_vision_dropped_frames": self._int_or_blank(timing.get("vision_dropped_frames", -1)),
				"timing_control_compute_start_wall_sec": self._num(timing.get("control_compute_start_wall_sec")),
				"timing_control_compute_end_wall_sec": self._num(timing.get("control_compute_end_wall_sec")),
				"timing_control_compute_duration_ms": self._num(timing.get("control_compute_duration_ms")),
				"timing_control_period_wall_sec": self._num(timing.get("control_period_wall_sec")),
				"timing_control_dt_vision_sec": self._num(timing.get("control_dt_vision_sec")),
				"timing_flow_age_at_control_wall_ms": self._num(timing.get("flow_age_at_control_wall_ms")),
				"timing_px4_publish_wall_sec": self._num(timing.get("px4_publish_wall_sec")),
				"timing_px4_publish_period_wall_sec": self._num(timing.get("px4_publish_period_wall_sec")),
				"timing_command_age_at_px4_publish_ms": self._num(timing.get("command_age_at_px4_publish_ms")),
				"timing_flow_age_at_px4_publish_wall_ms": self._num(timing.get("flow_age_at_px4_publish_wall_ms")),
				"timing_px4_publish_count": self._int_or_blank(timing.get("px4_publish_count", -1)),
				"timing_sim_rtf_estimate": self._num(timing.get("sim_rtf_estimate")),
				"timing_px4_rtf_estimate": self._num(timing.get("px4_rtf_estimate")),
			})

		# PX4 mode/arming: the ground truth for whether offboard is actually
		# active. If commands look ignored, check px4_nav_state == 14 (OFFBOARD).
		row["px4_nav_state"] = px4_nav_state if px4_nav_state is not None else ""
		row["px4_arming_state"] = px4_arming_state if px4_arming_state is not None else ""
		row["px4_failsafe"] = self._bool_int(px4_failsafe) if px4_failsafe is not None else ""

		# Clock-family offsets (WALL - PX4) and (WALL - SIM), in seconds. These
		# are diagnostics-only desync monitors: in healthy SITL they sit near a
		# constant; visible drift means the uXRCE-DDS timesync or the sim clock
		# is wandering relative to the wall clock the PX4 stream is stamped on.
		row["px4_wall_offset_sec"] = self._num(px4_wall_offset_sec) if px4_wall_offset_sec is not None else ""
		row["sim_wall_offset_sec"] = self._num(sim_wall_offset_sec) if sim_wall_offset_sec is not None else ""

		self._writer.writerow(row)
		if self._flush_every_row:
			self._file.flush()

	def close(self):
		if getattr(self, "_file", None) is not None and not self._file.closed:
			self._file.flush()
			self._file.close()

	@staticmethod
	def _fieldnames():
		return [
			"t_sec",
			"wall_timestamp",
			"target_timestamp_sec",
			"target_found",
			"target_confidence",
			"target_offset_x",
			"target_offset_y",
			"target_detection_width_px",
			"target_detection_height_px",
			"target_area_fraction",
			"target_fov_saturated",
			"flow_timestamp_sec",
			"flow_valid",
			"flow_mean_x_px_s",
			"flow_mean_y_px_s",
			"flow_mean_x_norm_s",
			"flow_mean_y_norm_s",
			"flow_divergence_1_s",
			"flow_raw_divergence_1_s",
			"flow_fit_quality",
			"flow_derotated",
			"flow_mean_x_raw_px_s",
			"flow_mean_y_raw_px_s",
			"flow_divergence_prederotation_1_s",
			"flow_roi_x0",
			"flow_roi_y0",
			"flow_roi_x1",
			"flow_roi_y1",
			"command_timestamp_sec",
			"command_roll_rad",
			"command_pitch_rad",
			"command_yaw_rad",
			"command_thrust",
			"command_thrust_integral",
			"vehicle_timestamp_sec",
			"vehicle_px4_timestamp_sec",
			"vehicle_x_m",
			"vehicle_y_m",
			"vehicle_z_m",
			"vehicle_vx_m_s",
			"vehicle_vy_m_s",
			"vehicle_vz_m_s",
			"vehicle_yaw_rad",
			"vehicle_attitude_timestamp_sec",
			"vehicle_roll_rad",
			"vehicle_pitch_rad",
			"vehicle_attitude_yaw_rad",
			"vehicle_attitude_source",
			"platform_x_m",
			"platform_y_m",
			"platform_z_m",
			"platform_vx_m_s",
			"platform_vy_m_s",
			"platform_vz_m_s",
			"relative_x_m",
			"relative_y_m",
			"relative_z_m",
			"relative_vx_m_s",
			"relative_vy_m_s",
			"relative_vz_m_s",
			"calibration_axis",
			"mission_substate",
			"mission_divergence_setpoint_1_s",
			"mission_thrust_gain_k",
			"mission_lateral_p_scale",
			"mission_lateral_d_scale",
			"mission_k_min",
			"mission_k_explore",
			"mission_h_crit_m",
			"mission_h_pred_m",
			"mission_peak_accel_m_s2",
			"mission_feasible",
			"mission_k_ceiling_leg",
			"mission_k_target",
			"mission_k_floor",
			"mission_k_probe",
			"mission_k_descend_start",
			"mission_near_field_height_m",
			"mission_ceiling_margin",
			"mission_k_over_ceiling_leg",
			"mission_probe_active",
			"mission_probe_phase",
			"mission_probe_accel_m_s2",
			"mission_probe_mean_accel_m_s2",
			"mission_probe_residual_accel_m_s2",
			"mission_probe_percentile_accel_m_s2",
			"mission_probe_peak_accel_at_handoff_m_s2",
			"mission_probe_peak_decay_tau_sec",
			"mission_probe_elapsed_sec",
			"mission_probe_total_elapsed_sec",
			"timing_camera_cb_start_wall_sec",
			"timing_camera_cb_end_wall_sec",
			"timing_camera_cb_duration_ms",
			"timing_stage_bridge_ms",
			"timing_stage_rotate_ms",
			"timing_stage_show_camera_ms",
			"timing_stage_body_rate_ms",
			"timing_stage_target_acquisition_ms",
			"timing_stage_optical_flow_ms",
			"timing_worker_target_acquisition_ms",
			"timing_worker_optical_flow_ms",
			"timing_ipc_in_ms",
			"timing_ipc_out_ms",
			"timing_frame_to_available_wall_ms",
			"timing_frame_to_command_wall_ms",
			"timing_vision_result_period_wall_ms",
			"timing_vision_dropped_frames",
			"timing_control_compute_start_wall_sec",
			"timing_control_compute_end_wall_sec",
			"timing_control_compute_duration_ms",
			"timing_control_period_wall_sec",
			"timing_control_dt_vision_sec",
			"timing_flow_age_at_control_wall_ms",
			"timing_px4_publish_wall_sec",
			"timing_px4_publish_period_wall_sec",
			"timing_command_age_at_px4_publish_ms",
			"timing_flow_age_at_px4_publish_wall_ms",
			"timing_px4_publish_count",
			"timing_sim_rtf_estimate",
			"timing_px4_rtf_estimate",
			"px4_nav_state",
			"px4_arming_state",
			"px4_failsafe",
			"px4_wall_offset_sec",
			"sim_wall_offset_sec",
		]

	@staticmethod
	def _raw_ts(value: Any):
		try:
			v = float(value)
		except (TypeError, ValueError):
			return ""
		return "" if v <= 0.0 else v

	@staticmethod
	def _num(value: Any):
		try:
			return float(value)
		except (TypeError, ValueError):
			return ""

	@staticmethod
	def _bool_int(value: Any) -> int:
		return 1 if bool(value) else 0

	@staticmethod
	def _int_or_blank(value: Any):
		try:
			v = int(value)
		except (TypeError, ValueError):
			return ""
		return "" if v < 0 else v