"""
CSV diagnostics writer for the bee landing controller.

This module is intentionally independent from the controller logic.

It logs, at each control tick:

	- normalized time
	- target acquisition outputs
	- optical-flow outputs
	- attitude/thrust commands
	- vehicle state, for diagnostics only

The vehicle state is not used by the control law.
"""

import csv
import os
import time
from datetime import datetime
from typing import Optional

try:
	from .state import (
		AttitudeSetpoint,
		FlowResult,
		TargetEstimate,
		VehicleState,
	)
except ImportError:
	from state import (
		AttitudeSetpoint,
		FlowResult,
		TargetEstimate,
		VehicleState,
	)


class DiagnosticsWriter:
	def __init__(
		self,
		output_dir: str = "logs",
		filename: Optional[str] = None,
		flush_every_row: bool = True,
	):
		"""
		Create a diagnostics CSV writer.

		output_dir:
			Folder where the CSV file will be created.

		filename:
			If None, a timestamped filename is generated automatically.

		flush_every_row:
			Useful for debugging. If the program crashes, the last rows are
			still written to disk.
		"""
		os.makedirs(output_dir, exist_ok=True)

		if filename is None:
			date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
			filename = f"bee_diagnostics_{date_str}.csv"

		self.filepath = os.path.join(output_dir, filename)
		self._flush_every_row = bool(flush_every_row)

		self._file = open(self.filepath, mode="w", newline="")
		self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
		self._writer.writeheader()

		self._t0 = None
		self._row_count = 0

	def write(
		self,
		wall_timestamp: Optional[float],
		target: Optional[TargetEstimate],
		flow: Optional[FlowResult],
		setpoint: Optional[AttitudeSetpoint],
		vehicle_state: Optional[VehicleState],
		calibration_axis: Optional[str] = None,
	):
		"""
		Write one diagnostics row.

		This should be called after the control law computes the latest
		setpoint, usually once per control timer tick.

		calibration_axis: only meaningful for calibration_node.py — which
		axis ("roll"/"pitch"/"thrust"/"settle") the step sequence was
		exercising for this row. None (the default, and what bee_node.py
		always passes implicitly by omitting it) writes an empty field.
		fit_axis_models.py uses this to fit each axis only from its own
		matching rows — without it, a row logged while e.g. roll is the
		one genuinely open-loop axis under test, but thrust is being
		actively damped against drift (see calibration_node.py), would
		look to the thrust fit like an open-loop sample when it's
		actually closed-loop, re-introducing the exact cause/effect
		entanglement this whole logging setup exists to avoid.
		"""
		if wall_timestamp is None:
			wall_timestamp = time.time()

		if self._t0 is None:
			self._t0 = float(wall_timestamp)

		t_sec = float(wall_timestamp) - self._t0

		row = {
			# --------------------------------------------------------
			# Time
			# --------------------------------------------------------
			"t_sec": t_sec,
			"wall_timestamp": float(wall_timestamp),

			# --------------------------------------------------------
			# Target acquisition
			# --------------------------------------------------------
			"target_timestamp_sec": self._normalize_optional_timestamp(
				getattr(target, "timestamp", None)
			),
			"target_found": self._safe_bool(getattr(target, "found", False)),
			"target_confidence": self._safe_float(getattr(target, "confidence", 0.0)),
			"target_offset_x": self._safe_float(getattr(target, "offset_x", 0.0)),
			"target_offset_y": self._safe_float(getattr(target, "offset_y", 0.0)),
			"target_detection_width_px": self._safe_float(
				getattr(target, "detection_width", 0.0)
			),
			"target_detection_height_px": self._safe_float(
				getattr(target, "detection_height", 0.0)
			),
			"target_area_fraction": self._safe_float(
				getattr(target, "area_fraction", 0.0)
			),

			# --------------------------------------------------------
			# Optical flow
			# --------------------------------------------------------
			"flow_timestamp_sec": self._normalize_optional_timestamp(
				getattr(flow, "timestamp", None)
			),
			"flow_valid": self._safe_bool(getattr(flow, "valid", False)),
			"flow_mean_x_px_s": self._safe_float(getattr(flow, "mean_flow_x", 0.0)),
			"flow_mean_y_px_s": self._safe_float(getattr(flow, "mean_flow_y", 0.0)),
			"flow_mean_x_norm_s": self._safe_float(getattr(flow, "mean_flow_x_norm", 0.0)),
			"flow_mean_y_norm_s": self._safe_float(getattr(flow, "mean_flow_y_norm", 0.0)),
			"flow_divergence_1_s": self._safe_float(
				getattr(flow, "divergence", 0.0)
			),
			"flow_raw_divergence_1_s": self._safe_float(
				getattr(flow, "raw_divergence", getattr(flow, "divergence", 0.0))
			),
			"flow_roi_x0": self._safe_int(getattr(flow, "roi_x0", -1)),
			"flow_roi_y0": self._safe_int(getattr(flow, "roi_y0", -1)),
			"flow_roi_x1": self._safe_int(getattr(flow, "roi_x1", -1)),
			"flow_roi_y1": self._safe_int(getattr(flow, "roi_y1", -1)),

			# --------------------------------------------------------
			# Controller command
			# --------------------------------------------------------
			"command_timestamp_sec": self._normalize_optional_timestamp(
				getattr(setpoint, "timestamp", None)
			),
			"command_roll_rad": self._safe_float(getattr(setpoint, "roll", 0.0)),
			"command_pitch_rad": self._safe_float(getattr(setpoint, "pitch", 0.0)),
			"command_yaw_rad": self._safe_float(getattr(setpoint, "yaw", 0.0)),
			"command_thrust": self._safe_float(getattr(setpoint, "thrust", 0.0)),

			# --------------------------------------------------------
			# Vehicle state, for diagnostics only
			# --------------------------------------------------------
			"vehicle_timestamp_sec": self._normalize_optional_timestamp(
				getattr(vehicle_state, "timestamp", None)
			),
			"vehicle_x_m": self._safe_float(getattr(vehicle_state, "x", 0.0)),
			"vehicle_y_m": self._safe_float(getattr(vehicle_state, "y", 0.0)),
			"vehicle_z_m": self._safe_float(getattr(vehicle_state, "z", 0.0)),
			"vehicle_vx_m_s": self._safe_float(getattr(vehicle_state, "vx", 0.0)),
			"vehicle_vy_m_s": self._safe_float(getattr(vehicle_state, "vy", 0.0)),
			"vehicle_vz_m_s": self._safe_float(getattr(vehicle_state, "vz", 0.0)),
			"vehicle_yaw_rad": self._safe_float(getattr(vehicle_state, "yaw", 0.0)),
			"vehicle_attitude_timestamp_sec": self._normalize_optional_timestamp(
				getattr(vehicle_state, "attitude_timestamp", None)
			),
			"vehicle_roll_rad": self._safe_float(getattr(vehicle_state, "roll", 0.0)),
			"vehicle_pitch_rad": self._safe_float(getattr(vehicle_state, "pitch", 0.0)),
			"vehicle_attitude_yaw_rad": self._safe_float(
				getattr(vehicle_state, "attitude_yaw", 0.0)
			),
			"vehicle_attitude_source": getattr(vehicle_state, "attitude_source", "") or "",

			# --------------------------------------------------------
			# Calibration-only metadata (empty outside calibration_node.py)
			# --------------------------------------------------------
			"calibration_axis": calibration_axis if calibration_axis else "",
		}

		self._writer.writerow(row)
		self._row_count += 1

		if self._flush_every_row:
			self._file.flush()

	def close(self):
		if not self._file.closed:
			self._file.flush()
			self._file.close()

	def row_count(self) -> int:
		return self._row_count

	def _normalize_optional_timestamp(self, timestamp):
		"""
		Normalize a timestamp using the first CSV write time as t=0.

		Returns an empty string if the timestamp is missing or zero.
		"""
		if timestamp is None:
			return ""

		try:
			timestamp = float(timestamp)
		except (TypeError, ValueError):
			return ""

		if timestamp <= 0.0 or self._t0 is None:
			return ""

		return timestamp - self._t0

	@staticmethod
	def _safe_float(value) -> float:
		try:
			return float(value)
		except (TypeError, ValueError):
			return 0.0

	@staticmethod
	def _safe_bool(value) -> int:
		return 1 if bool(value) else 0

	@staticmethod
	def _safe_int(value) -> int:
		try:
			return int(value)
		except (TypeError, ValueError):
			return -1

	@staticmethod
	def _fieldnames():
		return [
			# Time
			"t_sec",
			"wall_timestamp",

			# Target acquisition
			"target_timestamp_sec",
			"target_found",
			"target_confidence",
			"target_offset_x",
			"target_offset_y",
			"target_detection_width_px",
			"target_detection_height_px",
			"target_area_fraction",

			# Optical flow
			"flow_timestamp_sec",
			"flow_valid",
			"flow_mean_x_px_s",
			"flow_mean_y_px_s",
			"flow_mean_x_norm_s",
			"flow_mean_y_norm_s",
			"flow_divergence_1_s",
			"flow_raw_divergence_1_s",
			"flow_roi_x0",
			"flow_roi_y0",
			"flow_roi_x1",
			"flow_roi_y1",

			# Controller command
			"command_timestamp_sec",
			"command_roll_rad",
			"command_pitch_rad",
			"command_yaw_rad",
			"command_thrust",

			# Vehicle state, diagnostics only
			"vehicle_timestamp_sec",
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

			# Calibration-only metadata
			"calibration_axis",
		]