"""BEE_LAND paired-log analyser.

The analyser consumes two independent CSV files:

* bee_controller_*.csv: sparse/event-oriented controller and vision log.
* bee_truth_*.csv: dense Gazebo-native physical truth log.

All physical comparisons use Gazebo simulation time.  The truth stream is never
reconstructed from PX4, receipt, or wall timestamps.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_LABELS = {
	"center": "CENTER",
	"approach_probe": "APPROACH PROBE",
	"final_probe": "FINAL PROBE",
	"descend": "DESCENT",
	"infeasible": "INFEASIBLE",
	"landed": "LANDED",
}


PHASE_COLORS = {
	"center": "#4C78A8",
	"approach_probe": "#F2CF5B",
	"final_probe": "#F28E2B",
	"descend": "#59A14F",
	"landed": "#B07AA1",
	"infeasible": "#E15759",
}


PHASE_ALPHA = 0.055


@dataclass(frozen=True)
class AnalysisData:
	controller: pd.DataFrame
	control: pd.DataFrame
	truth: pd.DataFrame
	t0: float
	t1: float


def _num(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
	if name not in df.columns:
		return pd.Series(default, index=df.index, dtype=float)
	return pd.to_numeric(df[name], errors="coerce")


def _bool(df: pd.DataFrame, name: str) -> pd.Series:
	return _num(df, name, 0.0).fillna(0.0) > 0.5


def _clean_string(series: pd.Series) -> pd.Series:
	return series.fillna("").astype(str).str.strip().str.lower()


def _read_csv(path: Path) -> pd.DataFrame:
	try:
		return pd.read_csv(path, low_memory=False)
	except Exception as exc:
		raise RuntimeError(f"Could not read {path}: {exc}") from exc


def _classify_file(df: pd.DataFrame) -> str:
	cols = set(df.columns)
	if "truth_sim_time_sec" in cols and "truth_drone_position_z_m" in cols:
		return "truth"
	if "flow_sim_timestamp_sec" in cols and "controller_phase" in cols:
		return "controller"
	return "unknown"


def _load_pair(path_a: Path, path_b: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
	a = _read_csv(path_a)
	b = _read_csv(path_b)
	kind_a = _classify_file(a)
	kind_b = _classify_file(b)
	if kind_a == "controller" and kind_b == "truth":
		return a, b
	if kind_a == "truth" and kind_b == "controller":
		return b, a
	raise ValueError(
		"Expected one controller CSV and one truth CSV. "
		f"Detected {path_a.name}={kind_a}, {path_b.name}={kind_b}."
	)


def _prepare(controller: pd.DataFrame, truth: pd.DataFrame) -> AnalysisData:
	controller = controller.copy()
	truth = truth.copy()

	controller["_sim_time"] = _num(controller, "flow_sim_timestamp_sec")
	missing = ~np.isfinite(controller["_sim_time"])
	controller.loc[missing, "_sim_time"] = _num(
		controller.loc[missing], "command_source_sim_timestamp_sec"
	)
	missing = ~np.isfinite(controller["_sim_time"])
	controller.loc[missing, "_sim_time"] = _num(
		controller.loc[missing], "contact_truth_sim_timestamp_sec"
	)

	truth["_sim_time"] = _num(truth, "truth_sim_time_sec")
	truth = truth[np.isfinite(truth["_sim_time"])].copy()
	truth = truth.sort_values("_sim_time").drop_duplicates("_sim_time", keep="last")

	# Control rows are the rows produced by fresh visual results.  Explicit
	# events may carry the previous cached visual result, so exclude them from
	# sampled control curves except mission-transition rows when needed only for
	# phase boundary detection.
	event = _clean_string(controller.get("event", pd.Series("", index=controller.index)))
	control_mask = (
		np.isfinite(controller["_sim_time"])
		& _bool(controller, "flow_valid")
		& _num(controller, "vision_sequence").notna()
		& (event == "")
	)
	control = controller[control_mask].copy()
	if control.empty:
		# Compatibility fallback for logs whose normal control rows have an
		# explicit "control" event.
		control_mask = (
			np.isfinite(controller["_sim_time"])
			& _bool(controller, "flow_valid")
			& event.isin(["", "control"])
		)
		control = controller[control_mask].copy()

	control = control.sort_values("_sim_time")
	if "vision_sequence" in control.columns:
		control = control.drop_duplicates("vision_sequence", keep="last")
	else:
		control = control.drop_duplicates("_sim_time", keep="last")

	finite_control = control["_sim_time"].to_numpy(float)
	finite_truth = truth["_sim_time"].to_numpy(float)
	if finite_truth.size == 0:
		raise ValueError("Truth CSV contains no finite truth_sim_time_sec values.")

	if finite_control.size:
		# Begin with the first controller sample for a mission-focused view, but
		# retain the truth tail through node shutdown.  This is essential for
		# post-touchdown settling and disarmed-state diagnostics.
		t0 = max(float(np.nanmin(finite_truth)), float(np.nanmin(finite_control)))
		t1 = float(np.nanmax(finite_truth))
	else:
		t0 = float(np.nanmin(finite_truth))
		t1 = float(np.nanmax(finite_truth))
	if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
		raise ValueError("Controller and truth logs do not have an overlapping SIM-time interval.")

	return AnalysisData(controller=controller, control=control, truth=truth, t0=t0, t1=t1)


def _clip_max_time(data: AnalysisData, max_time: Optional[float]) -> AnalysisData:
	"""Clip both logs to a relative plot-time limit.

	``max_time`` is expressed in simulated seconds after ``data.t0``. The
	clipping is applied before plotting and summary generation.
	"""
	if max_time is None:
		return data
	if not np.isfinite(max_time) or max_time <= 0.0:
		raise ValueError("--max-time must be a finite value greater than zero.")

	clipped_t1 = min(data.t1, data.t0 + float(max_time))
	controller = data.controller[
		np.isfinite(data.controller["_sim_time"])
		& (data.controller["_sim_time"] >= data.t0)
		& (data.controller["_sim_time"] <= clipped_t1)
	].copy()
	control = data.control[
		np.isfinite(data.control["_sim_time"])
		& (data.control["_sim_time"] >= data.t0)
		& (data.control["_sim_time"] <= clipped_t1)
	].copy()
	truth = data.truth[
		np.isfinite(data.truth["_sim_time"])
		& (data.truth["_sim_time"] >= data.t0)
		& (data.truth["_sim_time"] <= clipped_t1)
	].copy()
	if truth.empty:
		raise ValueError("--max-time leaves no Gazebo truth samples to plot.")
	return AnalysisData(controller=controller, control=control, truth=truth, t0=data.t0, t1=clipped_t1)


def _relative_time(values: Iterable[float], t0: float) -> np.ndarray:
	return np.asarray(values, dtype=float) - t0


def _interp_truth(data: AnalysisData, column: str, at_sim_time: np.ndarray) -> np.ndarray:
	if column not in data.truth.columns:
		return np.full_like(at_sim_time, np.nan, dtype=float)
	t = data.truth["_sim_time"].to_numpy(float)
	y = _num(data.truth, column).to_numpy(float)
	good = np.isfinite(t) & np.isfinite(y)
	if good.sum() < 2:
		return np.full_like(at_sim_time, np.nan, dtype=float)
	t = t[good]
	y = y[good]
	order = np.argsort(t)
	t = t[order]
	y = y[order]
	out = np.interp(at_sim_time, t, y)
	out[(at_sim_time < t[0]) | (at_sim_time > t[-1])] = np.nan
	return out


def _phase_intervals(data: AnalysisData) -> list[tuple[float, float, str]]:
	df = data.control
	if df.empty or "mission_substate" not in df.columns:
		return []
	t = df["_sim_time"].to_numpy(float)
	phase = _clean_string(df["mission_substate"]).to_numpy()
	valid = np.isfinite(t) & (phase != "")
	t = t[valid]
	phase = phase[valid]
	if len(t) == 0:
		return []

	intervals: list[tuple[float, float, str]] = []
	start = t[0]
	current = phase[0]
	for i in range(1, len(t)):
		if phase[i] != current:
			intervals.append((start, t[i], current))
			start = t[i]
			current = phase[i]
	intervals.append((start, max(t[-1], data.t1), current))

	# Append terminal LANDED interval from the continuing post-touchdown rows.
	cphase = _clean_string(data.controller.get("controller_phase", pd.Series("", index=data.controller.index)))
	mphase = _clean_string(data.controller.get("mission_substate", pd.Series("", index=data.controller.index)))
	event = _clean_string(data.controller.get("event", pd.Series("", index=data.controller.index)))
	landed_mask = (cphase == "landed") | (mphase == "landed") | event.isin(["landed", "landed_state"])
	landed_rows = data.controller[landed_mask & np.isfinite(data.controller["_sim_time"])]
	if not landed_rows.empty:
		landed_start = float(landed_rows["_sim_time"].min())
		if not intervals or landed_start > intervals[-1][0]:
			if intervals and landed_start < intervals[-1][1]:
				s, _, p = intervals[-1]
				intervals[-1] = (s, landed_start, p)
			intervals.append((landed_start, max(data.t1, landed_start), "landed"))
	return intervals


def _shade_phases(ax, data_or_intervals, labels=True):
	"""Draw readable mission-phase bands and labels.

	Bands use a stronger fill than before and labels sit near the lower edge
	of each axis, where they remain visible without obscuring the main traces.
	"""
	# Backward-compatible API: existing callers pass AnalysisData and a
	# ``labels=...`` keyword, while direct callers may pass precomputed intervals.
	is_analysis_data = isinstance(data_or_intervals, AnalysisData)
	intervals = (
		_phase_intervals(data_or_intervals)
		if is_analysis_data
		else data_or_intervals
	)
	time_origin = data_or_intervals.t0 if is_analysis_data else 0.0

	if not intervals:
		return

	# _phase_intervals returns (start_sim, end_sim, phase).
	for start_sim, end_sim, phase in intervals:
		start = float(start_sim) - time_origin
		end = float(end_sim) - time_origin
		if not np.isfinite(start) or not np.isfinite(end) or end <= start:
			continue

		color = PHASE_COLORS.get(phase, "#9C9C9C")
		ax.axvspan(
			start,
			end,
			facecolor=color,
			alpha=0.20,
			edgecolor=color,
			linewidth=0.9,
			zorder=-20,
		)

		width = end - start
		label = PHASE_LABELS.get(phase, phase.replace("_", " ").upper())
		if width <= 0.15:
			continue

		if not labels:
			continue

		# x uses data coordinates; y uses axis coordinates.  This keeps every
		# phase label aligned in a stable lower strip regardless of y-limits.
		ax.text(
			start + 0.5 * width,
			0.035,
			label,
			transform=ax.get_xaxis_transform(),
			ha="center",
			va="bottom",
			fontsize=8.5,
			fontweight="bold",
			color=color,
			alpha=0.98,
			rotation=0,
			clip_on=True,
			zorder=30,
			bbox={
				"boxstyle": "round,pad=0.18",
				"facecolor": "white",
				"edgecolor": color,
				"linewidth": 0.7,
				"alpha": 0.82,
			},
		)


def _finish_figure(fig: plt.Figure, axes: Iterable[plt.Axes], data: AnalysisData, path: Path) -> None:
	axes = list(axes)
	for i, ax in enumerate(axes):
		ax.grid(True, alpha=0.25)
		_shade_phases(ax, data, labels=(i == 0))
		ax.set_xlim(0.0, max(0.0, data.t1 - data.t0))
	fig.tight_layout()
	fig.savefig(path, dpi=170, bbox_inches="tight")
	plt.close(fig)


def _legend(ax: plt.Axes, *, loc: str = "best", ncol: int = 1) -> None:
	handles, labels = ax.get_legend_handles_labels()
	if handles:
		ax.legend(loc=loc, ncol=ncol, framealpha=0.92)


def plot_detections_boxes_fov(data: AnalysisData, out: Path) -> None:
	c = data.control
	t = _relative_time(c["_sim_time"], data.t0)
	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

	axes[0].plot(t, _num(c, "target_detection_width_px"), label="Detection width")
	axes[0].plot(t, _num(c, "target_detection_height_px"), label="Detection height")
	axes[0].set_ylabel("Bounding box [px]")
	axes[0].set_title("Target bounding box and field-of-view saturation")
	_legend(axes[0], ncol=2)

	axes[1].plot(t, 100.0 * _num(c, "target_area_fraction"), label="Detected image area")
	axes[1].set_ylabel("Area [% of image]")
	_legend(axes[1])

	axes[2].plot(t, _num(c, "target_confidence"), label="Detection confidence")
	sat = _bool(c, "target_fov_saturated").to_numpy()
	if sat.any():
		axes[2].fill_between(t, 0.0, 1.0, where=sat, step="mid", alpha=0.16, label="FOV saturated")
	axes[2].set_ylim(-0.03, 1.05)
	axes[2].set_ylabel("Confidence / flag")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[2], ncol=2)

	_finish_figure(fig, axes, data, out / "detections_boxes_fov.png")


def plot_drone_platform_position(data: AnalysisData, out: Path) -> None:
	tr = data.truth
	t = _relative_time(tr["_sim_time"], data.t0)
	fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
	for ax, axis in zip(axes[:3], "xyz"):
		ax.plot(t, _num(tr, f"truth_drone_position_{axis}_m"), label=f"Drone {axis}")
		if axis == "z":
			platform_col = "truth_deck_point_z_m"
			platform_label = "Deck top z"
		else:
			platform_col = f"truth_platform_position_{axis}_m"
			platform_label = f"Platform {axis}"
		ax.plot(t, _num(tr, platform_col), label=platform_label)
		ax.set_ylabel(f"{axis.upper()} [m]")
		_legend(ax, ncol=2)
	axes[0].set_title("Gazebo truth: drone and platform world position")

	axes[3].plot(t, _num(tr, "truth_min_pad_signed_distance_m"), label="Minimum skid-to-deck distance")
	axes[3].axhline(0.0, linestyle="--", linewidth=1.0, label="Deck contact plane")
	axes[3].set_ylabel("Clearance [m]")
	axes[3].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[3], ncol=2)
	_finish_figure(fig, axes, data, out / "drone_platform_position.png")


def plot_gain_schedule(data: AnalysisData, out: Path) -> None:
	c = data.control
	t = _relative_time(c["_sim_time"], data.t0)
	truth_h = _interp_truth(data, "truth_min_pad_signed_distance_m", c["_sim_time"].to_numpy(float))
	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

	axes[0].plot(t, _num(c, "mission_thrust_gain_k"), label="Applied vertical gain K")
	for col, label, style in [
		("mission_k_min", "Estimated disturbance floor", "--"),
		("mission_k_explore", "Exploration gain", ":"),
		("mission_k_floor", "Scheduled floor", "-."),
	]:
		y = _num(c, col)
		if np.isfinite(y).any():
			axes[0].plot(t, y, linestyle=style, label=label)
	axes[0].set_ylabel("Gain K")
	axes[0].set_title("Mission gain schedule")
	_legend(axes[0], ncol=2)

	axes[1].plot(t, _num(c, "mission_lateral_p_scale"), label="Lateral P scale")
	axes[1].plot(t, _num(c, "mission_lateral_d_scale"), label="Lateral D scale")
	axes[1].set_ylabel("Scale")
	_legend(axes[1], ncol=2)

	axes[2].plot(t, truth_h, label="True minimum skid clearance")
	pred = _num(c, "mission_h_pred_m")
	if np.isfinite(pred).any():
		axes[2].plot(t, pred, label="Mission open-loop height prediction")
	axes[2].axhline(0.0, linestyle="--", linewidth=1.0, label="Contact plane")
	axes[2].set_ylabel("Height [m]")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[2], ncol=2)
	_finish_figure(fig, axes, data, out / "gain_schedule.png")


def plot_lateral_control(data: AnalysisData, out: Path) -> None:
	c = data.control
	t = _relative_time(c["_sim_time"], data.t0)
	tc = c["_sim_time"].to_numpy(float)
	rel_x = _interp_truth(data, "truth_drone_position_x_m", tc) - _interp_truth(data, "truth_platform_position_x_m", tc)
	rel_y = _interp_truth(data, "truth_drone_position_y_m", tc) - _interp_truth(data, "truth_platform_position_y_m", tc)

	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
	axes[0].plot(t, _num(c, "target_offset_x"), label="Image offset x")
	axes[0].plot(t, _num(c, "target_offset_y"), label="Image offset y")
	axes[0].axhline(0.0, linewidth=1.0, linestyle="--")
	axes[0].set_ylabel("Normalized offset")
	axes[0].set_title("Lateral visual control versus Gazebo truth")
	_legend(axes[0], ncol=2)

	axes[1].plot(t, rel_x, label="True drone-platform Δx")
	axes[1].plot(t, rel_y, label="True drone-platform Δy")
	axes[1].axhline(0.0, linewidth=1.0, linestyle="--")
	axes[1].set_ylabel("Relative position [m]")
	_legend(axes[1], ncol=2)

	axes[2].plot(t, np.degrees(_num(c, "command_roll_rad")), label="Roll command")
	axes[2].plot(t, np.degrees(_num(c, "command_pitch_rad")), label="Pitch command")
	axes[2].set_ylabel("Command [deg]")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[2], ncol=2)
	_finish_figure(fig, axes, data, out / "lateral_control.png")


def plot_probe_acceleration(data: AnalysisData, out: Path) -> None:
	c = data.control
	t = _relative_time(c["_sim_time"], data.t0)
	tc = c["_sim_time"].to_numpy(float)
	drone_az = _interp_truth(data, "truth_drone_linear_acceleration_z_m_s2", tc)
	platform_az = _interp_truth(data, "truth_platform_linear_acceleration_z_m_s2", tc)
	relative_az = drone_az - platform_az

	fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
	axes[0].plot(t, _num(c, "mission_probe_accel_m_s2"), label="Command-derived probe acceleration")
	if np.isfinite(relative_az).any():
		axes[0].plot(t, relative_az, alpha=0.72, label="True vertical relative acceleration")
	axes[0].set_ylabel("Acceleration [m/s²]")
	axes[0].set_title("Probe disturbance estimate versus Gazebo truth")
	_legend(axes[0], ncol=2)

	for col, label, style in [
		("mission_probe_mean_accel_m_s2", "Probe mean", "-"),
		("mission_probe_residual_accel_m_s2", "Probe residual", "--"),
		("mission_probe_percentile_accel_m_s2", "Probe percentile", ":"),
		("mission_peak_accel_m_s2", "Peak used by gate", "-."),
	]:
		y = _num(c, col)
		if np.isfinite(y).any():
			axes[1].plot(t, y, linestyle=style, label=label)
			
	if np.isfinite(relative_az).any():
		axes[1].plot(t, relative_az, alpha=0.52, label="True vertical relative acceleration")
	axes[1].set_ylabel("Estimator terms [m/s²]")
	axes[1].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[1], ncol=2)
	_finish_figure(fig, axes, data, out / "probe_acceleration.png")


def plot_vertical_descent(data: AnalysisData, out: Path) -> None:
	c = data.control
	t = _relative_time(c["_sim_time"], data.t0)
	tc = c["_sim_time"].to_numpy(float)
	h_pad = _interp_truth(data, "truth_min_pad_signed_distance_m", tc)
	h_camera = _interp_truth(data, "truth_camera_normal_distance_m", tc)
	closing = _interp_truth(data, "truth_contact_pad_closing_rate_m_s", tc)

	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
	axes[0].plot(t, h_pad, label="True minimum skid clearance")
	axes[0].plot(t, h_camera, label="True camera-to-deck distance", alpha=0.8)
	pred = _num(c, "mission_h_pred_m")
	if np.isfinite(pred).any():
		axes[0].plot(t, pred, linestyle="--", label="Mission predicted height")
	axes[0].axhline(0.0, linestyle=":", linewidth=1.0, label="Contact plane")
	axes[0].set_ylabel("Distance [m]")
	axes[0].set_title("Vertical descent against Gazebo truth")
	_legend(axes[0], ncol=2)

	axes[1].plot(t, closing, label="True pad closing rate (+ toward deck)")
	axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
	contact_t = _first_rising_time(data.truth, "truth_any_contact")
	if np.isfinite(contact_t) and data.t0 <= contact_t <= data.t1:
		contact_rate = _precontact_mean(
			data,
			"truth_contact_pad_closing_rate_m_s",
			contact_t,
			CONTACT_RATE_AVERAGE_WINDOW_SEC,
		)
		if np.isfinite(contact_rate):
			axes[1].scatter(
				[contact_t - data.t0],
				[contact_rate],
				s=72,
				marker="o",
				edgecolors="black",
				linewidths=0.8,
				alpha=0.5,
				zorder=12,
				label=(
					f"Contact mean ({CONTACT_RATE_AVERAGE_WINDOW_SEC:.1f} s): "
					f"{contact_rate:+.3f} m/s"
				),
			)
			axes[1].annotate(
				(
					f"{CONTACT_RATE_AVERAGE_WINDOW_SEC:.1f} s pre-contact mean\n"
					f"{contact_rate:+.3f} m/s"
				),
				xy=(contact_t - data.t0, contact_rate),
				xytext=(8, 10),
				textcoords="offset points",
				fontsize=9,
				fontweight="bold",
			)
	axes[1].set_ylabel("Closing rate [m/s]")
	_legend(axes[1])

	axes[2].plot(t, _num(c, "command_thrust"), label="Commanded thrust")
	integ = _num(c, "command_thrust_integral")
	if np.isfinite(integ).any():
		axes[2].plot(t, integ, label="Thrust integral contribution")
	axes[2].set_ylabel("Normalized thrust")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[2], ncol=2)
	_finish_figure(fig, axes, data, out / "vertical_descent.png")


def plot_vertical_divergence(data: AnalysisData, out: Path) -> None:
	c = data.control
	t = _relative_time(c["_sim_time"], data.t0)
	tc = c["_sim_time"].to_numpy(float)
	truth_d = _interp_truth(data, "truth_normal_expansion_rate_1_s", tc)
	truth_valid = _interp_truth(data, "truth_expansion_truth_valid", tc) > 0.5
	truth_d[~truth_valid] = np.nan
	measured = _num(c, "flow_divergence_1_s").to_numpy(float)
	raw = _num(c, "flow_raw_divergence_1_s").to_numpy(float)
	setpoint = _num(c, "mission_divergence_setpoint_1_s").to_numpy(float)

	fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
	axes[0].plot(t, measured, label="Measured divergence (filtered)")
	axes[0].plot(t, raw, alpha=0.42, label="Measured divergence (raw)")
	axes[0].plot(t, truth_d, linewidth=1.8, label="Gazebo truth c/h")
	axes[0].plot(t, setpoint, linestyle="--", label="Mission setpoint D*")
	axes[0].set_ylabel("Divergence [1/s]")
	axes[0].set_title("Vertical divergence tracking and truth comparison")
	_legend(axes[0], ncol=2)

	error = measured - truth_d
	axes[1].plot(t, error, label="Measurement error: D measured − D truth")
	axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
	axes[1].set_ylabel("Error [1/s]")
	axes[1].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[1])
	_finish_figure(fig, axes, data, out / "vertical_divergence.png")


def plot_platform_motion(data: AnalysisData, out: Path) -> None:
	tr = data.truth
	t = _relative_time(tr["_sim_time"], data.t0)
	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
	for axis in "xyz":
		axes[0].plot(t, _num(tr, f"truth_platform_position_{axis}_m"), label=f"{axis.upper()} position")
		axes[1].plot(t, _num(tr, f"truth_platform_linear_velocity_{axis}_m_s"), label=f"{axis.upper()} velocity")
		axes[2].plot(t, _num(tr, f"truth_platform_linear_acceleration_{axis}_m_s2"), label=f"{axis.upper()} acceleration")
	axes[0].set_title("Gazebo truth: platform motion")
	axes[0].set_ylabel("Position [m]")
	axes[1].set_ylabel("Velocity [m/s]")
	axes[2].set_ylabel("Acceleration [m/s²]")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	for ax in axes:
		_legend(ax, ncol=3)
	_finish_figure(fig, axes, data, out / "platform_motion.png")


def plot_relative_motion(data: AnalysisData, out: Path) -> None:
	tr = data.truth
	t = _relative_time(tr["_sim_time"], data.t0)
	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
	for axis in "xyz":
		rel = _num(tr, f"truth_drone_position_{axis}_m") - _num(tr, f"truth_platform_position_{axis}_m")
		axes[0].plot(t, rel, label=f"Δ{axis}")
		rel_v = _num(tr, f"truth_drone_linear_velocity_{axis}_m_s") - _num(tr, f"truth_platform_linear_velocity_{axis}_m_s")
		axes[1].plot(t, rel_v, label=f"Δv{axis}")
	axes[2].plot(t, _num(tr, "truth_left_pad_signed_distance_m"), label="Left skid clearance")
	axes[2].plot(t, _num(tr, "truth_right_pad_signed_distance_m"), label="Right skid clearance")
	axes[2].plot(t, _num(tr, "truth_contact_pad_closing_rate_m_s"), label="Lowest-pad closing rate")
	axes[0].set_title("Gazebo truth: drone-platform relative motion")
	axes[0].set_ylabel("Relative position [m]")
	axes[1].set_ylabel("Relative velocity [m/s]")
	axes[2].set_ylabel("Clearance / rate")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[0], ncol=3)
	_legend(axes[1], ncol=3)
	_legend(axes[2], ncol=3)
	_finish_figure(fig, axes, data, out / "relative_motion.png")


def plot_target_detection(data: AnalysisData, out: Path) -> None:
	c = data.control
	t = _relative_time(c["_sim_time"], data.t0)
	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
	axes[0].plot(t, _num(c, "target_offset_x"), label="Offset x")
	axes[0].plot(t, _num(c, "target_offset_y"), label="Offset y")
	axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
	axes[0].set_ylabel("Normalized offset")
	axes[0].set_title("Target-detection quality")
	_legend(axes[0], ncol=2)

	axes[1].plot(t, _num(c, "target_confidence"), label="Confidence")
	axes[1].plot(t, _bool(c, "target_found").astype(float), label="Target found")
	axes[1].set_ylabel("Score / flag")
	_legend(axes[1], ncol=2)

	axes[2].plot(t, _num(c, "flow_fit_quality"), label="Affine-flow fit quality")
	axes[2].plot(t, _bool(c, "flow_valid").astype(float), label="Flow valid")
	axes[2].set_ylabel("Quality / flag")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[2], ncol=2)
	_finish_figure(fig, axes, data, out / "target_detection.png")


def _first_rising_time(df: pd.DataFrame, column: str) -> float:
	if column not in df.columns:
		return np.nan
	mask = _bool(df, column).to_numpy()
	t = df["_sim_time"].to_numpy(float)
	idx = np.flatnonzero(mask & np.isfinite(t))
	return float(t[idx[0]]) if idx.size else np.nan


CONTACT_RATE_AVERAGE_WINDOW_SEC = 0.10


def _precontact_mean(
	data: AnalysisData,
	column: str,
	contact_time: float,
	window_sec: float = CONTACT_RATE_AVERAGE_WINDOW_SEC,
) -> float:
	"""Mean a dense truth quantity over the interval immediately before contact."""
	if not np.isfinite(contact_time) or column not in data.truth.columns:
		return np.nan

	times = data.truth["_sim_time"].to_numpy(float)
	values = _num(data.truth, column).to_numpy(float)
	mask = (
		np.isfinite(times)
		& np.isfinite(values)
		& (times >= contact_time - float(window_sec))
		& (times < contact_time)
	)
	return float(np.mean(values[mask])) if np.any(mask) else np.nan


def _finite_stats(values) -> tuple[int, float, float, float, float]:
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return 0, np.nan, np.nan, np.nan, np.nan
	return (
		int(arr.size),
		float(np.mean(arr)),
		float(np.median(arr)),
		float(np.percentile(arr, 95)),
		float(np.max(arr)),
	)


def _append_vision_delay_table(data: AnalysisData, lines: list[str]) -> None:
	"""Append end-to-end and internal vision-delay decompositions.

	New v4.5 logs expose the complete source-frame path and the internal
	OpticalFlowEstimator stages. Older logs remain supported: transport is
	reconstructed from its legs when possible, then from total-target-flow.

	Wall-clock durations are the quantities relevant to the controller delay.
	``optical_flow_total_cpu_ms`` is reported separately because process CPU
	time can exceed wall time when OpenCV / BLAS use several native threads;
	it is a compute-load indicator, not an additional serial delay.
	"""
	c = data.control
	if c.empty:
		return

	def values(name: str) -> np.ndarray:
		return _num(c, name).to_numpy(float)

	def append_table(
		title: str,
		rows: list[tuple[str, np.ndarray]],
		*,
		note: Optional[str] = None,
	) -> None:
		formatted = []
		max_n = 0
		for label, row_values in rows:
			n, mean, median, p95, vmax = _finite_stats(row_values)
			if n == 0:
				continue
			max_n = max(max_n, n)
			formatted.append(
				f"{label:42s}{mean:9.2f}{median:9.2f}{p95:9.2f}{vmax:9.2f}"
			)
		if not formatted:
			return

		lines.extend([
			"",
			title,
			"-" * len(title),
			f"Computed over: {max_n} fresh controller/vision rows in analysed interval",
			f"{'stage':42s}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}",
		])
		lines.extend(formatted)
		if note:
			lines.append(note)

	total = values("timing_frame_to_result_ms")
	prequeue = values("timing_camera_prequeue_ms")
	target = values("timing_vision_worker_target_acquisition_ms")
	flow = values("timing_vision_worker_optical_flow_ms")

	ipc_in = values("timing_vision_ipc_in_ms")
	ipc_out = values("timing_vision_ipc_out_ms")
	transport = values("timing_vision_transport_total_ms")
	if not np.isfinite(transport).any():
		if np.isfinite(ipc_in).any() or np.isfinite(ipc_out).any():
			transport = np.nan_to_num(ipc_in, nan=0.0) + np.nan_to_num(ipc_out, nan=0.0)
		else:
			transport = total - target - flow
			transport[~np.isfinite(total)] = np.nan

	frame_to_command = values("timing_frame_to_command_ms")
	control_compute = values("timing_control_compute_ms")

	phases = _clean_string(c.get("mission_substate", pd.Series("", index=c.index)))
	descend_n = int((phases == "descend").sum())

	append_table(
		"Vision end-to-end delay decomposition [ms]",
		[
			("TOTAL: camera receipt -> processed result", total),
			("camera callback before queue", prequeue),
			("target_acquisition.update()", target),
			("optical_flow.update()", flow),
			("IPC inbound", ipc_in),
			("IPC outbound", ipc_out),
			("transport total (IPC in + IPC out)", transport),
			("TOTAL: camera receipt -> command formed", frame_to_command),
			("mission + control-law compute", control_compute),
		],
		note=(
			f"Mission-phase subset: {descend_n} DESCENT rows."
			if descend_n else None
		),
	)

	# Internal optical-flow wall-clock decomposition. These rows are not
	# expected to sum perfectly because instrumentation itself and tiny
	# unlisted Python operations remain inside total_wall.
	#
	# Keep a single derotation timing source:
	# timing_optical_flow_derotation_ms. Older logs may not contain this column;
	# in that case report a zero-cost disabled stage rather than omitting it.
	derotation_ms = values("timing_optical_flow_derotation_ms")
	if not np.isfinite(derotation_ms).any():
		derotation_ms = np.zeros(len(c), dtype=float)

	append_table(
		"Optical-flow internal wall-time decomposition [ms]",
		[
			("TOTAL optical_flow.update()", values("timing_optical_flow_total_wall_ms")),
			("grayscale conversion", values("timing_optical_flow_grayscale_ms")),
			("ROI setup", values("timing_optical_flow_roi_setup_ms")),
			("adaptive downsample resize", values("timing_optical_flow_downsample_resize_ms")),
			("Farneback dense flow", values("timing_optical_flow_farneback_ms")),
			("flow scaling / upsample", values("timing_optical_flow_flow_scaling_upsample_ms")),
			("derotation", derotation_ms),
			("mean-flow reductions", values("timing_optical_flow_mean_flow_ms")),
			("gradient weights", values("timing_optical_flow_gradient_ms")),
			("robust affine divergence fit", values("timing_optical_flow_affine_fit_ms")),
			("pre-derotation diagnostic fit", values("timing_optical_flow_prederotation_fit_ms")),
			("divergence filter", values("timing_optical_flow_divergence_filter_ms")),
			("result + previous-frame state", values("timing_optical_flow_result_and_state_ms")),
		],
		note="Rows are measured wall-clock stages; small instrumentation gaps may remain.",
	)

	append_table(
		"Robust affine-fit internal decomposition [ms]",
		[
			("TOTAL robust affine fit", values("timing_optical_flow_affine_fit_ms")),
			("array/design setup", values("timing_optical_flow_affine_setup_ms")),
			("initial weighted solve", values("timing_optical_flow_affine_initial_solve_ms")),
			("residual + inlier quantile", values("timing_optical_flow_affine_residual_quantile_ms")),
			("trimmed weighted refit", values("timing_optical_flow_affine_refit_ms")),
		],
	)

	cpu = values("timing_optical_flow_total_cpu_ms")
	if np.isfinite(cpu).any():
		n, mean, median, p95, vmax = _finite_stats(cpu)
		lines.extend([
			"",
			"Optical-flow aggregate process CPU time [ms]",
			"--------------------------------------------",
			f"n={n}, mean={mean:.2f}, median={median:.2f}, p95={p95:.2f}, max={vmax:.2f}",
			"Note: this can exceed wall time when native OpenCV/BLAS threads run in parallel;",
			"it indicates compute/thread load and must not be added to the delay chain.",
		])

	# Compact operating-point summary helps interpret why the stage costs vary.
	operating_rows = [
		("ROI width [px]", values("timing_optical_flow_roi_width_px")),
		("ROI height [px]", values("timing_optical_flow_roi_height_px")),
		("working width [px]", values("timing_optical_flow_working_width_px")),
		("working height [px]", values("timing_optical_flow_working_height_px")),
		("working flow vectors", values("timing_optical_flow_working_flow_vectors")),
		("downsample scale [-]", values("timing_optical_flow_downsample_scale")),
		("affine fit stride", values("timing_optical_flow_affine_fit_stride")),
		("affine sampled points", values("timing_optical_flow_affine_sampled_points")),
		("affine finite points", values("timing_optical_flow_affine_finite_points")),
		("affine points used", values("timing_optical_flow_affine_points_used")),
	]
	formatted = []
	for label, row_values in operating_rows:
		n, mean, median, p95, vmax = _finite_stats(row_values)
		if n:
			formatted.append(
				f"{label:28s}{mean:11.2f}{median:11.2f}{p95:11.2f}{vmax:11.2f}"
			)
	if formatted:
		lines.extend([
			"",
			"Optical-flow operating point",
			"----------------------------",
			f"{'quantity':28s}{'mean':>11}{'median':>11}{'p95':>11}{'max':>11}",
		])
		lines.extend(formatted)

	dropped = values("timing_vision_dropped_frames")
	finite_dropped = dropped[np.isfinite(dropped)]
	if finite_dropped.size:
		lines.extend([
			"",
			f"Frames dropped at input queue (cumulative): {int(np.nanmax(finite_dropped))}",
		])


def _estimate_logged_rates(data: AnalysisData) -> tuple[float, float]:
	"""Estimate effective processed-camera and PX4 publish rates from the log.

	Camera rate uses unique fresh vision results in Gazebo SIM time, so it is the
	effective processed-frame rate after any queue drops. PX4 rate uses changes
	in publish sequence over parent-process monotonic time, which recovers the
	actual publication cadence even though controller rows sample that stream
	sparsely and may repeat the latest publish record.
	"""
	camera_fps = np.nan
	if not data.control.empty:
		vision = data.control.copy()
		if "vision_sequence" in vision.columns:
			vision = vision.drop_duplicates("vision_sequence", keep="last")
		t = _num(vision, "flow_sim_timestamp_sec").to_numpy(float)
		t = np.sort(t[np.isfinite(t)])
		dt = np.diff(t)
		dt = dt[np.isfinite(dt) & (dt > 0.0)]
		if dt.size:
			camera_fps = float(1.0 / np.mean(dt))

	px4_hz = np.nan
	seq = _num(data.controller, "px4_publish_sequence").to_numpy(float)
	mono = _num(data.controller, "px4_publish_monotonic_timestamp_sec").to_numpy(float)
	good = np.isfinite(seq) & np.isfinite(mono)
	if np.count_nonzero(good) >= 2:
		publish = pd.DataFrame({"seq": seq[good], "mono": mono[good]})
		publish = (
			publish.sort_values("mono")
			.drop_duplicates("seq", keep="last")
		)
		if len(publish) >= 2:
			dseq = float(publish["seq"].iloc[-1] - publish["seq"].iloc[0])
			dtime = float(publish["mono"].iloc[-1] - publish["mono"].iloc[0])
			if dseq > 0.0 and dtime > 0.0:
				px4_hz = dseq / dtime

	return camera_fps, px4_hz


def write_summary(data: AnalysisData, out: Path, controller_path: Path, truth_path: Path) -> None:
	c = data.control
	tr = data.truth
	camera_fps, px4_hz = _estimate_logged_rates(data)
	rate_parts = []
	if np.isfinite(camera_fps):
		rate_parts.append(f"processed camera: {camera_fps:.2f} fps (SIM time)")
	if np.isfinite(px4_hz):
		rate_parts.append(f"PX4 publish: {px4_hz:.2f} Hz (monotonic wall time)")
	rate_line = "Estimated logged rates: " + "; ".join(rate_parts) if rate_parts else "Estimated logged rates: unavailable"
	lines = [
		"BEE_LAND paired-log analysis",
		"================================",
		f"Controller file: {controller_path}",
		f"Truth file:      {truth_path}",
		f"Analysed SIM span: {data.t1 - data.t0:.3f} s ({data.t0:.6f} to {data.t1:.6f})",
		f"Plot-time range: 0.000 to {data.t1 - data.t0:.3f} s SIM",
		f"Unique controller samples: {len(c)}",
		f"Truth samples: {len(tr)}",
		rate_line,
	]

	truth_sim = tr["_sim_time"].to_numpy(float)
	truth_sim = truth_sim[np.isfinite(truth_sim)]
	truth_dt = np.diff(truth_sim)
	truth_dt = truth_dt[np.isfinite(truth_dt) & (truth_dt > 0.0)]
	receipt_wall = _num(tr, "truth_receipt_wall_timestamp_sec").to_numpy(float)
	receipt_wall = receipt_wall[np.isfinite(receipt_wall)]
	wall_dt = np.diff(receipt_wall)
	wall_dt = wall_dt[np.isfinite(wall_dt) & (wall_dt > 0.0)]
	if truth_dt.size:
		lines.append(f"Mean truth SIM period: {np.mean(truth_dt):.6f} s")
		lines.append(f"Median truth SIM period: {np.median(truth_dt):.6f} s")
		lines.append(f"Effective truth rate in SIM time: {1.0/np.mean(truth_dt):.2f} Hz")
	if wall_dt.size:
		lines.append(f"Mean truth receipt wall period: {np.mean(wall_dt):.6f} s")
		lines.append(f"Effective truth receipt rate in wall time: {1.0/np.mean(wall_dt):.2f} Hz")
	if truth_dt.size and wall_dt.size:
		n = min(truth_dt.size, wall_dt.size)
		valid = (truth_dt[:n] > 0.0) & (wall_dt[:n] > 0.0)
		if valid.any():
			rtf = np.sum(truth_dt[:n][valid]) / np.sum(wall_dt[:n][valid])
			lines.append(f"Effective Gazebo real-time factor (SIM/wall): {rtf:.3f}")

	phases = _clean_string(c.get("mission_substate", pd.Series("", index=c.index)))
	if len(phases):
		lines.append("Mission samples by phase:")
		for phase, count in phases[phases != ""].value_counts(sort=False).items():
			lines.append(f"  {phase}: {count}")

	contact_t = _first_rising_time(tr, "truth_any_contact")
	confirmed_t = _first_rising_time(tr, "truth_contact_confirmed")
	if np.isfinite(contact_t):
		lines.append(f"First Gazebo contact: {contact_t:.6f} s SIM ({contact_t-data.t0:.3f} s plot time)")
		contact_closing = _precontact_mean(
			data,
			"truth_contact_pad_closing_rate_m_s",
			contact_t,
			CONTACT_RATE_AVERAGE_WINDOW_SEC,
		)
		contact_clearance = _interp_truth(
			data, "truth_min_pad_signed_distance_m", np.array([contact_t])
		)[0]
		lines.append(
			"True pre-contact closing-rate mean "
			f"({CONTACT_RATE_AVERAGE_WINDOW_SEC:.2f} s window): "
			f"{contact_closing:+.4f} m/s"
		)
		lines.append(f"True minimum pad clearance at first contact: {contact_clearance:+.5f} m")
	if np.isfinite(confirmed_t):
		lines.append(f"Confirmed contact: {confirmed_t:.6f} s SIM ({confirmed_t-data.t0:.3f} s plot time)")
		confirmed_closing = _interp_truth(
			data, "truth_contact_pad_closing_rate_m_s", np.array([confirmed_t])
		)[0]
		lines.append(f"True closing rate at confirmed contact: {confirmed_closing:+.4f} m/s")

	tc = c["_sim_time"].to_numpy(float)
	measured = _num(c, "flow_divergence_1_s").to_numpy(float)
	truth_d = _interp_truth(data, "truth_normal_expansion_rate_1_s", tc)
	valid = _interp_truth(data, "truth_expansion_truth_valid", tc) > 0.5
	good = np.isfinite(measured) & np.isfinite(truth_d) & valid
	if good.any():
		err = measured[good] - truth_d[good]
		lines.append(f"Divergence truth comparison samples: {good.sum()}")
		lines.append(f"Divergence bias (measured-truth): {np.mean(err):+.5f} 1/s")
		lines.append(f"Divergence RMSE: {math.sqrt(np.mean(err**2)):.5f} 1/s")
		if good.sum() >= 3:
			lines.append(f"Divergence correlation: {np.corrcoef(measured[good], truth_d[good])[0,1]:.4f}")

	_append_vision_delay_table(data, lines)

	lines += [
		"",
		"Interpretation note:",
		"  Gazebo truth is used directly in SIM time. No PX4/platform clock",
		"  reconstruction and no differentiated cross-stream positions are used.",
	]
	(out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compare a BEE_LAND controller log with its Gazebo truth log."
	)
	parser.add_argument("csv_a", type=Path, help="Controller or truth CSV")
	parser.add_argument("csv_b", type=Path, help="The paired truth or controller CSV")
	parser.add_argument("output_dir", type=Path, help="Directory for generated plots")
	parser.add_argument(
		"--full",
		action="store_true",
		help="Also generate platform_motion, relative_motion, and target_detection.",
	)
	parser.add_argument(
		"--max-time",
		type=float,
		default=None,
		metavar="SECONDS",
		help=(
			"Stop all plots and summary calculations at this many simulated "
			"seconds after the common plot start. Example: --max-time 55."
		),
	)
	return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
	args = parse_args(argv)
	controller, truth = _load_pair(args.csv_a, args.csv_b)
	data = _prepare(controller, truth)
	data = _clip_max_time(data, args.max_time)
	args.output_dir.mkdir(parents=True, exist_ok=True)

	main_plots = [
		plot_detections_boxes_fov,
		plot_drone_platform_position,
		plot_gain_schedule,
		plot_lateral_control,
		plot_probe_acceleration,
		plot_vertical_descent,
		plot_vertical_divergence,
	]
	for fn in main_plots:
		fn(data, args.output_dir)
		print(f"saved {args.output_dir / (fn.__name__.replace('plot_', '') + '.png')}")

	if args.full:
		for fn in [plot_platform_motion, plot_relative_motion, plot_target_detection]:
			fn(data, args.output_dir)
			print(f"saved {args.output_dir / (fn.__name__.replace('plot_', '') + '.png')}")

	controller_path = args.csv_a if _classify_file(controller) == "controller" else args.csv_b
	# The dataframes are already ordered by role; determine source paths safely.
	if _classify_file(_read_csv(args.csv_a).head(2)) == "controller":
		controller_path, truth_path = args.csv_a, args.csv_b
	else:
		controller_path, truth_path = args.csv_b, args.csv_a
	write_summary(data, args.output_dir, controller_path, truth_path)
	print(f"saved {args.output_dir / 'summary.txt'}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except (ValueError, RuntimeError) as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		raise SystemExit(2)
