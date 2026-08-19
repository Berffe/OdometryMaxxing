"""BEE_LAND three-stream analyser.

The analyser consumes three independent CSV files:

* bee_controller_*.csv: sparse/event-oriented controller and vision log.
* bee_truth_*.csv: dense Gazebo-native physical truth log.
* bee_wind_*.csv: deterministic WindController command diagnostics.

All physical comparisons use Gazebo simulation time. Truth and wind are never
reconstructed from PX4, receipt, or wall timestamps.

Plot groups
-----------
The default invocation produces the mission/controller plots that remain useful
for both calm and windy runs. ``--wind`` keeps every default plot and adds only
wind-specific diagnostics (visual-center/acceleration biases, lateral command
trim decomposition, and reconstructed wind acceleration contribution).
``--full`` remains the opt-in group for lower-level truth/vision diagnostics.

Folder mode may be used when one run directory contains the matching three CSVs.
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


# Height-free visual trackability diagnostic.  The derivative is reconstructed
# causally from the already-filtered divergence using a short least-squares
# line fit.  The 2.0 s^-2 threshold is intentionally a diagnostic candidate:
# it separates the currently validated feasible and high-frequency infeasible
# runs, but it is not yet part of the mission feasibility verdict.
VISUAL_MISMATCH_DERIVATIVE_WINDOW_SEC = 0.20
VISUAL_MISMATCH_PERCENTILE = 95.0
VISUAL_MISMATCH_MIN_SAMPLES = 10
VISUAL_MISMATCH_LIMIT_S2 = 2.0


# Current ControlLaw allocation constants used only to convert the logged final
# attitude/collective-thrust command back into acceleration-domain diagnostics.
# These are not controller inputs and never affect the flight.  Keeping the
# inverse mapping here lets command plots remain expressed in the same m/s²
# coordinates as the mission probes and frozen lateral feedforward.
CONTROL_HOVER_THRUST = 0.73
G_ACCEL = 9.80665


@dataclass(frozen=True)
class AnalysisData:
	controller: pd.DataFrame
	control: pd.DataFrame
	truth: pd.DataFrame
	wind: pd.DataFrame
	t0: float
	t1: float
	wind_force_scale: float = 1.0


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
	if "wind_sim_time_sec" in cols and "wind_command_x_enu_m_s" in cols:
		return "wind"
	if "flow_sim_timestamp_sec" in cols and "controller_phase" in cols:
		return "controller"
	return "unknown"


def _load_triplet(*paths: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if len(paths) != 3:
		raise ValueError("Expected exactly three CSV paths: controller, truth and wind.")
	frames = [(_read_csv(path), path) for path in paths]
	by_kind: dict[str, tuple[pd.DataFrame, Path]] = {}
	for frame, path in frames:
		kind = _classify_file(frame)
		if kind == "unknown":
			raise ValueError(f"Could not classify {path.name} as controller, truth or wind.")
		if kind in by_kind:
			raise ValueError(f"Received more than one {kind} CSV.")
		by_kind[kind] = (frame, path)
	missing = [kind for kind in ("controller", "truth", "wind") if kind not in by_kind]
	if missing:
		raise ValueError(f"Missing required CSV stream(s): {', '.join(missing)}.")
	return by_kind["controller"][0], by_kind["truth"][0], by_kind["wind"][0]


def _prepare(
	controller: pd.DataFrame,
	truth: pd.DataFrame,
	wind: pd.DataFrame,
	*,
	wind_force_scale: float = 1.0,
) -> AnalysisData:
	controller = controller.copy()
	truth = truth.copy()
	wind = wind.copy()

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

	wind["_sim_time"] = _num(wind, "wind_sim_time_sec")
	wind = wind[np.isfinite(wind["_sim_time"])].copy()
	wind = wind.sort_values("_sim_time").drop_duplicates("_sim_time", keep="last")
	if wind.empty:
		raise ValueError("Wind CSV contains no finite wind_sim_time_sec values.")

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

	return AnalysisData(
		controller=controller, control=control, truth=truth, wind=wind,
		t0=t0, t1=t1, wind_force_scale=float(wind_force_scale),
	)


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
	wind = data.wind[
		np.isfinite(data.wind["_sim_time"])
		& (data.wind["_sim_time"] >= data.t0)
		& (data.wind["_sim_time"] <= clipped_t1)
	].copy()
	if truth.empty:
		raise ValueError("--max-time leaves no Gazebo truth samples to plot.")
	if wind.empty:
		raise ValueError("--max-time leaves no wind-command samples to plot.")
	return AnalysisData(
		controller=controller, control=control, truth=truth, wind=wind,
		t0=data.t0, t1=clipped_t1, wind_force_scale=data.wind_force_scale,
	)


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


def _interp_wind(data: AnalysisData, column: str, at_sim_time: np.ndarray) -> np.ndarray:
	if column not in data.wind.columns:
		return np.full_like(at_sim_time, np.nan, dtype=float)
	t = data.wind["_sim_time"].to_numpy(float)
	y = _num(data.wind, column).to_numpy(float)
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


def _quaternion_to_euler_xyz(
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Convert Gazebo quaternions to roll, pitch and yaw [rad].

	The returned angles follow the standard XYZ roll-pitch-yaw convention used
	for Gazebo poses. Quaternions are normalized sample by sample, invalid samples
	remain NaN, and roll/yaw are unwrapped to show a continuous angle evolution.
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	z = np.asarray(z, dtype=float)
	w = np.asarray(w, dtype=float)

	roll = np.full_like(x, np.nan, dtype=float)
	pitch = np.full_like(x, np.nan, dtype=float)
	yaw = np.full_like(x, np.nan, dtype=float)

	norm = np.sqrt(x * x + y * y + z * z + w * w)
	good = (
		np.isfinite(x)
		& np.isfinite(y)
		& np.isfinite(z)
		& np.isfinite(w)
		& (norm > 1e-12)
	)
	if not np.any(good):
		return roll, pitch, yaw

	qx = x[good] / norm[good]
	qy = y[good] / norm[good]
	qz = z[good] / norm[good]
	qw = w[good] / norm[good]

	sin_roll = 2.0 * (qw * qx + qy * qz)
	cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)
	roll_good = np.arctan2(sin_roll, cos_roll)

	sin_pitch = 2.0 * (qw * qy - qz * qx)
	pitch_good = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

	sin_yaw = 2.0 * (qw * qz + qx * qy)
	cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
	yaw_good = np.arctan2(sin_yaw, cos_yaw)

	roll[good] = np.unwrap(roll_good)
	pitch[good] = pitch_good
	yaw[good] = np.unwrap(yaw_good)
	return roll, pitch, yaw


def _truth_euler_angles(
	data: AnalysisData,
	entity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return one truth entity's roll, pitch and yaw arrays [rad]."""
	tr = data.truth
	prefix = f"truth_{entity}_orientation"
	return _quaternion_to_euler_xyz(
		_num(tr, f"{prefix}_x").to_numpy(float),
		_num(tr, f"{prefix}_y").to_numpy(float),
		_num(tr, f"{prefix}_z").to_numpy(float),
		_num(tr, f"{prefix}_w").to_numpy(float),
	)


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


def _finish_figure(
	fig: plt.Figure,
	axes: Iterable[plt.Axes],
	data: AnalysisData,
	path: Path,
	*,
	x_end: Optional[float] = None,
) -> None:
	axes = list(axes)
	plot_end = (
		max(0.0, data.t1 - data.t0)
		if x_end is None
		else max(0.0, float(x_end))
	)
	for i, ax in enumerate(axes):
		ax.grid(True, alpha=0.25)
		_shade_phases(ax, data, labels=(i == 0))
		ax.set_xlim(0.0, plot_end)
	fig.tight_layout()
	fig.savefig(path, dpi=170, bbox_inches="tight")
	plt.close(fig)


def _legend(ax: plt.Axes, *, loc: str = "best", ncol: int = 1) -> None:
	handles, labels = ax.get_legend_handles_labels()
	if handles:
		ax.legend(loc=loc, ncol=ncol, framealpha=0.92)


def _last_finite_value(df: pd.DataFrame, column: str) -> float:
	values = _num(df, column).to_numpy(float)
	values = values[np.isfinite(values)]
	return float(values[-1]) if values.size else np.nan


def _probe_plot_end_sim_time(data: AnalysisData) -> float:
	"""First DESCENT/INFEASIBLE timestamp, or the analysed end if neither occurs."""
	df = data.controller
	if df.empty or "mission_substate" not in df.columns:
		return float(data.t1)

	phase = _clean_string(df["mission_substate"])
	times = _num(df, "_sim_time").to_numpy(float)
	terminal = phase.isin(["descend", "infeasible"]).to_numpy()
	valid = terminal & np.isfinite(times)
	if not np.any(valid):
		return float(data.t1)
	return min(float(data.t1), float(np.min(times[valid])))


def _causal_linear_derivative(
	times: np.ndarray,
	values: np.ndarray,
	window_sec: float,
	*,
	min_points: int = 4,
) -> np.ndarray:
	"""Estimate dy/dt from a causal trailing least-squares line fit.

	The estimator only uses samples at or before the current timestamp, so the
	diagnostic can later be reproduced online without relying on acausal
	smoothing.  A 0.20 s window is long enough to suppress frame-to-frame
	divergence noise while remaining short relative to the platform periods used
	in the landing tests.
	"""
	t = np.asarray(times, dtype=float)
	y = np.asarray(values, dtype=float)
	out = np.full_like(y, np.nan, dtype=float)
	left = 0

	for i in range(len(t)):
		if not np.isfinite(t[i]) or not np.isfinite(y[i]):
			continue
		while left < i and (t[i] - t[left]) > float(window_sec):
			left += 1

		idx = np.arange(left, i + 1)
		good = np.isfinite(t[idx]) & np.isfinite(y[idx])
		idx = idx[good]
		if idx.size < int(min_points):
			continue

		t_local = t[idx]
		y_local = y[idx]
		t_centered = t_local - float(np.mean(t_local))
		den = float(np.dot(t_centered, t_centered))
		if den <= 1e-12:
			continue

		y_centered = y_local - float(np.mean(y_local))
		out[i] = float(np.dot(t_centered, y_centered) / den)

	return out


def _visual_mismatch_series(
	data: AnalysisData,
	control: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return measured and Gazebo-truth chi = D_dot - D^2 [1/s^2]."""
	t = control["_sim_time"].to_numpy(float)
	measured_d = _num(control, "flow_divergence_1_s").to_numpy(float)
	measured_d_dot = _causal_linear_derivative(
		t,
		measured_d,
		VISUAL_MISMATCH_DERIVATIVE_WINDOW_SEC,
	)
	measured_chi = measured_d_dot - measured_d * measured_d

	truth_d = _interp_truth(data, "truth_normal_expansion_rate_1_s", t)
	truth_valid = _interp_truth(data, "truth_expansion_truth_valid", t) > 0.5
	truth_d[~truth_valid] = np.nan
	truth_d_dot = _causal_linear_derivative(
		t,
		truth_d,
		VISUAL_MISMATCH_DERIVATIVE_WINDOW_SEC,
	)
	truth_chi = truth_d_dot - truth_d * truth_d
	return measured_chi, truth_chi


def _visual_mismatch_probe_statistics(
	control: pd.DataFrame,
	chi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Cumulative FINAL_PROBE mismatch statistics in controller-row coordinates.

	The robust probe statistic follows the same interpretation used for the
	acceleration probe: persistent mismatch is represented by |mean| and the
	oscillatory component by the percentile of |chi - mean|. Their sum is the
	scalar quantity compared with the candidate trackability limit.
	"""
	mean_mag = np.full(len(control), np.nan, dtype=float)
	residual = np.full(len(control), np.nan, dtype=float)
	percentile = np.full(len(control), np.nan, dtype=float)
	robust = np.full(len(control), np.nan, dtype=float)

	phase = _clean_string(
		control.get("mission_substate", pd.Series("", index=control.index))
	).to_numpy()
	probe_indices = np.flatnonzero((phase == "final_probe") & np.isfinite(chi))
	values: list[float] = []

	for i in probe_indices:
		values.append(float(chi[i]))
		arr = np.asarray(values, dtype=float)
		mean = float(np.mean(arr))
		mean_mag[i] = abs(mean)
		residual[i] = abs(float(chi[i]) - mean)

		if arr.size >= VISUAL_MISMATCH_MIN_SAMPLES:
			p = float(np.percentile(np.abs(arr - mean), VISUAL_MISMATCH_PERCENTILE))
			percentile[i] = p
			robust[i] = abs(mean) + p

	return mean_mag, residual, percentile, robust


def _plot_visual_mismatch_legacy(data: AnalysisData, out: Path) -> None:
	"""Legacy fallback for logs recorded before mission chi telemetry existed."""
	c_all = data.control
	probe_end_sim = _probe_plot_end_sim_time(data)
	if probe_end_sim < data.t1 - 1e-9:
		c = c_all[c_all["_sim_time"] < probe_end_sim].copy()
	else:
		c = c_all.copy()

	if c.empty:
		return

	t = _relative_time(c["_sim_time"], data.t0)
	measured_chi, truth_chi = _visual_mismatch_series(data, c)
	mean_mag, residual, percentile, robust = _visual_mismatch_probe_statistics(c, measured_chi)

	fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

	axes[0].plot(
		t,
		measured_chi,
		linewidth=1.6,
		label=r"Visual mismatch $\chi=\dot{D}-D^2$",
	)
	if np.isfinite(truth_chi).any():
		axes[0].plot(
			t,
			truth_chi,
			alpha=0.72,
			linewidth=1.5,
			label="Gazebo-truth normalized relative acceleration",
		)
	axes[0].axhline(
		VISUAL_MISMATCH_LIMIT_S2,
		linestyle="--",
		linewidth=1.5,
		label=f"Candidate trackability limit: ±{VISUAL_MISMATCH_LIMIT_S2:.2f} s⁻²",
	)
	axes[0].axhline(
		-VISUAL_MISMATCH_LIMIT_S2,
		linestyle="--",
		linewidth=1.5,
		label="_nolegend_",
	)
	axes[0].axhline(0.0, linestyle=":", linewidth=1.0)
	axes[0].set_ylabel("Mismatch χ [s⁻²]")
	axes[0].set_title("Visual mismatch probe: height-free dynamic trackability")
	_legend(axes[0], ncol=2)

	axes[1].plot(t, mean_mag, label=r"Persistent term $|\bar{\chi}|$")
	axes[1].plot(t, residual, alpha=0.55, label=r"Instantaneous residual $|\chi-\bar{\chi}|$")
	axes[1].plot(
		t,
		percentile,
		linestyle=":",
		linewidth=1.7,
		label=f"Residual P{VISUAL_MISMATCH_PERCENTILE:.0f}",
	)
	axes[1].plot(
		t,
		robust,
		linewidth=2.0,
		label=r"Probe statistic $|\bar{\chi}|+P_{95}(|\chi-\bar{\chi}|)$",
	)
	axes[1].axhline(
		VISUAL_MISMATCH_LIMIT_S2,
		linestyle="--",
		linewidth=1.7,
		label=f"Candidate limit: {VISUAL_MISMATCH_LIMIT_S2:.2f} s⁻²",
	)
	axes[1].set_ylabel("Probe statistic [s⁻²]")
	axes[1].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[1], ncol=2)

	_finish_figure(
		fig,
		axes,
		data,
		out / "visual_mismatch_z_probe.png",
		x_end=probe_end_sim - data.t0,
	)




def _visual_mismatch_columns(axis: str) -> dict[str, str]:
	"""Telemetry columns for one visual-mismatch axis.

	``z`` keeps the original unqualified mission fields for backward
	compatibility.  The lateral probes use the explicit x/y fields introduced
	with the three-axis trackability gate.
	"""
	axis = str(axis).strip().lower()
	if axis == "z":
		return {
			"chi": "mission_chi",
			"abs": "mission_chi_abs_1_s2",
			"percentile": "mission_chi_percentile_1_s2",
			"peak": "mission_chi_peak_1_s2",
			"limit": "mission_chi_limit_1_s2",
			"observed": "mission_chi_observed_sec",
			"ready": "mission_tracking_z_ready",
			"decision_peak": "mission_tracking_z_decision_chi_peak_1_s2",
			"legacy_ready": "mission_tracking_ready",
			"legacy_decision_peak": "mission_tracking_decision_chi_peak_1_s2",
		}
	if axis == "x":
		return {
			"chi": "mission_chi_x",
			"abs": "mission_chi_x_abs_1_s2",
			"percentile": "mission_chi_x_percentile_1_s2",
			"peak": "mission_chi_x_peak_1_s2",
			"limit": "mission_chi_x_limit_1_s2",
			"observed": "mission_chi_x_observed_sec",
			"ready": "mission_tracking_x_ready",
			"decision_peak": "mission_tracking_x_decision_chi_peak_1_s2",
		}
	if axis == "y":
		return {
			"chi": "mission_chi_y",
			"abs": "mission_chi_y_abs_1_s2",
			"percentile": "mission_chi_y_percentile_1_s2",
			"peak": "mission_chi_y_peak_1_s2",
			"limit": "mission_chi_y_limit_1_s2",
			"observed": "mission_chi_y_observed_sec",
			"ready": "mission_tracking_y_ready",
			"decision_peak": "mission_tracking_y_decision_chi_peak_1_s2",
		}
	raise ValueError(f"Unknown visual-mismatch axis: {axis!r}")


def _logged_visual_mismatch_available(
	control: pd.DataFrame,
	axis: str = "z",
) -> bool:
	"""True when the log contains the mission's online chi estimator for ``axis``."""
	column = _visual_mismatch_columns(axis)["chi"]
	return column in control.columns and np.isfinite(_num(control, column)).any()


def _first_true_sample_time(df: pd.DataFrame, column: str) -> float:
	"""Return the first SIM timestamp where a logged boolean telemetry flag is true."""
	if column not in df.columns:
		return np.nan
	mask = _bool(df, column).to_numpy()
	t = _num(df, "_sim_time").to_numpy(float)
	idx = np.flatnonzero(mask & np.isfinite(t))
	return float(t[idx[0]]) if idx.size else np.nan


def _tracking_decision_value(control: pd.DataFrame, column: str) -> float:
	"""Read a FINAL_PROBE gate value only from rows where tracking_ready is latched."""
	if column not in control.columns:
		return np.nan
	ready = _bool(control, "mission_tracking_ready").to_numpy()
	value = _num(control, column).to_numpy(float)
	good = ready & np.isfinite(value)
	if np.any(good):
		return float(value[np.flatnonzero(good)[0]])
	return np.nan


def _tracking_decision_sample(
	df: pd.DataFrame,
	axis: str = "z",
) -> tuple[float, float, float]:
	"""Return ``(time, decision_peak, observed_sec)`` at an axis gate decision.

	The full controller log is used because the actual FINAL_PROBE verdict may
	be emitted on the phase-transition event row and therefore be absent from
	the fresh-control subset.  For z, the original unqualified tracking fields
	remain accepted as a compatibility fallback.
	"""
	cols = _visual_mismatch_columns(axis)
	ready_column = cols["ready"]
	peak_column = cols["decision_peak"]

	if ready_column not in df.columns and axis == "z":
		ready_column = cols["legacy_ready"]
	if peak_column not in df.columns and axis == "z":
		peak_column = cols["legacy_decision_peak"]
	if ready_column not in df.columns:
		return np.nan, np.nan, np.nan

	ready = _bool(df, ready_column).to_numpy()
	t = _num(df, "_sim_time").to_numpy(float)
	peak = _num(df, peak_column).to_numpy(float)
	obs = _num(df, cols["observed"]).to_numpy(float)
	good = ready & np.isfinite(t)
	idx = np.flatnonzero(good)
	if not idx.size:
		return np.nan, np.nan, np.nan
	i = int(idx[0])
	return (
		float(t[i]),
		float(peak[i]) if np.isfinite(peak[i]) else np.nan,
		float(obs[i]) if np.isfinite(obs[i]) else np.nan,
	)


def _plot_visual_mismatch_axis(
	data: AnalysisData,
	out: Path,
	*,
	axis: str,
) -> None:
	"""Plot one ONLINE visual-mismatch probe using the common two-panel layout.

	Panel 1 shows the online mismatch itself over the full analysed mission.
	Panel 2 shows only FINAL_PROBE and DESCENT, because the robust envelope and
	the frozen decision are meaningful there.  The former estimator/gate flag
	panel is intentionally omitted.

	No derivative trace is plotted: it is an internal ingredient of chi, not an
	independent trackability result.
	"""
	axis = axis.lower()
	c = data.control
	if c.empty:
		return

	# Historical logs only contain the vertical online mismatch. Preserve the
	# existing offline reconstruction for that case, but never invent x/y probes.
	if not _logged_visual_mismatch_available(c, axis):
		if axis == "z":
			_plot_visual_mismatch_legacy(data, out)
		return

	cols = _visual_mismatch_columns(axis)
	t = _relative_time(c["_sim_time"], data.t0)
	tc = c["_sim_time"].to_numpy(float)

	chi = _num(c, cols["chi"]).to_numpy(float)
	abs_chi = _num(c, cols["abs"]).to_numpy(float)
	percentile = _num(c, cols["percentile"]).to_numpy(float)
	live_peak = _num(c, cols["peak"]).to_numpy(float)
	limit = _num(c, cols["limit"]).to_numpy(float)

	phase = _clean_string(
		c.get("mission_substate", pd.Series("", index=c.index))
	).to_numpy()
	probe_descent = np.isin(phase, ["final_probe", "descend"])

	# Keep the second panel visually scoped to the phases where this diagnostic
	# is used for commitment and then monitored after commitment.
	abs_display = np.where(probe_descent, abs_chi, np.nan)
	percentile_display = np.where(probe_descent, percentile, np.nan)
	peak_display = np.where(probe_descent, live_peak, np.nan)
	limit_display = np.where(probe_descent, limit, np.nan)

	axis_titles = {
		"z": "Vertical visual mismatch",
		"x": "Visual mismatch X — roll channel",
		"y": "Visual mismatch Y — pitch channel",
	}
	formulas = {
		"z": r"$\chi_z=\dot{\omega}_z-\omega_z^2$",
		"x": r"$\chi_x=\dot{\omega}_x-\omega_x\omega_z$",
		"y": r"$\chi_y=\dot{\omega}_y-\omega_y\omega_z$",
	}

	fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

	# 1) Online mismatch. For z only, retain the independent Gazebo-truth
	# validation curve that the original figure already provided.
	axes[0].plot(t, chi, linewidth=1.8, label=f"Online {formulas[axis]}")
	if axis == "z":
		truth_d = _interp_truth(data, "truth_normal_expansion_rate_1_s", tc)
		truth_valid = _interp_truth(data, "truth_expansion_truth_valid", tc) > 0.5
		truth_d[~truth_valid] = np.nan
		truth_d_dot = _causal_linear_derivative(
			tc, truth_d, VISUAL_MISMATCH_DERIVATIVE_WINDOW_SEC,
		)
		truth_chi = truth_d_dot - truth_d * truth_d
		if np.isfinite(truth_chi).any():
			axes[0].plot(
				t,
				truth_chi,
				alpha=0.52,
				linewidth=1.3,
				label="Gazebo-truth normalized relative acceleration",
			)

	axes[0].axhline(0.0, linestyle=":", linewidth=1.0)
	axes[0].set_ylabel("Mismatch χ [s⁻²]")
	axes[0].set_title(f"{axis_titles[axis]}: online height-free bandwidth diagnostic")
	_legend(axes[0], ncol=2)

	# 2) Exact robust quantities used by VisualMismatchProbe, restricted to
	# FINAL_PROBE + DESCENT.  The live peak may keep evolving after commitment;
	# the latched decision peak remains the immutable pre-descent verdict.
	if np.isfinite(abs_display).any():
		axes[1].plot(t, abs_display, alpha=0.42, label=r"Instantaneous $|\chi|$")
	if np.isfinite(percentile_display).any():
		axes[1].plot(
			t,
			percentile_display,
			linestyle=":",
			linewidth=1.8,
			label="Rolling |χ| percentile",
		)
	if np.isfinite(peak_display).any():
		axes[1].plot(
			t,
			peak_display,
			linewidth=2.0,
			label="Live leaky mismatch peak",
		)

	decision_time_sim, committed_peak, _ = _tracking_decision_sample(
		data.controller, axis
	)
	if np.isfinite(decision_time_sim) and np.isfinite(committed_peak):
		decision_time = decision_time_sim - data.t0
		decision_plot = np.where(
			probe_descent & (t >= decision_time),
			committed_peak,
			np.nan,
		)
		axes[1].plot(
			t,
			decision_plot,
			linestyle="-.",
			linewidth=2.1,
			label="Latched FINAL_PROBE decision peak",
		)
		for ax in axes:
			ax.axvline(
				decision_time,
				linestyle=":",
				linewidth=1.0,
				alpha=0.65,
			)

	if np.isfinite(limit_display).any():
		axes[1].plot(
			t,
			limit_display,
			linestyle="--",
			linewidth=1.8,
			label="FINAL_PROBE χ limit",
		)
	axes[1].set_ylabel("Mismatch magnitude [s⁻²]")
	axes[1].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[1], ncol=2)

	_finish_figure(
		fig,
		axes,
		data,
		out / f"visual_mismatch_{axis}_probe.png",
	)


def plot_visual_mismatch_z_probe(data: AnalysisData, out: Path) -> None:
	_plot_visual_mismatch_axis(data, out, axis="z")


def plot_visual_mismatch_x_probe(data: AnalysisData, out: Path) -> None:
	_plot_visual_mismatch_axis(data, out, axis="x")


def plot_visual_mismatch_y_probe(data: AnalysisData, out: Path) -> None:
	_plot_visual_mismatch_axis(data, out, axis="y")

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
	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

	# Vertical gain: use the same four-curve semantics as the lateral panels:
	# applied gain, rejection lower bound, scheduled floor, touchdown ceiling.
	axes[0].plot(t, _num(c, "mission_thrust_gain_k"), label="Applied vertical gain K")
	for col, label, style in [
		("mission_k_min", "Estimated disturbance floor", "--"),
		("mission_k_floor", "Scheduled floor", "-."),
		("mission_k_ceiling_leg", "Touchdown stability ceiling", ":"),
	]:
		y = _num(c, col)
		if np.isfinite(y).any():
			axes[0].plot(t, y, linestyle=style, label=label)
	axes[0].set_ylabel("Vertical gain K")
	axes[0].set_title("Mission gain schedule")
	_legend(axes[0], ncol=2)

	def actual_lateral_d_gain(axis: str) -> pd.Series:
		"""Reconstruct the D gain that ControlLaw actually receives.

		The mission historically commands lateral *scales*. The independent
		``mission_<axis>_k_applied`` quantity is the scheduled physical K used by
		the lateral gate/descent logic and is not necessarily populated in CENTER,
		APPROACH_PROBE or FINAL_PROBE. Plotting it directly therefore creates gaps
		or apparent jumps that do not exist in the controller.

		For every active phase the physical coefficient is instead

		    K_D,actual = K_D,base * effective_axis_d_scale,

		where the effective scale is the independent axis scale when available and
		the historical shared lateral D scale otherwise.

		Recover the immutable base K_D from rows where both the independently
		scheduled K and the exact per-axis scale are logged. In DESCEND their ratio
		is identically the base controller gain. If a run never reaches DESCEND, use
		the FINAL_PROBE/PROBE_HOLD value k_probe and its contemporaneous scale.
		"""
		prefix = f"mission_{axis}"
		axis_scale = _num(c, f"{prefix}_d_scale")
		shared_scale = _num(c, "mission_lateral_d_scale")

		# CENTER / APPROACH_PROBE / FINAL_PROBE still use the historical shared
		# lateral scale in some mission versions. DESCEND supplies independent
		# per-axis scales. Reconstruct the scale exactly as ControlLaw sees it:
		# prefer the axis-specific value when present, otherwise use the shared one.
		scale = axis_scale.where(np.isfinite(axis_scale), shared_scale)
		scheduled = _num(c, f"{prefix}_k_applied")

		scale_values = scale.to_numpy(float)
		scheduled_values = scheduled.to_numpy(float)
		base_gain = np.nan

		# Preferred identification: DESCEND's scheduled K divided by the exact
		# per-axis scale sent to ControlLaw. This remains valid even when roll and
		# pitch have different independently scheduled floors.
		valid = (
			np.isfinite(scheduled_values)
			& np.isfinite(scale_values)
			& (np.abs(scale_values) > 1e-9)
		)
		if np.any(valid):
			ratios = scheduled_values[valid] / scale_values[valid]
			ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
			if ratios.size:
				base_gain = float(np.nanmedian(ratios))

		# Probe-only / infeasible-run fallback: k_probe is the actual D gain held
		# during FINAL_PROBE, so k_probe / d_scale identifies the same base K_D.
		if not np.isfinite(base_gain):
			k_probe = _num(c, f"{prefix}_k_probe").to_numpy(float)
			phase = _clean_string(
				c.get("mission_substate", pd.Series("", index=c.index))
			).to_numpy()
			probe_phase = np.isin(phase, ["final_probe", "probe_hold", "infeasible"])
			valid_probe = (
				probe_phase
				& np.isfinite(k_probe)
				& np.isfinite(scale_values)
				& (np.abs(scale_values) > 1e-9)
			)
			if np.any(valid_probe):
				ratios = k_probe[valid_probe] / scale_values[valid_probe]
				ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
				if ratios.size:
					base_gain = float(np.nanmedian(ratios))

		if np.isfinite(base_gain) and np.isfinite(scale_values).any():
			return pd.Series(base_gain * scale_values, index=c.index, dtype=float)

		# Older logs may only contain the physical scheduled K. It is preferable
		# to show the available actual-gain samples than to relabel a unitless
		# legacy scale as a physical gain.
		return scheduled

	def plot_lateral_axis(ax: plt.Axes, axis: str) -> None:
		prefix = f"mission_{axis}"
		actual = actual_lateral_d_gain(axis)
		if np.isfinite(actual).any():
			ax.plot(t, actual, linewidth=1.9, label=f"Actual {axis} D gain")

		# All gate quantities below already live in the same physical D-gain
		# coordinates, so they can now be compared directly with the solid curve.
		for suffix, label, style in [
			("k_min", "Estimated disturbance floor", "--"),
			("k_floor", "Scheduled floor", "-."),
			("k_ceiling_leg", "Touchdown stability ceiling", ":"),
		]:
			y = _num(c, f"{prefix}_{suffix}")
			if np.isfinite(y).any():
				ax.plot(t, y, linestyle=style, label=label)

		# Very old logs have neither an independent per-axis scale nor physical K.
		# Retain a clearly marked diagnostic fallback without pretending it has K
		# units. Current logs should never take this branch.
		if not np.isfinite(actual).any():
			legacy = _num(c, "mission_lateral_d_scale")
			if np.isfinite(legacy).any():
				ax.plot(
					t, legacy, alpha=0.75,
					label=f"{axis.capitalize()} D scale (legacy, unitless)",
				)

		ax.set_ylabel(f"{axis.capitalize()} D gain")
		_legend(ax, ncol=2)

	plot_lateral_axis(axes[1], "roll")
	plot_lateral_axis(axes[2], "pitch")
	axes[2].set_xlabel("Time since common log start [s SIM]")

	_finish_figure(fig, axes, data, out / "gain_schedule.png")



def _lateral_static_offset_reference(
	c: pd.DataFrame,
	*,
	axis: str,
) -> np.ndarray:
	"""Return the visual reference actually used by the lateral P branch.

	Current wind-capable logs expose ``mission_*_offset_setpoint`` directly.
	That reference already contains the geometric tilt compensation and the
	adaptive visual-center bias, so it is the correct quantity to use in the
	lateral decomposition.

	Older logs may not contain it.  In that case fall back to the historical
	slow trim mean so old result folders remain analysable.
	"""
	axis = axis.lower()
	if axis == "x":
		setpoint_col = "mission_roll_offset_setpoint"
		trim_mean_col = "mission_center_trim_mean_x"
	elif axis == "y":
		setpoint_col = "mission_pitch_offset_setpoint"
		trim_mean_col = "mission_center_trim_mean_y"
	else:
		raise ValueError(f"Unknown lateral image axis: {axis!r}")

	setpoint = _num(c, setpoint_col).to_numpy(float)
	if np.isfinite(setpoint).any():
		return setpoint

	trim_mean = _num(c, trim_mean_col).to_numpy(float)
	if np.isfinite(trim_mean).any():
		return trim_mean

	return np.zeros(len(c), dtype=float)

def _allocated_lateral_accelerations(c: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
	"""Invert the logged final PX4 setpoint into allocated lateral accelerations.

	This mirrors ControlLaw._shape_commands after filtering, authority allocation
	and tilt compensation, so the resulting curves are the acceleration actually
	represented by the final roll/pitch/collective-thrust command.
	"""
	roll = _num(c, "command_roll_rad").to_numpy(float)
	pitch = _num(c, "command_pitch_rad").to_numpy(float)
	thrust = _num(c, "command_thrust").to_numpy(float)

	specific_thrust = G_ACCEL * thrust / max(CONTROL_HOVER_THRUST, 1e-9)
	roll_accel = specific_thrust * np.sin(roll)
	pitch_accel = specific_thrust * np.cos(roll) * np.sin(pitch)
	return roll_accel, pitch_accel


def plot_lateral_match(data: AnalysisData, out: Path) -> None:
	"""Visual lateral matching: raw offset, static/dynamic split, truth position."""
	c = data.control
	if c.empty:
		return

	t = _relative_time(c["_sim_time"], data.t0)
	tc = c["_sim_time"].to_numpy(float)
	offset_x = _num(c, "target_offset_x").to_numpy(float)
	offset_y = _num(c, "target_offset_y").to_numpy(float)
	static_x = _lateral_static_offset_reference(c, axis="x")
	static_y = _lateral_static_offset_reference(c, axis="y")
	omega_x = _num(c, "flow_mean_x_norm_s").to_numpy(float)
	omega_y = _num(c, "flow_mean_y_norm_s").to_numpy(float)

	rel_x = (
		_interp_truth(data, "truth_drone_position_x_m", tc)
		- _interp_truth(data, "truth_platform_position_x_m", tc)
	)
	rel_y = (
		_interp_truth(data, "truth_drone_position_y_m", tc)
		- _interp_truth(data, "truth_platform_position_y_m", tc)
	)

	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

	# 1) What the camera actually sees.
	axes[0].plot(t, offset_x, label=r"Measured $e_x$")
	axes[0].plot(t, offset_y, label=r"Measured $e_y$")
	axes[0].axhline(0.0, linewidth=1.0, linestyle="--")
	axes[0].set_ylabel("Normalized offset")
	axes[0].set_title("Lateral visual matching: static trim and dynamic mismatch")
	_legend(axes[0], ncol=2)

# for ax, column, axis_name, channel_label in [
# 		(axes[0], "flow_mean_x_norm_s", "x", "Image X / roll channel"),
# 		(axes[1], "flow_mean_y_norm_s", "y", "Image Y / pitch channel"),
# 	]:
# 		values = _num(c, column)
# 		ax.plot(
# 			t, values, linewidth=1.8,
# 			label=fr"{channel_label}: $\omega_{{{axis_name}}}$",
# 		)
# 		ax.axhline(0.0, linewidth=1.0, linestyle="--")
# 		ax.set_ylabel(rf"$\omega_{{{axis_name}}}$ [1/s]")
# 		_legend(ax)

# 	axes[0].set_title("Lateral optical-flow evolution")
# 	axes[1].set_xlabel("Time since common log start [s SIM]")

	# 2) Independent Gazebo truth, retained from the former lateral_control plot.
	axes[1].plot(t, rel_x, label="True drone-platform Δx")
	axes[1].plot(t, rel_y, label="True drone-platform Δy")
	axes[1].axhline(0.0, linewidth=1.0, linestyle="--")
	axes[1].set_ylabel("Relative position [m]")
	_legend(axes[1], ncol=2)

	# 3) Measured optical flows.
	axes[2].plot(t, omega_x, label="Image X / roll channel")
	axes[2].plot(t, omega_y, label="Image Y / pitch channel")
	axes[2].axhline(0.0, linewidth=1.0, linestyle="--")
	axes[2].set_ylabel("Lateral optical-flow evolution [1/s]")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[2], ncol=2)

	_finish_figure(fig, axes, data, out / "lateral_match.png")



def plot_lateral_decomposition(data: AnalysisData, out: Path) -> None:
	"""Separate roll/pitch visual-reference decomposition.

	The dotted curve is the actual visual setpoint sent to ``ControlLaw``.
	The solid curve is therefore the position-like P error that remains after
	geometric tilt compensation and any adaptive-center bias.
	"""
	c = data.control
	if c.empty:
		return

	t = _relative_time(c["_sim_time"], data.t0)
	offset_x = _num(c, "target_offset_x").to_numpy(float)
	offset_y = _num(c, "target_offset_y").to_numpy(float)
	reference_x = _lateral_static_offset_reference(c, axis="x")
	reference_y = _lateral_static_offset_reference(c, axis="y")
	error_x = offset_x - reference_x
	error_y = offset_y - reference_y
	phase = _clean_string(
		c.get("mission_substate", pd.Series("", index=c.index))
	).to_numpy()
	p_active = np.isin(phase, ["center", "approach_probe"])
	error_x = np.where(p_active, error_x, np.nan)
	error_y = np.where(p_active, error_y, np.nan)

	fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

	for ax, axis_name, measured, reference, error in [
		(axes[0], "Roll / image X", offset_x, reference_x, error_x),
		(axes[1], "Pitch / image Y", offset_y, reference_y, error_y),
	]:
		ax.plot(t, measured, alpha=0.55, linewidth=1.4, label="Measured target offset")
		ax.plot(t, reference, linestyle=":", linewidth=2.0, label="Visual reference setpoint")
		ax.plot(t, error, linewidth=1.8, label="Active P-branch visual error")
		ax.axhline(0.0, linewidth=1.0, linestyle="--")
		ax.set_ylabel("Normalized offset")
		_legend(ax, ncol=3)

	axes[0].set_title("Lateral visual decomposition: measured offset, adaptive reference and P error")
	axes[1].set_xlabel("Time since common log start [s SIM]")
	_finish_figure(fig, axes, data, out / "lateral_decomposition.png")


def plot_lateral_commands(data: AnalysisData, out: Path) -> None:
	"""Baseline acceleration-domain lateral commands.

	The final shaped PX4 attitude/collective command is inverted into the
	acceleration domain.  Static wind-trim decomposition is intentionally kept
	out of this default figure; ``--wind`` adds ``wind_lateral_commands.png``.
	"""
	c = data.control
	if c.empty:
		return

	t = _relative_time(c["_sim_time"], data.t0)
	roll_total, pitch_total = _allocated_lateral_accelerations(c)

	fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
	for ax, axis_name, total in [
		(axes[0], "Roll", roll_total),
		(axes[1], "Pitch", pitch_total),
	]:
		ax.plot(t, total, linewidth=1.9, label=f"Allocated {axis_name.lower()} acceleration command")
		ax.axhline(0.0, linewidth=1.0, linestyle=":")
		ax.set_ylabel("Acceleration [m/s²]")
		_legend(ax)

	axes[0].set_title("Lateral acceleration commands")
	axes[1].set_xlabel("Time since common log start [s SIM]")
	_finish_figure(fig, axes, data, out / "lateral_commands.png")


def _wind_axis_components(
	data: AnalysisData,
	at_sim_time: np.ndarray,
	*,
	truth_axis: str,
	truth_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return relative acceleration, wind compensation and their sum.

	``truth_sign`` maps the Gazebo world axis into the controller channel:
	vertical -> +Z, roll/image-X -> -Y, pitch/image-Y -> -X.
	"""
	tc = np.asarray(at_sim_time, dtype=float)
	drone_accel = _interp_truth(
		data, f"truth_drone_linear_acceleration_{truth_axis}_m_s2", tc
	)
	platform_accel = _interp_truth(
		data, f"truth_platform_linear_acceleration_{truth_axis}_m_s2", tc
	)
	drone_velocity = _interp_truth(
		data, f"truth_drone_linear_velocity_{truth_axis}_m_s", tc
	)
	wind_velocity = _interp_wind(
		data, f"wind_command_{truth_axis}_enu_m_s", tc
	)
	wind_enabled = _interp_wind(data, "wind_enabled", tc)

	relative_accel = float(truth_sign) * (drone_accel - platform_accel)
	wind_physical_accel = data.wind_force_scale * (wind_velocity - drone_velocity)
	wind_compensation = -float(truth_sign) * wind_physical_accel
	wind_compensation[np.isfinite(wind_enabled) & (wind_enabled <= 0.5)] = 0.0
	return relative_accel, wind_compensation, relative_accel + wind_compensation


def _plot_probe_axis(
	data: AnalysisData,
	out: Path,
	*,
	axis_name: str,
	truth_axis: str,
	truth_sign: float,
	probe_prefix: str,
	peak_column: str,
	capacity_ceiling_column: str,
	filename: str,
) -> None:
	"""Plot the probe quantities that matter independently of wind.

	Panel 1 compares the command-derived probe acceleration with the full
	truth acceleration in the gate coordinates: the Gazebo drone-platform
	dynamic acceleration shifted by the probe's estimated static mean.
	Panel 2 shows the slowly varying probe mean, gate envelope and stability
	capacity in the exact coordinates used by the feasibility logic.

	The reconstructed WindEffects contribution deliberately lives in the
	``--wind`` plot group instead of being mixed into every probe figure.
	"""
	c_all = data.control
	probe_end_sim = _probe_plot_end_sim_time(data)

	if probe_end_sim < data.t1 - 1e-9:
		c = c_all[c_all["_sim_time"] < probe_end_sim].copy()
	else:
		c = c_all.copy()

	if c.empty:
		return

	t = _relative_time(c["_sim_time"], data.t0)
	tc = c["_sim_time"].to_numpy(float)
	relative_accel, _, _ = _wind_axis_components(
		data, tc, truth_axis=truth_axis, truth_sign=truth_sign
	)

	static_mean = _num(c, f"mission_{probe_prefix}_mean_accel_m_s2").to_numpy(float)
	# The feasibility envelope is expressed around this static operating point.
	# Put Gazebo's dynamic relative acceleration in the same coordinates before
	# comparing the truth trace with the probe command and gate envelope.
	full_truth_accel = relative_accel + static_mean
	peak_used = _num(c, peak_column).to_numpy(float)
	capacity_ceiling = _last_finite_value(c_all, capacity_ceiling_column)

	upper_peak = lower_peak = None
	if np.isfinite(peak_used).any() and np.isfinite(static_mean).any():
		upper_peak = static_mean + peak_used
		lower_peak = static_mean - peak_used

	upper_capacity = lower_capacity = None
	if np.isfinite(capacity_ceiling) and np.isfinite(static_mean).any():
		upper_capacity = static_mean + capacity_ceiling
		lower_capacity = static_mean - capacity_ceiling

	fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

	axes[0].plot(
		t,
		_num(c, f"mission_{probe_prefix}_accel_m_s2"),
		linewidth=1.8,
		label="Command-derived probe acceleration",
	)
	if np.isfinite(full_truth_accel).any():
		axes[0].plot(
			t,
			full_truth_accel,
			alpha=0.78,
			linewidth=1.6,
			label="Full truth acceleration (Gazebo dynamic + probe static mean)",
		)
	if upper_peak is not None:
		peak_line = axes[0].plot(
			t, upper_peak, linestyle="-.", linewidth=1.8, label="Gate envelope"
		)[0]
		axes[0].plot(
			t, lower_peak, linestyle="-.", linewidth=1.2, alpha=0.70,
			color=peak_line.get_color(), label="_nolegend_",
		)
	axes[0].axhline(0.0, linestyle=":", linewidth=0.9, alpha=0.55)
	axes[0].set_ylabel("Acceleration [m/s²]")
	axes[0].set_title(f"{axis_name} probe: demand, truth response and gate envelope")
	_legend(axes[0], ncol=3)

	if np.isfinite(static_mean).any():
		axes[1].plot(
			t, static_mean, linestyle=":", linewidth=2.0, label="Probe mean acceleration"
		)
	if upper_peak is not None:
		peak_line_mid = axes[1].plot(
			t, upper_peak, linestyle="-.", linewidth=1.8, label="Gate envelope"
		)[0]
		axes[1].plot(
			t, lower_peak, linestyle="-.", linewidth=1.2, alpha=0.70,
			color=peak_line_mid.get_color(), label="_nolegend_",
		)
	if upper_capacity is not None:
		cap_line = axes[1].plot(
			t, upper_capacity, linestyle="--", linewidth=2.0,
			label="Stability capacity envelope",
		)[0]
		axes[1].plot(
			t, lower_capacity, linestyle="--", linewidth=1.3, alpha=0.75,
			color=cap_line.get_color(), label="_nolegend_",
		)
	axes[1].axhline(0.0, linestyle=":", linewidth=0.9, alpha=0.55)
	axes[1].set_ylabel("Probe coordinates [m/s²]")
	axes[1].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[1], ncol=3)

	_finish_figure(
		fig, axes, data, out / filename, x_end=probe_end_sim - data.t0
	)

def plot_probe_vertical(data: AnalysisData, out: Path) -> None:
	_plot_probe_axis(
		data,
		out,
		axis_name="vertical",
		truth_axis="z",
		truth_sign=1.0,
		probe_prefix="probe",
		peak_column="mission_peak_accel_m_s2",
		capacity_ceiling_column="mission_vertical_accel_capacity_ceiling_m_s2",
		filename="probe_vertical.png",
	)


def plot_probe_roll(data: AnalysisData, out: Path) -> None:
	_plot_probe_axis(
		data,
		out,
		axis_name="roll channel (image X -> world Y)",
		truth_axis="y",
		truth_sign=-1.0,
		probe_prefix="roll_probe",
		peak_column="mission_roll_peak_accel_m_s2",
		capacity_ceiling_column="mission_roll_accel_capacity_ceiling_m_s2",
		filename="probe_roll.png",
	)


def plot_probe_pitch(data: AnalysisData, out: Path) -> None:
	_plot_probe_axis(
		data,
		out,
		axis_name="pitch channel (image Y -> world X)",
		truth_axis="x",
		truth_sign=-1.0,
		probe_prefix="pitch_probe",
		peak_column="mission_pitch_peak_accel_m_s2",
		capacity_ceiling_column="mission_pitch_accel_capacity_ceiling_m_s2",
		filename="probe_pitch.png",
	)



def plot_wind_biases(data: AnalysisData, out: Path) -> None:
	"""Wind-oriented adaptive-bias diagnostics.

	Panel 1 is the adaptive visual-center bias used while lateral P control is
	active. Panel 2 shows whether that bias is actually driving the tilt-corrected
	physical centering error toward zero. Panel 3 is the near-field static
	acceleration trim that takes over when lateral P is disabled.
	"""
	c = data.control
	if c.empty:
		return

	t = _relative_time(c["_sim_time"], data.t0)
	bx = _num(c, "mission_center_visual_bias_x").to_numpy(float)
	by = _num(c, "mission_center_visual_bias_y").to_numpy(float)
	phys_x = _num(c, "mission_center_physical_mean_x").to_numpy(float)
	phys_y = _num(c, "mission_center_physical_mean_y").to_numpy(float)
	roll_ff = _num(c, "mission_roll_accel_feedforward_m_s2", 0.0).fillna(0.0).to_numpy(float)
	pitch_ff = _num(c, "mission_pitch_accel_feedforward_m_s2", 0.0).fillna(0.0).to_numpy(float)
	roll_dev = _num(c, "mission_descent_roll_bias_deviation_m_s2").to_numpy(float)
	pitch_dev = _num(c, "mission_descent_pitch_bias_deviation_m_s2").to_numpy(float)

	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

	if np.isfinite(bx).any():
		axes[0].plot(t, bx, linewidth=1.9, label=r"Adaptive visual bias $b_x$")
	if np.isfinite(by).any():
		axes[0].plot(t, by, linewidth=1.9, label=r"Adaptive visual bias $b_y$")
	axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
	axes[0].set_ylabel("Normalized bias")
	axes[0].set_title("Wind-specific adaptive biases")
	_legend(axes[0], ncol=2)

	if np.isfinite(phys_x).any():
		axes[1].plot(t, phys_x, linewidth=1.7, label=r"Tilt-corrected physical mean $e_x$")
	if np.isfinite(phys_y).any():
		axes[1].plot(t, phys_y, linewidth=1.7, label=r"Tilt-corrected physical mean $e_y$")
	axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
	axes[1].set_ylabel("Physical center error")
	_legend(axes[1], ncol=2)

	axes[2].plot(t, roll_ff, linewidth=1.9, label="Roll static acceleration trim")
	axes[2].plot(t, pitch_ff, linewidth=1.9, label="Pitch static acceleration trim")
	if np.isfinite(roll_dev).any():
		axes[2].plot(
			t, roll_dev, linestyle=":", alpha=0.70,
			label="Roll DESCENT adaptation from FINAL_PROBE value",
		)
	if np.isfinite(pitch_dev).any():
		axes[2].plot(
			t, pitch_dev, linestyle=":", alpha=0.70,
			label="Pitch DESCENT adaptation from FINAL_PROBE value",
		)
	axes[2].axhline(0.0, linestyle="--", linewidth=1.0)
	axes[2].set_ylabel("Acceleration [m/s²]")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[2], ncol=2)

	_finish_figure(fig, axes, data, out / "wind_biases.png")


def plot_wind_lateral_commands(data: AnalysisData, out: Path) -> None:
	"""Wind-specific lateral command decomposition: static trim + feedback."""
	c = data.control
	if c.empty:
		return

	t = _relative_time(c["_sim_time"], data.t0)
	roll_total, pitch_total = _allocated_lateral_accelerations(c)
	roll_static = _num(
		c, "mission_roll_accel_feedforward_m_s2", 0.0
	).fillna(0.0).to_numpy(float)
	pitch_static = _num(
		c, "mission_pitch_accel_feedforward_m_s2", 0.0
	).fillna(0.0).to_numpy(float)
	roll_dynamic = roll_total - roll_static
	pitch_dynamic = pitch_total - pitch_static

	fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
	for ax, axis_name, total, static, dynamic in [
		(axes[0], "Roll", roll_total, roll_static, roll_dynamic),
		(axes[1], "Pitch", pitch_total, pitch_static, pitch_dynamic),
	]:
		ax.plot(t, total, linewidth=1.9, label=f"Total {axis_name.lower()} acceleration")
		ax.plot(t, static, linestyle=":", linewidth=2.1, label="Static acceleration trim")
		ax.plot(t, dynamic, linestyle="--", linewidth=1.7, label="Dynamic feedback remainder")
		ax.axhline(0.0, linewidth=1.0, linestyle=":")
		ax.set_ylabel("Acceleration [m/s²]")
		_legend(ax, ncol=3)

	axes[0].set_title("Wind-specific lateral command decomposition")
	axes[1].set_xlabel("Time since common log start [s SIM]")
	_finish_figure(fig, axes, data, out / "wind_lateral_commands.png")


def plot_wind_acceleration_contribution(data: AnalysisData, out: Path) -> None:
	"""Reconstruct the WindEffects contribution in all three control channels."""
	c = data.control
	if c.empty:
		return

	t = _relative_time(c["_sim_time"], data.t0)
	tc = c["_sim_time"].to_numpy(float)
	channels = [
		("Vertical / Z", "z", 1.0),
		("Roll channel / -Y", "y", -1.0),
		("Pitch channel / -X", "x", -1.0),
	]

	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
	for ax, (label, truth_axis, truth_sign) in zip(axes, channels):
		relative_accel, wind_compensation, total = _wind_axis_components(
			data,
			tc,
			truth_axis=truth_axis,
			truth_sign=truth_sign,
		)
		if np.isfinite(relative_accel).any():
			ax.plot(
				t, relative_accel, linewidth=1.6,
				label="Drone-platform relative acceleration",
			)
		if np.isfinite(wind_compensation).any():
			ax.plot(
				t, wind_compensation, linewidth=1.8,
				label=(
					"Wind contribution to required control "
					f"(-a_w, k_w={data.wind_force_scale:g})"
				),
			)
		if np.isfinite(total).any():
			ax.plot(
				t, total, linestyle="--", linewidth=1.9,
				label="Relative acceleration + wind compensation",
			)
		ax.axhline(0.0, linestyle=":", linewidth=1.0)
		ax.set_ylabel(f"{label}\n[m/s²]")
		_legend(ax, ncol=3)

	axes[0].set_title("Wind contribution to required acceleration")
	axes[-1].set_xlabel("Time since common log start [s SIM]")
	_finish_figure(fig, axes, data, out / "wind_acceleration_contribution.png")


def plot_vertical_descent(data: AnalysisData, out: Path) -> None:
	"""Vertical geometry/contact diagnostics only; command details live separately."""
	c = data.control
	t = _relative_time(c["_sim_time"], data.t0)
	tc = c["_sim_time"].to_numpy(float)
	h_pad = _interp_truth(data, "truth_min_pad_signed_distance_m", tc)
	h_camera = _interp_truth(data, "truth_camera_normal_distance_m", tc)
	closing = _interp_truth(data, "truth_contact_pad_closing_rate_m_s", tc)

	fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
	axes[0].plot(t, h_pad, label="True minimum skid clearance")
	axes[0].plot(t, h_camera, label="True camera-to-deck distance", alpha=0.8)
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
	axes[1].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[1])
	_finish_figure(fig, axes, data, out / "vertical_descent.png")


def plot_vertical_commands(data: AnalysisData, out: Path) -> None:
	"""Collective-thrust command and its exact geometric decomposition.

	The logged command is the total collective thrust after tilt compensation.
	Projecting it back onto world vertical gives the actual vertical component.
	Therefore

	    u_total = u_hover + Δu_visual + Δu_tilt

	with Δu_visual = u_vertical - u_hover and
	Δu_tilt = u_total - u_vertical.  This uses only logged final commands plus
	the controller's fixed hover-thrust calibration; no truth state enters.
	"""
	c = data.control
	if c.empty:
		return

	t = _relative_time(c["_sim_time"], data.t0)
	thrust = _num(c, "command_thrust").to_numpy(float)
	roll = _num(c, "command_roll_rad").to_numpy(float)
	pitch = _num(c, "command_pitch_rad").to_numpy(float)
	vertical_component = thrust * np.cos(roll) * np.cos(pitch)
	hover = np.full_like(thrust, CONTROL_HOVER_THRUST, dtype=float)
	visual_delta = vertical_component - hover
	tilt_delta = thrust - vertical_component

	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

	# Panel 1: total command and hover baseline only.
	axes[0].plot(t, thrust, linewidth=2.0, label="Collective thrust command")
	axes[0].plot(t, hover, linestyle=":", linewidth=1.8, label="Hover baseline")
	axes[0].set_ylabel("Normalized thrust")
	axes[0].set_title("Vertical command decomposition")
	_legend(axes[0], ncol=2)

	# Panel 2: additive contributions separated from the absolute command.
	axes[1].plot(t, visual_delta, linestyle="--", linewidth=1.8, label="Vertical visual contribution Δu")
	axes[1].plot(t, tilt_delta, linestyle="-.", linewidth=1.8, label="Tilt-compensation contribution Δu")
	axes[1].axhline(0.0, linestyle=":", linewidth=1.0)
	axes[1].set_ylabel("Contribution Δu")
	_legend(axes[1], ncol=2)

	# Panel 3 deliberately shows the integral STATE, matching controller telemetry.
	integ = _num(c, "command_thrust_integral")
	if np.isfinite(integ).any():
		axes[2].plot(t, integ, linewidth=1.8, label="Thrust integral")
	axes[2].axhline(0.0, linestyle=":", linewidth=1.0)
	axes[2].set_ylabel("Divergence integral state")
	axes[2].set_xlabel("Time since common log start [s SIM]")
	_legend(axes[2])

	_finish_figure(fig, axes, data, out / "vertical_commands.png")


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


def plot_platform_angles(data: AnalysisData, out: Path) -> None:
	"""Plot the platform's Gazebo-truth roll, pitch and yaw evolution."""
	tr = data.truth
	t = _relative_time(tr["_sim_time"], data.t0)
	roll, pitch, yaw = _truth_euler_angles(data, "platform")

	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
	for ax, name, angle in zip(
		axes,
		("Roll", "Pitch", "Yaw"),
		(roll, pitch, yaw),
	):
		ax.plot(t, np.degrees(angle), label=f"Platform {name.lower()} truth")
		ax.axhline(0.0, linestyle="--", linewidth=1.0)
		ax.set_ylabel(f"{name} [deg]")
		_legend(ax)
	axes[0].set_title("Gazebo truth: platform attitude evolution")
	axes[-1].set_xlabel("Time since common log start [s SIM]")
	_finish_figure(fig, axes, data, out / "platform_angles.png")


def plot_drone_angles(data: AnalysisData, out: Path) -> None:
	"""Plot true drone attitude together with the commanded Euler angles."""
	tr = data.truth
	c = data.control
	t_truth = _relative_time(tr["_sim_time"], data.t0)
	t_command = _relative_time(c["_sim_time"], data.t0)
	roll, pitch, yaw = _truth_euler_angles(data, "drone")

	fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
	# PX4's pitch-command convention is opposite to the Gazebo Euler pitch sign
	# in this logging setup. Map truth to the command convention only in this
	# command-tracking figure; the platform-attitude plot remains raw Gazebo truth.
	for ax, name, angle, command_column, truth_sign in zip(
		axes,
		("Roll", "Pitch", "Yaw"),
		(roll, pitch, yaw),
		("command_roll_rad", "command_pitch_rad", "command_yaw_rad"),
		(1.0, -1.0, 1.0),
	):
		truth_label = f"True drone {name.lower()}"
		if truth_sign < 0.0:
			truth_label += " (mapped to command sign)"
		ax.plot(
			t_truth,
			np.degrees(truth_sign * angle),
			label=truth_label,
		)
		command = _num(c, command_column).to_numpy(float)
		if np.isfinite(command).any():
			ax.plot(
				t_command,
				np.degrees(command),
				alpha=0.78,
				label=f"Commanded {name.lower()}",
			)
		ax.axhline(0.0, linestyle="--", linewidth=1.0)
		ax.set_ylabel(f"{name} [deg]")
		_legend(ax, ncol=2)
	axes[0].set_title("Drone attitude evolution: Gazebo truth and commanded angles")
	axes[-1].set_xlabel("Time since common log start [s SIM]")
	_finish_figure(fig, axes, data, out / "drone_angles.png")


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


def write_summary(
	data: AnalysisData,
	out: Path,
	controller_path: Path,
	truth_path: Path,
	wind_path: Path,
	*,
	wind_details: bool = False,
) -> None:
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
		"BEE_LAND three-stream analysis",
		"================================",
		f"Controller file: {controller_path}",
		f"Truth file:      {truth_path}",
		f"Wind file:       {wind_path}",
		f"Analysed SIM span: {data.t1 - data.t0:.3f} s ({data.t0:.6f} to {data.t1:.6f})",
		f"Plot-time range: 0.000 to {data.t1 - data.t0:.3f} s SIM",
		f"Unique controller samples: {len(c)}",
		f"Truth samples: {len(tr)}",
		f"Wind samples: {len(data.wind)}",
		f"Wind-specific diagnostics enabled: {'YES' if wind_details else 'NO'}",
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


	if wind_details:
		wind_t = data.wind["_sim_time"].to_numpy(float)
		wind_t = wind_t[np.isfinite(wind_t)]
		wind_dt = np.diff(wind_t)
		wind_dt = wind_dt[np.isfinite(wind_dt) & (wind_dt > 0.0)]
		wind_norm = _num(data.wind, "wind_command_norm_m_s").to_numpy(float)
		wind_norm = wind_norm[np.isfinite(wind_norm)]
		lines.extend(["", "Wind diagnostics", "----------------"])
		lines.append(
			f"WindEffects force approximation scale: {data.wind_force_scale:g}"
		)
		if wind_dt.size:
			lines.append(
				f"Effective wind diagnostic rate in SIM time: "
				f"{1.0/np.mean(wind_dt):.2f} Hz"
			)
		if wind_norm.size:
			lines.append(
				f"Commanded wind speed: mean={np.mean(wind_norm):.3f}, "
				f"min={np.min(wind_norm):.3f}, max={np.max(wind_norm):.3f} m/s"
			)

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

	# Final three-axis feasibility result and immediate failure explanation.
	feasible_values = _num(c, "mission_feasible").to_numpy(float)
	feasible_values = feasible_values[np.isfinite(feasible_values)]
	if feasible_values.size:
		lines.extend([
			"",
			"Landing feasibility",
			"-------------------",
			f"Combined vertical/roll/pitch/tracking verdict: {'FEASIBLE' if feasible_values[-1] > 0.5 else 'INFEASIBLE'}",
		])
		for axis in ("vertical", "roll", "pitch"):
			flag = _last_finite_value(c, f"mission_{axis}_feasible")
			if np.isfinite(flag):
				lines.append(f"{axis.capitalize()} probe: {'OK' if flag > 0.5 else 'FAILED'}")
		reasons = c.get("mission_infeasible_reason", pd.Series("", index=c.index))
		reasons = reasons.fillna("").astype(str).str.strip()
		reasons = reasons[reasons != ""]
		if len(reasons):
			lines.append(f"Reason: {reasons.iloc[-1]}")

	# Height-free bandwidth / trackability telemetry. Current logs expose the
	# exact online estimator for z/x/y.  Report the three channels separately so
	# the summary mirrors the vertical/roll/pitch acceleration-probe structure.
	if any(_logged_visual_mismatch_available(c, axis) for axis in ("z", "x", "y")):
		lines.extend([
			"",
			"Visual mismatch / bandwidth gates",
			"---------------------------------",
			"Z: chi_z = omega_z_dot - omega_z^2",
			"X / roll: chi_x = omega_x_dot - omega_x*omega_z",
			"Y / pitch: chi_y = omega_y_dot - omega_y*omega_z",
		])

		phase = _clean_string(c.get("mission_substate", pd.Series("", index=c.index)))
		descend_mask = (phase == "descend").to_numpy()

		for axis, label in (
			("z", "Z / vertical"),
			("x", "X / roll"),
			("y", "Y / pitch"),
		):
			if not _logged_visual_mismatch_available(c, axis):
				continue

			cols = _visual_mismatch_columns(axis)
			abs_chi = _num(c, cols["abs"]).to_numpy(float)
			live_peak = _num(c, cols["peak"]).to_numpy(float)
			limit = _last_finite_value(c, cols["limit"])
			decision_time, decision_peak, observed_at_decision = _tracking_decision_sample(
				data.controller, axis
			)

			ready_column = cols["ready"]
			if ready_column not in data.controller.columns and axis == "z":
				ready_column = cols["legacy_ready"]
			tracking_ready = _last_finite_value(data.controller, ready_column)

			feasible_column = {
				"z": "mission_tracking_z_feasible",
				"x": "mission_tracking_x_feasible",
				"y": "mission_tracking_y_feasible",
			}[axis]
			if feasible_column not in data.controller.columns and axis == "z":
				feasible_column = "mission_tracking_feasible"
			tracking_feasible = _last_finite_value(
				data.controller, feasible_column
			)

			descend_peak = live_peak[descend_mask & np.isfinite(live_peak)]
			max_descend_peak = (
				float(np.max(descend_peak)) if descend_peak.size else np.nan
			)
			finite_abs = abs_chi[np.isfinite(abs_chi)]

			lines.append(f"{label}:")
			if finite_abs.size:
				lines.append(
					f"  Maximum instantaneous |chi|: {np.max(finite_abs):.4f} s^-2"
				)
			if np.isfinite(decision_peak):
				lines.append(
					f"  FINAL_PROBE decision peak: {decision_peak:.4f} s^-2"
				)
			if np.isfinite(limit):
				lines.append(f"  FINAL_PROBE chi limit: {limit:.4f} s^-2")
			if np.isfinite(decision_time):
				lines.append(
					f"  FINAL_PROBE decision time: {decision_time-data.t0:.3f} s plot time"
				)
			if np.isfinite(observed_at_decision):
				lines.append(
					f"  Observation accumulated at decision: "
					f"{observed_at_decision:.3f} s"
				)
			if np.isfinite(tracking_ready):
				lines.append(
					f"  Gate ready: {'YES' if tracking_ready > 0.5 else 'NO'}"
				)
			if np.isfinite(tracking_feasible):
				lines.append(
					f"  Verdict: "
					f"{'FEASIBLE' if tracking_feasible > 0.5 else 'INFEASIBLE'}"
				)
			if np.isfinite(max_descend_peak):
				lines.append(
					f"  Maximum live chi peak during DESCENT: "
					f"{max_descend_peak:.4f} s^-2 "
					"(diagnostic only; commitment already made)"
				)
	else:
		measured_chi, truth_chi = _visual_mismatch_series(data, c)
		mean_mag, residual, percentile, robust = _visual_mismatch_probe_statistics(
			c, measured_chi
		)
		finite_robust = robust[np.isfinite(robust)]
		if finite_robust.size:
			final_visual_mismatch = float(finite_robust[-1])
			lines.extend([
				"",
				"Visual mismatch diagnostic (legacy reconstruction)",
				"--------------------------------------------------",
				(
					"Definition: chi_z = D_dot - D^2, with D_dot reconstructed "
					"from a causal "
					f"{VISUAL_MISMATCH_DERIVATIVE_WINDOW_SEC:.2f} s linear fit"
				),
				(
					f"FINAL_PROBE reconstructed mismatch: "
					f"{final_visual_mismatch:.4f} s^-2 "
					f"(|mean| + P{VISUAL_MISMATCH_PERCENTILE:.0f} residual)"
				),
				f"Legacy candidate trackability limit: "
				f"{VISUAL_MISMATCH_LIMIT_S2:.4f} s^-2",
			])

	_append_vision_delay_table(data, lines)

	lines += [
		"",
		"Interpretation note:",
		"  Gazebo truth is used directly in SIM time.",
	]
	if wind_details:
		lines += [
			"  Wind diagnostics use the commanded WindController stream in SIM time.",
			"  Wind contribution uses a_w = k_w (W - v_drone) and plots -a_w",
			"  as the control effort required to reject the WindEffects disturbance.",
		]
	(out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_run_triplet(run_dir: Path) -> tuple[Path, Path, Path]:
	"""Find exactly one complete controller/truth/wind run inside ``run_dir``.

	The three diagnostics files share the same suffix, e.g.::

	    bee_controller_20260817_145438.csv
	    bee_truth_20260817_145438.csv
	    bee_wind_20260817_145438.csv

	Folder mode deliberately refuses to guess when more than one complete run is
	present. Put one run per folder or use the explicit three-CSV invocation.
	"""
	run_dir = run_dir.expanduser()
	if not run_dir.is_dir():
		raise ValueError(f"Run path is not a directory: {run_dir}")

	prefixes = {
		"controller": "bee_controller_",
		"truth": "bee_truth_",
		"wind": "bee_wind_",
	}
	runs: dict[str, dict[str, Path]] = {}

	for kind, prefix in prefixes.items():
		for path in run_dir.glob(f"{prefix}*.csv"):
			suffix = path.name[len(prefix):]
			runs.setdefault(suffix, {})[kind] = path

	complete = [
		(suffix, files)
		for suffix, files in runs.items()
		if all(kind in files for kind in ("controller", "truth", "wind"))
	]

	if not complete:
		found = {
			kind: len(list(run_dir.glob(f"{prefix}*.csv")))
			for kind, prefix in prefixes.items()
		}
		raise ValueError(
			"No complete BEE_LAND run triplet found in "
			f"{run_dir}. Found controller={found['controller']}, "
			f"truth={found['truth']}, wind={found['wind']} CSV(s)."
		)

	if len(complete) > 1:
		run_ids = ", ".join(sorted(suffix.removesuffix(".csv") for suffix, _ in complete))
		raise ValueError(
			f"More than one complete run found in {run_dir}: {run_ids}. "
			"Use one run per folder or pass the three CSV files explicitly."
		)

	_, files = complete[0]
	return files["controller"], files["truth"], files["wind"]


def _resolve_input_paths(paths: list[Path]) -> tuple[Path, Path, Path, Path]:
	"""Resolve folder mode or the legacy explicit-three-CSV mode."""
	if len(paths) == 2:
		run_dir, output_dir = paths
		controller_path, truth_path, wind_path = _discover_run_triplet(run_dir)
		return controller_path, truth_path, wind_path, output_dir

	if len(paths) == 4:
		csv_a, csv_b, csv_c, output_dir = paths
		for path in (csv_a, csv_b, csv_c):
			if not path.is_file():
				raise ValueError(f"CSV path does not exist or is not a file: {path}")
		return csv_a, csv_b, csv_c, output_dir

	raise ValueError(
		"Expected either RUN_DIR OUTPUT_DIR or "
		"CSV_A CSV_B CSV_C OUTPUT_DIR."
	)



def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compare BEE_LAND controller, Gazebo truth and WindController logs. "
			"Default mode generates the core mission/control figures; --wind "
			"adds wind-specific bias and acceleration diagnostics; --full adds "
			"lower-level truth/vision plots."
		),
		epilog=(
			"Examples:\n"
			"  py analyse_log.py .\\logs\\run\\ .\\results\\run\n"
			"  py analyse_log.py .\\logs\\wind\\ .\\results\\wind --wind\n"
			"  py analyse_log.py bee_controller_RUN.csv bee_truth_RUN.csv "
			"bee_wind_RUN.csv results --wind --full"
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument(
		"paths",
		type=Path,
		nargs="+",
		help=(
			"Either RUN_DIR OUTPUT_DIR, or "
			"CSV_A CSV_B CSV_C OUTPUT_DIR."
		),
	)
	parser.add_argument(
		"--wind",
		action="store_true",
		help=(
			"Add wind-specific diagnostics on top of the default plots: "
			"adaptive visual/acceleration biases, lateral static/dynamic command "
			"decomposition, and reconstructed WindEffects acceleration contribution."
		),
	)
	parser.add_argument(
		"--wind-force-scale",
		type=float,
		default=1.0,
		metavar="K_W",
		help=(
			"Gazebo WindEffects force_approximation_scaling_factor used by "
			"--wind acceleration reconstruction (default: 1.0)."
		),
	)
	parser.add_argument(
		"--full",
		action="store_true",
		help=(
			"Also generate platform_motion, platform_angles, drone_angles, "
			"relative_motion, and target_detection."
		),
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
	csv_a, csv_b, csv_c, output_dir = _resolve_input_paths(args.paths)

	controller, truth, wind = _load_triplet(csv_a, csv_b, csv_c)
	data = _prepare(
		controller, truth, wind, wind_force_scale=args.wind_force_scale
	)
	data = _clip_max_time(data, args.max_time)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Keep result folders mode-clean when they are reused. Optional plot files
	# from a previous --wind/--full run are removed when that group is disabled.
	legacy_files = ["lateral_control.png"]
	wind_files = [
		"wind_biases.png",
		"wind_lateral_commands.png",
		"wind_acceleration_contribution.png",
	]
	full_files = [
		"platform_motion.png",
		"platform_angles.png",
		"drone_angles.png",
		"relative_motion.png",
		"target_detection.png",
	]
	for filename in legacy_files:
		path = output_dir / filename
		if path.exists():
			path.unlink()
	if not args.wind:
		for filename in wind_files:
			path = output_dir / filename
			if path.exists():
				path.unlink()
	if not args.full:
		for filename in full_files:
			path = output_dir / filename
			if path.exists():
				path.unlink()

	default_plots = [
		("detections_boxes_fov.png", plot_detections_boxes_fov),
		("drone_platform_position.png", plot_drone_platform_position),
		("gain_schedule.png", plot_gain_schedule),
		("lateral_match.png", plot_lateral_match),
		("lateral_decomposition.png", plot_lateral_decomposition),
		("lateral_commands.png", plot_lateral_commands),
		("probe_vertical.png", plot_probe_vertical),
		("probe_roll.png", plot_probe_roll),
		("probe_pitch.png", plot_probe_pitch),
		("visual_mismatch_z_probe.png", plot_visual_mismatch_z_probe),
		("visual_mismatch_x_probe.png", plot_visual_mismatch_x_probe),
		("visual_mismatch_y_probe.png", plot_visual_mismatch_y_probe),
		("vertical_descent.png", plot_vertical_descent),
		("vertical_commands.png", plot_vertical_commands),
		("vertical_divergence.png", plot_vertical_divergence),
	]
	wind_plots = [
		("wind_biases.png", plot_wind_biases),
		("wind_lateral_commands.png", plot_wind_lateral_commands),
		("wind_acceleration_contribution.png", plot_wind_acceleration_contribution),
	]
	full_plots = [
		("platform_motion.png", plot_platform_motion),
		("platform_angles.png", plot_platform_angles),
		("drone_angles.png", plot_drone_angles),
		("relative_motion.png", plot_relative_motion),
		("target_detection.png", plot_target_detection),
	]

	def run_group(group_name: str, plots: list[tuple[str, object]]) -> None:
		if plots:
			print(f"[{group_name}]")
		for filename, fn in plots:
			fn(data, output_dir)
			path = output_dir / filename
			if path.exists():
				print(f"  saved {path}")
			else:
				print(f"  skipped {filename} (required telemetry unavailable)")

	run_group("default", default_plots)
	if args.wind:
		run_group("wind", wind_plots)
	if args.full:
		run_group("full", full_plots)

	paths_by_kind = {}
	for path in (csv_a, csv_b, csv_c):
		paths_by_kind[_classify_file(pd.read_csv(path, nrows=2))] = path
	controller_path = paths_by_kind["controller"]
	truth_path = paths_by_kind["truth"]
	wind_path = paths_by_kind["wind"]
	write_summary(
		data,
		output_dir,
		controller_path,
		truth_path,
		wind_path,
		wind_details=args.wind,
	)
	print(f"saved {output_dir / 'summary.txt'}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except (ValueError, RuntimeError) as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		raise SystemExit(2)
