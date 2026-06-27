"""
Analyse BEE_LAND diagnostics CSV logs with a coherent timestamp basis.

Default output is intentionally synthetic: a small set of high-signal plots and
one text summary. Extra legacy/detail plots can be enabled with --full.

Usage:

	# One CSV -> write plots directly into results/test9
	python analyse_log.py logs/bee_diagnostics_XXXXXXXX.csv results/test9

	# Whole folder -> write one subfolder per CSV: results/test1, test2, ...
	python analyse_log.py logs

	# Current moving-platform tests
	python analyse_log.py logs/bee_diagnostics_XXXXXXXX.csv results/test9 \
		--platform-frequency-hz 0.2

Default generated files:

	- summary.txt
	- detection_boxes_fov.png             field-of-view reconstruction
	- target_detection_summary.png
	- lateral_control.png
	- vertical_control.png
	- platform_motion_frequency.png       when platform_z_m is available
	- drone_platform_position_xyz.png      drone and platform positions on x/y/z
	- closing_rate_spectrum.png           when relative_vz_m_s or vehicle_vz_m_s is available

Optional with --full:

	- vehicle_position_xyz.png
	- platform_position_xyz.png
	- platform_velocity_xyz.png
	- relative_motion_xyz.png

Timestamp policy:

	The controller now uses image / visual timestamps for target acquisition,
	optical flow, divergence, and control dt. This analyser follows the same
	convention by default: it uses flow_timestamp_sec if valid, then
	target_timestamp_sec, then command_timestamp_sec. It never mixes PX4 epoch
	timestamps with visual timestamps by absolute value. Wall time is used only
	as a fallback and for estimating real-time factor.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# I/O and columns
# ---------------------------------------------------------------------------


def read_log(csv_path: str | Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if df.empty:
		raise ValueError(f"CSV file is empty: {csv_path}")
	return df


def numeric_column(df: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
	if name not in df.columns:
		return np.full(len(df), default, dtype=float)
	return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def bool_column(df: pd.DataFrame, name: str, default: bool = False) -> np.ndarray:
	if name not in df.columns:
		return np.full(len(df), default, dtype=bool)

	raw = df[name]
	if raw.dtype == bool:
		return raw.to_numpy(dtype=bool)
	if raw.dtype == object:
		cleaned = raw.astype(str).str.lower().str.strip()
		return cleaned.isin(["1", "true", "yes", "y"]).to_numpy(dtype=bool)
	return pd.to_numeric(raw, errors="coerce").fillna(0).to_numpy(dtype=float) > 0.5


def ensure_output_dir(output_dir: str | Path):
	os.makedirs(output_dir, exist_ok=True)


def save_current_figure(output_dir: str | Path, filename: str):
	path = Path(output_dir) / filename
	plt.tight_layout()
	plt.savefig(path, dpi=160)
	plt.close()
	print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Timebase handling
# ---------------------------------------------------------------------------


TIME_BASE_COLUMNS = {
	"flow": "flow_timestamp_sec",
	"target": "target_timestamp_sec",
	"command": "command_timestamp_sec",
	"vehicle_px4": "vehicle_px4_timestamp_sec",
	"vehicle": "vehicle_timestamp_sec",
	"wall": "wall_timestamp",
	"t_sec": "t_sec",
}


def _valid_time_column(df: pd.DataFrame, column: str) -> Tuple[bool, np.ndarray, str]:
	if column not in df.columns:
		return False, np.arange(len(df), dtype=float), f"missing {column}"

	raw = numeric_column(df, column)
	finite = np.isfinite(raw)
	if np.count_nonzero(finite) < max(3, int(0.25 * len(df))):
		return False, raw, f"too few finite samples in {column}"

	# Use finite values in row order. Repeated timestamps are tolerated, but the
	# column must have positive elapsed time and mostly non-negative increments.
	values = raw[finite]
	diffs = np.diff(values)
	positive_span = float(np.nanmax(values) - np.nanmin(values))
	if not np.isfinite(positive_span) or positive_span <= 1e-9:
		return False, raw, f"degenerate span in {column}"

	nonnegative_fraction = float(np.mean(diffs >= -1e-9)) if len(diffs) else 1.0
	if nonnegative_fraction < 0.90:
		return False, raw, f"not mostly monotonic in {column}"

	return True, raw, "ok"


def choose_time_base(df: pd.DataFrame, requested: str = "auto") -> Tuple[np.ndarray, str, str]:
	"""Return elapsed time, source column, and a human-readable description."""
	if requested != "auto":
		column = TIME_BASE_COLUMNS[requested]
		ok, raw, reason = _valid_time_column(df, column)
		if not ok:
			raise ValueError(f"Requested time base '{requested}' is invalid: {reason}")
		return _normalize_elapsed(raw), column, f"requested {requested} ({column})"

	# Match the current controller's actual visual timebase first. PX4 time is
	# deliberately late in the list because its absolute epoch may be unrelated
	# to Gazebo image time.
	priority = [
		"flow_timestamp_sec",
		"target_timestamp_sec",
		"command_timestamp_sec",
		"t_sec",
		"wall_timestamp",
		"vehicle_timestamp_sec",
		"vehicle_px4_timestamp_sec",
	]

	reasons = []
	for column in priority:
		ok, raw, reason = _valid_time_column(df, column)
		if ok:
			return _normalize_elapsed(raw), column, f"auto-selected {column}"
		reasons.append(f"{column}: {reason}")

	return np.arange(len(df), dtype=float), "sample_index", "fallback sample index; " + "; ".join(reasons)


def _normalize_elapsed(raw: np.ndarray) -> np.ndarray:
	raw = np.asarray(raw, dtype=float)
	finite = np.isfinite(raw)
	if not np.any(finite):
		return np.arange(len(raw), dtype=float)
	t0 = float(raw[finite][0])
	return raw - t0


def median_positive_dt(t: np.ndarray) -> float:
	finite = np.isfinite(t)
	values = t[finite]
	if len(values) < 2:
		return float("nan")
	diffs = np.diff(values)
	diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
	if len(diffs) == 0:
		return float("nan")
	return float(np.median(diffs))


def estimate_rtf(df: pd.DataFrame, analysis_t: np.ndarray) -> Optional[float]:
	"""Estimate sim/visual time advance divided by wall time advance."""
	wall_t = None
	if "t_sec" in df.columns:
		wall_t = _normalize_elapsed(numeric_column(df, "t_sec"))
	elif "wall_timestamp" in df.columns:
		wall_t = _normalize_elapsed(numeric_column(df, "wall_timestamp"))

	if wall_t is None:
		return None

	sim_dt = np.diff(analysis_t)
	wall_dt = np.diff(wall_t)
	mask = np.isfinite(sim_dt) & np.isfinite(wall_dt) & (sim_dt > 1e-9) & (wall_dt > 1e-9)
	if np.count_nonzero(mask) < 3:
		return None
	return float(np.median(sim_dt[mask] / wall_dt[mask]))


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------


def sine_fit_at_frequency(t: np.ndarray, y: np.ndarray, frequency_hz: float) -> Optional[dict]:
	if frequency_hz is None or frequency_hz <= 0.0:
		return None
	mask = np.isfinite(t) & np.isfinite(y)
	if np.count_nonzero(mask) < 8:
		return None

	tt = t[mask]
	yy = y[mask]
	w = 2.0 * math.pi * frequency_hz
	X = np.column_stack([np.ones_like(tt), np.sin(w * tt), np.cos(w * tt)])
	coeff, *_ = np.linalg.lstsq(X, yy, rcond=None)
	offset, a_sin, b_cos = [float(v) for v in coeff]
	y_hat = X @ coeff
	amp = float(math.hypot(a_sin, b_cos))
	phase = float(math.atan2(b_cos, a_sin))
	rmse = float(np.sqrt(np.mean((yy - y_hat) ** 2)))
	return {
		"frequency_hz": float(frequency_hz),
		"period_s": 1.0 / float(frequency_hz),
		"amplitude": amp,
		"offset": offset,
		"phase_rad": phase,
		"rmse": rmse,
		"t": tt,
		"fit": y_hat,
	}


def dominant_frequency_fft(
	t: np.ndarray,
	y: np.ndarray,
	min_frequency_hz: float = 0.01,
	max_frequency_hz: float = 2.0,
) -> Optional[dict]:
	mask = np.isfinite(t) & np.isfinite(y)
	if np.count_nonzero(mask) < 20:
		return None

	tt = t[mask]
	yy = y[mask]
	order = np.argsort(tt)
	tt = tt[order]
	yy = yy[order]

	span = float(tt[-1] - tt[0])
	dt = median_positive_dt(tt)
	if not np.isfinite(dt) or dt <= 1e-9 or span <= 0.0:
		return None

	n = int(max(32, math.floor(span / dt) + 1))
	t_uniform = np.linspace(tt[0], tt[-1], n)
	y_uniform = np.interp(t_uniform, tt, yy)
	y_uniform = y_uniform - float(np.mean(y_uniform))

	freqs = np.fft.rfftfreq(n, d=(t_uniform[1] - t_uniform[0]))
	spec = np.abs(np.fft.rfft(y_uniform))
	valid = (freqs >= min_frequency_hz) & (freqs <= max_frequency_hz)
	if not np.any(valid):
		return None

	idxs = np.where(valid)[0]
	idx = int(idxs[np.argmax(spec[valid])])
	return {
		"frequency_hz": float(freqs[idx]),
		"period_s": float(1.0 / freqs[idx]) if freqs[idx] > 1e-12 else float("inf"),
		"freqs": freqs,
		"spectrum": spec,
	}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def compute_summary(
	df: pd.DataFrame,
	t: np.ndarray,
	time_column: str,
	time_description: str,
	expected_platform_frequency_hz: Optional[float],
) -> str:
	lines = []
	lines.append("BEE_LAND log summary")
	lines.append("====================")
	lines.append(f"Rows: {len(df)}")
	lines.append(f"Time base: {time_description}")
	lines.append(f"Analysis time span: {np.nanmin(t):.3f} s -> {np.nanmax(t):.3f} s")
	lines.append(f"Median analysis dt: {median_positive_dt(t):.4f} s")

	rtf = estimate_rtf(df, t)
	if rtf is not None:
		lines.append(f"Median visual/sim time per wall time: {rtf:.3f}")

	if "target_found" in df.columns:
		target_found = bool_column(df, "target_found")
		lines.append(f"Target found ratio: {np.mean(target_found):.3f}")
	if "flow_valid" in df.columns:
		flow_valid = bool_column(df, "flow_valid")
		lines.append(f"Flow valid ratio: {np.mean(flow_valid):.3f}")
	if "target_fov_saturated" in df.columns:
		sat = bool_column(df, "target_fov_saturated")
		lines.append(f"FOV saturated samples: {np.count_nonzero(sat)} / {len(sat)}")

	for col, label in [
		("command_roll_rad", "roll command [rad]"),
		("command_pitch_rad", "pitch command [rad]"),
		("command_thrust", "thrust command [-]"),
		("flow_divergence_1_s", "flow divergence [1/s]"),
		("relative_z_m", "relative z [m]"),
		("relative_vz_m_s", "relative vz [m/s]"),
	]:
		if col in df.columns:
			y = numeric_column(df, col)
			finite = y[np.isfinite(y)]
			if len(finite):
				lines.append(
					f"{label}: min={np.min(finite):.4g}, median={np.median(finite):.4g}, max={np.max(finite):.4g}"
				)

	if "platform_z_m" in df.columns:
		z = numeric_column(df, "platform_z_m")
		finite = z[np.isfinite(z)]
		if len(finite):
			lines.append(
				f"Platform z range: {np.min(finite):.4f} m -> {np.max(finite):.4f} m "
				f"(peak-to-peak {np.max(finite) - np.min(finite):.4f} m)"
			)

		dom = dominant_frequency_fft(t, z, min_frequency_hz=0.01, max_frequency_hz=2.0)
		if dom is not None:
			lines.append(
				f"Platform z dominant frequency: {dom['frequency_hz']:.4f} Hz "
				f"(period {dom['period_s']:.3f} s)"
			)

		fit = sine_fit_at_frequency(t, z, expected_platform_frequency_hz)
		if fit is not None:
			lines.append(
				f"Platform z fit at expected frequency {expected_platform_frequency_hz:.4f} Hz: "
				f"amplitude={fit['amplitude']:.4f} m, offset={fit['offset']:.4f} m, "
				f"RMSE={fit['rmse']:.5f} m"
			)

	return "\n".join(lines) + "\n"


def write_summary(output_dir: str | Path, summary: str):
	path = Path(output_dir) / "summary.txt"
	path.write_text(summary, encoding="utf-8")
	print(summary.rstrip())
	print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plotting: synthetic defaults
# ---------------------------------------------------------------------------


def _mark_bool_false(ax, t: np.ndarray, mask_false: np.ndarray, label: str):
	if np.any(mask_false):
		ax.scatter(t[mask_false], np.zeros(np.count_nonzero(mask_false)), marker="x", s=24, label=label)


def plot_target_detection_summary(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	required_any = ["target_offset_x", "target_offset_y", "target_area_fraction", "target_confidence"]
	if not any(c in df.columns for c in required_any):
		print("Skipping target detection summary. No target columns found.")
		return

	target_found = bool_column(df, "target_found", default=True)
	fov_sat = bool_column(df, "target_fov_saturated", default=False)

	fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
	fig.suptitle("Target detection summary")

	if "target_offset_x" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_x"), label="offset x")
	if "target_offset_y" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_y"), label="offset y")
	_mark_bool_false(axes[0], t, ~target_found, "target not found")
	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("offset [-]")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	if "target_area_fraction" in df.columns:
		axes[1].plot(t, numeric_column(df, "target_area_fraction"), label="area fraction")
	if np.any(fov_sat):
		axes[1].scatter(t[fov_sat], numeric_column(df, "target_area_fraction")[fov_sat], marker="o", s=18, label="FOV saturated")
	axes[1].set_ylabel("area fraction [-]")
	axes[1].grid(True)
	axes[1].legend(loc="best")

	if "target_confidence" in df.columns:
		axes[2].plot(t, numeric_column(df, "target_confidence"), label="confidence")
	axes[2].fill_between(t, 0, 1, where=~target_found, alpha=0.15, label="not found")
	axes[2].set_ylim(-0.05, 1.05)
	axes[2].set_ylabel("confidence [-]")
	axes[2].set_xlabel("time [s]")
	axes[2].grid(True)
	axes[2].legend(loc="best")

	save_current_figure(output_dir, "target_detection_summary.png")


def plot_lateral_control(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	if not any(c in df.columns for c in ["target_offset_x", "target_offset_y", "command_roll_rad", "command_pitch_rad"]):
		print("Skipping lateral control plot. Missing lateral target/command columns.")
		return

	fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
	fig.suptitle("Lateral visual control")

	if "target_offset_x" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_x"), label="target_offset_x")
	if "target_offset_y" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_y"), label="target_offset_y")
	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("image offset [-]")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	if "command_roll_rad" in df.columns:
		axes[1].plot(t, numeric_column(df, "command_roll_rad"), label="roll command")
	if "command_pitch_rad" in df.columns:
		axes[1].plot(t, numeric_column(df, "command_pitch_rad"), label="pitch command")
	axes[1].axhline(0.0, linestyle="--", linewidth=1)
	axes[1].set_ylabel("command [rad]")
	axes[1].set_xlabel("time [s]")
	axes[1].grid(True)
	axes[1].legend(loc="best")

	save_current_figure(output_dir, "lateral_control.png")


def plot_vertical_control(df: pd.DataFrame, t: np.ndarray, output_dir: str, divergence_setpoint: Optional[float]):
	available = any(c in df.columns for c in ["flow_divergence_1_s", "relative_vz_m_s", "vehicle_vz_m_s", "command_thrust", "relative_z_m"])
	if not available:
		print("Skipping vertical control plot. Missing vertical/divergence/command columns.")
		return

	fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
	fig.suptitle("Vertical / divergence control")

	if "flow_divergence_1_s" in df.columns:
		axes[0].plot(t, numeric_column(df, "flow_divergence_1_s"), label="filtered divergence")
	if "flow_raw_divergence_1_s" in df.columns:
		axes[0].plot(t, numeric_column(df, "flow_raw_divergence_1_s"), label="raw divergence", alpha=0.75)
	if divergence_setpoint is not None:
		axes[0].axhline(divergence_setpoint, linestyle=":", linewidth=1.4, label=f"setpoint {divergence_setpoint:g}")
	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("divergence [1/s]")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	left_handles = []
	left_labels = []
	if "relative_vz_m_s" in df.columns:
		line, = axes[1].plot(t, numeric_column(df, "relative_vz_m_s"), label="relative_vz closing rate")
		left_handles.append(line)
		left_labels.append(line.get_label())
	elif "vehicle_vz_m_s" in df.columns:
		line, = axes[1].plot(t, numeric_column(df, "vehicle_vz_m_s"), label="vehicle_vz")
		left_handles.append(line)
		left_labels.append(line.get_label())

	right_handles = []
	right_labels = []
	if "relative_z_m" in df.columns:
		ax_alt = axes[1].twinx()
		line, = ax_alt.plot(t, numeric_column(df, "relative_z_m"), label="relative_z", alpha=0.45)
		right_handles.append(line)
		right_labels.append(line.get_label())
		ax_alt.set_ylabel("relative z [m]")

	axes[1].axhline(0.0, linestyle="--", linewidth=1)
	axes[1].set_ylabel("velocity [m/s]")
	axes[1].grid(True)
	if left_handles or right_handles:
		axes[1].legend(left_handles + right_handles, left_labels + right_labels, loc="best")

	if "command_thrust" in df.columns:
		axes[2].plot(t, numeric_column(df, "command_thrust"), label="thrust command")
		axes[2].axhline(0.73, linestyle="--", linewidth=1, label="hover ref 0.73")
	axes[2].set_ylabel("thrust [-]")
	axes[2].set_xlabel("time [s]")
	axes[2].grid(True)
	axes[2].legend(loc="best")

	save_current_figure(output_dir, "vertical_control.png")


def plot_platform_motion_frequency(
	df: pd.DataFrame,
	t: np.ndarray,
	output_dir: str,
	expected_frequency_hz: Optional[float],
):
	if "platform_z_m" not in df.columns:
		print("Skipping platform motion plot. Missing platform_z_m.")
		return

	z = numeric_column(df, "platform_z_m")
	if np.count_nonzero(np.isfinite(z)) < 8:
		print("Skipping platform motion plot. Not enough platform_z_m samples.")
		return

	fit = sine_fit_at_frequency(t, z, expected_frequency_hz)
	dom = dominant_frequency_fft(t, z, min_frequency_hz=0.01, max_frequency_hz=2.0)

	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
	fig.suptitle("Platform vertical motion and frequency")

	axes[0].plot(t, z, label="platform_z_m")
	if fit is not None:
		order = np.argsort(fit["t"])
		axes[0].plot(
			fit["t"][order],
			fit["fit"][order],
			linestyle="--",
			label=(
				f"fit {fit['frequency_hz']:.3f} Hz, "
				f"A={fit['amplitude']:.3f} m, RMSE={fit['rmse']:.4f} m"
			),
		)
	axes[0].set_xlabel("time [s]")
	axes[0].set_ylabel("platform z [m]")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	if dom is not None:
		axes[1].plot(dom["freqs"], dom["spectrum"], label="|FFT(platform_z)|")
		axes[1].axvline(dom["frequency_hz"], linestyle="--", label=f"dominant {dom['frequency_hz']:.4f} Hz")
	if expected_frequency_hz is not None and expected_frequency_hz > 0.0:
		axes[1].axvline(expected_frequency_hz, linestyle=":", label=f"expected {expected_frequency_hz:.4f} Hz")
	axes[1].set_xlim(left=0.0, right=1.0)
	axes[1].set_xlabel("frequency [Hz]")
	axes[1].set_ylabel("|FFT|")
	axes[1].grid(True)
	axes[1].legend(loc="best")

	save_current_figure(output_dir, "platform_motion_frequency.png")



def plot_drone_platform_position_xyz(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path):
	"""Three-panel position comparison: drone and platform x/y/z.

	vehicle_z_m is PX4 NED (up is negative), while platform_z_m is Gazebo/SDF
	ENU (up is positive). For a physically readable overlay, the drone z trace
	is converted to the platform convention as drone_z_enu = -vehicle_z_m.
	This keeps the bottom panel in one common vertical coordinate system.
	"""
	axis_specs = [
		("x", "vehicle_x_m", "platform_x_m", lambda v: v + 1.5, "position x [m]"),
		("y", "vehicle_y_m", "platform_y_m", lambda v: v + 1.5, "position y [m]"),
		("z", "vehicle_z_m", "platform_z_m", lambda v: -v, "position z [m] (up-positive)"),
	]

	has_any = any(
		vehicle_col in df.columns or platform_col in df.columns
		for _, vehicle_col, platform_col, _, _ in axis_specs
	)
	if not has_any:
		print("Skipping drone/platform position plot. No vehicle/platform position columns found.")
		return

	fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
	fig.suptitle("Drone and platform position by axis")

	for ax, (axis_name, vehicle_col, platform_col, vehicle_transform, ylabel) in zip(axes, axis_specs):
		plotted = False

		if vehicle_col in df.columns:
			vehicle = numeric_column(df, vehicle_col)
			if vehicle_transform is not None:
				vehicle = vehicle_transform(vehicle)
				vehicle_label = "drone z = -vehicle_z_m"
			else:
				vehicle_label = f"drone {axis_name}"
			ax.plot(t, vehicle, label=vehicle_label)
			plotted = True

		if platform_col in df.columns:
			platform = numeric_column(df, platform_col)
			ax.plot(t, platform, label=f"platform {axis_name}")
			plotted = True

		ax.axhline(0.0, linestyle="--", linewidth=1)
		ax.set_ylabel(ylabel)
		ax.grid(True)
		if plotted:
			ax.legend(loc="best")

	axes[-1].set_xlabel("time [s]")
	save_current_figure(output_dir, "drone_platform_position_xyz.png")

def plot_closing_rate_spectrum(
	df: pd.DataFrame,
	t: np.ndarray,
	output_dir: str,
	expected_frequency_hz: Optional[float],
):
	if "relative_vz_m_s" in df.columns and not np.all(np.isnan(numeric_column(df, "relative_vz_m_s"))):
		sig = numeric_column(df, "relative_vz_m_s")
		source = "relative_vz_m_s"
	elif "vehicle_vz_m_s" in df.columns:
		sig = numeric_column(df, "vehicle_vz_m_s")
		source = "vehicle_vz_m_s"
	else:
		print("Skipping closing-rate spectrum. No relative_vz_m_s or vehicle_vz_m_s.")
		return

	dom = dominant_frequency_fft(t, sig, min_frequency_hz=0.01, max_frequency_hz=2.0)
	if dom is None:
		print("Skipping closing-rate spectrum. Not enough valid samples.")
		return

	plt.figure(figsize=(11, 5))
	plt.title(f"Spectrum of closing-rate response ({source})")
	plt.plot(dom["freqs"], dom["spectrum"], label=f"|FFT({source})|")
	plt.axvline(dom["frequency_hz"], linestyle="--", label=f"dominant {dom['frequency_hz']:.4f} Hz")
	if expected_frequency_hz is not None and expected_frequency_hz > 0.0:
		plt.axvline(expected_frequency_hz, linestyle=":", label=f"expected platform {expected_frequency_hz:.4f} Hz")
	plt.xlim(left=0.0, right=1.0)
	plt.xlabel("frequency [Hz]")
	plt.ylabel("|FFT|")
	plt.grid(True)
	plt.legend(loc="best")
	save_current_figure(output_dir, "closing_rate_spectrum.png")


# ---------------------------------------------------------------------------
# Optional full/detail plots
# ---------------------------------------------------------------------------


def plot_detection_boxes_fov(
	df: pd.DataFrame,
	t: np.ndarray,
	image_width: int,
	image_height: int,
	output_dir: str | Path,
	max_boxes: int = 120,
):
	required = [
		"target_offset_x",
		"target_offset_y",
		"target_detection_width_px",
		"target_detection_height_px",
	]
	missing = [name for name in required if name not in df.columns]
	if missing:
		print(f"Skipping detection box plot. Missing columns: {missing}")
		return

	target_found = bool_column(df, "target_found", default=True)
	offset_x = numeric_column(df, "target_offset_x")
	offset_y = numeric_column(df, "target_offset_y")
	box_w = numeric_column(df, "target_detection_width_px")
	box_h = numeric_column(df, "target_detection_height_px")

	valid = (
		target_found
		& np.isfinite(offset_x)
		& np.isfinite(offset_y)
		& np.isfinite(box_w)
		& np.isfinite(box_h)
		& (box_w > 0.0)
		& (box_h > 0.0)
	)
	indices = np.where(valid)[0]
	if len(indices) == 0:
		print("Skipping detection box plot. No valid target detections.")
		return
	if len(indices) > max_boxes:
		indices = np.linspace(indices[0], indices[-1], max_boxes).astype(int)

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.set_title("Detection boxes in camera FOV")
	ax.set_xlabel("image x [px]")
	ax.set_ylabel("image y [px]")
	ax.set_xlim(0, image_width)
	ax.set_ylim(image_height, 0)
	ax.set_aspect("equal", adjustable="box")
	ax.add_patch(Rectangle((0, 0), image_width, image_height, fill=False, linewidth=2))
	ax.axvline(image_width / 2.0, linestyle="--", linewidth=1)
	ax.axhline(image_height / 2.0, linestyle="--", linewidth=1)

	cmap = plt.get_cmap("viridis")
	centers_x, centers_y = [], []
	for k, idx in enumerate(indices):
		color = cmap(k / max(len(indices) - 1, 1))
		alpha = 0.25 + 0.75 * k / max(len(indices) - 1, 1)
		cx = (0.5 * offset_x[idx] + 0.5) * image_width
		cy = (0.5 * offset_y[idx] + 0.5) * image_height
		rect = Rectangle(
			(cx - 0.5 * box_w[idx], cy - 0.5 * box_h[idx]),
			box_w[idx],
			box_h[idx],
			fill=False,
			linewidth=1.2,
			edgecolor=(color[0], color[1], color[2], alpha),
		)
		ax.add_patch(rect)
		centers_x.append(cx)
		centers_y.append(cy)

	ax.plot(centers_x, centers_y, marker=".", linewidth=1.2, label="detection center")
	ax.legend(loc="best")
	save_current_figure(output_dir, "detection_boxes_fov.png")


def plot_multi_column(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path, columns: Sequence[Tuple[str, str]], title: str, filename: str):
	available = [(name, label) for name, label in columns if name in df.columns]
	if not available:
		print(f"Skipping {filename}. No required columns found.")
		return

	fig, axes = plt.subplots(len(available), 1, figsize=(11, 2.5 * len(available)), sharex=True)
	if len(available) == 1:
		axes = [axes]
	fig.suptitle(title)

	for ax, (name, label) in zip(axes, available):
		y = numeric_column(df, name)
		ax.plot(t, y, label=label)
		ax.axhline(0.0, linestyle="--", linewidth=1)
		ax.set_ylabel(label)
		ax.grid(True)
		ax.legend(loc="best")
	axes[-1].set_xlabel("time [s]")
	save_current_figure(output_dir, filename)


def make_default_plots(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path, args):
	plot_target_detection_summary(df, t, output_dir)
	plot_detection_boxes_fov(df, t, args.image_width, args.image_height, output_dir, args.max_boxes)
	plot_lateral_control(df, t, output_dir)
	plot_vertical_control(df, t, output_dir, divergence_setpoint=args.divergence_setpoint)
	plot_platform_motion_frequency(df, t, output_dir, expected_frequency_hz=args.platform_frequency_hz)
	plot_drone_platform_position_xyz(df, t, output_dir)
	plot_closing_rate_spectrum(df, t, output_dir, expected_frequency_hz=args.platform_frequency_hz)


def make_full_plots(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path, args):
	plot_multi_column(
		df,
		t,
		output_dir,
		[("vehicle_x_m", "vehicle x [m]"), ("vehicle_y_m", "vehicle y [m]"), ("vehicle_z_m", "vehicle z [m]")],
		"Vehicle position",
		"vehicle_position_xyz.png",
	)
	plot_multi_column(
		df,
		t,
		output_dir,
		[("platform_x_m", "platform x [m]"), ("platform_y_m", "platform y [m]"), ("platform_z_m", "platform z [m]")],
		"Platform position",
		"platform_position_xyz.png",
	)
	plot_multi_column(
		df,
		t,
		output_dir,
		[("platform_vx_m_s", "platform vx [m/s]"), ("platform_vy_m_s", "platform vy [m/s]"), ("platform_vz_m_s", "platform vz [m/s]")],
		"Platform velocity",
		"platform_velocity_xyz.png",
	)
	plot_multi_column(
		df,
		t,
		output_dir,
		[
			("relative_x_m", "relative x [m]"),
			("relative_y_m", "relative y [m]"),
			("relative_z_m", "relative z [m]"),
			("relative_vx_m_s", "relative vx [m/s]"),
			("relative_vy_m_s", "relative vy [m/s]"),
			("relative_vz_m_s", "relative vz [m/s]"),
		],
		"Relative motion",
		"relative_motion_xyz.png",
	)


# ---------------------------------------------------------------------------
# CLI and path handling
# ---------------------------------------------------------------------------


def _looks_like_csv_path(path_like: str) -> bool:
	return Path(path_like).suffix.lower() == ".csv"


def _path_contains_csv(path_like: str) -> bool:
	path = Path(path_like)
	return path.is_dir() and any(child.suffix.lower() == ".csv" for child in path.iterdir())


def _split_inputs_and_output_dir(raw_paths, output_dir_arg: str, default_output_dir: str):
	paths = [str(path) for path in raw_paths]
	output_dir = output_dir_arg

	if output_dir_arg == default_output_dir and len(paths) >= 2:
		last = paths[-1]
		last_is_input = _looks_like_csv_path(last) or _path_contains_csv(last)
		if not last_is_input:
			output_dir = last
			paths = paths[:-1]

	if not paths:
		raise ValueError("No CSV file or log folder was provided.")
	return paths, output_dir


def collect_csv_paths(inputs: Iterable[str]) -> list[Path]:
	csv_paths = []
	for item in inputs:
		path = Path(item)
		if path.is_dir():
			csv_paths.extend(sorted(path.glob("*.csv")))
		elif path.is_file() and path.suffix.lower() == ".csv":
			csv_paths.append(path)
		else:
			raise FileNotFoundError(f"Input is neither a CSV file nor a folder containing CSV logs: {item}")
	if not csv_paths:
		raise FileNotFoundError("No CSV files found in the provided input(s).")
	return csv_paths


def output_dir_for_csv(csv_path: Path, output_root: Path, total_csv_count: int, index: int, explicit_positional_output: bool):
	if total_csv_count == 1 and explicit_positional_output:
		return output_root
	if total_csv_count == 1 and output_root.name.lower().startswith("test"):
		return output_root
	return output_root / f"test{index}"


def parse_args():
	parser = argparse.ArgumentParser(description="Analyse BEE_LAND diagnostics CSV with coherent timestamps.")
	parser.add_argument(
		"paths",
		nargs="+",
		help=(
			"CSV file(s), folder(s) containing CSV logs, and optionally an output folder "
			"as the final positional argument. Examples: analyse_log.py logs/file.csv results/test9 ; analyse_log.py logs"
		),
	)
	parser.add_argument("--image-width", type=int, default=640, help="Camera image width in pixels. Default: 640.")
	parser.add_argument("--image-height", type=int, default=480, help="Camera image height in pixels. Default: 480.")
	parser.add_argument("--max-boxes", type=int, default=120, help="Maximum detection boxes drawn in detection_boxes_fov.png.")
	parser.add_argument("--output-dir", default="results", help="Directory where plots will be saved. Default: results.")
	parser.add_argument(
		"--time-base",
		choices=["auto", "flow", "target", "command", "vehicle_px4", "vehicle", "wall", "t_sec"],
		default="auto",
		help="X-axis time base. Default: auto, preferring visual/flow timestamps.",
	)
	parser.add_argument(
		"--platform-frequency-hz",
		type=float,
		default=0.2,
		help="Expected platform frequency for reference/fitting. Default: 0.2 Hz. Pass 0 to disable reference fit.",
	)
	parser.add_argument(
		"--divergence-setpoint",
		type=float,
		default=0.01,
		help="Divergence setpoint drawn on vertical_control.png. Default: 0.01.",
	)
	parser.add_argument(
		"--full",
		action="store_true",
		help="Generate optional detailed/legacy plots in addition to the concise default set.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	default_output_dir = "results"
	positional_output_requested = (
		args.output_dir == default_output_dir
		and len(args.paths) >= 2
		and not _looks_like_csv_path(args.paths[-1])
		and not _path_contains_csv(args.paths[-1])
	)

	input_paths, output_dir = _split_inputs_and_output_dir(args.paths, args.output_dir, default_output_dir)
	csv_paths = collect_csv_paths(input_paths)
	output_root = Path(output_dir)
	ensure_output_dir(output_root)

	expected_freq = args.platform_frequency_hz if args.platform_frequency_hz and args.platform_frequency_hz > 0.0 else None

	for index, csv_path in enumerate(csv_paths, start=1):
		df = read_log(csv_path)
		t, time_column, time_description = choose_time_base(df, requested=args.time_base)
		output_complete = output_dir_for_csv(csv_path, output_root, len(csv_paths), index, positional_output_requested)
		ensure_output_dir(output_complete)

		print(f"Loaded: {csv_path}")
		print(f"Output directory: {output_complete}")

		summary = compute_summary(df, t, time_column, time_description, expected_freq)
		write_summary(output_complete, summary)

		make_default_plots(df, t, output_complete, args)
		if args.full:
			make_full_plots(df, t, output_complete, args)

		print("Done.")


if __name__ == "__main__":
	main()
