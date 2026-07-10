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
	- flow_derotation.png                 raw vs de-rotated optical flow (ego-rotation removal)
	- gain_schedule.png
	- divergence_consistency.png          vision divergence vs. kinematic ground truth
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


def truncate_to_duration(df: pd.DataFrame, max_duration_sec: Optional[float], csv_path: str | Path) -> pd.DataFrame:
	"""Drop rows beyond max_duration_sec of WALL-clock run time (t_sec), e.g.
	after accidentally leaving the sim running. Deliberately keys on t_sec
	specifically -- the one column every log has that is always wall-clock
	(see diagnostics_writer.py) -- not on --time-base's chosen column, which
	may be sim-time and run at a different rate under the sim's real-time
	factor (see mission_routine.py's CLOCK note)."""
	if max_duration_sec is None:
		return df
	if "t_sec" not in df.columns:
		print(f"  --max-duration-sec ignored for {csv_path}: no t_sec column in this log.")
		return df

	t_sec = pd.to_numeric(df["t_sec"], errors="coerce")
	keep = t_sec <= float(max_duration_sec)
	n_dropped = int((~keep).sum())
	if n_dropped == 0:
		return df

	truncated = df.loc[keep].reset_index(drop=True)
	if truncated.empty:
		raise ValueError(
			f"--max-duration-sec={max_duration_sec:g} leaves no rows for {csv_path} "
			f"(t_sec starts at {t_sec.min():g}). Check the value or drop the flag for this log."
		)
	print(f"  --max-duration-sec={max_duration_sec:g}: dropped {n_dropped}/{len(df)} rows "
	      f"beyond t_sec={max_duration_sec:g} (log ran to t_sec={t_sec.max():g}).")
	return truncated


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

	_append_mission_summary(df, t, lines)
	_append_touchdown_summary(df, t, lines)
	_append_latency_summary(df, lines)

	return "\n".join(lines) + "\n"


def _last_finite(df: pd.DataFrame, col: str) -> Optional[float]:
	if col not in df.columns:
		return None
	y = numeric_column(df, col)
	y = y[np.isfinite(y)]
	return float(y[-1]) if len(y) else None


def _first_true_index(mask: np.ndarray) -> Optional[int]:
	idx = np.where(np.asarray(mask, dtype=bool))[0]
	return int(idx[0]) if len(idx) else None


def _after_descend_start_mask(df: pd.DataFrame) -> np.ndarray:
	"""Rows at/after the first DESCEND sample, or all rows for older logs.

	Touchdown heuristics should not accidentally trigger on a pre-flight zero
	thrust sample, so when mission_substate is available we only search once the
	mission has actually entered DESCEND.
	"""
	mask = np.ones(len(df), dtype=bool)
	if "mission_substate" not in df.columns:
		return mask
	sub = df["mission_substate"].astype(str).fillna("").to_numpy()
	descend_idxs = np.where(sub == "descend")[0]
	if len(descend_idxs) == 0:
		return mask
	mask[:] = False
	mask[int(descend_idxs[0]):] = True
	return mask


def _detect_touchdown(df: pd.DataFrame, t: np.ndarray) -> Optional[dict]:
	"""Best-effort touchdown detector for log analysis.

	Preferred signal is an explicit touchdown/landed boolean if present. The
	current logs usually expose touchdown indirectly by command_thrust dropping
	to zero after DESCEND, so that is the main fallback. If neither is available,
	use |relative_z| approaching zero after DESCEND.
	"""
	after_descend = _after_descend_start_mask(df)

	for col in (
		"touchdown",
		"touchdown_detected",
		"landed",
		"vehicle_landed",
		"contact_detected",
		"in_contact",
	):
		if col in df.columns:
			idx = _first_true_index(bool_column(df, col) & after_descend)
			if idx is not None:
				return {"idx": idx, "t": float(t[idx]), "source": col}

	if "command_thrust" in df.columns:
		thrust = numeric_column(df, "command_thrust")
		finite = np.isfinite(thrust)
		seen_positive = np.maximum.accumulate(finite & (thrust > 1e-4))
		mask = finite & seen_positive & after_descend & (thrust <= 1e-6)
		idx = _first_true_index(mask)
		if idx is not None:
			return {"idx": idx, "t": float(t[idx]), "source": "command_thrust<=0"}

	if "relative_z_m" in df.columns:
		relz = numeric_column(df, "relative_z_m")
		height = np.abs(relz)
		finite = np.isfinite(height)
		# Avoid triggering on a log that begins already close to zero by requiring
		# that the same post-DESCEND segment previously had a meaningful gap.
		seen_above = np.maximum.accumulate(finite & after_descend & (height > 0.10))
		mask = finite & after_descend & seen_above & (height <= 0.03)
		idx = _first_true_index(mask)
		if idx is not None:
			return {"idx": idx, "t": float(t[idx]), "source": "|relative_z|<=0.03m"}

	return None


def _smooth_until_index(t: np.ndarray, y: np.ndarray, end_idx: Optional[int], window_sec: float = 0.6) -> np.ndarray:
	"""Smooth y only up to end_idx inclusive; keep the rest NaN.

	For touchdown velocity, this deliberately prevents post-contact samples from
	being used by the centered rolling window at the end of the pre-contact trace.
	"""
	y = np.asarray(y, dtype=float)
	y_smooth = np.full_like(y, np.nan, dtype=float)
	if len(y) == 0:
		return y_smooth
	end = len(y) if end_idx is None else max(0, min(len(y), int(end_idx) + 1))
	if end <= 0:
		return y_smooth
	y_smooth[:end] = _rolling_smooth(np.asarray(t[:end], dtype=float), y[:end], window_sec=window_sec)
	return y_smooth


def _touchdown_velocity(df: pd.DataFrame, t: np.ndarray, window_sec: float = 0.6) -> Optional[dict]:
	"""Return raw and smoothed relative z-velocity at touchdown, if detectable."""
	touch = _detect_touchdown(df, t)
	if touch is None or "relative_vz_m_s" not in df.columns:
		return None
	vz = numeric_column(df, "relative_vz_m_s")
	vz_s = _smooth_until_index(t, vz, touch["idx"], window_sec=window_sec)
	idx = int(touch["idx"])
	raw = float(vz[idx]) if idx < len(vz) and np.isfinite(vz[idx]) else float("nan")
	smoothed = float(vz_s[idx]) if idx < len(vz_s) and np.isfinite(vz_s[idx]) else raw
	out = dict(touch)
	out.update({"relative_vz_raw": raw, "relative_vz_smoothed": smoothed, "smooth_window_sec": float(window_sec)})
	if "relative_z_m" in df.columns:
		relz = numeric_column(df, "relative_z_m")
		if idx < len(relz) and np.isfinite(relz[idx]):
			out["relative_z_m"] = float(relz[idx])
	return out


def _append_touchdown_summary(df: pd.DataFrame, t: np.ndarray, lines: list):
	touch = _touchdown_velocity(df, t)
	if touch is None:
		return

	z_note = ""
	if "relative_z_m" in touch:
		z_note = f", relative_z={touch['relative_z_m']:+.3f} m"
	lines.append(
		f"Touchdown relative z-velocity (closing +, smoothed {touch['smooth_window_sec']:.1f}s): "
		f"{touch['relative_vz_smoothed']:+.4f} m/s "
		f"(raw {touch['relative_vz_raw']:+.4f} m/s, t={touch['t']:.3f}s, "
		f"source={touch['source']}{z_note})"
	)


def _append_mission_summary(df: pd.DataFrame, t: np.ndarray, lines: list):
	"""Mission routine: bounds, feasibility verdict, phase durations, and the
	open-loop height-prediction error -- the numbers needed to tune the bounds."""
	if "mission_substate" not in df.columns:
		return
	sub = df["mission_substate"].astype(str).fillna("").to_numpy()
	if not np.any((sub != "") & (sub != "nan")):
		return

	lines.append("")
	lines.append("Mission (probe -> gate -> scheduled-gain descent)")
	lines.append("-------------------------------------------------")

	peak = _last_finite(df, "mission_peak_accel_m_s2")
	k_min = _last_finite(df, "mission_k_min")
	k_explore = _last_finite(df, "mission_k_explore")
	h_crit = _last_finite(df, "mission_h_crit_m")
	feasible = _last_finite(df, "mission_feasible")
	if peak is not None:
		lines.append(f"Probe peak platform accel: {peak:.4f} m/s^2")
	if k_min is not None and k_explore is not None:
		lines.append(f"k_min (Herisse floor): {k_min:.4f}   k_explore (hand-tuned initial gain): {k_explore:.4f}")
	if h_crit is not None:
		lines.append(f"h_crit (gain hits floor): {h_crit:.4f} m")
	if feasible is not None:
		lines.append(f"Feasibility verdict: {'FEASIBLE' if feasible >= 0.5 else 'INFEASIBLE'}")

	spans = _mission_phase_spans(df, t)
	if spans:
		durations: dict = {}
		for t0, t1, s in spans:
			durations[s] = durations.get(s, 0.0) + max(0.0, float(t1 - t0))
		dur_str = ", ".join(f"{s}={d:.1f}s" for s, d in durations.items())
		lines.append(f"Phase durations: {dur_str}")

	if "mission_thrust_gain_k" in df.columns:
		k = numeric_column(df, "mission_thrust_gain_k")
		kf = k[np.isfinite(k)]
		if len(kf):
			lines.append(f"Commanded thrust gain k(t) range: {np.min(kf):.4f} -> {np.max(kf):.4f}")

	if "mission_h_pred_m" in df.columns and "relative_z_m" in df.columns:
		h_pred = numeric_column(df, "mission_h_pred_m")
		truth = np.abs(numeric_column(df, "relative_z_m"))
		mask = np.isfinite(h_pred) & np.isfinite(truth)
		if np.count_nonzero(mask) >= 3:
			err = h_pred[mask] - truth[mask]
			lines.append(
				f"Open-loop height prediction error (h_pred - |relative_z|): "
				f"median={np.median(err):+.3f} m, RMS={np.sqrt(np.mean(err**2)):.3f} m, "
				f"max|err|={np.max(np.abs(err)):.3f} m"
			)
			lines.append(
				"  (large/growing error here means the gain is scheduled at the "
				"wrong height -- use a more pessimistic descent or a live estimator)"
			)

	consistency = _divergence_consistency(df, t)
	if consistency is not None:
		if "onset_t" in consistency:
			lines.append(
				f"Divergence/kinematics decorrelation: SUSTAINED mismatch from "
				f"t={consistency['onset_t']:.1f}s (height={consistency['onset_height']:.2f} m) "
				f"through end of log -- flow_divergence stops reflecting the true closing "
				f"rate there (see divergence_consistency.png). This is a SENSING issue, "
				f"not a gain issue."
			)
		else:
			lines.append(
				"Divergence/kinematics decorrelation: none detected -- flow_divergence "
				"tracked relative_vz/|relative_z| to the end of the log."
			)


def _select_latency_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
	"""Restrict to the regime that actually drives VISION_PROCESSING_LATENCY_
	BUDGET_SEC: DESCEND phase, pre-touchdown. Far-field/on-ground/post-
	touchdown frames are systematically cheaper (smaller or trivial ROI, no
	real motion) and would understate the number that matters if left in.
	Falls back gracefully to the whole log if mission_substate/command_thrust
	aren't present (older logs, or logs from phases before descent)."""
	subset = df
	description = f"all {len(df)} rows"

	if "mission_substate" in df.columns:
		mask = df["mission_substate"].astype(str) == "descend"
		if mask.any():
			subset = df.loc[mask]
			description = f"{len(subset)} DESCEND-phase rows"
			if "command_thrust" in subset.columns:
				thrust = numeric_column(subset, "command_thrust")
				pre_touchdown = thrust > 0
				# Only narrow further if this actually excludes something --
				# an all-True or all-NaN thrust column means the filter has
				# nothing to add, so leave description as just DESCEND.
				if 0 < int(np.sum(pre_touchdown)) < len(subset):
					subset = subset.loc[pre_touchdown]
					description = f"{len(subset)} DESCEND-phase rows, pre-touchdown (command_thrust>0)"

	return subset, description


# Stage-by-stage on_camera timing (see diagnostics_writer.py's timing_stage_*
# columns / bee_node.py's on_camera instrumentation). This is a PERMANENT
# instrumentation feature, not one of the removed investigation-only
# diagnostics -- every log going forward should have these columns.
#
# v2.0 note: target_acquisition/optical_flow moved out of on_camera into a
# separate vision_worker process (see vision_worker.py / bee_node.py's
# _ship_frame_to_vision + _drain_vision_results). timing_stage_target_
# acquisition_ms and timing_stage_optical_flow_ms are therefore EXPECTED
# BLANK in v2.0-and-later logs -- on_camera no longer runs those two calls, so
# there is nothing to time there. They are left in this table (not removed)
# so older, pre-v2.0 logs -- where on_camera ran them inline -- still get a
# complete on_camera breakdown when re-analysed. The honest v2.0 replacement
# numbers are in VISION_WORKER_COLUMNS / _append_vision_worker_summary below:
# worker_*_ms is the same two calls' cost, measured in the worker process
# instead of on_camera, and is NOT part of the on_camera total anymore.
LATENCY_STAGE_COLUMNS = [
	("timing_stage_bridge_ms", "cv_bridge conversion"),
	("timing_stage_rotate_ms", "cv2.rotate"),
	("timing_stage_show_camera_ms", "imshow/waitKey debug window"),
	("timing_stage_body_rate_ms", "body-rate buffer lookup"),
	("timing_stage_target_acquisition_ms", "target_acquisition.update() [pre-v2.0 only]"),
	("timing_stage_optical_flow_ms", "optical_flow.update() [pre-v2.0 only]"),
	("timing_camera_cb_duration_ms", "on_camera TOTAL"),
]

# v2.0 out-of-process vision pipeline (see vision_worker.py). worker_* are the
# worker's own measurement of the two calls it now owns -- comparable to the
# pre-v2.0 timing_stage_target_acquisition_ms/timing_stage_optical_flow_ms
# numbers above, but measured in a different process, so they are reported
# separately rather than folded into the on_camera table (that folding was
# the original v2.0-rollout bug: a worker-side number appearing as an
# on_camera sub-stage that summed to more than on_camera's own total).
# frame_to_available/frame_to_command are the real cross-process latencies
# the pre-v2.0 stage timers never captured -- frame_to_command is the number
# to compare against a pre-v2.0 log's on_camera TOTAL to judge whether moving
# vision out of process actually helped the control loop.
VISION_WORKER_COLUMNS = [
	("timing_worker_target_acquisition_ms", "worker: target_acquisition.update()"),
	("timing_worker_optical_flow_ms", "worker: optical_flow.update()"),
	("timing_frame_to_available_wall_ms", "frame arrival -> result available"),
	("timing_frame_to_command_wall_ms", "frame arrival -> command computed"),
	("timing_vision_result_period_wall_ms", "period between processed results"),
]


def _append_latency_summary(df: pd.DataFrame, lines: list):
	"""Per-stage vision-pipeline latency: mean/median/p95/max in ms, over the
	DESCEND-phase pre-touchdown rows when available (see _select_latency_rows).
	No-op (nothing appended) for older logs that predate this instrumentation."""
	if not any(col in df.columns for col, _ in LATENCY_STAGE_COLUMNS):
		return

	subset, region_description = _select_latency_rows(df)

	table_rows = []
	for col, label in LATENCY_STAGE_COLUMNS:
		if col not in subset.columns:
			continue
		values = numeric_column(subset, col)
		finite = values[np.isfinite(values)]
		if len(finite) == 0:
			continue
		table_rows.append(
			f"{label:40s}{np.mean(finite):8.2f}{np.median(finite):8.2f}"
			f"{np.percentile(finite, 95):8.2f}{np.max(finite):8.2f}"
		)
	if not table_rows:
		return

	lines.append("")
	lines.append("Vision pipeline latency (on_camera stage breakdown, ms)")
	lines.append("--------------------------------------------------------")
	lines.append(f"Computed over: {region_description}")
	lines.append(f"{'stage':40s}{'mean':>8}{'median':>8}{'p95':>8}{'max':>8}")
	lines.extend(table_rows)
	if (
		"timing_stage_target_acquisition_ms" not in subset.columns
		or numeric_column(subset, "timing_stage_target_acquisition_ms")[
			np.isfinite(numeric_column(subset, "timing_stage_target_acquisition_ms"))
		].size == 0
	):
		lines.append(
			"(target_acquisition/optical_flow rows blank/absent above: this is a "
			"v2.0-or-later log -- those two calls now run in vision_worker, not "
			"on_camera. See the vision worker table below.)"
		)

	_append_vision_worker_summary(subset, region_description, lines)

	# Legacy scheduling-delay diagnostic (camera interarrival jitter): only
	# present in logs captured during the in-process-contention investigation
	# and since removed from live instrumentation. Included opportunistically
	# if an old log still has it, so nothing is silently dropped when
	# re-analysing past runs, but absent from current/future logs.
	if "timing_camera_interarrival_jitter_ms" in subset.columns:
		values = numeric_column(subset, "timing_camera_interarrival_jitter_ms")
		finite = values[np.isfinite(values)]
		if len(finite):
			lines.append("")
			lines.append(
				f"(legacy, this log only) camera interarrival jitter: "
				f"mean={np.mean(finite):.2f} median={np.median(finite):.2f} "
				f"p95={np.percentile(finite, 95):.2f} max={np.max(finite):.2f} ms"
			)


def _append_vision_worker_summary(subset: pd.DataFrame, region_description: str, lines: list):
	"""v2.0 out-of-process vision worker: per-call cost measured in the worker
	process, plus the real frame->available and frame->command cross-process
	latencies (see VISION_WORKER_COLUMNS's docstring above). No-op for logs
	that predate v2.0 (none of these columns exist yet)."""
	if not any(col in subset.columns for col, _ in VISION_WORKER_COLUMNS):
		return

	table_rows = []
	for col, label in VISION_WORKER_COLUMNS:
		if col not in subset.columns:
			continue
		values = numeric_column(subset, col)
		finite = values[np.isfinite(values)]
		if len(finite) == 0:
			continue
		table_rows.append(
			f"{label:40s}{np.mean(finite):8.2f}{np.median(finite):8.2f}"
			f"{np.percentile(finite, 95):8.2f}{np.max(finite):8.2f}"
		)
	if not table_rows:
		return

	lines.append("")
	lines.append("Vision worker latency (v2.0 out-of-process pipeline, ms)")
	lines.append("--------------------------------------------------------")
	lines.append(f"Computed over: {region_description}")
	lines.append(f"{'stage':40s}{'mean':>8}{'median':>8}{'p95':>8}{'max':>8}")
	lines.extend(table_rows)

	if "timing_vision_dropped_frames" in subset.columns:
		dropped = numeric_column(subset, "timing_vision_dropped_frames")
		dropped = dropped[np.isfinite(dropped)]
		if len(dropped):
			# Monotonically-increasing counter (bee_node._vision_dropped_frames),
			# so the count over this region is the increase across it, not a
			# mean/max of the running total.
			net_dropped = int(np.max(dropped) - np.min(dropped))
			lines.append(
				f"Frames dropped at the input queue (worker fell behind): {net_dropped} "
				f"over {len(dropped)} rows in this region"
			)


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
	have_flow = any(c in df.columns for c in ["flow_mean_x_norm_s", "flow_mean_y_norm_s"])
	if not any(c in df.columns for c in ["target_offset_x", "target_offset_y", "command_roll_rad", "command_pitch_rad"]):
		print("Skipping lateral control plot. Missing lateral target/command columns.")
		return

	n_rows = 3 if have_flow else 2
	fig, axes = plt.subplots(n_rows, 1, figsize=(11, 6.5 + (2.2 if have_flow else 0)), sharex=True)
	fig.suptitle("Lateral visual control")

	shade_mission_phases(axes[0], df, t)
	if "target_offset_x" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_x"), label="target_offset_x")
	if "target_offset_y" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_y"), label="target_offset_y")
	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("image offset [-]\n(P-term input)")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	flow_axis = 1 if have_flow else None
	if have_flow:
		ax = axes[1]
		shade_mission_phases(ax, df, t)
		if "flow_mean_x_norm_s" in df.columns:
			ax.plot(t, numeric_column(df, "flow_mean_x_norm_s"), label="flow_mean_x_norm", color="tab:green")
		if "flow_mean_y_norm_s" in df.columns:
			ax.plot(t, numeric_column(df, "flow_mean_y_norm_s"), label="flow_mean_y_norm", color="tab:red")
		ax.axhline(0.0, linestyle="--", linewidth=1)
		# Same [-1,1]-per-frame-half-width/height normalization as offset_x/y
		# above (see optical_flow.py / target_acquisition.py) -- units here are
		# that same normalized scale per second, i.e. directly comparable in
		# SPACE to the offset panel, differing only by the /s (this is the
		# D-term input; compare its noise/lag directly against the P-term
		# panel above when investigating derivative-term behavior).
		ax.set_ylabel("normalized flow [1/s]\n(D-term input)")
		ax.grid(True)
		ax.legend(loc="best")

	cmd_axis = axes[-1]
	if "command_roll_rad" in df.columns:
		cmd_axis.plot(t, numeric_column(df, "command_roll_rad"), label="roll command")
	if "command_pitch_rad" in df.columns:
		cmd_axis.plot(t, numeric_column(df, "command_pitch_rad"), label="pitch command")
	shade_mission_phases(cmd_axis, df, t)
	cmd_axis.axhline(0.0, linestyle="--", linewidth=1)
	cmd_axis.set_ylabel("command [rad]")
	cmd_axis.set_xlabel("time [s]")
	cmd_axis.grid(True)
	cmd_axis.legend(loc="best")

	save_current_figure(output_dir, "lateral_control.png")


def plot_vertical_control(df: pd.DataFrame, t: np.ndarray, output_dir: str, divergence_setpoint: Optional[float] = None):
	available = any(c in df.columns for c in ["flow_divergence_1_s", "relative_vz_m_s", "vehicle_vz_m_s", "command_thrust", "relative_z_m"])
	if not available:
		print("Skipping vertical control plot. Missing vertical/divergence/command columns.")
		return

	have_integral = "command_thrust_integral" in df.columns and np.any(
		np.isfinite(numeric_column(df, "command_thrust_integral"))
	)
	n_rows = 4 if have_integral else 3
	fig, axes = plt.subplots(n_rows, 1, figsize=(11, 8 + (2 if have_integral else 0)), sharex=True)
	fig.suptitle("Vertical / divergence control")
	touch = _touchdown_velocity(df, t)

	if "flow_divergence_1_s" in df.columns:
		axes[0].plot(t, numeric_column(df, "flow_divergence_1_s"), label="filtered divergence")
	if "flow_raw_divergence_1_s" in df.columns:
		axes[0].plot(t, numeric_column(df, "flow_raw_divergence_1_s"), label="raw divergence", alpha=0.75)
	# Prefer the actual per-tick commanded setpoint (probe D*=0 -> descent D*,
	# possibly ramping -- see mission_divergence_setpoint_1_s), falling back to
	# the static CLI value for old logs.
	if "mission_divergence_setpoint_1_s" in df.columns and np.any(
		np.isfinite(numeric_column(df, "mission_divergence_setpoint_1_s"))
	):
		axes[0].plot(t, numeric_column(df, "mission_divergence_setpoint_1_s"),
		             linestyle=":", linewidth=1.6, label="commanded D* (logged)")
	elif divergence_setpoint is not None:
		axes[0].axhline(divergence_setpoint, linestyle=":", linewidth=1.4, label=f"setpoint {divergence_setpoint:g}")
	shade_mission_phases(axes[0], df, t)
	if touch is not None:
		axes[0].axvline(touch["t"], linestyle="--", linewidth=1.1, alpha=0.75)
	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("divergence [1/s]")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	left_handles = []
	left_labels = []
	if "relative_vz_m_s" in df.columns:
		vz = numeric_column(df, "relative_vz_m_s")
		touch_idx = touch["idx"] if touch is not None else None
		vz_smooth = _smooth_until_index(t, vz, touch_idx, window_sec=0.6)
		line_raw, = axes[1].plot(t, vz, label="relative_vz raw", alpha=0.35)
		line_smooth, = axes[1].plot(t, vz_smooth, label="relative_vz smoothed until touchdown", linewidth=1.9)
		left_handles.extend([line_raw, line_smooth])
		left_labels.extend([line_raw.get_label(), line_smooth.get_label()])
		if touch is not None and np.isfinite(touch.get("relative_vz_smoothed", np.nan)):
			idx = int(touch["idx"])
			marker = axes[1].scatter(
				[touch["t"]], [touch["relative_vz_smoothed"]],
				marker="o", s=42, zorder=5, label="touchdown vz"
			)
			axes[1].axvline(touch["t"], linestyle="--", linewidth=1.1, alpha=0.75)
			axes[1].annotate(
				f"touchdown\n{touch['relative_vz_smoothed']:+.2f} m/s",
				xy=(touch["t"], touch["relative_vz_smoothed"]),
				xytext=(8, 12), textcoords="offset points", fontsize=8,
			)
			left_handles.append(marker)
			left_labels.append(marker.get_label())
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

	shade_mission_phases(axes[1], df, t)
	axes[1].axhline(0.0, linestyle="--", linewidth=1)
	axes[1].set_ylabel("velocity [m/s]")
	axes[1].grid(True)
	if left_handles or right_handles:
		axes[1].legend(left_handles + right_handles, left_labels + right_labels, loc="best")

	if "command_thrust" in df.columns:
		axes[2].plot(t, numeric_column(df, "command_thrust"), label="thrust command")
		axes[2].axhline(0.73, linestyle="--", linewidth=1, label="hover ref 0.73")
	shade_mission_phases(axes[2], df, t)
	if touch is not None:
		axes[2].axvline(touch["t"], linestyle="--", linewidth=1.1, alpha=0.75)
	axes[2].set_ylabel("thrust [-]")
	axes[2].grid(True)
	axes[2].legend(loc="best")

	if have_integral:
		integral = numeric_column(df, "command_thrust_integral")
		axes[3].plot(t, integral, color="tab:purple", label="divergence integral (raw)")
		shade_mission_phases(axes[3], df, t)
		if touch is not None:
			axes[3].axvline(touch["t"], linestyle="--", linewidth=1.1, alpha=0.75)
		axes[3].axhline(0.0, linestyle="--", linewidth=1, color="0.4")
		finite = integral[np.isfinite(integral)]
		if len(finite):
			# Flag likely clamp saturation: sustained runs pinned at the series'
			# own extreme, which is what windup against divergence_integral_limit
			# looks like from the outside (the limit itself is not logged).
			hi, lo = float(np.nanmax(finite)), float(np.nanmin(finite))
			pinned_hi = np.isclose(integral, hi, atol=max(1e-3, 0.01 * abs(hi)))
			pinned_lo = np.isclose(integral, lo, atol=max(1e-3, 0.01 * abs(lo)))
			frac_pinned = float(np.mean(pinned_hi | pinned_lo))
			note = ""
			if frac_pinned > 0.05 and (hi - lo) > 1e-6:
				note = f"  ({frac_pinned*100:.0f}% of samples pinned near an extreme -- possible clamp saturation)"
			axes[3].set_title(f"range=[{lo:+.3f}, {hi:+.3f}]{note}", fontsize=9)
		axes[3].set_ylabel("thrust_integral_gain_const *\nintegral(divergence error)")
	axes[-1].set_xlabel("time [s]")

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
		("x", "vehicle_x_m", "platform_y_m", lambda v: v + 1.5, "position x [m]"),
		("y", "vehicle_y_m", "platform_x_m", lambda v: v + 1.5, "position y [m]"),
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


def _rolling_smooth(t: np.ndarray, y: np.ndarray, window_sec: float) -> np.ndarray:
	"""Centered rolling median over ~window_sec of samples (sample count
	derived from the median sample spacing, so this doesn't assume a
	uniform dt). Used to separate a SUSTAINED trend from per-sample
	optical-flow/velocity noise -- median rather than mean so a handful of
	spiky outliers (common in relative_vz near the ground, see below)
	don't drag the smoothed curve toward them."""
	dt = median_positive_dt(t)
	if not np.isfinite(dt) or dt <= 1e-9:
		return y.copy()
	n = max(1, int(round(window_sec / dt)))
	if n <= 1:
		return y.copy()
	return pd.Series(y).rolling(window=n, center=True, min_periods=max(1, n // 3)).median().to_numpy()


def _divergence_consistency(df: pd.DataFrame, t: np.ndarray, min_height_m: float = 0.05) -> Optional[dict]:
	"""Compare the VISION-based divergence estimate against a 'physical'
	proxy computed purely from ground-truth kinematics:

		proxy = relative_vz / |relative_z|        (closing_rate / height)

	i.e. the same quantity flow_divergence is meant to estimate from optical
	flow, computed instead from the logged ground truth. This is a sanity
	check on the SENSING pipeline, independent of any control-law gain: if
	flow_divergence and the proxy disagree, the controller is being fed a
	number that no longer reflects reality, and no gain retune fixes that --
	only fixing (or working around) the sensing does.

	Scans for a SUSTAINED mismatch by walking backward from the end of the
	series: if the smoothed |error| is above threshold at the last sample
	and stays above threshold in an unbroken run back to some onset point,
	that onset is reported. This deliberately ignores an isolated mid-descent
	noise spike that recovers -- it is built to catch exactly the terminal,
	does-not-recover-before-touchdown breakdown (close-range optical-flow
	degradation stacking with the gain schedule's own k_min-authority loss
	right at the end of DESCEND), not every noisy sample.

	Returns None if the required columns/data aren't present. Otherwise a
	dict with the raw/smoothed series (for plotting) and, if found,
	onset_t / onset_height for the start of the trailing mismatch run.
	"""
	required = ["flow_divergence_1_s", "relative_vz_m_s", "relative_z_m"]
	if not all(c in df.columns for c in required):
		return None

	div = numeric_column(df, "flow_divergence_1_s")
	vz = numeric_column(df, "relative_vz_m_s")
	relz = numeric_column(df, "relative_z_m")
	if not (np.any(np.isfinite(div)) and np.any(np.isfinite(vz)) and np.any(np.isfinite(relz))):
		return None

	height = np.abs(relz)
	# Floor the denominator, not the numerator: right at touchdown height ->
	# 0 and the true ratio is genuinely unbounded, which is exactly the
	# regime this is meant to expose -- clip only hard enough to keep the
	# proxy finite/plottable, not to hide the blow-up.
	proxy = vz / np.clip(height, min_height_m, None)

	div_s = _rolling_smooth(t, div, window_sec=0.6)
	proxy_s = _rolling_smooth(t, proxy, window_sec=0.6)
	err_s = np.abs(div_s - proxy_s)

	# Flag threshold: the larger of a fixed floor (so a quiet near-zero hold
	# doesn't trip on tiny absolute noise) and a fraction of the proxy's own
	# typical scale over the series (so a fast, large-divergence descent
	# gets a proportionally larger tolerance).
	finite_proxy = proxy_s[np.isfinite(proxy_s)]
	scale = float(np.nanmedian(np.abs(finite_proxy))) if len(finite_proxy) else 0.0
	threshold = max(0.15, 0.75 * scale)

	# Both the merge-gap tolerance and the minimum reportable run length are
	# tied to the smoothing window above (not separate magic numbers): a
	# "sustained" mismatch shouldn't be trusted as sustained if it's shorter
	# than the window used to smooth it, and a gap tolerant enough to bridge
	# a single flicker but not two genuinely separate incidents should be a
	# fraction of that same window.
	smoothing_window_sec = 0.6
	merge_gap_sec = 0.5 * smoothing_window_sec
	min_run_sec = smoothing_window_sec

	onset_idx = None
	finite = np.isfinite(err_s) & np.isfinite(height)
	idxs = np.where(finite)[0]
	if len(idxs) >= 5:
		bad = err_s[idxs] > threshold

		# Group into contiguous "bad" runs, then merge runs separated by a
		# short time gap -- a single sample of accidental agreement (e.g.
		# right at touchdown, when a disarmed/near-zero vz and a near-zero
		# divergence both trivially settle near 0 together for a moment)
		# should not fragment one real sustained breakdown into pieces that
		# each look too short to flag.
		runs = []
		i = 0
		n = len(idxs)
		while i < n:
			if bad[i]:
				j = i
				while j + 1 < n and bad[j + 1]:
					j += 1
				runs.append([i, j])
				i = j + 1
			else:
				i += 1

		merged = []
		for r in runs:
			if merged and (t[idxs[r[0]]] - t[idxs[merged[-1][1]]]) <= merge_gap_sec:
				merged[-1][1] = r[1]
			else:
				merged.append(list(r))

		# Only the LAST merged run can support a "mismatch persists through
		# the end of the log" claim -- if the tail of the data is back
		# within tolerance, there is nothing sustained to report even if an
		# earlier run was long. Require it to reach the final finite sample
		# and to be longer than min_run_sec, so a single blip surviving the
		# merge doesn't get reported as an onset.
		if merged:
			last = merged[-1]
			reaches_end = last[1] == n - 1
			duration = t[idxs[last[1]]] - t[idxs[last[0]]]
			if reaches_end and duration >= min_run_sec:
				onset_idx = idxs[last[0]]

	result = {
		"height": height,
		"proxy": proxy,
		"div_smoothed": div_s,
		"proxy_smoothed": proxy_s,
		"err_smoothed": err_s,
		"threshold": threshold,
	}
	if onset_idx is not None:
		result["onset_t"] = float(t[onset_idx])
		result["onset_height"] = float(height[onset_idx])
	return result


def plot_divergence_consistency(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path):
	"""Vision-based divergence vs. a ground-truth kinematic proxy
	(relative_vz / |relative_z|) across the whole log, with DESCEND
	highlighted. Isolates the SENSING half of the terminal-approach story
	from the gain-schedule half (mission_routine.py's k_min-authority-loss
	note covers the latter): flow_divergence undershooting D* for most of
	DESCEND is the expected, documented gain-schedule gap, but if it instead
	COLLAPSES toward zero/negative while the true closing rate (relative_vz)
	is climbing hard right before touchdown, that's the vision pipeline
	itself decorrelating from reality, not a control response -- retuning
	gains will not fix it."""
	consistency = _divergence_consistency(df, t)
	if consistency is None:
		print("Skipping divergence consistency plot. Missing flow_divergence_1_s / "
		      "relative_vz_m_s / relative_z_m, or no finite data.")
		return

	div = numeric_column(df, "flow_divergence_1_s")
	dstar = numeric_column(df, "mission_divergence_setpoint_1_s") if "mission_divergence_setpoint_1_s" in df.columns else None

	fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
	fig.suptitle("Divergence sensing consistency: vision estimate vs. kinematic ground truth")

	ax = axes[0]
	shade_mission_phases(ax, df, t)
	ax.plot(t, div, color="tab:blue", alpha=0.30, linewidth=1, label="flow_divergence")
	ax.plot(t, consistency["div_smoothed"], color="tab:blue", linewidth=1.8, label="flow_divergence (smoothed)")
	ax.plot(t, consistency["proxy"], color="tab:orange", alpha=0.30, linewidth=1,
	        label="proxy = relative_vz / |relative_z|")
	ax.plot(t, consistency["proxy_smoothed"], color="tab:orange", linewidth=1.8, label="proxy (smoothed)")
	if dstar is not None and np.any(np.isfinite(dstar)):
		ax.plot(t, dstar, linestyle=":", linewidth=1.4, color="0.3", label="commanded D*")
	ax.axhline(0.0, linestyle="--", linewidth=0.8, color="0.5")
	if "onset_t" in consistency:
		ax.axvline(consistency["onset_t"], color="tab:red", linestyle="--", linewidth=1.3)
	ax.set_ylabel("divergence [1/s]")
	ax.grid(True, alpha=0.4)
	ax.legend(loc="best", fontsize=8)

	ax = axes[1]
	shade_mission_phases(ax, df, t)
	ax.plot(t, consistency["err_smoothed"], color="tab:red", linewidth=1.6,
	        label="|flow_divergence - proxy| (smoothed)")
	ax.axhline(consistency["threshold"], linestyle=":", linewidth=1.3, color="0.3",
	           label=f"flag threshold {consistency['threshold']:.2f}")
	if "onset_t" in consistency:
		ax.axvline(consistency["onset_t"], color="tab:red", linestyle="--", linewidth=1.3)
		ax.annotate(
			f"sustained mismatch from\nt={consistency['onset_t']:.1f}s, "
			f"h={consistency['onset_height']:.2f}m",
			xy=(consistency["onset_t"], consistency["threshold"]),
			xytext=(8, 10), textcoords="offset points", fontsize=8, color="tab:red",
		)
	ax.set_ylabel("|error| [1/s]")
	ax.grid(True, alpha=0.4)
	ax.legend(loc="best", fontsize=8)

	ax = axes[2]
	shade_mission_phases(ax, df, t)
	ax.plot(t, consistency["height"], color="tab:purple", linewidth=1.6,
	        label="|relative_z| (height above platform)")
	if "onset_t" in consistency:
		ax.axvline(consistency["onset_t"], color="tab:red", linestyle="--", linewidth=1.3)
		ax.axhline(consistency["onset_height"], color="tab:red", linestyle=":", linewidth=1.1)
	ax.set_ylabel("height [m]")
	ax.set_xlabel("time [s]")
	ax.grid(True, alpha=0.4)
	ax.legend(loc="best", fontsize=8)

	save_current_figure(output_dir, "divergence_consistency.png")


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


def _mission_phase_spans(df: pd.DataFrame, t: np.ndarray):
	"""Yield (t_start, t_end, substate) spans of contiguous mission substate.

	Used to shade probe / descend / infeasible regions on time-axis plots so
	the probe->descend handover (and any abort) is visible on every figure.
	"""
	if "mission_substate" not in df.columns:
		return []
	sub = df["mission_substate"].astype(str).fillna("").to_numpy()
	spans = []
	start_idx = None
	for i in range(len(sub)):
		s = sub[i]
		if s in ("", "nan"):
			if start_idx is not None:
				spans.append((t[start_idx], t[i - 1], sub[start_idx]))
				start_idx = None
			continue
		if start_idx is None:
			start_idx = i
		elif sub[i] != sub[start_idx]:
			# Close at the boundary sample (t[i]) so the next span starts where
			# this one ends -- no one-sample unshaded gap between phases.
			spans.append((t[start_idx], t[i], sub[start_idx]))
			start_idx = i
	if start_idx is not None:
		spans.append((t[start_idx], t[len(sub) - 1], sub[start_idx]))
	return spans


_PHASE_COLORS = {
	"center": ("tab:purple", 0.06),
	"probe": ("tab:blue", 0.06),
	"probe_hold": ("tab:cyan", 0.08),
	"descend": ("tab:green", 0.06),
	"infeasible": ("tab:red", 0.10),
}


def shade_mission_phases(ax, df: pd.DataFrame, t: np.ndarray, label_once: bool = True):
	"""Shade mission substate spans on a time-axis Axes; mark handovers."""
	seen = set()
	for t0, t1, sub in _mission_phase_spans(df, t):
		color, alpha = _PHASE_COLORS.get(sub, ("tab:gray", 0.06))
		lbl = None
		if label_once and sub not in seen:
			lbl = f"phase: {sub}"
			seen.add(sub)
		ax.axvspan(t0, t1, color=color, alpha=alpha, label=lbl, zorder=0)


def plot_gain_schedule(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	"""The probe-driven gain schedule: thrust k(t), lateral scale, and bounds.

	This is the figure that shows the Ho/de Croon profile directly -- k(t)
	should trace an exponential decay in time (== linear in height) clamped
	between k_explore (top) and k_min (floor), and the lateral scale should be
	the same curve normalized to [k_min/k_explore, 1].
	"""
	have = [c for c in ("mission_thrust_gain_k", "mission_lateral_p_scale", "mission_lateral_d_scale") if c in df.columns]
	if not have:
		print("Skipping gain schedule plot. No mission_* gain columns (old log?).")
		return

	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
	fig.suptitle("Probe-driven gain schedule (Herisse floor / de Croon ceiling)")

	# --- thrust gain k(t) with k_min / k_explore reference bands ---
	ax = axes[0]
	shade_mission_phases(ax, df, t)
	if "mission_thrust_gain_k" in df.columns:
		ax.plot(t, numeric_column(df, "mission_thrust_gain_k"), label="k(t) thrust gain", linewidth=1.8)
	for col, style, lbl in (
		("mission_k_explore", (0, (4, 3)), "k_explore (hand-tuned initial gain)"),
		("mission_k_min", (0, (1, 2)), "k_min (Herisse floor)"),
	):
		if col in df.columns:
			y = numeric_column(df, col)
			finite = y[np.isfinite(y)]
			if len(finite):
				ax.axhline(float(np.nanmedian(finite)), linestyle=style, linewidth=1.3, label=lbl)
	ax.set_ylabel("thrust gain k [m/s]")
	ax.grid(True)
	ax.legend(loc="best", fontsize=8)

	# --- lateral P/D scales (INDEPENDENT since the CENTER-phase P/D-split fix;
	# during DESCEND both track k(t)/k_explore identically -- see mission_
	# routine.py's _do_descend, so they overlap there; during CENTER they
	# differ, reflecting the two different historical scale factors being
	# reversed -- see CENTER_LATERAL_P_SCALE/CENTER_LATERAL_D_SCALE) ---
	ax = axes[1]
	shade_mission_phases(ax, df, t)
	if "mission_lateral_p_scale" in df.columns:
		ax.plot(t, numeric_column(df, "mission_lateral_p_scale"), label="lateral P scale", linewidth=1.8)
	if "mission_lateral_d_scale" in df.columns:
		ax.plot(t, numeric_column(df, "mission_lateral_d_scale"), label="lateral D scale", linewidth=1.4, linestyle="--")
	# overlay normalized thrust gain as a cross-check that DESCEND's lateral
	# scales ride the same ramp as thrust (should overlap P/D there)
	if {"mission_thrust_gain_k", "mission_k_explore"}.issubset(df.columns):
		k = numeric_column(df, "mission_thrust_gain_k")
		ke = numeric_column(df, "mission_k_explore")
		with np.errstate(divide="ignore", invalid="ignore"):
			norm = np.where(ke > 1e-9, k / ke, np.nan)
		ax.plot(t, norm, linestyle=":", alpha=0.6, label="k(t)/k_explore (cross-check)")
	ax.set_ylabel("lateral scale [-]")
	ax.set_xlabel("time [s]")
	ax.grid(True)
	ax.legend(loc="best", fontsize=8)

	save_current_figure(output_dir, "gain_schedule.png")


def plot_height_prediction(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	"""Open-loop predicted height h_pred vs measured height above the platform.

	The descent schedules the gain against the OPEN-LOOP prediction
	h(t)=h0*exp(-D* t), NOT a live height estimate, so this is the single most
	important descent-tuning diagnostic: if h_pred and the true height above the
	deck diverge, the clock-scheduled gain is being evaluated at the wrong
	height and the schedule needs a more pessimistic descent assumption (or a
	live estimator).

	IMPORTANT dimensional note: h_pred is height above the PLATFORM (h0 is seeded
	as takeoff_altitude - platform_height). The only correct ground-truth to
	compare against is therefore relative_z_m (vehicle-to-platform gap). |vehicle_z|
	is height above the GROUND and differs from h_pred by the platform height, so
	it is shown only as dashed context and the error panel is suppressed for it --
	comparing the two directly would report a spurious constant bias.
	"""
	if "mission_h_pred_m" not in df.columns:
		print("Skipping height prediction plot. No mission_h_pred_m (old log?).")
		return

	h_pred = numeric_column(df, "mission_h_pred_m")
	if np.count_nonzero(np.isfinite(h_pred)) < 3:
		print("Skipping height prediction plot. No descent rows with h_pred.")
		return

	# Correct ground truth is relative_z (height above the platform). Only then
	# is h_pred - truth a meaningful prediction error.
	truth = None
	truth_label = None
	truth_is_comparable = False
	if "relative_z_m" in df.columns and np.any(np.isfinite(numeric_column(df, "relative_z_m"))):
		truth = np.abs(numeric_column(df, "relative_z_m"))
		truth_label = "|relative_z| (above platform, ground truth)"
		truth_is_comparable = True
	elif "vehicle_z_m" in df.columns:
		# Dimensionally NOT comparable to h_pred (ground- vs platform-referenced).
		truth = np.abs(numeric_column(df, "vehicle_z_m"))
		truth_label = "|vehicle_z| (above GROUND -- offset by platform height)"
		truth_is_comparable = False

	# Median h_crit (the schedule floor) and the time h_pred first reaches it.
	h_crit_val = None
	if "mission_h_crit_m" in df.columns:
		hc = numeric_column(df, "mission_h_crit_m")
		finite = hc[np.isfinite(hc)]
		if len(finite):
			h_crit_val = float(np.nanmedian(finite))
	t_floor = None
	if h_crit_val is not None and h_crit_val > 0:
		below = np.isfinite(h_pred) & (h_pred <= h_crit_val)
		if np.any(below):
			t_floor = float(t[np.argmax(below)])  # first True

	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
	fig.suptitle("Open-loop height prediction vs measured (descent diagnostic)")

	# --- Top: h_pred vs truth, with the h_crit floor and its crossing. ---
	ax = axes[0]
	shade_mission_phases(ax, df, t)
	ax.plot(t, h_pred, color="tab:blue", linewidth=1.9, label="h_pred = h0*exp(-D* t)")
	if truth is not None:
		style = "-" if truth_is_comparable else "--"
		ax.plot(t, truth, color="tab:orange", alpha=0.85, linestyle=style, label=truth_label)
	if h_crit_val is not None:
		ax.axhline(h_crit_val, color="tab:red", linestyle=":", linewidth=1.4,
		           label=f"h_crit = {h_crit_val:.2f} m (schedule floor)")
	if t_floor is not None:
		ax.axvline(t_floor, color="tab:red", linestyle="--", linewidth=1.1, alpha=0.7)
		ax.annotate(f"h_pred=h_crit\n@ t={t_floor:.1f}s", xy=(t_floor, h_crit_val),
		            xytext=(6, 10), textcoords="offset points", fontsize=8, color="tab:red")
	ax.set_ylabel("height above platform [m]")
	ax.grid(True, alpha=0.4)
	ax.legend(loc="best", fontsize=8)

	# --- Bottom: prediction error, ONLY when the comparison is dimensionally valid. ---
	ax = axes[1]
	shade_mission_phases(ax, df, t)
	if truth is not None and truth_is_comparable:
		err = h_pred - truth
		ax.plot(t, err, color="tab:red", linewidth=1.5, label="h_pred - truth")
		ax.axhline(0.0, linestyle="--", linewidth=1, color="0.4")
		finite = err[np.isfinite(err)]
		if len(finite):
			med = float(np.nanmedian(finite))
			rms = float(np.sqrt(np.nanmean(finite ** 2)))
			ax.set_title(f"prediction error over descent: median={med:+.3f} m, RMS={rms:.3f} m  "
			             f"(positive = clock thinks it is higher than it is -> gain decays late)",
			             fontsize=9)
	elif truth is not None:
		ax.text(0.5, 0.5, "error suppressed: only a GROUND-referenced height is logged\n"
		                  "(not comparable to platform-referenced h_pred)",
		        ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.35")
	else:
		ax.text(0.5, 0.5, "no ground-truth height column to compare against",
		        ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.35")
	ax.set_ylabel("error [m]")
	ax.set_xlabel("time [s]")
	ax.grid(True, alpha=0.4)
	ax.legend(loc="best", fontsize=8)

	save_current_figure(output_dir, "height_prediction.png")


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


def plot_flow_derotation(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path):
	"""Raw vs de-rotated optical flow -- the acceptance plot for ego-rotation
	removal (derotation.py / optical_flow.py).

	Reading it: over a segment with strong body RATE but little translation (a
	hover wobble), the de-rotated traces should collapse toward zero while the
	raw traces track the rotation, and the dotted "rotational component removed"
	should overlay the raw trace. The wrong-sign failure is loud here -- a
	de-rotated trace with LARGER amplitude than raw means that axis's rotation is
	being added instead of subtracted; flip that column of R_body_to_optical (see
	derotation.py's validation banner) and re-run.
	"""
	if "flow_mean_x_raw_px_s" not in df.columns and "flow_mean_y_raw_px_s" not in df.columns:
		print("Skipping flow de-rotation plot. No pre-de-rotation columns "
		      "(de-rotation logging off, or an older log).")
		return

	valid = bool_column(df, "flow_valid", default=True)

	def masked(col: str) -> np.ndarray:
		y = numeric_column(df, col)
		return np.where(valid, y, np.nan)

	raw_x, der_x = masked("flow_mean_x_raw_px_s"), masked("flow_mean_x_px_s")
	raw_y, der_y = masked("flow_mean_y_raw_px_s"), masked("flow_mean_y_px_s")

	have_div = "flow_divergence_prederotation_1_s" in df.columns
	n_rows = 3 if have_div else 2
	fig, axes = plt.subplots(n_rows, 1, figsize=(11, 8 if have_div else 6), sharex=True)
	fig.suptitle("Optical-flow de-rotation: raw vs corrected")

	# Flag logs where de-rotation never actually engaged (raw == corrected), so
	# an overlapping plot isn't misread as "correction had no effect".
	frac_derot = float(np.mean(bool_column(df, "flow_derotated", default=False)))
	if frac_derot < 0.01:
		axes[0].set_title(
			"de-rotation inactive in this log (raw == corrected) -- no body "
			"rates buffered, or derotator disabled",
			fontsize=9,
		)

	axes[0].plot(t, raw_x, label="mean flow x -- raw", color="tab:blue", alpha=0.65)
	axes[0].plot(t, der_x, label="mean flow x -- de-rotated", color="tab:blue", linewidth=1.8)
	axes[0].plot(t, raw_x - der_x, label="rotational component removed",
	             color="tab:orange", linestyle=":", alpha=0.85)
	shade_mission_phases(axes[0], df, t)
	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("flow x [px/s]")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	axes[1].plot(t, raw_y, label="mean flow y -- raw", color="tab:green", alpha=0.65)
	axes[1].plot(t, der_y, label="mean flow y -- de-rotated", color="tab:green", linewidth=1.8)
	axes[1].plot(t, raw_y - der_y, label="rotational component removed",
	             color="tab:orange", linestyle=":", alpha=0.85)
	shade_mission_phases(axes[1], df, t)
	axes[1].axhline(0.0, linestyle="--", linewidth=1)
	axes[1].set_ylabel("flow y [px/s]")
	axes[1].grid(True)
	axes[1].legend(loc="best")

	if have_div:
		pre = masked("flow_divergence_prederotation_1_s")
		# Compare against the (unfiltered) de-rotated divergence so both traces
		# are the same estimator, differing only by de-rotation.
		post_col = "flow_raw_divergence_1_s" if "flow_raw_divergence_1_s" in df.columns else "flow_divergence_1_s"
		post = masked(post_col)
		axes[2].plot(t, pre, label="divergence -- pre-de-rotation", color="tab:red", alpha=0.65)
		axes[2].plot(t, post, label="divergence -- de-rotated (unfiltered)", color="tab:red", linewidth=1.8)
		shade_mission_phases(axes[2], df, t)
		axes[2].axhline(0.0, linestyle="--", linewidth=1)
		axes[2].set_ylabel("divergence [1/s]")
		axes[2].grid(True)
		axes[2].legend(loc="best")

	axes[-1].set_xlabel("time [s]")
	save_current_figure(output_dir, "flow_derotation.png")


def make_default_plots(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path, args):
	plot_target_detection_summary(df, t, output_dir)
	plot_detection_boxes_fov(df, t, args.image_width, args.image_height, output_dir, args.max_boxes)
	plot_lateral_control(df, t, output_dir)
	plot_vertical_control(df, t, output_dir, divergence_setpoint=None)
	plot_flow_derotation(df, t, output_dir)
	plot_gain_schedule(df, t, output_dir)
	plot_divergence_consistency(df, t, output_dir)
	plot_height_prediction(df, t, output_dir)
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
		"--max-duration-sec",
		type=float,
		default=None,
		help=(
			"Hard cutoff on wall-clock run duration: rows with t_sec beyond this are "
			"dropped before any analysis/plotting (e.g. after accidentally leaving the "
			"sim running). Uses t_sec specifically (the one WALL-clock column every log "
			"has), regardless of --time-base. Default: no limit."
		),
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
		df = truncate_to_duration(df, args.max_duration_sec, csv_path)
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