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

Default generated files -- the core loop: did we SEE the target, did the PROBE
measure the platform, did the GATE pick a sane gain window, did the DESCENT do
what was asked:

	- summary.txt
	- target_detection_summary.png        target offsets/area/confidence (phase-shaded)
	- detection_boxes_fov.png             field-of-view reconstruction
	- probe_acceleration.png              what peak_accel is built from, step by step
	- gain_schedule.png                   k(t) inside the [k_min, k_ceiling] window
	- vertical_divergence.png             the loop's ERROR signal: D vs D*, + integral
	- vertical_descent.png                the PHYSICAL outcome: closing rate, height, thrust
	- closing_rate_spectrum.png           when relative_vz_m_s or vehicle_vz_m_s is available

Optional with --full -- narrower or more diagnostic questions. Not less useful,
just not what you look at on every run:

	- lateral_control.png
	- flow_derotation.png                 raw vs de-rotated optical flow (ego-rotation removal)
	- divergence_consistency.png          vision divergence vs. kinematic ground truth
	- height_prediction.png               open-loop h_pred vs measured (diagnostic only)
	- platform_motion_frequency.png       when platform_z_m is available
	- drone_platform_position_xyz.png     drone and platform positions on x/y/z (phase-shaded)
	- vehicle_position_xyz.png
	- platform_position_xyz.png
	- platform_velocity_xyz.png
	- relative_motion_xyz.png

Timestamp policy:

	The controller now uses image / visual timestamps for target acquisition,
	optical flow, divergence, and control dt. This analyser follows the same
	convention by default: it uses flow_source_timestamp_sec if valid, then target_source_timestamp_sec,
	then command_source_flow_timestamp_sec (with legacy aliases for older logs). It never mixes PX4 epoch
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
# Display labels
# ---------------------------------------------------------------------------
#
# ONE place that maps a CSV column to the words that appear on a figure. Plots
# call label_for("mission_k_floor") rather than hardcoding a string, so a name
# is chosen once and every figure agrees. Anything not listed falls back to a
# readable de-snake_cased version of the column, so a new column still plots
# sensibly before it gets a curated name here.
#
# Conventions: state what the quantity IS, then its role in parentheses where
# the role is the reason it is on the plot ("k_min (Herisse floor)"). Units go
# on the axis label, not the series label.

COLUMN_LABELS: dict[str, str] = {
	# --- target / vision ---
	"target_offset_x": "horizontal offset (P-term input)",
	"target_offset_y": "vertical offset (P-term input)",
	"target_area_fraction": "target area fraction",
	"target_confidence": "detection confidence",
	"target_fov_saturated": "FOV saturated (area no longer tracks range)",
	"target_detection_width_px": "detection width",
	"target_detection_height_px": "detection height",

	# --- optical flow ---
	"flow_divergence_1_s": "divergence (filtered, control input)",
	"flow_raw_divergence_1_s": "divergence (raw, unfiltered)",
	"flow_divergence_prederotation_1_s": "divergence before de-rotation",
	"flow_mean_x_norm_s": "horizontal flow (D-term input)",
	"flow_mean_y_norm_s": "vertical flow (D-term input)",
	"flow_mean_x_px_s": "horizontal flow",
	"flow_mean_y_px_s": "vertical flow",
	"flow_mean_x_raw_px_s": "horizontal flow before de-rotation",
	"flow_mean_y_raw_px_s": "vertical flow before de-rotation",
	"flow_fit_quality": "divergence fit quality (weighted R\u00b2)",

	# --- commands ---
	"command_thrust": "thrust command",
	"command_thrust_integral": "divergence integral",
	"command_roll_rad": "roll command",
	"command_pitch_rad": "pitch command",

	# --- vehicle / platform ---
	"vehicle_vz_m_s": "vehicle vertical velocity",
	"relative_z_m": "signed feet/deck coordinate (- above, 0 touchdown, + penetration)",
	"relative_vz_m_s": "closing rate (measured)",
	"platform_z_m": "platform height",
	"platform_vz_m_s": "platform vertical velocity",

	# --- mission: gain window ---
	# The descent gain rides the de Croon ceiling AT LEG HEIGHT; k_min survives
	# only as a hard floor beneath it. Naming keeps that hierarchy legible.
	"mission_thrust_gain_k": "k(t) (scheduled thrust gain)",
	"mission_k_explore": "k_explore (exploration gain, schedule start)",
	"mission_k_min": "k_min (Herisse floor: peak accel / D*)",
	"mission_k_ceiling_leg": "k_ceiling (de Croon limit at leg height)",
	"mission_k_target": "k_target (margin \u00d7 ceiling: what k(t) aims for)",
	"mission_k_floor": "k_floor (schedule asymptote = max(k_min, k_target))",
	"mission_k_probe": "k_probe (near-field probe gain; descent starts here)",
	"mission_k_descend_start": "k_descend_start (= k_probe)",
	"mission_k_over_ceiling_leg": "k(t) / k_ceiling (fraction of the limit used)",
	"mission_ceiling_margin": "ceiling margin",
	"mission_h_crit_m": "h_crit (height where ceiling meets floor)",
	"mission_h_pred_m": "h_pred (open-loop prediction, diagnostic only)",
	"mission_divergence_setpoint_1_s": "D* (commanded divergence)",
	"mission_lateral_p_scale": "lateral P scale",
	"mission_lateral_d_scale": "lateral D scale",

	# --- mission: platform probe ---
	"mission_peak_accel_m_s2": "peak accel (envelope \u2192 gate)",
	"mission_probe_accel_m_s2": "commanded accel",
	"mission_probe_mean_accel_m_s2": "EMA bias removed (hover trim + descent term)",
	"mission_probe_residual_accel_m_s2": "residual |accel \u2212 bias| (the measurement)",
	"mission_probe_percentile_accel_m_s2": "window percentile (envelope target)",
	"mission_probe_peak_accel_at_handoff_m_s2": "peak at far\u2192near handoff",
}


def label_for(column: str) -> str:
	"""Figure label for a CSV column. Falls back to a de-snake_cased name."""
	if column in COLUMN_LABELS:
		return COLUMN_LABELS[column]
	return column.replace("_", " ")


# ---------------------------------------------------------------------------
# I/O and columns
# ---------------------------------------------------------------------------


def read_log(csv_path: str | Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if df.empty:
		raise ValueError(f"CSV file is empty: {csv_path}")
	return normalize_diagnostics_schema(df)


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


def _simulation_time_column(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[str]]:
	"""Best available Gazebo / visual simulation clock.

	Physical positions, velocities and frequencies in SITL must be evaluated
	against simulation time, not host wall time.  The visual timestamps are
	Gazebo-clock timestamps in the current stack.  Repeated values are allowed:
	they simply mean the diagnostics row reused the latest processed frame.
	"""
	for column in ("flow_timestamp_sec", "target_timestamp_sec", "command_timestamp_sec"):
		ok, raw, _ = _valid_time_column(df, column)
		if ok:
			return _normalize_elapsed(raw), column
	return None, None


def estimate_rtf(df: pd.DataFrame, analysis_t: np.ndarray) -> Optional[dict]:
	"""Return effective simulation/wall timing statistics.

	The run-wide ratio includes wall time spent while simulation/frame time is
	stalled.  The active-step median excludes those stalls and is reported only
	as a scheduler diagnostic; it must not be used to rescale physical velocity.
	"""
	sim_t, sim_column = _simulation_time_column(df)
	if sim_t is None:
		return None

	if "t_sec" in df.columns:
		wall_t = _normalize_elapsed(numeric_column(df, "t_sec"))
	elif "wall_timestamp" in df.columns:
		wall_t = _normalize_elapsed(numeric_column(df, "wall_timestamp"))
	else:
		return None

	common = np.isfinite(sim_t) & np.isfinite(wall_t)
	idx = np.where(common)[0]
	if len(idx) < 3:
		return None
	i0, i1 = int(idx[0]), int(idx[-1])
	sim_span = float(sim_t[i1] - sim_t[i0])
	wall_span = float(wall_t[i1] - wall_t[i0])
	if sim_span <= 0.0 or wall_span <= 0.0:
		return None

	dsim = np.diff(sim_t)
	dwall = np.diff(wall_t)
	active = np.isfinite(dsim) & np.isfinite(dwall) & (dsim > 1e-9) & (dwall > 1e-9)
	active_median = float(np.median(dsim[active] / dwall[active])) if np.count_nonzero(active) >= 3 else float("nan")
	return {
		"global": sim_span / wall_span,
		"active_step_median": active_median,
		"sim_span": sim_span,
		"wall_span": wall_span,
		"sim_column": sim_column,
	}


# ---------------------------------------------------------------------------
# TOuchdown helpers
# ---------------------------------------------------------------------------

def _first_true_index(mask: np.ndarray) -> Optional[int]:
	idx = np.where(np.asarray(mask, dtype=bool))[0]
	return int(idx[0]) if len(idx) else None


def _flying_mask(df: pd.DataFrame) -> np.ndarray:
	"""Rows where the vehicle is actually FLYING -- i.e. not the post-touchdown
	LANDED hold.

	Any statistic about the descent (achieved divergence, thrust, tracking error,
	touchdown velocity) must exclude LANDED rows, which are the vehicle sitting on
	the platform at zero thrust. Before the LANDED substate existed those rows were
	labelled "descend" and silently dominated: in one run 1437 of 2065 "descend"
	rows were post-touchdown, and the descent's achieved divergence read 0.012
	instead of its true 0.238.
	"""
	if "mission_substate" not in df.columns:
		return np.ones(len(df), dtype=bool)
	return df["mission_substate"].astype(str).fillna("").to_numpy() != "landed"


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

	# BEST signal: the relative_z=0 SURFACE CROSSING. relative_z is now
	# belly-coherent (feet-to-deck-surface), so its zero-crossing is the true
	# "legs touch the platform" instant -- earlier and better-defined than the
	# LANDED substate / thrust->0 / contact event, all of which latch after the
	# skids have already driven into the deck. We still need a coarse touchdown to
	# search back from, so find the node's own latch first, then refine to the
	# crossing that precedes it.
	coarse = None
	if "mission_substate" in df.columns:
		sub = df["mission_substate"].astype(str).fillna("").to_numpy()
		idx = _first_true_index(sub == "landed")
		if idx is not None:
			coarse = {"idx": idx, "t": float(t[idx]), "source": "mission_substate=landed"}
	if coarse is None and "command_thrust" in df.columns:
		thrust = numeric_column(df, "command_thrust")
		finite = np.isfinite(thrust)
		seen_positive = np.maximum.accumulate(finite & (thrust > 1e-4))
		mask = finite & seen_positive & after_descend & (thrust <= 1e-6)
		idx = _first_true_index(mask)
		if idx is not None:
			coarse = {"idx": idx, "t": float(t[idx]), "source": "command_thrust<=0"}

	if coarse is not None:
		crossing = _surface_crossing_index(df, t, coarse["idx"])
		if crossing is not None:
			return {
				"idx": crossing,
				"t": float(t[crossing]),
				"source": "relative_z=0 surface crossing",
				"contact_idx": coarse["idx"],
				"contact_t": coarse["t"],
				"contact_source": coarse["source"],
			}
		return coarse

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
		seen_above = np.maximum.accumulate(finite & after_descend & (height > 0.10))
		mask = finite & after_descend & seen_above & (height <= 0.03)
		idx = _first_true_index(mask)
		if idx is not None:
			return {"idx": idx, "t": float(t[idx]), "source": "|relative_z|<=0.03m"}

	return None


def _surface_crossing_index(df: pd.DataFrame, t: np.ndarray, contact_idx: int, window_sec: float = 4.0) -> Optional[int]:
	"""Index where the feet first reach the deck surface: relative_z crossing 0
	from below (feet above surface, relative_z<0) to at/through it (relative_z>=0).

	relative_z is now belly-coherent (platform_motion.relative_motion): 0 == feet
	exactly on the surface. That zero-crossing is the TRUE touchdown instant --
	earlier, and physically better-defined, than either the Gazebo contact event or
	the thrust->0 latch, both of which fire after the skids have already driven
	some centimetres into the deck.

	Searches backward from contact_idx over `window_sec`. Returns None if
	relative_z is unavailable or never crosses (old logs, aborted runs), so callers
	fall back to contact_idx.
	"""
	if "relative_z_m" not in df.columns:
		return None
	relz = numeric_column(df, "relative_z_m")
	n = len(relz)
	if contact_idx <= 0 or contact_idx >= n:
		return None
	lo_t = t[contact_idx] - float(window_sec)
	win = np.where((t >= lo_t) & (t <= t[contact_idx]) & np.isfinite(relz))[0]
	if len(win) < 3:
		return None
	# Last index in the window where feet were still above the surface (relz<0)
	# immediately followed by at/through it: that boundary is the crossing.
	below = relz[win] < 0.0
	# Walk from the end back to the last below->notbelow boundary.
	for k in range(len(win) - 1, 0, -1):
		if below[k - 1] and not below[k]:
			return int(win[k])
	# No clean crossing in-window (already through it throughout, or never below).
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


def _physics_time_column(df: pd.DataFrame, t: np.ndarray) -> np.ndarray:
	"""Clock used for physical derivatives and spectra.

	In Gazebo SITL, physical dynamics evolve in simulation time even when the
	host runs slower than real time.  Wall time remains appropriate for CPU
	latency and throughput only.  This function deliberately ignores the chosen
	plot x-axis and retrieves the simulation clock directly.
	"""
	sim_t, _ = _simulation_time_column(df)
	if sim_t is not None:
		return sim_t
	return np.asarray(t, dtype=float)


def _rolling_median_time(t_real: np.ndarray, y: np.ndarray, window_sec: float, center: bool = True) -> np.ndarray:
	"""Rolling median over a window measured in REAL ELAPSED SECONDS (not a
	sample count), for outlier rejection before differentiating.

	Uses np.searchsorted on t_real (assumed sorted, as any time column is) to
	find each row's window boundary by actual timestamp, not by counting a
	fixed number of neighboring rows -- see _rolling_regression_slope's
	docstring for why sample-count windows are wrong here (message arrival on
	/platform/pose is jittery: dt on a real log ranged 2ms to 196ms against a
	20ms median, so a fixed N-sample window silently spanned anywhere from
	0.3s to 1.3s of real time for a nominal "0.6s" window).

	This is a genuine median (not the regression-slope's mean-based formula),
	so a single-sample outlier in y is rejected rather than pulling a rolling
	average toward it -- the "discard outliers" half of the ask; the
	regression slope applied to this pre-filtered series afterward is the
	"increase the smoothing slightly, but properly" half.
	"""
	n = len(t_real)
	out = np.full(n, np.nan)
	t_real = np.asarray(t_real, dtype=float)
	y = np.asarray(y, dtype=float)
	valid = np.isfinite(t_real) & np.isfinite(y)
	for i in range(n):
		if not valid[i]:
			continue
		if center:
			lo_b, hi_b = t_real[i] - window_sec / 2.0, t_real[i] + window_sec / 2.0
		else:
			lo_b, hi_b = t_real[i] - window_sec, t_real[i]
		lo = np.searchsorted(t_real, lo_b, side="left")
		hi = np.searchsorted(t_real, hi_b, side="right")
		seg_valid = valid[lo:hi]
		if not np.any(seg_valid):
			continue
		out[i] = float(np.median(y[lo:hi][seg_valid]))
	return out


def _rolling_regression_slope(t_real: np.ndarray, y: np.ndarray, window_sec: float, center: bool,
                                min_pts: int = 3) -> np.ndarray:
	"""Per-sample slope of a local linear fit of y against REAL time, over a
	window measured in ELAPSED REAL SECONDS -- not an approximated sample
	count. This is "differentiate", done properly: a windowed least-squares
	slope is a low-pass-filtered derivative, robust to per-sample jitter in y,
	unlike a raw point-to-point difference (which divides by a single dt and
	amplifies noise by 1/dt) or unlike smoothing an already-differentiated
	signal (which can't undo distortion from whatever upstream process
	produced that signal's derivative in the first place).

	TIME-based, not sample-count-based: an earlier version converted
	window_sec to a fixed number of rows via one global median dt, then used
	pandas' sample-count rolling. That silently mismatched the requested
	window whenever sampling was irregular -- confirmed on a real log,
	/platform/pose message dt ranged 2ms to 196ms against a 20ms median (5% of
	samples were >3x the median gap), so a nominal "0.6s" window actually
	spanned 0.29s to 1.30s of real time depending on local message density.
	That inconsistency was large enough to visibly distort platform_vz's
	derived curve (it should trace a clean sinusoid; it didn't). Finding each
	row's window by its actual TIMESTAMP (via np.searchsorted on the sorted
	real-time column) fixes this: window_sec now always means real elapsed
	seconds, regardless of how bunched or sparse the underlying samples are.

	Implementation: prefix sums of t, y, t*y, t^2 (and a valid-count), so each
	row's windowed regression is an O(1) subtraction after an O(n) prefix pass
	plus an O(log n) searchsorted -- overall O(n log n), not the O(n*window)
	a naive per-row polyfit loop would cost.

	center=True: centered window (no lag for the bulk of a curve, at the cost
	    of needing future samples).
	center=False: trailing/causal window (every point uses only itself and
	    earlier samples -- needed at a boundary like a touchdown instant, where
	    a centered window would silently use post-touchdown data).
	"""
	n = len(t_real)
	out = np.full(n, np.nan)
	t_real = np.asarray(t_real, dtype=float)
	y = np.asarray(y, dtype=float)
	valid = np.isfinite(t_real) & np.isfinite(y)
	if not np.any(valid):
		return out

	tt = np.where(valid, t_real, 0.0)
	yy = np.where(valid, y, 0.0)
	c_t = np.cumsum(tt)
	c_y = np.cumsum(yy)
	c_ty = np.cumsum(tt * yy)
	c_tt = np.cumsum(tt * tt)
	c_n = np.cumsum(valid.astype(np.int64))

	for i in range(n):
		if not valid[i]:
			continue
		if center:
			lo_b, hi_b = t_real[i] - window_sec / 2.0, t_real[i] + window_sec / 2.0
		else:
			lo_b, hi_b = t_real[i] - window_sec, t_real[i]
		lo = np.searchsorted(t_real, lo_b, side="left")
		hi = np.searchsorted(t_real, hi_b, side="right") - 1
		if hi < lo:
			continue
		cnt = c_n[hi] - (c_n[lo - 1] if lo > 0 else 0)
		if cnt < min_pts:
			continue
		s_t = c_t[hi] - (c_t[lo - 1] if lo > 0 else 0.0)
		s_y = c_y[hi] - (c_y[lo - 1] if lo > 0 else 0.0)
		s_ty = c_ty[hi] - (c_ty[lo - 1] if lo > 0 else 0.0)
		s_tt = c_tt[hi] - (c_tt[lo - 1] if lo > 0 else 0.0)
		mean_t = s_t / cnt
		mean_y = s_y / cnt
		var_t = s_tt / cnt - mean_t * mean_t
		if var_t <= 1e-10:
			continue
		cov_ty = s_ty / cnt - mean_t * mean_y
		out[i] = cov_ty / var_t
	return out


def _derivative_from_position(t_real: np.ndarray, position: np.ndarray, window_sec: float, center: bool,
                                end_idx: Optional[int] = None, prefilter_sec: float = 0.15) -> np.ndarray:
	"""THE two-step estimator: reject outliers in `position` with a short
	rolling median (prefilter_sec, real-time-windowed), THEN take a longer
	rolling regression slope (window_sec) of the filtered position. Used
	everywhere this module needs a velocity derived from a position column
	(closing rate, platform vx/vy/vz) -- see _rolling_regression_slope's
	docstring for why position, not an already-differentiated velocity
	column, is the right thing to differentiate in the first place.

	prefilter_sec should be well SHORTER than window_sec: its only job is to
	reject single-sample position spikes before they enter the regression, not
	to do the smoothing itself (increasing prefilter_sec would start
	distorting the true position trajectory via median-filter flattening,
	rather than just rejecting outliers).

	end_idx truncates exactly like _smooth_until_index: nothing past end_idx
	is used, so a value read at end_idx never looks into the future.
	"""
	n = len(position)
	out = np.full(n, np.nan)
	end = n if end_idx is None else max(0, min(n, int(end_idx) + 1))
	if end <= 0:
		return out
	t_seg = t_real[:end]
	pos_seg = position[:end]
	filtered = _rolling_median_time(t_seg, pos_seg, window_sec=prefilter_sec, center=center)
	out[:end] = _rolling_regression_slope(t_seg, filtered, window_sec=window_sec, center=center)
	return out


def _closing_rate_from_position(df: pd.DataFrame, t: np.ndarray, end_idx: Optional[int] = None,
                                  window_sec: float = 0.6) -> np.ndarray:
	"""THE closing-rate curve: a rolling regression slope of relative_z_m against
	real time, truncated (like _smooth_until_index) so nothing past end_idx is
	used. This supersedes reading/smoothing relative_vz_m_s directly.

	WHY: relative_vz_m_s = vehicle_vz + platform_vz, and platform_vz is itself
	produced upstream by finite-differencing the platform's POSE (see
	platform_motion.py) -- i.e. it is already a numerical derivative before this
	module ever sees it, and differentiation always amplifies noise. Smoothing
	that already-differentiated signal cannot recover what the first
	differentiation distorted (confirmed on a real log: smoothing
	vehicle_vz+platform_vz separately then combining still gave a NEGATIVE
	closing rate at a touchdown instant, opposite the sign a monotonic
	relative_z trend implied). relative_z_m, by contrast, is built from direct
	POSITION measurements (vehicle position estimate + platform pose, neither
	itself a derivative), so differentiating IT -- once, here, in a controlled
	way -- avoids compounding two rounds of differentiation noise. On the same
	log this reads a small but consistently POSITIVE closing rate, in line with
	relative_z's own visibly monotonic trend.

	Uses a CENTERED window for the bulk of the curve (no lag, matching
	_rolling_smooth's convention elsewhere) and truncates at end_idx exactly the
	way _smooth_until_index does, so the value AT end_idx itself only ever used
	data up to and including that sample -- never a look-ahead into the future.

	Delegates to _derivative_from_position, which adds a short outlier-
	rejecting median pre-filter (0.15s) on relative_z_m before the regression
	slope -- discarding single-sample position spikes rather than letting them
	pull the fit.
	"""
	out = np.full(len(t), np.nan)
	if "relative_z_m" not in df.columns:
		return out
	relz = numeric_column(df, "relative_z_m")
	t_real = _physics_time_column(df, t)
	return _derivative_from_position(t_real, relz, window_sec=window_sec, center=True, end_idx=end_idx)



def _touchdown_velocity(df: pd.DataFrame, t: np.ndarray, window_sec: float = 0.6) -> Optional[dict]:
	"""Return the approach (closing) velocity at touchdown, if detectable.

	Reads _closing_rate_from_position -- a rolling regression slope of
	relative_z_m -- at the touchdown instant, rather than smoothing
	relative_vz_m_s directly. See that function's docstring for why: platform_vz
	(and therefore relative_vz) is itself the output of an upstream finite
	difference of the platform's pose, so it is noisier than relative_z, and
	smoothing an already-differentiated signal cannot undo that. This IS the
	same curve plot_vertical_descent draws (see mark_touchdown/annotate), so the
	reported number and the visible curve can never disagree, by construction --
	the same guarantee an earlier relative_vz-smoothing version was meant to
	give, now built on a genuinely low-noise estimator instead of a noisy one.

	relative_vz_raw/relative_vz_smoothed are kept as secondary diagnostic fields
	(the raw point sample and a light smooth of relative_vz_m_s itself) so a
	large disagreement between them and approach_velocity_m_s remains visible in
	the summary -- that disagreement is itself informative (see
	_closing_rate_from_position's docstring).

	Returned sign convention: closing (descending toward the platform) is
	POSITIVE, matching relative_vz_m_s.
	"""
	touch = _detect_touchdown(df, t)
	if touch is None:
		return None

	idx = int(touch["idx"])
	out = dict(touch)

	closing_curve = _closing_rate_from_position(df, t, end_idx=idx, window_sec=window_sec)
	approach = float(closing_curve[idx]) if idx < len(closing_curve) and np.isfinite(closing_curve[idx]) else float("nan")

	raw = smoothed = float("nan")
	if "relative_vz_m_s" in df.columns:
		vz = numeric_column(df, "relative_vz_m_s")
		# _smooth_until_index never lets samples AFTER idx into the window, so this
		# is a causal (trailing-truncated) smooth ending exactly at touchdown.
		vz_s = _smooth_until_index(t, vz, idx, window_sec=window_sec)
		raw = float(vz[idx]) if idx < len(vz) and np.isfinite(vz[idx]) else float("nan")
		smoothed = float(vz_s[idx]) if idx < len(vz_s) and np.isfinite(vz_s[idx]) else float("nan")

	out.update({
		"approach_velocity_m_s": approach,
		"relative_vz_raw": raw,
		"relative_vz_smoothed": smoothed,
		"smooth_window_sec": float(window_sec),
	})

	if "relative_z_m" in df.columns:
		relz = numeric_column(df, "relative_z_m")
		# Penetration is read at the CONTACT latch (skids pressed into the deck),
		# which can be a later index than the reported touchdown when that was
		# refined to an earlier relative_z=0 crossing (see _detect_touchdown).
		pen_idx = int(touch.get("contact_idx", idx))
		if pen_idx < len(relz) and np.isfinite(relz[pen_idx]):
			out["relative_z_m"] = float(relz[pen_idx])
		if idx < len(relz) and np.isfinite(relz[idx]):
			out["relative_z_at_velocity_m"] = float(relz[idx])
	return out


def _append_touchdown_summary(df: pd.DataFrame, t: np.ndarray, lines: list):
	touch = _touchdown_velocity(df, t)
	if touch is None:
		return

	# The APPROACH velocity: a rolling regression slope of relative_z_m (position),
	# read at the touchdown instant -- NOT a smooth of relative_vz_m_s itself. See
	# _closing_rate_from_position's docstring: platform_vz is itself an upstream
	# finite difference of the platform's pose, so relative_vz inherits that
	# differentiation noise, and smoothing it afterward cannot undo it.
	v = touch.get("approach_velocity_m_s", float("nan"))
	raw = touch.get("relative_vz_raw", float("nan"))
	smoothed_vz = touch.get("relative_vz_smoothed", float("nan"))
	lag_note = ""
	if "contact_t" in touch:
		lag = touch["contact_t"] - touch["t"]
		lag_note = f", contact latched +{lag:.2f}s later ({touch.get('contact_source','')})"
	lines.append(
		f"Touchdown approach velocity (closing +): {v:+.4f} m/s "
		f"[relative_z slope, {touch.get('smooth_window_sec', 0.6):.1f}s window, at "
		f"t={touch['t']:.3f}s via {touch['source']}{lag_note}]"
	)
	# relative_vz_m_s (raw/smoothed) is diagnostic context, not the reported
	# number. A large gap here means platform_vz's own noise is significant at
	# this instant -- worth knowing, not worth trusting as the touchdown speed.
	if np.isfinite(raw) and np.isfinite(v) and abs(raw - v) > 0.10:
		lines.append(
			f"  (relative_vz_m_s disagrees: raw {raw:+.4f} m/s, smoothed {smoothed_vz:+.4f} m/s "
			f"-- platform_vz noise at this instant; relative_z slope above is the trusted number)"
		)

	# The PENETRATION reading. relative_z_m is now belly-coherent (see
	# platform_motion.relative_motion): it measures the drone's FEET to the
	# platform's TOP surface, so relative_z = 0 is a clean feet-on-surface
	# touchdown and relative_z > 0 means the skids have gone THROUGH the deck by
	# that many metres. (Convention: negative while descending -- feet still above
	# the surface -- crossing 0 at contact, positive into penetration.)
	if "relative_z_m" in touch:
		z_contact = touch["relative_z_m"]
		penetration = z_contact  # positive = feet below surface
		if penetration > 0.02:
			note = (
				f"  <-- feet {penetration*100:.0f} cm THROUGH the deck: the contact "
				"event latched this far past the surface (detection lag, not a "
				"control error -- the approach above was clean)"
			)
		elif penetration < -0.02:
			note = "  (feet still above the surface at the detected index)"
		else:
			note = "  (clean: feet within +/-2 cm of the surface)"
		lines.append(
			f"Penetration at contact: relative_z = {z_contact:+.3f} m "
			f"(0 = feet on deck surface, + = through it){note}"
		)
	if "relative_z_at_velocity_m" in touch:
		lines.append(
			f"  relative_z at touchdown = {touch['relative_z_at_velocity_m']:+.3f} m "
			f"(feet above surface)"
		)

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
		lines.append(
			f"Simulation clock: {rtf['sim_column']}; simulated {rtf['sim_span']:.3f} s "
			f"in {rtf['wall_span']:.3f} s wall time"
		)
		lines.append(f"Effective run-wide real-time factor (sim/wall): {rtf['global']:.3f}")
		if np.isfinite(rtf["active_step_median"]):
			lines.append(
				f"Active-step median sim/wall ratio: {rtf['active_step_median']:.3f} "
				"(excludes simulation timestamp stalls; do not use for velocity scaling)"
			)

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

	# --- CALIBRATION READOUT for NEAR_FIELD_HEIGHT_M ---------------------------
	# The whole gain schedule is anchored on the height at which the near-field
	# trigger fires (bee_node.NEAR_FIELD_HEIGHT_M): k_probe = margin *
	# k_ceiling(that height). It is a camera-geometry constant, so unlike h0 it is
	# directly measurable -- here it is, from relative_z_m at the FINAL_PROBE
	# transition. Copy the measured number back into NEAR_FIELD_HEIGHT_M.
	if "relative_z_m" in df.columns:
		entry_idx = _first_true_index(sub == "final_probe")
		if entry_idx is not None:
			h_meas = numeric_column(df, "relative_z_m")[entry_idx]
			if np.isfinite(h_meas):
				h_meas = abs(float(h_meas))
				assumed = _last_finite(df, "mission_near_field_height_m")
				line = f"Height at FINAL_PROBE entry (MEASURED): {h_meas:.3f} m"
				if assumed is not None and np.isfinite(assumed) and assumed > 0.0:
					err = 100.0 * (assumed - h_meas) / h_meas
					line += f"   vs NEAR_FIELD_HEIGHT_M = {assumed:.3f} m ({err:+.0f}%)"
					if assumed > h_meas * 1.15:
						line += "\n  WARNING: the anchor is HIGHER than reality -> k_probe was set"
						line += "\n  above the true ceiling at the probe height. Lower it."
				lines.append(line)

	k_probe = _last_finite(df, "mission_k_probe")
	k_ceiling = _last_finite(df, "mission_k_ceiling_leg")
	k_floor = _last_finite(df, "mission_k_floor")
	if k_probe is not None and k_floor is not None:
		lines.append(
			f"Gain walk-down: k_explore {k_explore:.2f} -> k_probe {k_probe:.2f} "
			f"(flat through FINAL_PROBE) -> k_floor {k_floor:.2f}"
		)
	if k_ceiling is not None:
		lines.append(f"k_ceiling at leg height: {k_ceiling:.4f}")

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
		truth = np.maximum(-numeric_column(df, "relative_z_m"), 0.0)
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
				"tracked the relative_z-slope proxy to the end of the log."
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

	# Phase shading: detection quality is only interpretable against WHICH PHASE
	# it happened in -- e.g. area_fraction crossing FOV_NEAR_AREA_FRACTION is what
	# TRIGGERS the FINAL_PROBE boundary drawn here, so the two must be read
	# together.
	for ax in axes:
		shade_mission_phases(ax, df, t, label_once=(ax is axes[0]))
		mark_touchdown(ax, df, t, label_once=(ax is axes[0]))

	if "target_offset_x" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_x"), label=label_for("target_offset_x"))
	if "target_offset_y" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_y"), label=label_for("target_offset_y"))
	_mark_bool_false(axes[0], t, ~target_found, "target not found")
	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("offset [-]")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	if "target_area_fraction" in df.columns:
		axes[1].plot(t, numeric_column(df, "target_area_fraction"), label=label_for("target_area_fraction"))
	if np.any(fov_sat):
		axes[1].scatter(t[fov_sat], numeric_column(df, "target_area_fraction")[fov_sat], marker="o", s=18, label=label_for("target_fov_saturated"))
	axes[1].set_ylabel("area fraction [-]")
	axes[1].grid(True)
	axes[1].legend(loc="best")

	if "target_confidence" in df.columns:
		axes[2].plot(t, numeric_column(df, "target_confidence"), label=label_for("target_confidence"))
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
	mark_touchdown(axes[0], df, t)
	if "target_offset_x" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_x"), label=label_for("target_offset_x"))
	if "target_offset_y" in df.columns:
		axes[0].plot(t, numeric_column(df, "target_offset_y"), label=label_for("target_offset_y"))
	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("image offset [-]\n(P-term input)")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	flow_axis = 1 if have_flow else None
	if have_flow:
		ax = axes[1]
		shade_mission_phases(ax, df, t)
		mark_touchdown(ax, df, t)
		if "flow_mean_x_norm_s" in df.columns:
			ax.plot(t, numeric_column(df, "flow_mean_x_norm_s"), label=label_for("flow_mean_x_norm_s"), color="tab:green")
		if "flow_mean_y_norm_s" in df.columns:
			ax.plot(t, numeric_column(df, "flow_mean_y_norm_s"), label=label_for("flow_mean_y_norm_s"), color="tab:red")
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
		cmd_axis.plot(t, numeric_column(df, "command_roll_rad"), label=label_for("command_roll_rad"))
	if "command_pitch_rad" in df.columns:
		cmd_axis.plot(t, numeric_column(df, "command_pitch_rad"), label=label_for("command_pitch_rad"))
	shade_mission_phases(cmd_axis, df, t)
	mark_touchdown(cmd_axis, df, t)
	cmd_axis.axhline(0.0, linestyle="--", linewidth=1)
	cmd_axis.set_ylabel("command [rad]")
	cmd_axis.set_xlabel("time [s]")
	cmd_axis.grid(True)
	cmd_axis.legend(loc="best")

	save_current_figure(output_dir, "lateral_control.png")


def plot_vertical_divergence(df: pd.DataFrame, t: np.ndarray, output_dir: str, divergence_setpoint: Optional[float] = None):
	"""The vertical loop's ERROR SIGNAL: what divergence we asked for, what we
	measured, and what the integrator did about the gap.

	Split out of the old 4-panel vertical_control figure, which crammed the loop's
	input and its physical outcome onto one unreadable axis stack. This is the
	"what did the controller SEE" half; plot_vertical_descent is the "what did the
	vehicle DO" half.
	"""
	available = any(c in df.columns for c in ["flow_divergence_1_s", "mission_divergence_setpoint_1_s", "command_thrust_integral"])
	if not available:
		print("Skipping vertical divergence plot. Missing divergence columns.")
		return

	have_integral = "command_thrust_integral" in df.columns and np.any(
		np.isfinite(numeric_column(df, "command_thrust_integral"))
	)
	n_rows = 2 if have_integral else 1
	fig, axes = plt.subplots(n_rows, 1, figsize=(11, 4.5 * n_rows), sharex=True, squeeze=False)
	axes = axes[:, 0]
	fig.suptitle("Vertical loop: divergence tracking")

	ax = axes[0]
	if "flow_divergence_1_s" in df.columns:
		ax.plot(t, numeric_column(df, "flow_divergence_1_s"), label=label_for("flow_divergence_1_s"))
	if "flow_raw_divergence_1_s" in df.columns:
		ax.plot(t, numeric_column(df, "flow_raw_divergence_1_s"), label=label_for("flow_raw_divergence_1_s"), alpha=0.75)
	# Prefer the actual per-tick commanded setpoint (probe D*=0 -> descent D*,
	# possibly ramping), falling back to the static CLI value for old logs.
	if "mission_divergence_setpoint_1_s" in df.columns and np.any(
		np.isfinite(numeric_column(df, "mission_divergence_setpoint_1_s"))
	):
		ax.plot(t, numeric_column(df, "mission_divergence_setpoint_1_s"),
		        linestyle=":", linewidth=1.8, color="k", label=label_for("mission_divergence_setpoint_1_s"))
	elif divergence_setpoint is not None:
		ax.axhline(divergence_setpoint, linestyle=":", linewidth=1.4, label=f"setpoint {divergence_setpoint:g}")
	shade_mission_phases(ax, df, t)
	mark_touchdown(ax, df, t)
	ax.axhline(0.0, linestyle="--", linewidth=1, color="0.4")
	ax.set_ylabel("divergence [1/s]")
	ax.grid(True)
	ax.legend(loc="best", fontsize=8)

	# Achieved-vs-commanded, stated numerically, over the FLYING descend rows only
	# (post-touchdown LANDED rows would otherwise drag the mean to zero).
	if {"mission_substate", "mission_divergence_setpoint_1_s", "flow_divergence_1_s"}.issubset(df.columns):
		sub = df["mission_substate"].astype(str).fillna("").to_numpy()
		d = (sub == "descend") & _flying_mask(df)
		if np.any(d):
			meas = numeric_column(df, "flow_divergence_1_s")[d]
			cmd = numeric_column(df, "mission_divergence_setpoint_1_s")[d]
			meas_m, cmd_m = np.nanmedian(meas), np.nanmedian(cmd)
			if np.isfinite(meas_m) and np.isfinite(cmd_m) and abs(cmd_m) > 1e-6:
				ax.set_title(
					f"DESCEND: measured D = {meas_m:+.3f} /s  vs  commanded D* = {cmd_m:+.3f} /s "
					f"({100.0 * meas_m / cmd_m:.0f}%)",
					fontsize=9,
				)

	if have_integral:
		ax = axes[1]
		integral = numeric_column(df, "command_thrust_integral")
		ax.plot(t, integral, color="tab:purple", label=label_for("command_thrust_integral"))
		shade_mission_phases(ax, df, t)
		mark_touchdown(ax, df, t)
		ax.axhline(0.0, linestyle="--", linewidth=1, color="0.4")
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
			ax.set_title(f"range=[{lo:+.3f}, {hi:+.3f}]{note}", fontsize=9)
		ax.set_ylabel("thrust_integral_gain_const *\nintegral(divergence error)")
		ax.grid(True)
		ax.legend(loc="best", fontsize=8)

	axes[-1].set_xlabel("time [s]")
	save_current_figure(output_dir, "vertical_divergence.png")


def plot_vertical_descent(df: pd.DataFrame, t: np.ndarray, output_dir: str):
    """Physical descent profile with online/offline estimates separated.

    Schema-3 logs use the synchronized native-stream model. The online composed
    closing rate remains visible but no longer controls the physical-outcome
    panel's scale. Robust viewport limits are disclosed and residual outliers are
    marked without modifying any source sample.
    """
    model = _synchronized_vertical_model(df)
    if model is None:
        # Preserve the established legacy behavior for logs without native clocks.
        available = any(c in df.columns for c in ["relative_vz_m_s", "vehicle_vz_m_s", "relative_z_m", "command_thrust"])
        if not available:
            print("Skipping vertical descent plot. Missing velocity/height/thrust columns.")
            return
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        fig.suptitle("Vertical loop: descent profile and thrust")
        if "relative_vz_m_s" in df.columns:
            raw = numeric_column(df, "relative_vz_m_s")
            derived = _closing_rate_from_position(df, t, window_sec=0.6)
            axes[0].plot(t, raw, alpha=0.22, label="online composed closing rate")
            axes[0].plot(t, derived, linewidth=1.9, label="closing rate from relative-position slope")
            _apply_robust_limits(axes[0], raw, derived)
        if "relative_z_m" in df.columns:
            alt = axes[0].twinx()
            alt.plot(t, -numeric_column(df, "relative_z_m"), alpha=0.45, label="height above deck")
            alt.set_ylabel("height [m]")
        if "command_thrust" in df.columns:
            axes[1].plot(t, numeric_column(df, "command_thrust"), label="thrust command")
        for k, ax in enumerate(axes):
            shade_mission_phases(ax, df, t, label_once=(k == 0))
            mark_touchdown(ax, df, t, label_once=(k == 0))
            ax.grid(True); ax.legend(fontsize=8)
        axes[0].set_ylabel("closing rate [m/s]")
        axes[1].set_ylabel("thrust [-]"); axes[1].set_xlabel("time [s]")
        save_current_figure(output_dir, "vertical_descent.png")
        return

    grid = model["t"]
    closing = model["closing_rate"]
    height = -model["relative_z"]
    sim_row, _ = _simulation_time_column(df)
    composed = numeric_column(df, "relative_vz_m_s") if "relative_vz_m_s" in df.columns else np.full(len(df), np.nan)
    if sim_row is not None:
        raw_col = next((c for c in ("flow_source_timestamp_sec", "flow_timestamp_sec") if c in df.columns and np.isfinite(numeric_column(df, c)).any()), None)
        origin_sim = numeric_column(df, raw_col)[np.isfinite(numeric_column(df, raw_col))][0] if raw_col else model["t_abs"][0]
        composed_g = _interp_no_extrapolation(sim_row + origin_sim, composed, model["t_abs"])
    else:
        composed_g = np.full(len(grid), np.nan)
    residual = composed_g - closing
    outliers = _hampel_mask(grid, residual, window_sec=1.0, threshold_sigma=4.0)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11.5), sharex=True)
    fig.suptitle("Vertical loop: physical descent and online-estimate consistency")

    # Physical outcome: trusted offline closing rate and actual relative height.
    axes[0].plot(grid, closing, linewidth=1.9, label="offline closing rate from synchronized position (0.60 s)")
    alt = axes[0].twinx()
    alt.plot(grid, height, alpha=0.48, label="height above deck")
    alt.set_ylabel("height above deck [m]")
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("closing rate [m/s]")
    _apply_robust_limits(axes[0], closing)
    h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = alt.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    # Consistency: online composition is deliberately secondary and transparent.
    axes[1].plot(grid, composed_g, alpha=0.22, linewidth=0.9, label="online composed closing rate (unaltered)")
    axes[1].plot(grid, closing, linewidth=1.8, label="offline synchronized reference")
    axes[1].scatter(grid[outliers], composed_g[outliers], marker="x", s=22,
                    label="residual Hampel outlier (4 sigma)", zorder=5)
    axes[1].set_ylabel("closing rate [m/s]")
    axes[1].set_title(_robust_comparison_metrics(closing, composed_g, outliers=outliers), fontsize=9)
    _apply_robust_limits(axes[1], composed_g, closing)
    axes[1].legend(fontsize=8, loc="best")

    # Command remains on its row/mission time base; both axes are elapsed SIM time.
    if "command_thrust" in df.columns:
        axes[2].plot(t, numeric_column(df, "command_thrust"), label="thrust command")
        axes[2].axhline(0.73, linestyle="--", linewidth=1, alpha=0.6, label="hover reference 0.73")
    axes[2].set_ylabel("thrust [-]")
    axes[2].set_xlabel("elapsed simulation time [s]")
    axes[2].legend(fontsize=8, loc="best")

    for k, ax in enumerate(axes):
        shade_mission_phases(ax, df, t, label_once=(k == 0))
        ax.grid(True)
    save_current_figure(output_dir, "vertical_descent.png")

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
	mark_touchdown(axes[0], df, t)
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
		# Shade first so the traces draw on top of it (shade_mission_phases uses
		# zorder=0, but fill order still matters for the legend).
		shade_mission_phases(ax, df, t, label_once=(ax is axes[0]))
		mark_touchdown(ax, df, t, label_once=(ax is axes[0]))
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

	t_physics = _physics_time_column(df, t)
	dom = dominant_frequency_fft(t_physics, sig, min_frequency_hz=0.01, max_frequency_hz=2.0)
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
	"""Centered rolling median over an actual elapsed-time window.

	This remains well defined when diagnostics rows are irregular or when a
	source timestamp repeats because the latest sample is being held.
	"""
	return _rolling_median_time(
		np.asarray(t, dtype=float), np.asarray(y, dtype=float),
		window_sec=window_sec, center=True,
	)


def _divergence_consistency(df: pd.DataFrame, t: np.ndarray, min_height_m: float = 0.05) -> Optional[dict]:
	"""Compare the VISION-based divergence estimate against a 'physical'
	proxy computed purely from ground-truth kinematics:

		proxy = closing_rate / |relative_z|        (closing_rate / height)

	i.e. the same quantity flow_divergence is meant to estimate from optical
	flow, computed instead from the logged ground truth. This is a sanity
	check on the SENSING pipeline, independent of any control-law gain: if
	flow_divergence and the proxy disagree, the controller is being fed a
	number that no longer reflects reality, and no gain retune fixes that --
	only fixing (or working around) the sensing does.

	closing_rate is _closing_rate_from_position (a rolling regression slope of
	relative_z_m), NOT relative_vz_m_s directly. relative_vz_m_s is itself built
	from platform_vz_m_s, which bee_node derives by finite-differencing raw
	/platform/pose messages against wall-clock RECEIPT time and passing the
	result through one EMA pass -- i.e. it is a noisy, once-differentiated
	signal before this module ever sees it (confirmed on a real log: its
	step-to-step noise is an order of magnitude larger than relative_z_m's).
	Building this "ground truth" proxy on top of that noise made the sensing
	check itself unreliable -- exactly the kind of contamination this function
	exists to rule out. relative_z_m is a direct position measurement, so a
	regression-slope derivative of it is the more trustworthy numerator; the
	old relative_vz-based proxy is kept as "proxy_raw" (see plot, low opacity)
	purely for context/comparison, not for the mismatch/onset detection below.

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
	vz_raw = numeric_column(df, "relative_vz_m_s")
	relz = numeric_column(df, "relative_z_m")
	if not (np.any(np.isfinite(div)) and np.any(np.isfinite(vz_raw)) and np.any(np.isfinite(relz))):
		return None

	height = np.maximum(-relz, 0.0)
	# Floor the denominator, not the numerator: right at touchdown height ->
	# 0 and the true ratio is genuinely unbounded, which is exactly the
	# regime this is meant to expose -- clip only hard enough to keep the
	# proxy finite/plottable, not to hide the blow-up.
	height_floor = np.clip(height, min_height_m, None)

	# CONTEXT ONLY, low opacity in the plot: the old relative_vz-based proxy,
	# noisy because platform_vz is noisy (see docstring above).
	proxy_raw = vz_raw / height_floor

	# PRIMARY / trusted proxy: closing rate derived from relative_z_m's own
	# regression-slope derivative, not from relative_vz_m_s.
	closing_rate = _closing_rate_from_position(df, t, end_idx=None, window_sec=0.6)
	proxy = closing_rate / height_floor

	div_s = _rolling_smooth(t, div, window_sec=0.6)
	err_s = np.abs(div_s - proxy)

	# Flag threshold: the larger of a fixed floor (so a quiet near-zero hold
	# doesn't trip on tiny absolute noise) and a fraction of the proxy's own
	# typical scale over the series (so a fast, large-divergence descent
	# gets a proportionally larger tolerance).
	finite_proxy = proxy[np.isfinite(proxy)]
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
		"proxy_raw": proxy_raw,
		"div_smoothed": div_s,
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
	mark_touchdown(ax, df, t)
	ax.plot(t, div, color="tab:blue", alpha=0.30, linewidth=1, label="flow_divergence")
	ax.plot(t, consistency["div_smoothed"], color="tab:blue", linewidth=1.8, label="flow_divergence (smoothed)")
	# CONTEXT, low opacity: the old relative_vz-based proxy. Still shown because
	# it is real, significant data (platform_vz noise is itself informative when
	# large), just not the number this plot trusts -- see _divergence_consistency's
	# docstring for why relative_vz is noisier than relative_z.
	ax.plot(t, consistency["proxy_raw"], color="tab:orange", alpha=0.20, linewidth=1,
	        label="proxy (relative_vz/height, noisy, context)")
	ax.plot(t, consistency["proxy"], color="tab:orange", linewidth=1.8,
	        label="proxy (relative_z slope/height, trusted)")
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
	mark_touchdown(ax, df, t)
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
	mark_touchdown(ax, df, t)
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


# Mission substates, in the order they occur. "probe"/"probe_hold" are kept for
# older logs written before the approach/final split.
_PHASE_COLORS = {
	"center": ("tab:purple", 0.06),
	"approach_probe": ("tab:blue", 0.06),
	"final_probe": ("tab:cyan", 0.09),
	"descend": ("tab:green", 0.06),
	"landed": ("tab:gray", 0.12),
	"infeasible": ("tab:red", 0.10),
	"probe": ("tab:blue", 0.06),        # legacy
	"probe_hold": ("tab:cyan", 0.08),   # legacy
}

_PHASE_LABELS = {
	"center": "CENTER (acquire + centre)",
	"approach_probe": "APPROACH_PROBE (descend at D*>0 while probing)",
	"final_probe": "FINAL_PROBE (near-field hold at D*=0)",
	"descend": "DESCEND (committed landing)",
	"landed": "LANDED (touchdown; zero thrust)",
	"infeasible": "INFEASIBLE (gate refused; holding)",
	"probe": "PROBE (legacy)",
	"probe_hold": "PROBE_HOLD (legacy)",
}


def shade_mission_phases(ax, df: pd.DataFrame, t: np.ndarray, label_once: bool = True):
    """Shade mission phases and write their names directly inside the axes.

    Phase spans are deliberately excluded from the legend: repeating CENTER /
    APPROACH / FINAL_PROBE / DESCEND entries on every figure obscured the actual
    signal labels. ``label_once`` is retained for backward compatibility and now
    controls whether transparent in-plot phase names are drawn on this axes.
    """
    y_positions: dict[str, int] = {}
    for t0, t1, sub in _mission_phase_spans(df, t):
        color, alpha = _PHASE_COLORS.get(sub, ("tab:gray", 0.06))
        ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=0)
        if not label_once or not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue
        # Alternate two shallow rows when the same phase appears more than once.
        occurrence = y_positions.get(sub, 0)
        y_positions[sub] = occurrence + 1
        y = 0.965 - 0.065 * (occurrence % 2)
        label = _PHASE_LABELS.get(sub, sub).split(" (")[0]
        ax.text(
            0.5 * (t0 + t1), y, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5,
            alpha=0.42, color=color, clip_on=True, zorder=1,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.30),
        )

_TOUCHDOWN_CACHE: dict = {}


def _touchdown_for(df: pd.DataFrame, t: np.ndarray) -> Optional[dict]:
	key = (id(df), id(t))
	if key not in _TOUCHDOWN_CACHE:
		_TOUCHDOWN_CACHE[key] = _touchdown_velocity(df, t)
	return _TOUCHDOWN_CACHE[key]


def mark_touchdown(ax, df: pd.DataFrame, t: np.ndarray, label_once: bool = True, annotate: bool = False):
	"""Draw the touchdown instant on any time-axis Axes.

	Two vertical lines, because the two moments are genuinely different and both
	matter when reading any signal against time:
	  - solid red  : feet-on-surface (relative_z=0 crossing) -- the true landing.
	  - dotted red : physical contact latch (Gazebo event / thrust->0), which
	                 fires later, after the skids have driven into the deck. Drawn
	                 only when it differs from the crossing (belly-coherent logs);
	                 on older logs the two coincide and only one line shows.

	label_once=True labels them for the legend (use on the top subplot of a stack,
	False on the rest so the legend is not repeated). annotate=True adds the
	approach-velocity text (used on the vertical_descent plot, which is about
	exactly that number; elsewhere the lines alone keep the figure uncluttered).
	"""
	touch = _touchdown_for(df, t)
	if touch is None:
		return

	td_t = touch.get("t")
	if td_t is not None and np.isfinite(td_t):
		ax.axvline(
			td_t, color="tab:red", linestyle="-", linewidth=1.2, alpha=0.8, zorder=4,
			label=("touchdown (feet on surface)" if label_once else None),
		)

	contact_t = touch.get("contact_t")
	if contact_t is not None and np.isfinite(contact_t) and abs(contact_t - (td_t or contact_t)) > 1e-3:
		ax.axvline(
			contact_t, color="tab:red", linestyle=":", linewidth=1.1, alpha=0.6, zorder=4,
			label=("contact latch (penetrated)" if label_once else None),
		)

	if annotate and td_t is not None and np.isfinite(td_t):
		v = touch.get("approach_velocity_m_s")
		if v is not None and np.isfinite(v):
			ax.annotate(
				f"touchdown\n{v:+.3f} m/s",
				xy=(td_t, ax.get_ylim()[1]), xytext=(6, -12),
				textcoords="offset points", fontsize=8, va="top", color="tab:red",
			)


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
	mark_touchdown(ax, df, t)
	if "mission_thrust_gain_k" in df.columns:
		ax.plot(t, numeric_column(df, "mission_thrust_gain_k"), label=label_for("mission_thrust_gain_k"), linewidth=1.8)
	# The gain WINDOW, top to bottom: k_explore (start) / k_ceiling at leg height
	# (the de Croon limit k(t) now aims just under) / k_floor (the asymptote it
	# actually settles on) / k_min (the Herisse floor, now only a hard backstop).
	for col, style, lbl in (
		("mission_k_explore", (0, (4, 3)), label_for("mission_k_explore")),
		("mission_k_probe", (0, (8, 2, 2, 2)), label_for("mission_k_probe")),
		("mission_k_ceiling_leg", (0, (6, 2)), label_for("mission_k_ceiling_leg")),
		("mission_k_floor", (0, (3, 1, 1, 1)), label_for("mission_k_floor")),
		("mission_k_min", (0, (1, 2)), label_for("mission_k_min")),
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
	mark_touchdown(ax, df, t)
	if "mission_lateral_p_scale" in df.columns:
		ax.plot(t, numeric_column(df, "mission_lateral_p_scale"), label=label_for("mission_lateral_p_scale"), linewidth=1.8)
	if "mission_lateral_d_scale" in df.columns:
		ax.plot(t, numeric_column(df, "mission_lateral_d_scale"), label=label_for("mission_lateral_d_scale"), linewidth=1.4, linestyle="--")
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


def plot_probe_acceleration(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	"""What the platform probe actually measured, step by step.

	The feasibility gate rests on ONE number -- peak_accel -- via k_min =
	peak_accel / D*. That number is a leaky-max envelope over a rolling percentile
	of |commanded accel - EMA bias|, so it can sit high for reasons that have
	nothing to do with the platform. This figure shows the whole chain it is built
	from, so the envelope can be judged instead of trusted.

	Top: the raw commanded accel and the EMA bias being subtracted from it.
	     If the BIAS visibly oscillates rather than sitting flat, the highpass tau
	     is short enough to be tracking -- and therefore cancelling -- the very
	     platform motion being measured, and peak_accel is biased LOW.
	     During APPROACH_PROBE the bias legitimately carries the slow contribution
	     of the D*>0 descent itself; that is what it is there to remove.

	Bottom: the residual (the actual measurement), the rolling-window percentile
	     the envelope chases, and the envelope itself.
	     The envelope should sit ON TOP of the residual's excursions. If it
	     visibly COASTS above them -- decaying smoothly, never re-raised by a
	     fresh sample -- the probe is running on a stale measurement: either the
	     window is too short to catch the platform's slow swing, or
	     peak_decay_tau is too long.

	The far->near handoff is marked: that is where the time constants are retuned
	and the near-field samples (better synchronized, hence more trustworthy) are
	expected to REVISE the carried-over far-field estimate. peak_accel_at_handoff
	is drawn as a reference line so the size and direction of that revision is one
	subtraction. If the two never separate, either the near hold is too short to
	see an excursion or NEAR_PROBE_DECAY_TAU_SEC is too long.
	"""
	needed = ("mission_probe_residual_accel_m_s2", "mission_peak_accel_m_s2")
	if not any(c in df.columns for c in needed):
		print("Skipping probe acceleration plot. No mission_probe_* columns (old log?).")
		return

	fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
	fig.suptitle("Platform probe: what peak_accel is built from")

	# --- raw commanded accel vs the bias being removed ---
	ax = axes[0]
	shade_mission_phases(ax, df, t)
	mark_touchdown(ax, df, t)
	for col, kwargs in (
		("mission_probe_accel_m_s2", dict(linewidth=1.0, alpha=0.65)),
		("mission_probe_mean_accel_m_s2", dict(linewidth=1.8, linestyle="--")),
	):
		if col in df.columns:
			ax.plot(t, numeric_column(df, col), label=label_for(col), **kwargs)
	ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
	ax.set_ylabel("acceleration [m/s$^2$]")
	ax.grid(True)
	ax.legend(loc="best", fontsize=8)

	# --- the measurement, the percentile, and the envelope the gate consumes ---
	ax = axes[1]
	shade_mission_phases(ax, df, t)
	mark_touchdown(ax, df, t)
	if "mission_probe_residual_accel_m_s2" in df.columns:
		ax.plot(
			t, numeric_column(df, "mission_probe_residual_accel_m_s2"),
			label=label_for("mission_probe_residual_accel_m_s2"),
			linewidth=0.9, alpha=0.55,
		)
	if "mission_probe_percentile_accel_m_s2" in df.columns:
		ax.plot(
			t, numeric_column(df, "mission_probe_percentile_accel_m_s2"),
			label=label_for("mission_probe_percentile_accel_m_s2"),
			linewidth=1.2, linestyle=":",
		)
	if "mission_peak_accel_m_s2" in df.columns:
		ax.plot(
			t, numeric_column(df, "mission_peak_accel_m_s2"),
			label=label_for("mission_peak_accel_m_s2"),
			linewidth=2.0, color="tab:red",
		)

	# Mark the far->near retune and the estimate it carried across.
	handoff_t = _probe_handoff_time(df, t)
	if handoff_t is not None:
		for a in axes:
			a.axvline(handoff_t, color="k", linestyle="-.", linewidth=1.2, alpha=0.7)
		axes[1].annotate(
			"far \u2192 near handoff\n(probe retuned, estimate kept)",
			xy=(handoff_t, ax.get_ylim()[1]),
			xytext=(6, -12), textcoords="offset points",
			fontsize=8, va="top",
		)

	handoff_peak = _last_finite(df, "mission_probe_peak_accel_at_handoff_m_s2")
	if handoff_peak is not None and np.isfinite(handoff_peak) and handoff_peak > 0.0:
		ax.axhline(
			handoff_peak, color="tab:orange", linestyle=(0, (5, 2)), linewidth=1.3,
			label=label_for("mission_probe_peak_accel_at_handoff_m_s2"),
		)

	# Where the envelope is COASTING: it is above the percentile it chases, i.e.
	# no fresh sample is holding it up and it is purely decaying off an old one.
	# Shaded because it is the single most important failure mode to catch -- a
	# probe that spends most of its life coasting is reporting a measurement it
	# took once, not one it keeps making.
	if {"mission_peak_accel_m_s2", "mission_probe_percentile_accel_m_s2"}.issubset(df.columns):
		peak = numeric_column(df, "mission_peak_accel_m_s2")
		pct = numeric_column(df, "mission_probe_percentile_accel_m_s2")
		active = bool_column(df, "mission_probe_active")
		coasting = active & np.isfinite(peak) & np.isfinite(pct) & (peak > pct * 1.02)
		if np.any(coasting):
			ax.fill_between(
				t, 0, 1, where=coasting, transform=ax.get_xaxis_transform(),
				color="tab:red", alpha=0.07, zorder=0,
				label=f"envelope coasting ({100.0 * np.mean(coasting[active]):.0f}% of probe)",
			)

	ax.set_ylabel("acceleration [m/s$^2$]")
	ax.set_xlabel("time [s]")
	ax.grid(True)
	ax.legend(loc="best", fontsize=8)

	# The gate's arithmetic, spelled out, so the figure is self-contained.
	final_peak = _last_finite(df, "mission_peak_accel_m_s2")
	k_min = _last_finite(df, "mission_k_min")
	bits = []
	if final_peak is not None and np.isfinite(final_peak):
		bits.append(f"peak_accel = {final_peak:.3f} m/s$^2$")
	if handoff_peak is not None and np.isfinite(handoff_peak) and handoff_peak > 0.0 \
			and final_peak is not None and np.isfinite(final_peak):
		delta = 100.0 * (final_peak - handoff_peak) / handoff_peak
		bits.append(f"near-field revision {delta:+.0f}%")
	if k_min is not None and np.isfinite(k_min):
		bits.append(f"k_min = {k_min:.2f}")
	if bits:
		ax.set_title("  |  ".join(bits), fontsize=9)

	save_current_figure(output_dir, "probe_acceleration.png")


def _probe_handoff_time(df: pd.DataFrame, t: np.ndarray) -> Optional[float]:
	"""Time of the far->near probe retune (first sample of the FINAL_PROBE hold)."""
	if "mission_probe_phase" not in df.columns:
		return None
	phase = df["mission_probe_phase"].astype(str).fillna("").to_numpy()
	idx = _first_true_index(phase == "near")
	return float(t[idx]) if idx is not None else None


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
		truth = np.maximum(-numeric_column(df, "relative_z_m"), 0.0)
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
	mark_touchdown(ax, df, t)
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
	mark_touchdown(ax, df, t)
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


def plot_platform_velocity_xyz(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path):
	"""Platform velocity, x/y/z, each panel showing the raw column at LOW
	opacity (still real, significant data -- worth seeing) plus a SMOOTHED
	overlay at full strength, so the two panels' worth of noise the raw
	channel actually carries doesn't drown the trend.

	The overlay is a rolling regression slope of the corresponding POSITION
	column (platform_x_m/y_m/z_m), not a smooth of the velocity column itself
	-- see _closing_rate_from_position's docstring for why: platform_v{x,y,z}
	is itself produced by finite-differencing raw /platform/pose messages
	against wall-clock receipt time, then one EMA pass (bee_node.py's
	on_platform_pose, PLATFORM_VELOCITY_SMOOTHING=0.7) -- i.e. it is already a
	once-differentiated, once-filtered signal before this module sees it, and
	smoothing it further cannot undo whatever the upstream differencing
	distorted. Differentiating the (never-differentiated) position column
	directly, via a windowed regression slope, avoids compounding two rounds
	of differentiation noise -- the same fix applied to the touchdown-velocity
	and divergence-consistency-proxy estimators.

	Two further corrections in the same spirit (_derivative_from_position):
	the window is measured in real ELAPSED SECONDS via each row's actual
	timestamp, not an approximate sample count (message arrival on
	/platform/pose is jittery enough -- 2ms to 196ms against a 20ms median dt
	on a real log -- that a sample-count window silently spanned anywhere from
	0.3s to 1.3s of real time for a nominal "0.6s" window); and a short
	outlier-rejecting median pre-filter runs on the position column before the
	regression, so a single-sample position spike is discarded rather than
	pulled into the fit.
	"""
	axis_specs = [
		("x", "platform_vx_m_s", "platform_x_m", "platform vx [m/s]"),
		("y", "platform_vy_m_s", "platform_y_m", "platform vy [m/s]"),
		("z", "platform_vz_m_s", "platform_z_m", "platform vz [m/s]"),
	]
	available = [spec for spec in axis_specs if spec[1] in df.columns]
	if not available:
		print("Skipping platform_velocity_xyz.png. No platform_v{x,y,z}_m_s columns found.")
		return

	fig, axes = plt.subplots(len(available), 1, figsize=(11, 2.8 * len(available)), sharex=True)
	if len(available) == 1:
		axes = [axes]
	fig.suptitle("Platform velocity (raw, low opacity, vs. smoothed from position)")

	t_real = _physics_time_column(df, t)
	for ax, (axis, vcol, pcol, label) in zip(axes, available):
		raw = numeric_column(df, vcol)
		ax.plot(t, raw, color="tab:blue", alpha=0.25, linewidth=1, label=f"{label} (raw)")
		if pcol in df.columns:
			pos = numeric_column(df, pcol)
			derived = _derivative_from_position(t_real, pos, window_sec=0.6, center=True)
			ax.plot(t, derived, color="tab:blue", linewidth=1.8, label=f"{label} (smoothed, from {pcol} slope)")
		ax.axhline(0.0, linestyle="--", linewidth=1, color="0.5")
		ax.set_ylabel(label)
		ax.grid(True)
		ax.legend(loc="best", fontsize=8)
	axes[-1].set_xlabel("time [s]")
	save_current_figure(output_dir, "platform_velocity_xyz.png")


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
	mark_touchdown(axes[0], df, t)
	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("flow x [px/s]")
	axes[0].grid(True)
	axes[0].legend(loc="best")

	axes[1].plot(t, raw_y, label="mean flow y -- raw", color="tab:green", alpha=0.65)
	axes[1].plot(t, der_y, label="mean flow y -- de-rotated", color="tab:green", linewidth=1.8)
	axes[1].plot(t, raw_y - der_y, label="rotational component removed",
	             color="tab:orange", linestyle=":", alpha=0.85)
	shade_mission_phases(axes[1], df, t)
	mark_touchdown(axes[1], df, t)
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
		mark_touchdown(axes[2], df, t)
		axes[2].axhline(0.0, linestyle="--", linewidth=1)
		axes[2].set_ylabel("divergence [1/s]")
		axes[2].grid(True)
		axes[2].legend(loc="best")

	axes[-1].set_xlabel("time [s]")
	save_current_figure(output_dir, "flow_derotation.png")



def _robust_limits(*signals: np.ndarray, lower: float = 0.5, upper: float = 99.5,
                   margin_fraction: float = 0.10) -> Optional[Tuple[float, float]]:
    """Display limits from central percentiles without deleting any samples."""
    finite_parts = []
    for signal in signals:
        if signal is None:
            continue
        values = np.asarray(signal, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            finite_parts.append(values)
    if not finite_parts:
        return None
    values = np.concatenate(finite_parts)
    if len(values) < 5:
        return None
    lo, hi = np.percentile(values, [lower, upper])
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 1e-9:
        span = max(abs(float(lo)), abs(float(hi)), 1.0)
    margin = margin_fraction * span
    return float(lo - margin), float(hi + margin)


def _apply_robust_limits(ax, *signals: np.ndarray, lower: float = 0.5, upper: float = 99.5,
                         note: bool = True) -> Optional[Tuple[float, float]]:
    """Apply robust display limits and disclose how many finite points are clipped.

    The underlying arrays and all reported metrics remain untouched. Only the
    viewport is robustified, which is why this is defensible for diagnostics.
    """
    limits = _robust_limits(*signals, lower=lower, upper=upper)
    if limits is None:
        return None
    ax.set_ylim(*limits)
    if note:
        total = outside = 0
        for signal in signals:
            if signal is None:
                continue
            values = np.asarray(signal, dtype=float)
            finite = np.isfinite(values)
            total += int(np.count_nonzero(finite))
            outside += int(np.count_nonzero(finite & ((values < limits[0]) | (values > limits[1]))))
        if outside:
            ax.text(
                0.995, 0.02,
                f"{outside}/{total} finite points outside displayed {lower:g}-{upper:g}% range",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                alpha=0.65,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.55),
            )
    return limits


def _hampel_mask(t: np.ndarray, values: np.ndarray, window_sec: float = 1.0,
                 threshold_sigma: float = 4.0, min_points: int = 7) -> np.ndarray:
    """Conservative rolling Hampel flag using median and scaled MAD.

    This returns a diagnostic mask only. It never replaces, clips or removes
    signal samples. A point is marked when its residual is unusually far from
    the local residual distribution.
    """
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.zeros(len(values), dtype=bool)
    if len(t) != len(values):
        return mask
    for i in range(len(values)):
        if not (np.isfinite(t[i]) and np.isfinite(values[i])):
            continue
        lo = np.searchsorted(t, t[i] - 0.5 * window_sec, side="left")
        hi = np.searchsorted(t, t[i] + 0.5 * window_sec, side="right")
        segment = values[lo:hi]
        segment = segment[np.isfinite(segment)]
        if len(segment) < min_points:
            continue
        median = float(np.median(segment))
        mad = float(np.median(np.abs(segment - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 1e-9:
            mask[i] = abs(values[i] - median) > threshold_sigma * robust_sigma
    return mask


def _robust_comparison_metrics(reference: np.ndarray, estimate: np.ndarray,
                               mask: Optional[np.ndarray] = None,
                               outliers: Optional[np.ndarray] = None) -> str:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(estimate)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if np.count_nonzero(valid) < 5:
        return "not enough common samples"
    residual = estimate[valid] - reference[valid]
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    p95 = float(np.percentile(np.abs(residual), 95))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    frac = float('nan')
    if outliers is not None:
        selected = np.asarray(outliers, dtype=bool)[valid]
        frac = 100.0 * float(np.mean(selected))
    suffix = f", Hampel={frac:.1f}%" if np.isfinite(frac) else ""
    return f"RMSE={rmse:.3f}, median={median:+.3f}, MAD={mad:.3f}, p95|e|={p95:.3f}{suffix} m/s"


def _comparison_metrics(reference: np.ndarray, estimate: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
	"""Compact consistency metrics for two same-unit traces."""
	reference = np.asarray(reference, dtype=float)
	estimate = np.asarray(estimate, dtype=float)
	valid = np.isfinite(reference) & np.isfinite(estimate)
	if mask is not None:
		valid &= np.asarray(mask, dtype=bool)
	if np.count_nonzero(valid) < 5:
		return "not enough common samples"
	err = estimate[valid] - reference[valid]
	corr = float(np.corrcoef(reference[valid], estimate[valid])[0, 1])
	return (
		f"RMSE={np.sqrt(np.mean(err**2)):.3f} m/s, "
		f"bias={np.mean(err):+.3f} m/s, corr={corr:.3f}"
	)


def plot_timebase_diagnostics(df: pd.DataFrame, output_dir: str | Path):
	"""Show how Gazebo simulation time maps to host wall time.

	Physical plots use the simulation clock.  Wall time is retained here only
	to diagnose real-time factor, stalls and compute throughput.
	"""
	sim_t, sim_column = _simulation_time_column(df)
	if sim_t is None:
		print("Skipping timebase diagnostics. No simulation timestamp column.")
		return
	if "t_sec" in df.columns:
		wall_t = _normalize_elapsed(numeric_column(df, "t_sec"))
	elif "wall_timestamp" in df.columns:
		wall_t = _normalize_elapsed(numeric_column(df, "wall_timestamp"))
	else:
		print("Skipping timebase diagnostics. No wall-time column.")
		return

	common = np.isfinite(sim_t) & np.isfinite(wall_t)
	if np.count_nonzero(common) < 5:
		print("Skipping timebase diagnostics. Not enough common clock samples.")
		return

	i = np.where(common)[0]
	sim_span = float(sim_t[i[-1]] - sim_t[i[0]])
	wall_span = float(wall_t[i[-1]] - wall_t[i[0]])
	global_rtf = sim_span / wall_span if wall_span > 0.0 else float("nan")
	rtf_roll = _rolling_regression_slope(
		wall_t, sim_t, window_sec=2.0, center=True, min_pts=8
	)

	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
	fig.suptitle(
		f"Clock coherence: {sim_column} vs wall time "
		f"(effective run-wide RTF={global_rtf:.3f})"
	)

	ax = axes[0]
	shade_mission_phases(ax, df, wall_t)
	ax.plot(wall_t, sim_t, linewidth=1.8, label="elapsed simulation time")
	limit = max(float(np.nanmax(wall_t)), float(np.nanmax(sim_t)))
	ax.plot([0.0, limit], [0.0, limit], linestyle="--", linewidth=1.0, label="real time (RTF=1)")
	ax.set_ylabel("simulation time [s]")
	ax.grid(True)
	ax.legend(loc="best", fontsize=8)

	ax = axes[1]
	shade_mission_phases(ax, df, wall_t)
	ax.plot(wall_t, rtf_roll, linewidth=1.6, label="local RTF (2 s regression)")
	ax.axhline(global_rtf, linestyle="--", linewidth=1.2, label=f"run-wide RTF {global_rtf:.3f}")
	ax.axhline(1.0, linestyle=":", linewidth=1.0, label="real time")
	if "timing_sim_rtf_estimate" in df.columns:
		ax.plot(
			wall_t, numeric_column(df, "timing_sim_rtf_estimate"),
			alpha=0.35, linewidth=1.0, label="logged timing_sim_rtf_estimate",
		)
	ax.set_ylabel("sim / wall [-]")
	ax.set_xlabel("wall time [s]")
	ax.grid(True)
	ax.legend(loc="best", fontsize=8)

	save_current_figure(output_dir, "timebase_diagnostics.png")


def _legacy_plot_vertical_kinematics_validation(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path):
	"""Validate z velocities against one controlled derivative of z positions.

	The derivative uses Gazebo simulation time and a 0.6 s local regression.
	It is a consistency check, not absolute vehicle ground truth: PX4 position
	and PX4 velocity can have different estimator filtering.  In simulation,
	an additional Gazebo vehicle twist channel should be logged for the true
	accuracy benchmark.
	"""
	t_sim = _physics_time_column(df, t)
	required = ("platform_z_m", "vehicle_z_m", "relative_z_m")
	if not all(c in df.columns for c in required):
		print("Skipping vertical kinematics validation. Missing z-position columns.")
		return

	platform_z = numeric_column(df, "platform_z_m")
	vehicle_z = numeric_column(df, "vehicle_z_m")
	relative_z = numeric_column(df, "relative_z_m")
	platform_vz_from_z = _derivative_from_position(t_sim, platform_z, window_sec=0.6, center=True)
	vehicle_vz_from_z = _derivative_from_position(t_sim, vehicle_z, window_sec=0.6, center=True)
	closing_from_z = _derivative_from_position(t_sim, relative_z, window_sec=0.6, center=True)
	flying = _flying_mask(df)

	fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
	fig.suptitle("Vertical kinematics validation on Gazebo simulation time")

	for k, ax in enumerate(axes):
		shade_mission_phases(ax, df, t_sim, label_once=(k == 0))
		mark_touchdown(ax, df, t_sim, label_once=(k == 0))
		ax.grid(True)

	ax = axes[0]
	ax.plot(t_sim, -relative_z, linewidth=1.6, label="height above deck = -relative_z")
	ax.axhline(0.0, linestyle="--", linewidth=1)
	ax.set_ylabel("height [m]\n(+ above deck)")
	ax.legend(loc="best", fontsize=8)

	ax = axes[1]
	if "platform_vz_m_s" in df.columns:
		platform_vz = numeric_column(df, "platform_vz_m_s")
		ax.plot(t_sim, platform_vz, alpha=0.28, linewidth=1.0, label="logged platform_vz")
		ax.plot(t_sim, platform_vz_from_z, linewidth=1.8, label="d(platform_z)/d(sim time)")
		ax.set_title(_comparison_metrics(platform_vz, platform_vz_from_z, flying), fontsize=9)
	else:
		ax.plot(t_sim, platform_vz_from_z, linewidth=1.8, label="d(platform_z)/d(sim time)")
	ax.axhline(0.0, linestyle="--", linewidth=1)
	ax.set_ylabel("platform vz [m/s]")
	ax.legend(loc="best", fontsize=8)

	ax = axes[2]
	if "vehicle_vz_m_s" in df.columns:
		vehicle_vz = numeric_column(df, "vehicle_vz_m_s")
		ax.plot(t_sim, vehicle_vz, alpha=0.35, linewidth=1.0, label="PX4 vehicle_vz")
		ax.plot(t_sim, vehicle_vz_from_z, linewidth=1.8, label="d(vehicle_z)/d(sim time)")
		ax.set_title(_comparison_metrics(vehicle_vz, vehicle_vz_from_z, flying), fontsize=9)
	else:
		ax.plot(t_sim, vehicle_vz_from_z, linewidth=1.8, label="d(vehicle_z)/d(sim time)")
	ax.axhline(0.0, linestyle="--", linewidth=1)
	ax.set_ylabel("vehicle vz [m/s]\n(NED + down)")
	ax.legend(loc="best", fontsize=8)

	ax = axes[3]
	if "relative_vz_m_s" in df.columns:
		relative_vz = numeric_column(df, "relative_vz_m_s")
		ax.plot(t_sim, relative_vz, alpha=0.30, linewidth=1.0, label="vehicle_vz + platform_vz")
		ax.plot(t_sim, closing_from_z, linewidth=1.8, label="d(relative_z)/d(sim time)")
		ax.set_title(_comparison_metrics(relative_vz, closing_from_z, flying), fontsize=9)
	else:
		ax.plot(t_sim, closing_from_z, linewidth=1.8, label="d(relative_z)/d(sim time)")
	ax.axhline(0.0, linestyle="--", linewidth=1)
	ax.set_ylabel("closing rate [m/s]\n(+ closing)")
	ax.set_xlabel("simulation time [s]")
	ax.legend(loc="best", fontsize=8)

	save_current_figure(output_dir, "vertical_kinematics_validation.png")


def make_default_plots(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path, args):
	"""The core loop: did we SEE the target, did the PROBE measure the platform,
	did the GATE pick a sane gain window, and did the DESCENT do what was asked.

	Everything that answers a narrower or more diagnostic question -- lateral
	control, de-rotation, height prediction, cross-checks against ground truth,
	platform/vehicle kinematics -- now lives behind --full. Those plots are not
	less useful, they are just not what you look at on every run, and having a
	dozen figures open by default made the four that matter harder to find.
	"""
	# One log per call: clear the touchdown cache so a reused id() from a prior
	# log's garbage-collected df/t can't return a stale touchdown.
	_TOUCHDOWN_CACHE.clear()
	plot_timebase_diagnostics(df, output_dir)
	plot_vertical_kinematics_validation(df, t, output_dir)
	plot_target_detection_summary(df, t, output_dir)
	plot_detection_boxes_fov(df, t, args.image_width, args.image_height, output_dir, args.max_boxes)
	plot_probe_acceleration(df, t, output_dir)
	plot_gain_schedule(df, t, output_dir)
	# vertical_control.png was one 4-panel stack mixing the loop's error signal with
	# its physical outcome. Split into the two questions it was really answering.
	plot_vertical_divergence(df, t, output_dir, divergence_setpoint=None)
	plot_vertical_descent(df, t, output_dir)
	plot_closing_rate_spectrum(df, t, output_dir, expected_frequency_hz=args.platform_frequency_hz)
	plot_drone_platform_position_xyz(df, t, output_dir)


def make_full_plots(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path, args):
	# --- Moved out of the default set (see make_default_plots' docstring). ---
	plot_lateral_control(df, t, output_dir)
	plot_flow_derotation(df, t, output_dir)
	plot_divergence_consistency(df, t, output_dir)
	plot_height_prediction(df, t, output_dir)
	plot_platform_motion_frequency(df, t, output_dir, expected_frequency_hz=args.platform_frequency_hz)

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
	plot_platform_velocity_xyz(df, t, output_dir)
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
# Diagnostics schema 3.0 compatibility and native-stream analysis
# ---------------------------------------------------------------------------

# Canonical internal names intentionally remain close to the historical schema,
# so all established mission/control plots continue to work.  For a schema 3.0
# log, these aliases are COPIES of the explicitly named source columns; the
# original columns remain untouched and are used by the native-stream routines
# below.  This makes old and new logs readable by one analyser without hiding
# the clock provenance in the source CSV.
_SCHEMA_ALIASES = {
    "t_sec": ("log_elapsed_wall_sec",),
    "wall_timestamp": ("log_write_wall_timestamp_sec",),
    "target_timestamp_sec": ("target_source_timestamp_sec",),
    "flow_timestamp_sec": ("flow_source_timestamp_sec",),
    "command_timestamp_sec": ("command_source_flow_timestamp_sec",),
    "vehicle_timestamp_sec": ("vehicle_local_position_receipt_wall_timestamp_sec",),
    "vehicle_px4_timestamp_sec": (
        "vehicle_local_position_sample_px4_timestamp_sec",
        "vehicle_local_position_publication_px4_timestamp_sec",
    ),
    "vehicle_attitude_timestamp_sec": (
        "vehicle_attitude_sample_px4_timestamp_sec",
        "vehicle_attitude_publication_px4_timestamp_sec",
    ),
    "platform_vx_m_s": ("platform_vx_derived_from_pose_m_s",),
    "platform_vy_m_s": ("platform_vy_derived_from_pose_m_s",),
    "platform_vz_m_s": ("platform_vz_derived_from_pose_m_s",),
    "relative_vx_m_s": ("relative_vx_composed_m_s",),
    "relative_vy_m_s": ("relative_vy_composed_m_s",),
    "relative_vz_m_s": ("relative_vz_composed_m_s",),
    "timing_camera_cb_duration_ms": ("timing_camera_callback_duration_ms",),
    "timing_frame_to_available_wall_ms": ("timing_frame_to_available_ms",),
    "timing_frame_to_command_wall_ms": ("timing_frame_to_command_ms",),
    "timing_vision_result_period_wall_ms": ("timing_vision_result_period_ms",),
    "timing_control_period_wall_sec": ("timing_control_period_monotonic_sec",),
    "timing_px4_publish_period_wall_sec": ("timing_px4_publish_period_monotonic_sec",),
    "timing_flow_age_at_control_wall_ms": ("timing_flow_age_at_control_ms",),
    "timing_flow_age_at_px4_publish_wall_ms": ("timing_flow_age_at_px4_publish_ms",),
    "timing_sim_rtf_estimate": ("clock_sim_local_rate_per_wall",),
    "timing_px4_rtf_estimate": ("clock_px4_local_rate_per_wall",),
}


def normalize_diagnostics_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add compatibility aliases while preserving all native schema columns."""
    df = df.copy()
    for canonical, candidates in _SCHEMA_ALIASES.items():
        if canonical in df.columns and df[canonical].notna().any():
            continue
        for candidate in candidates:
            if candidate in df.columns:
                df[canonical] = df[candidate]
                break
    if "diagnostics_schema_version" not in df.columns:
        df["diagnostics_schema_version"] = "legacy"
    return df


def _source_clock_is_sim(df: pd.DataFrame) -> bool:
    for col in ("vision_source_clock", "camera_source_clock", "command_source_clock"):
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.strip().str.lower()
        values = values[~values.isin(("", "nan"))]
        if len(values):
            return bool((values == "gazebo_sim").mean() >= 0.90)
    # Historical BEE_LAND logs used Gazebo image stamps in flow/target fields.
    return "flow_timestamp_sec" in df.columns


def _wall_time_column(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for col in ("log_elapsed_wall_sec", "t_sec"):
        ok, raw, _ = _valid_time_column(df, col)
        if ok:
            return _normalize_elapsed(raw), col
    for col in ("log_write_wall_timestamp_sec", "wall_timestamp"):
        ok, raw, _ = _valid_time_column(df, col)
        if ok:
            return _normalize_elapsed(raw), col
    return None, None


def _simulation_time_column(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not _source_clock_is_sim(df):
        return None, None
    for column in (
        "flow_source_timestamp_sec", "target_source_timestamp_sec",
        "command_source_flow_timestamp_sec", "camera_source_timestamp_sec",
        "flow_timestamp_sec", "target_timestamp_sec", "command_timestamp_sec",
    ):
        ok, raw, _ = _valid_time_column(df, column)
        if ok:
            return _normalize_elapsed(raw), column
    return None, None


def _unique_source_frame(
    df: pd.DataFrame,
    timestamp_column: str,
    value_columns: Sequence[str],
    counter_column: Optional[str] = None,
) -> pd.DataFrame:
    """Return one row per actual asynchronous source update."""
    columns = [timestamp_column, *[c for c in value_columns if c in df.columns]]
    if counter_column and counter_column in df.columns:
        columns.append(counter_column)
    if timestamp_column not in df.columns or len(columns) <= 1:
        return pd.DataFrame(columns=columns)
    out = df[columns].copy()
    out[timestamp_column] = pd.to_numeric(out[timestamp_column], errors="coerce")
    out = out[np.isfinite(out[timestamp_column])]
    if counter_column and counter_column in out.columns:
        counter = pd.to_numeric(out[counter_column], errors="coerce")
        if counter.notna().any():
            out = out.assign(_counter=counter).dropna(subset=["_counter"])
            out = out.drop_duplicates("_counter", keep="last").drop(columns="_counter")
        else:
            out = out.drop_duplicates(timestamp_column, keep="last")
    else:
        out = out.drop_duplicates(timestamp_column, keep="last")
    return out.sort_values(timestamp_column).reset_index(drop=True)


def _interp_no_extrapolation(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); x_new = np.asarray(x_new, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    xx, yy = x[good], y[good]
    order = np.argsort(xx); xx, yy = xx[order], yy[order]
    unique = np.r_[True, np.diff(xx) > 1e-9]
    xx, yy = xx[unique], yy[unique]
    if len(xx) < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    result = np.interp(x_new, xx, yy)
    result[(x_new < xx[0]) | (x_new > xx[-1])] = np.nan
    return result


def _rowwise_px4_to_sim_time(df: pd.DataFrame, px4_time: np.ndarray) -> np.ndarray:
    """Map PX4 source time to estimated SIM time using the logged affine fits."""
    needed = (
        "clock_px4_local_rate_per_wall", "clock_px4_fit_wall_reference_sec",
        "clock_px4_fit_source_reference_sec", "clock_sim_local_rate_per_wall",
        "clock_sim_fit_wall_reference_sec", "clock_sim_fit_source_reference_sec",
    )
    if not all(c in df.columns for c in needed):
        return np.full(len(df), np.nan)
    rp = numeric_column(df, needed[0]); wp = numeric_column(df, needed[1]); pp = numeric_column(df, needed[2])
    rs = numeric_column(df, needed[3]); ws = numeric_column(df, needed[4]); ss = numeric_column(df, needed[5])
    px4_time = np.asarray(px4_time, dtype=float)
    valid = np.isfinite(px4_time) & np.isfinite(rp) & (np.abs(rp) > 1e-9)
    valid &= np.isfinite(wp) & np.isfinite(pp) & np.isfinite(rs) & np.isfinite(ws) & np.isfinite(ss)
    wall = np.full(len(df), np.nan)
    wall[valid] = wp[valid] + (px4_time[valid] - pp[valid]) / rp[valid]
    sim = np.full(len(df), np.nan)
    sim[valid] = ss[valid] + rs[valid] * (wall[valid] - ws[valid])
    return sim


def _vehicle_stream_sim(df: pd.DataFrame) -> Optional[dict]:
    source_col = None
    for c in ("vehicle_local_position_sample_px4_timestamp_sec", "vehicle_local_position_publication_px4_timestamp_sec", "vehicle_px4_timestamp_sec"):
        if c in df.columns and np.isfinite(numeric_column(df, c)).sum() >= 3:
            source_col = c; break
    if source_col is None:
        return None
    px4 = numeric_column(df, source_col)
    sim_est = _rowwise_px4_to_sim_time(df, px4)
    work = df.copy(); work["_vehicle_sim_est"] = sim_est
    time_col = "_vehicle_sim_est" if np.isfinite(sim_est).sum() >= 3 else source_col
    stream = _unique_source_frame(
        work, time_col,
        ["vehicle_x_m", "vehicle_y_m", "vehicle_z_m", "vehicle_vx_m_s", "vehicle_vy_m_s", "vehicle_vz_m_s", source_col],
        "vehicle_local_position_message_count" if "vehicle_local_position_message_count" in df.columns else None,
    )
    if len(stream) < 3:
        return None
    raw_t = pd.to_numeric(stream[time_col], errors="coerce").to_numpy(float)
    return {"df": stream, "t_abs": raw_t, "clock": "estimated Gazebo SIM" if time_col == "_vehicle_sim_est" else "PX4 source"}


def _platform_stream_sim(df: pd.DataFrame) -> Optional[dict]:
    time_col = None
    for c in ("platform_pose_estimated_sim_timestamp_sec", "flow_source_timestamp_sec", "flow_timestamp_sec"):
        if c in df.columns and np.isfinite(numeric_column(df, c)).sum() >= 3:
            time_col = c; break
    if time_col is None:
        return None
    stream = _unique_source_frame(
        df, time_col,
        ["platform_x_m", "platform_y_m", "platform_z_m", "platform_vx_m_s", "platform_vy_m_s", "platform_vz_m_s"],
        "platform_pose_message_count" if "platform_pose_message_count" in df.columns else None,
    )
    if len(stream) < 3:
        return None
    return {"df": stream, "t_abs": pd.to_numeric(stream[time_col], errors="coerce").to_numpy(float), "clock": time_col}


def _synchronized_vertical_model(df: pd.DataFrame, grid_dt: Optional[float] = None) -> Optional[dict]:
    """Synchronize PX4 vehicle and Gazebo platform positions on one SIM grid."""
    vehicle = _vehicle_stream_sim(df); platform = _platform_stream_sim(df)
    if vehicle is None or platform is None:
        return None
    vd, pd_ = vehicle["df"], platform["df"]
    tv, tp = vehicle["t_abs"], platform["t_abs"]
    zv = pd.to_numeric(vd.get("vehicle_z_m"), errors="coerce").to_numpy(float)
    zp = pd.to_numeric(pd_.get("platform_z_m"), errors="coerce").to_numpy(float)
    good_v = np.isfinite(tv) & np.isfinite(zv); good_p = np.isfinite(tp) & np.isfinite(zp)
    if good_v.sum() < 3 or good_p.sum() < 3:
        return None
    lo = max(float(np.nanmin(tv[good_v])), float(np.nanmin(tp[good_p])))
    hi = min(float(np.nanmax(tv[good_v])), float(np.nanmax(tp[good_p])))
    if hi - lo < 0.5:
        return None
    if grid_dt is None:
        candidates = [median_positive_dt(tv), median_positive_dt(tp)]
        candidates = [x for x in candidates if np.isfinite(x) and x > 0]
        grid_dt = min(candidates) if candidates else 0.02
        grid_dt = float(np.clip(grid_dt, 0.005, 0.05))
    grid = np.arange(lo, hi + 0.25 * grid_dt, grid_dt)
    vehicle_z = _interp_no_extrapolation(tv, zv, grid)
    platform_z = _interp_no_extrapolation(tp, zp, grid)
    # Infer the constant feet/surface geometry offset from the logger's own
    # relative-position equation. This avoids duplicating model constants here.
    sim_row, _ = _simulation_time_column(df)
    offset = 0.0
    if sim_row is not None and "relative_z_m" in df.columns:
        rel_logged = numeric_column(df, "relative_z_m")
        raw_col = next((c for c in ("flow_source_timestamp_sec", "target_source_timestamp_sec", "command_source_flow_timestamp_sec", "flow_timestamp_sec") if c in df.columns and np.isfinite(numeric_column(df, c)).any()), None)
        sim_origin = numeric_column(df, raw_col)[np.isfinite(numeric_column(df, raw_col))][0] if raw_col else 0.0
        sim_row_abs = sim_row + sim_origin
        zv_row = _interp_no_extrapolation(tv, zv, sim_row_abs)
        zp_row = _interp_no_extrapolation(tp, zp, sim_row_abs)
        candidates = rel_logged - zv_row - zp_row
        candidates = candidates[np.isfinite(candidates)]
        if len(candidates):
            offset = float(np.nanmedian(candidates))
    relative_z = vehicle_z + platform_z + offset
    closing = _derivative_from_position(grid, relative_z, window_sec=0.6, center=True)
    return {
        "t_abs": grid, "t": grid - grid[0], "vehicle_z": vehicle_z, "platform_z": platform_z,
        "relative_z": relative_z, "closing_rate": closing, "geometry_offset": offset,
        "vehicle": vehicle, "platform": platform,
    }


def _closing_rate_from_position(df: pd.DataFrame, t: np.ndarray, end_idx: Optional[int] = None, window_sec: float = 0.6) -> np.ndarray:
    """Offline closing rate from synchronized position streams when available."""
    model = _synchronized_vertical_model(df)
    target_t = _physics_time_column(df, t)
    if model is not None:
        target_abs = target_t.copy()
        sim_raw, _ = _simulation_time_column(df)
        # _simulation_time_column returns elapsed values; restore the native
        # source origin using the first finite raw SIM stamp.
        raw_col = next((c for c in ("flow_source_timestamp_sec", "target_source_timestamp_sec", "command_source_flow_timestamp_sec", "flow_timestamp_sec") if c in df.columns and np.isfinite(numeric_column(df,c)).any()), None)
        if raw_col is not None:
            raw = numeric_column(df, raw_col); first = raw[np.isfinite(raw)][0]
            target_abs = target_t + first
        result = _interp_no_extrapolation(model["t_abs"], model["closing_rate"], target_abs)
        if end_idx is not None and 0 <= int(end_idx) < len(result):
            result[int(end_idx) + 1:] = np.nan
        return result
    if "relative_z_m" not in df.columns:
        return np.full(len(t), np.nan)
    relz = numeric_column(df, "relative_z_m")
    return _derivative_from_position(target_t, relz, window_sec=window_sec, center=True, end_idx=end_idx)


def dominant_frequency_fft(t: np.ndarray, y: np.ndarray, min_frequency_hz: float = 0.01, max_frequency_hz: float = 2.0) -> Optional[dict]:
    """Uniform-grid Hann-windowed amplitude spectrum."""
    mask = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(mask) < 20:
        return None
    tt, yy = np.asarray(t)[mask], np.asarray(y)[mask]
    order = np.argsort(tt); tt, yy = tt[order], yy[order]
    keep = np.r_[True, np.diff(tt) > 1e-9]; tt, yy = tt[keep], yy[keep]
    span = float(tt[-1] - tt[0]); dt = median_positive_dt(tt)
    if len(tt) < 20 or not np.isfinite(dt) or dt <= 1e-9 or span <= 0:
        return None
    n = int(max(32, math.floor(span / dt) + 1))
    tu = np.linspace(tt[0], tt[-1], n)
    yu = np.interp(tu, tt, yy); yu -= float(np.mean(yu))
    window = np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=tu[1] - tu[0])
    spec = np.abs(np.fft.rfft(yu * window)) * (2.0 / max(np.sum(window), 1e-12))
    valid = (freqs >= min_frequency_hz) & (freqs <= max_frequency_hz)
    if not np.any(valid):
        return None
    idxs = np.where(valid)[0]; idx = int(idxs[np.argmax(spec[valid])])
    return {"frequency_hz": float(freqs[idx]), "period_s": float(1.0 / freqs[idx]), "freqs": freqs, "spectrum": spec}


def plot_timebase_diagnostics(df: pd.DataFrame, output_dir: str | Path):
    sim_t, sim_col = _simulation_time_column(df); wall_t, wall_col = _wall_time_column(df)
    if sim_t is None or wall_t is None:
        print("Skipping timebase diagnostics. Missing coherent SIM or wall axis."); return
    common = np.isfinite(sim_t) & np.isfinite(wall_t)
    if common.sum() < 5:
        print("Skipping timebase diagnostics. Not enough common samples."); return
    idx = np.where(common)[0]; global_rtf = (sim_t[idx[-1]] - sim_t[idx[0]]) / (wall_t[idx[-1]] - wall_t[idx[0]])
    computed = _rolling_regression_slope(wall_t, sim_t, 2.0, True, min_pts=8)
    nrows = 3 if "clock_px4_local_rate_per_wall" in df.columns else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.1*nrows), sharex=True)
    fig.suptitle(f"Clock coherence ({sim_col} vs {wall_col}); run-wide SIM/wall={global_rtf:.3f}")
    axes[0].plot(wall_t, sim_t, label="elapsed SIM source time")
    limit=max(np.nanmax(wall_t),np.nanmax(sim_t)); axes[0].plot([0,limit],[0,limit],'--',label="RTF=1")
    axes[0].set_ylabel("SIM time [s]"); axes[0].grid(True); axes[0].legend(fontsize=8)
    axes[1].plot(wall_t, computed, label="analyser local rate (2 s fit)")
    if "clock_sim_local_rate_per_wall" in df.columns:
        axes[1].plot(wall_t, numeric_column(df,"clock_sim_local_rate_per_wall"), alpha=.55, label="logged SIM local rate")
    if "clock_sim_run_rate_per_wall" in df.columns:
        axes[1].plot(wall_t, numeric_column(df,"clock_sim_run_rate_per_wall"), '--', label="logged SIM run rate")
    axes[1].axhline(global_rtf, linestyle=':', label=f"run-wide {global_rtf:.3f}")
    axes[1].set_ylabel("SIM / wall [-]"); axes[1].grid(True); axes[1].legend(fontsize=8)
    if nrows == 3:
        px4=numeric_column(df,"clock_px4_local_rate_per_wall")
        sim=numeric_column(df,"clock_sim_local_rate_per_wall") if "clock_sim_local_rate_per_wall" in df.columns else np.full(len(df),np.nan)
        axes[2].plot(wall_t, px4, label="PX4 local rate")
        axes[2].plot(wall_t, sim, label="SIM local rate", alpha=.75)
        axes[2].plot(wall_t, px4-sim, label="PX4 - SIM rate", alpha=.55)
        axes[2].axhline(0, linestyle='--', linewidth=1)
        axes[2].set_ylabel("rate [-]"); axes[2].grid(True); axes[2].legend(fontsize=8)
    axes[-1].set_xlabel("wall elapsed time [s]")
    save_current_figure(output_dir,"timebase_diagnostics.png")


def plot_source_data_age(df: pd.DataFrame, output_dir: str | Path):
    wall_t, _ = _wall_time_column(df)
    if wall_t is None:
        print("Skipping source data age. No wall axis."); return
    log_wall_col = "log_write_wall_timestamp_sec" if "log_write_wall_timestamp_sec" in df.columns else "wall_timestamp"
    if log_wall_col not in df.columns:
        return
    log_wall = numeric_column(df, log_wall_col)
    specs = [
        ("camera_receipt_wall_timestamp_sec", "camera frame"),
        ("vision_result_available_wall_timestamp_sec", "vision result"),
        ("vehicle_local_position_receipt_wall_timestamp_sec", "PX4 local position"),
        ("vehicle_attitude_receipt_wall_timestamp_sec", "PX4 attitude"),
        ("platform_pose_receipt_wall_timestamp_sec", "platform pose"),
    ]
    available=[]
    for col,label in specs:
        if col in df.columns:
            age=1000.0*(log_wall-numeric_column(df,col)); age[~np.isfinite(age)]=np.nan
            if np.isfinite(age).sum(): available.append((age,label))
    if not available:
        print("Skipping source data age. New receipt timestamps not present."); return
    fig,ax=plt.subplots(figsize=(11,5.5)); fig.suptitle("Age of held asynchronous sources at CSV write")
    for age,label in available: ax.plot(wall_t,age,label=label,alpha=.8)
    ax.axhline(0,linestyle='--',linewidth=1); ax.set_ylabel("age [ms]"); ax.set_xlabel("wall elapsed time [s]")
    ax.grid(True); ax.legend(fontsize=8); save_current_figure(output_dir,"source_data_age.png")


def plot_vertical_kinematics_validation(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path):
    """Native-stream vertical validation with robust, disclosed visualization.

    The online estimates are never altered. The primary offline references are
    position derivatives evaluated on their native clocks. Extreme online points
    remain in metrics and are counted when the central 0.5--99.5 percentile
    viewport hides them. Hampel flags are applied to residuals, not velocities.
    """
    model = _synchronized_vertical_model(df)
    if model is None:
        return _legacy_plot_vertical_kinematics_validation(df, t, output_dir)

    grid = model["t"]
    vehicle, platform = model["vehicle"], model["platform"]
    vd, pd_ = vehicle["df"], platform["df"]
    tv_abs, tp_abs = vehicle["t_abs"], platform["t_abs"]
    origin = model["t_abs"][0]
    tv, tp = tv_abs - origin, tp_abs - origin

    zv = pd.to_numeric(vd["vehicle_z_m"], errors="coerce").to_numpy(float)
    zp = pd.to_numeric(pd_["platform_z_m"], errors="coerce").to_numpy(float)
    vv = pd.to_numeric(vd["vehicle_vz_m_s"], errors="coerce").to_numpy(float) if "vehicle_vz_m_s" in vd else np.full(len(vd), np.nan)
    pv = pd.to_numeric(pd_["platform_vz_m_s"], errors="coerce").to_numpy(float) if "platform_vz_m_s" in pd_ else np.full(len(pd_), np.nan)

    # Different bandwidths are intentional and disclosed: 0.25 s preserves more
    # vehicle dynamics; 0.60 s is appropriate for the slow platform and robust
    # touchdown/closing-rate estimate.
    vv_d_fast = _derivative_from_position(tv_abs, zv, 0.25, True, prefilter_sec=0.10)
    vv_d_slow = _derivative_from_position(tv_abs, zv, 0.60, True, prefilter_sec=0.15)
    pv_d = _derivative_from_position(tp_abs, zp, 0.60, True, prefilter_sec=0.15)

    vv_g = _interp_no_extrapolation(tv_abs, vv, model["t_abs"])
    vv_fast_g = _interp_no_extrapolation(tv_abs, vv_d_fast, model["t_abs"])
    vv_slow_g = _interp_no_extrapolation(tv_abs, vv_d_slow, model["t_abs"])
    pv_g = _interp_no_extrapolation(tp_abs, pv, model["t_abs"])
    pv_dg = _interp_no_extrapolation(tp_abs, pv_d, model["t_abs"])

    sim_row, _ = _simulation_time_column(df)
    composed = numeric_column(df, "relative_vz_m_s") if "relative_vz_m_s" in df.columns else np.full(len(df), np.nan)
    if sim_row is not None:
        raw_col = next((c for c in ("flow_source_timestamp_sec", "flow_timestamp_sec") if c in df.columns and np.isfinite(numeric_column(df, c)).any()), None)
        origin_sim = numeric_column(df, raw_col)[np.isfinite(numeric_column(df, raw_col))][0] if raw_col else origin
        composed_g = _interp_no_extrapolation(sim_row + origin_sim, composed, model["t_abs"])
    else:
        composed_g = np.full(len(grid), np.nan)

    closing = model["closing_rate"]
    vehicle_residual = vv_g - vv_fast_g
    closing_residual = composed_g - closing
    vehicle_outliers = _hampel_mask(grid, vehicle_residual, window_sec=1.0, threshold_sigma=4.0)
    closing_outliers = _hampel_mask(grid, closing_residual, window_sec=1.0, threshold_sigma=4.0)

    # Main quantitative region: DESCEND, before feet-on-surface. If unavailable,
    # fall back to all common samples rather than fabricating a phase boundary.
    metric_mask = np.ones(len(grid), dtype=bool)
    spans = _mission_phase_spans(df, t)
    descend_spans = [(a, b) for a, b, s in spans if s == "descend"]
    if descend_spans:
        metric_mask[:] = False
        for a, b in descend_spans:
            metric_mask |= (grid >= a) & (grid <= b)
    touch = _detect_touchdown(df, t)
    if touch is not None and np.isfinite(touch.get("t", np.nan)):
        metric_mask &= grid <= float(touch["t"])

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    fig.suptitle("Vertical kinematics: native streams, robust view, disclosed outliers")

    # Map row-aligned phase spans directly to this common elapsed-SIM grid.
    for k, ax in enumerate(axes):
        shade_mission_phases(ax, df, t, label_once=(k == 0))
        ax.grid(True)

    axes[0].plot(grid, -model["relative_z"], linewidth=1.7, label="reconstructed height above deck")
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("height [m]")
    axes[0].legend(fontsize=8)

    axes[1].plot(grid, pv_g, alpha=0.25, linewidth=0.9, label="online platform velocity from pose")
    axes[1].plot(grid, pv_dg, linewidth=1.8, label="offline d(platform pose)/dt_SIM (0.60 s)")
    axes[1].set_title(_robust_comparison_metrics(pv_dg, pv_g, metric_mask), fontsize=9)
    axes[1].set_ylabel("platform vz [m/s]")
    _apply_robust_limits(axes[1], pv_g, pv_dg)
    axes[1].legend(fontsize=8)

    axes[2].plot(grid, vv_g, alpha=0.25, linewidth=0.9, label="PX4 velocity state (unaltered)")
    axes[2].plot(grid, vv_fast_g, linewidth=1.8, label="offline d(PX4 position)/dt_PX4 (0.25 s)")
    axes[2].plot(grid, vv_slow_g, linestyle=":", linewidth=1.2, alpha=0.75, label="same derivative (0.60 s sensitivity)")
    axes[2].scatter(grid[vehicle_outliers], vv_g[vehicle_outliers], marker="x", s=22,
                    label="residual Hampel outlier (4 sigma)", zorder=5)
    axes[2].set_title(_robust_comparison_metrics(vv_fast_g, vv_g, metric_mask, vehicle_outliers), fontsize=9)
    axes[2].set_ylabel("vehicle vz [m/s]")
    _apply_robust_limits(axes[2], vv_g, vv_fast_g, vv_slow_g)
    axes[2].legend(fontsize=8)

    axes[3].plot(grid, composed_g, alpha=0.22, linewidth=0.9, label="online composed closing rate (unaltered)")
    axes[3].plot(grid, closing, linewidth=1.9, label="offline d(synchronized relative position)/dt_SIM (0.60 s)")
    axes[3].scatter(grid[closing_outliers], composed_g[closing_outliers], marker="x", s=22,
                    label="residual Hampel outlier (4 sigma)", zorder=5)
    axes[3].set_title(_robust_comparison_metrics(closing, composed_g, metric_mask, closing_outliers), fontsize=9)
    axes[3].set_ylabel("closing rate [m/s]")
    _apply_robust_limits(axes[3], composed_g, closing)
    axes[3].legend(fontsize=8)

    axes[4].plot(grid, vehicle_residual, alpha=0.75, label="PX4 velocity - 0.25 s position derivative")
    axes[4].plot(grid, closing_residual, alpha=0.75, label="online closing - offline closing")
    axes[4].scatter(grid[vehicle_outliers], vehicle_residual[vehicle_outliers], marker="x", s=20)
    axes[4].scatter(grid[closing_outliers], closing_residual[closing_outliers], marker="x", s=20)
    axes[4].axhline(0, linestyle="--", linewidth=1)
    axes[4].set_ylabel("residual [m/s]")
    axes[4].set_xlabel("elapsed simulation time [s]")
    _apply_robust_limits(axes[4], vehicle_residual, closing_residual)
    axes[4].legend(fontsize=8)

    save_current_figure(output_dir, "vertical_kinematics_validation.png")


# ---------------------------------------------------------------------------
# Physically plausible offline kinematics
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# The CSV row is already the node's coherent "latest-state" snapshot.  Rebuilding
# relative position by independently remapping PX4 and platform samples through a
# changing affine clock fit created artificial jumps: tiny time-map changes were
# multiplied by the two streams' position slopes and then amplified by
# differentiation.  The offline relative estimate below therefore uses the
# logged relative_z_m snapshot on the visual SIM clock, deduplicated by vision
# result, and differentiates that coherent position directly.
#
# Native PX4 position derivatives are still evaluated on the PX4 source clock,
# but all time vectors are translated close to zero before regression.  This is
# essential numerically: prefix/covariance calculations on Unix-epoch-sized
# timestamps (~1.8e9 s) suffer catastrophic cancellation and can produce an
# almost-zero derivative even while position visibly changes.


def _stable_rolling_regression_slope(
    t_real: np.ndarray,
    y: np.ndarray,
    window_sec: float,
    center: bool,
    min_pts: int = 3,
) -> np.ndarray:
    """Numerically conditioned local least-squares derivative.

    Each fit uses time relative to the local window mean, so source timestamps
    may be Unix epoch, PX4 epoch, boot time, or Gazebo time without changing the
    result.  The window remains defined in elapsed seconds and supports
    irregular sampling.
    """
    t_real = np.asarray(t_real, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t_real)
    out = np.full(n, np.nan, dtype=float)
    valid_t = np.isfinite(t_real)
    if np.count_nonzero(valid_t) < min_pts:
        return out

    # Keep search values well conditioned too. Translation preserves intervals.
    origin = float(t_real[valid_t][0])
    ts = t_real - origin

    for i in range(n):
        if not (np.isfinite(ts[i]) and np.isfinite(y[i])):
            continue
        if center:
            lo_b = ts[i] - 0.5 * window_sec
            hi_b = ts[i] + 0.5 * window_sec
        else:
            lo_b = ts[i] - window_sec
            hi_b = ts[i]
        lo = int(np.searchsorted(ts, lo_b, side="left"))
        hi = int(np.searchsorted(ts, hi_b, side="right"))
        tv = ts[lo:hi]
        yv = y[lo:hi]
        good = np.isfinite(tv) & np.isfinite(yv)
        if np.count_nonzero(good) < min_pts:
            continue
        tv = tv[good]
        yv = yv[good]
        tc = tv - float(np.mean(tv))
        yc = yv - float(np.mean(yv))
        denom = float(np.dot(tc, tc))
        if denom <= 1e-12:
            continue
        out[i] = float(np.dot(tc, yc) / denom)
    return out


# Override the earlier implementation globally. Existing helpers and legacy plots
# now benefit from the same numerical conditioning.
_rolling_regression_slope = _stable_rolling_regression_slope


def _derivative_from_position(
    t_real: np.ndarray,
    position: np.ndarray,
    window_sec: float,
    center: bool,
    end_idx: Optional[int] = None,
    prefilter_sec: float = 0.15,
) -> np.ndarray:
    """Robust position derivative with a conditioned time axis.

    A short time-windowed median rejects isolated position glitches; a longer
    local regression provides the derivative.  No point is differentiated across
    a non-finite timestamp or beyond ``end_idx``.
    """
    t_real = np.asarray(t_real, dtype=float)
    position = np.asarray(position, dtype=float)
    n = len(position)
    out = np.full(n, np.nan, dtype=float)
    end = n if end_idx is None else max(0, min(n, int(end_idx) + 1))
    if end < 3:
        return out
    t_seg = t_real[:end]
    pos_seg = position[:end]
    finite_t = np.isfinite(t_seg)
    if np.count_nonzero(finite_t) < 3:
        return out
    t0 = float(t_seg[finite_t][0])
    t_local = t_seg - t0
    filtered = _rolling_median_time(
        t_local, pos_seg, window_sec=prefilter_sec, center=center
    )
    out[:end] = _stable_rolling_regression_slope(
        t_local, filtered, window_sec=window_sec, center=center
    )
    return out


def _strictly_increasing_mask(t: np.ndarray, max_gap_sec: Optional[float] = None) -> np.ndarray:
    """Keep finite samples in row order without sorting clock resets into data."""
    t = np.asarray(t, dtype=float)
    keep = np.zeros(len(t), dtype=bool)
    last = None
    positive = np.diff(t[np.isfinite(t)])
    positive = positive[(positive > 1e-9) & np.isfinite(positive)]
    median_dt = float(np.median(positive)) if len(positive) else float("nan")
    if max_gap_sec is None and np.isfinite(median_dt):
        max_gap_sec = max(0.5, 25.0 * median_dt)
    for i, value in enumerate(t):
        if not np.isfinite(value):
            continue
        if last is None:
            keep[i] = True
            last = float(value)
            continue
        dt = float(value - last)
        if dt <= 1e-9:
            continue
        # Large jumps are clock discontinuities. Keep the new sample as the start
        # of a later segment only in repaired-time helpers; do not bridge it here.
        if max_gap_sec is not None and dt > max_gap_sec:
            continue
        keep[i] = True
        last = float(value)
    return keep


def _dominant_timestamp_scale_mask(t: np.ndarray) -> np.ndarray:
    """Reject isolated boot-time/epoch-time unit or origin switches.

    Some logs contain one or two PX4 stamps near 90 s among thousands near
    1.8e9 s.  Sorting those values creates an artificial multi-decade time span.
    We retain the dominant order-of-magnitude cluster and disclose the count.
    """
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(t) & (np.abs(t) > 1e-12)
    mask = np.zeros(len(t), dtype=bool)
    if np.count_nonzero(finite) < 3:
        return finite
    decades = np.floor(np.log10(np.abs(t[finite])))
    values, counts = np.unique(decades, return_counts=True)
    dominant = values[int(np.argmax(counts))]
    mask[finite] = decades == dominant
    return mask


def _repaired_elapsed_source_time(
    raw_source_time: np.ndarray,
    receipt_monotonic_time: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build a continuous elapsed source clock without inventing dynamics.

    Normal source increments are retained exactly. Isolated origin switches,
    backwards steps, or implausibly large jumps are replaced by the expected
    increment inferred from monotonic receipt time and the median valid
    source/receipt rate. This only repairs the clock coordinate; position and
    velocity values remain untouched.
    """
    raw = np.asarray(raw_source_time, dtype=float)
    n = len(raw)
    receipt = (
        np.asarray(receipt_monotonic_time, dtype=float)
        if receipt_monotonic_time is not None
        else np.full(n, np.nan)
    )
    dominant = _dominant_timestamp_scale_mask(raw)

    d_raw = np.diff(raw)
    d_receipt = np.diff(receipt)
    candidate = (
        dominant[:-1] & dominant[1:]
        & np.isfinite(d_raw) & (d_raw > 1e-6) & (d_raw < 0.5)
    )
    rate_candidates = (
        d_raw[candidate] / d_receipt[candidate]
        if np.any(candidate & np.isfinite(d_receipt) & (d_receipt > 1e-6))
        else np.array([], dtype=float)
    )
    rate_candidates = rate_candidates[
        np.isfinite(rate_candidates) & (rate_candidates > 0.02) & (rate_candidates < 5.0)
    ]
    median_rate = float(np.median(rate_candidates)) if len(rate_candidates) else 1.0

    valid_raw_dt = d_raw[candidate]
    nominal_dt = (
        float(np.median(valid_raw_dt))
        if len(valid_raw_dt)
        else 0.02
    )

    elapsed = np.zeros(n, dtype=float)
    repaired = np.zeros(n, dtype=bool)
    for i in range(1, n):
        expected = nominal_dt
        if (
            np.isfinite(receipt[i]) and np.isfinite(receipt[i - 1])
            and receipt[i] > receipt[i - 1]
        ):
            expected = float((receipt[i] - receipt[i - 1]) * median_rate)
        expected = float(np.clip(expected, 1e-4, 0.25))

        raw_dt = raw[i] - raw[i - 1] if np.isfinite(raw[i]) and np.isfinite(raw[i - 1]) else np.nan
        use_raw = (
            dominant[i] and dominant[i - 1]
            and np.isfinite(raw_dt) and raw_dt > 1e-6 and raw_dt < 0.5
            and 0.25 * expected <= raw_dt <= 4.0 * expected
        )
        dt = float(raw_dt) if use_raw else expected
        repaired[i] = not use_raw
        elapsed[i] = elapsed[i - 1] + dt

    info = {
        "median_source_per_receipt_rate": median_rate,
        "nominal_dt_sec": nominal_dt,
        "repaired_count": int(np.count_nonzero(repaired)),
        "sample_count": int(n),
        "discarded_scale_count": int(np.count_nonzero(~dominant & np.isfinite(raw))),
    }
    return elapsed, repaired, info


def _coherent_relative_state_model(df: pd.DataFrame) -> Optional[dict]:
    """Relative kinematics from one coherent logged state snapshot.

    This intentionally does NOT recombine separately time-mapped PX4 and platform
    positions. ``relative_z_m`` and ``relative_vz_m_s`` were computed online from
    the latest states available to the controller and logged together with the
    visual source stamp. Deduplicating by vision result gives one coherent sample
    per control update and avoids zero-order-held row weighting.
    """
    time_col = next(
        (
            c for c in (
                "flow_source_timestamp_sec",
                "target_source_timestamp_sec",
                "command_source_flow_timestamp_sec",
                "flow_timestamp_sec",
            )
            if c in df.columns and np.isfinite(numeric_column(df, c)).sum() >= 3
        ),
        None,
    )
    if time_col is None or "relative_z_m" not in df.columns:
        return None

    values = ["relative_z_m"]
    velocity_col = "relative_vz_m_s" if "relative_vz_m_s" in df.columns else None
    if velocity_col:
        values.append(velocity_col)
    if "mission_substate" in df.columns:
        values.append("mission_substate")

    counter = "vision_result_count" if "vision_result_count" in df.columns else None
    stream = _unique_source_frame(df, time_col, values, counter)
    if len(stream) < 8:
        return None

    t_abs = pd.to_numeric(stream[time_col], errors="coerce").to_numpy(float)
    z_raw = pd.to_numeric(stream["relative_z_m"], errors="coerce").to_numpy(float)
    v_raw = (
        pd.to_numeric(stream[velocity_col], errors="coerce").to_numpy(float)
        if velocity_col and velocity_col in stream.columns
        else np.full(len(stream), np.nan)
    )
    good = np.isfinite(t_abs) & np.isfinite(z_raw)
    t_abs, z_raw, v_raw = t_abs[good], z_raw[good], v_raw[good]
    if len(t_abs) < 8:
        return None
    order = np.argsort(t_abs)
    t_abs, z_raw, v_raw = t_abs[order], z_raw[order], v_raw[order]
    unique = np.r_[True, np.diff(t_abs) > 1e-9]
    t_abs, z_raw, v_raw = t_abs[unique], z_raw[unique], v_raw[unique]
    if len(t_abs) < 8:
        return None

    t = t_abs - t_abs[0]

    # Height is only lightly de-spiked. The logged relative position is already
    # physically coherent and should not be replaced by an independently
    # synchronized recombination.
    z_smoothed = _rolling_median_time(t, z_raw, window_sec=0.18, center=True)

    # A 0.45 s local slope preserves the slow platform/descent dynamics while
    # rejecting frame-to-frame position jitter. A 0.60 s sensitivity curve is
    # retained for validation; neither is called ground truth.
    v_position = _derivative_from_position(
        t, z_raw, window_sec=0.45, center=True, prefilter_sec=0.12
    )
    v_position_slow = _derivative_from_position(
        t, z_raw, window_sec=0.60, center=True, prefilter_sec=0.15
    )
    v_online_smoothed = _rolling_median_time(
        t, v_raw, window_sec=0.18, center=True
    )

    return {
        "t_abs": t_abs,
        "t": t,
        "relative_z": z_smoothed,
        "relative_z_raw": z_raw,
        "closing_rate": v_position,
        "closing_rate_slow": v_position_slow,
        "online_closing_rate": v_raw,
        "online_closing_rate_smoothed": v_online_smoothed,
        "source_time_column": time_col,
        "sample_count": len(t),
        "geometry_offset": float("nan"),  # retained only for legacy wrapper compatibility
    }


# Keep the established function name so all downstream plots/summary helpers use
# the coherent relative-state model automatically.
def _synchronized_vertical_model(df: pd.DataFrame, grid_dt: Optional[float] = None) -> Optional[dict]:
    return _coherent_relative_state_model(df)


def _closing_rate_from_position(
    df: pd.DataFrame,
    t: np.ndarray,
    end_idx: Optional[int] = None,
    window_sec: float = 0.45,
) -> np.ndarray:
    model = _coherent_relative_state_model(df)
    if model is None:
        if "relative_z_m" not in df.columns:
            return np.full(len(t), np.nan)
        return _derivative_from_position(
            _physics_time_column(df, t),
            numeric_column(df, "relative_z_m"),
            window_sec=window_sec,
            center=True,
            end_idx=end_idx,
            prefilter_sec=min(0.15, window_sec / 3.0),
        )

    target_t = _physics_time_column(df, t)
    result = _interp_no_extrapolation(model["t"], model["closing_rate"], target_t)
    if end_idx is not None and 0 <= int(end_idx) < len(result):
        result[int(end_idx) + 1:] = np.nan
    return result


def _vehicle_validation_model(df: pd.DataFrame, relative_origin_abs: float) -> Optional[dict]:
    """PX4 state and position derivative on a repaired native PX4 clock."""
    source_col = next(
        (
            c for c in (
                "vehicle_local_position_sample_px4_timestamp_sec",
                "vehicle_local_position_publication_px4_timestamp_sec",
                "vehicle_px4_timestamp_sec",
            )
            if c in df.columns and np.isfinite(numeric_column(df, c)).sum() >= 3
        ),
        None,
    )
    if source_col is None or "vehicle_z_m" not in df.columns:
        return None

    plot_col = next(
        (
            c for c in ("flow_source_timestamp_sec", "flow_timestamp_sec")
            if c in df.columns and np.isfinite(numeric_column(df, c)).sum() >= 3
        ),
        None,
    )
    values = ["vehicle_z_m", "vehicle_vz_m_s", source_col]
    for c in (
        "vehicle_local_position_receipt_monotonic_timestamp_sec",
        plot_col,
    ):
        if c and c in df.columns and c not in values:
            values.append(c)

    counter = (
        "vehicle_local_position_message_count"
        if "vehicle_local_position_message_count" in df.columns
        else None
    )
    # Deduplicate in row order first; do not sort a clock-origin glitch.
    cols = [c for c in values if c in df.columns]
    if counter:
        cols.append(counter)
    stream = df[cols].copy()
    if counter:
        stream = stream.drop_duplicates(counter, keep="last")
    else:
        stream = stream.drop_duplicates(source_col, keep="last")
    stream = stream.reset_index(drop=True)
    if len(stream) < 8:
        return None

    raw_t = pd.to_numeric(stream[source_col], errors="coerce").to_numpy(float)
    receipt = (
        pd.to_numeric(
            stream["vehicle_local_position_receipt_monotonic_timestamp_sec"],
            errors="coerce",
        ).to_numpy(float)
        if "vehicle_local_position_receipt_monotonic_timestamp_sec" in stream.columns
        else None
    )
    native_t, repaired, repair_info = _repaired_elapsed_source_time(raw_t, receipt)
    z = pd.to_numeric(stream["vehicle_z_m"], errors="coerce").to_numpy(float)
    v = (
        pd.to_numeric(stream["vehicle_vz_m_s"], errors="coerce").to_numpy(float)
        if "vehicle_vz_m_s" in stream.columns
        else np.full(len(stream), np.nan)
    )

    if plot_col and plot_col in stream.columns:
        plot_abs = pd.to_numeric(stream[plot_col], errors="coerce").to_numpy(float)
        plot_t = plot_abs - relative_origin_abs
    else:
        plot_t = native_t - native_t[0]

    d_fast = _derivative_from_position(
        native_t, z, window_sec=0.25, center=True, prefilter_sec=0.10
    )
    d_slow = _derivative_from_position(
        native_t, z, window_sec=0.60, center=True, prefilter_sec=0.15
    )
    return {
        "plot_t": plot_t,
        "native_t": native_t,
        "z": z,
        "v": v,
        "d_fast": d_fast,
        "d_slow": d_slow,
        "repaired_mask": repaired,
        "repair_info": repair_info,
        "source_column": source_col,
    }


def _platform_validation_model(df: pd.DataFrame, relative_origin_abs: float) -> Optional[dict]:
    """Platform pose derivative on its estimated Gazebo SIM clock."""
    time_col = next(
        (
            c for c in (
                "platform_pose_estimated_sim_timestamp_sec",
                "flow_source_timestamp_sec",
                "flow_timestamp_sec",
            )
            if c in df.columns and np.isfinite(numeric_column(df, c)).sum() >= 3
        ),
        None,
    )
    if time_col is None or "platform_z_m" not in df.columns:
        return None

    values = ["platform_z_m", "platform_vz_m_s"]
    counter = "platform_pose_message_count" if "platform_pose_message_count" in df.columns else None
    stream = _unique_source_frame(df, time_col, values, counter)
    if len(stream) < 8:
        return None
    t_abs = pd.to_numeric(stream[time_col], errors="coerce").to_numpy(float)
    z = pd.to_numeric(stream["platform_z_m"], errors="coerce").to_numpy(float)
    v = (
        pd.to_numeric(stream["platform_vz_m_s"], errors="coerce").to_numpy(float)
        if "platform_vz_m_s" in stream.columns
        else np.full(len(stream), np.nan)
    )
    d = _derivative_from_position(
        t_abs - t_abs[0], z, window_sec=0.60, center=True, prefilter_sec=0.15
    )
    return {
        "plot_t": t_abs - relative_origin_abs,
        "native_t": t_abs - t_abs[0],
        "z": z,
        "v": v,
        "d": d,
        "source_column": time_col,
    }


def plot_vertical_kinematics_validation(
    df: pd.DataFrame,
    t: np.ndarray,
    output_dir: str | Path,
):
    """Plausible offline validation without cross-stream position recombination."""
    relative = _coherent_relative_state_model(df)
    if relative is None:
        return _legacy_plot_vertical_kinematics_validation(df, t, output_dir)

    grid = relative["t"]
    origin_abs = float(relative["t_abs"][0])
    vehicle = _vehicle_validation_model(df, origin_abs)
    platform = _platform_validation_model(df, origin_abs)

    raw_z = relative["relative_z_raw"]
    smooth_z = relative["relative_z"]
    online = relative["online_closing_rate"]
    online_s = relative["online_closing_rate_smoothed"]
    position_v = relative["closing_rate"]
    position_v_slow = relative["closing_rate_slow"]

    residual = online_s - position_v
    outliers = _hampel_mask(
        grid, residual, window_sec=1.0, threshold_sigma=4.0
    )

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    fig.suptitle(
        "Vertical kinematics: coherent offline estimates and native-clock validation"
    )
    for k, ax in enumerate(axes):
        shade_mission_phases(ax, df, t, label_once=(k == 0))
        ax.grid(True)

    # Height: raw coherent online snapshot plus a very light robust de-spike.
    axes[0].plot(
        grid, -raw_z, alpha=0.22, linewidth=0.8,
        label="logged height above deck (coherent snapshot)",
    )
    axes[0].plot(
        grid, -smooth_z, linewidth=1.8,
        label="offline robust height (0.18 s median)",
    )
    axes[0].axhline(0.0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("height [m]")
    axes[0].legend(fontsize=8)

    # Platform pose is naturally on estimated SIM time.
    if platform is not None:
        pg = _interp_no_extrapolation(platform["plot_t"], platform["v"], grid)
        pdg = _interp_no_extrapolation(platform["plot_t"], platform["d"], grid)
        axes[1].plot(grid, pg, alpha=0.25, linewidth=0.9,
                     label="online platform velocity from pose")
        axes[1].plot(grid, pdg, linewidth=1.8,
                     label="offline d(platform pose)/dt_SIM (0.60 s)")
        axes[1].set_title(
            _robust_comparison_metrics(pdg, pg), fontsize=9
        )
        _apply_robust_limits(axes[1], pg, pdg)
    else:
        axes[1].text(0.5, 0.5, "platform native stream unavailable",
                     transform=axes[1].transAxes, ha="center")
    axes[1].set_ylabel("platform vz [m/s]")
    axes[1].legend(fontsize=8)

    # PX4 derivative uses repaired native elapsed time. No epoch-sized arithmetic
    # and no rowwise PX4->SIM affine remapping enters the derivative.
    vehicle_residual_grid = np.full(len(grid), np.nan)
    if vehicle is not None:
        vg = _interp_no_extrapolation(vehicle["plot_t"], vehicle["v"], grid)
        vdg = _interp_no_extrapolation(vehicle["plot_t"], vehicle["d_fast"], grid)
        vdsg = _interp_no_extrapolation(vehicle["plot_t"], vehicle["d_slow"], grid)
        vehicle_residual_grid = vg - vdg
        vehicle_outliers = _hampel_mask(
            grid, vehicle_residual_grid, window_sec=1.0, threshold_sigma=4.0
        )
        axes[2].plot(grid, vg, alpha=0.25, linewidth=0.9,
                     label="PX4 velocity state (unaltered)")
        axes[2].plot(grid, vdg, linewidth=1.8,
                     label="offline d(PX4 position)/dt_PX4 (0.25 s)")
        axes[2].plot(grid, vdsg, linestyle=":", linewidth=1.2, alpha=0.75,
                     label="same derivative (0.60 s sensitivity)")
        axes[2].scatter(
            grid[vehicle_outliers], vg[vehicle_outliers],
            marker="x", s=22, zorder=5,
            label="residual Hampel outlier (4 sigma)",
        )
        title = _robust_comparison_metrics(
            vdg, vg, outliers=vehicle_outliers
        )
        info = vehicle["repair_info"]
        title += (
            f"; clock increments repaired "
            f"{info['repaired_count']}/{info['sample_count']}"
        )
        axes[2].set_title(title, fontsize=9)
        _apply_robust_limits(axes[2], vg, vdg, vdsg)
    else:
        axes[2].text(0.5, 0.5, "vehicle native stream unavailable",
                     transform=axes[2].transAxes, ha="center")
    axes[2].set_ylabel("vehicle vz [m/s]")
    axes[2].legend(fontsize=8)

    # Relative closing: online composition is a real controller estimate; the
    # position slope is an independent consistency estimate, not "ground truth".
    axes[3].plot(grid, online, alpha=0.20, linewidth=0.8,
                 label="online composed closing rate (raw)")
    axes[3].plot(grid, online_s, linewidth=1.55,
                 label="online composed closing rate (0.18 s median)")
    axes[3].plot(grid, position_v, linewidth=1.7,
                 label="offline closing from coherent relative-position slope (0.45 s)")
    axes[3].plot(grid, position_v_slow, linestyle=":", linewidth=1.1, alpha=0.7,
                 label="position-slope sensitivity (0.60 s)")
    axes[3].scatter(
        grid[outliers], online_s[outliers],
        marker="x", s=22, zorder=5,
        label="online-vs-position residual outlier",
    )
    axes[3].set_title(
        _robust_comparison_metrics(position_v, online_s, outliers=outliers),
        fontsize=9,
    )
    axes[3].set_ylabel("closing rate [m/s]")
    _apply_robust_limits(axes[3], online, online_s, position_v)
    axes[3].legend(fontsize=8)

    axes[4].plot(
        grid, vehicle_residual_grid, alpha=0.75,
        label="PX4 velocity - native position derivative",
    )
    axes[4].plot(
        grid, residual, alpha=0.75,
        label="smoothed online closing - relative-position slope",
    )
    axes[4].axhline(0.0, linestyle="--", linewidth=1)
    axes[4].set_ylabel("residual [m/s]")
    axes[4].set_xlabel("elapsed simulation time [s]")
    _apply_robust_limits(axes[4], vehicle_residual_grid, residual)
    axes[4].legend(fontsize=8)

    save_current_figure(output_dir, "vertical_kinematics_validation.png")


def plot_vertical_descent(
    df: pd.DataFrame,
    t: np.ndarray,
    output_dir: str,
):
    """Descent plot using the coherent logged relative state.

    The online composed closing rate is not demoted below a faulty cross-stream
    reconstruction. The independent position-slope curve is shown as a
    consistency estimate, and height is only lightly de-spiked.
    """
    model = _coherent_relative_state_model(df)
    if model is None:
        print("Skipping vertical descent plot. Coherent relative state unavailable.")
        return

    grid = model["t"]
    raw_z = model["relative_z_raw"]
    smooth_z = model["relative_z"]
    online = model["online_closing_rate"]
    online_s = model["online_closing_rate_smoothed"]
    position_v = model["closing_rate"]
    residual = online_s - position_v
    outliers = _hampel_mask(grid, residual, 1.0, 4.0)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11.5), sharex=True)
    fig.suptitle("Vertical loop: plausible relative motion and thrust")

    # Height receives its own axis so its remaining millimetric noise cannot
    # complicate the closing-rate scale.
    axes[0].plot(grid, -raw_z, alpha=0.20, linewidth=0.8,
                 label="logged height above deck")
    axes[0].plot(grid, -smooth_z, linewidth=1.8,
                 label="offline robust height (0.18 s median)")
    axes[0].axhline(0.0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("height [m]")
    axes[0].legend(fontsize=8)

    axes[1].plot(grid, online, alpha=0.20, linewidth=0.8,
                 label="online composed closing rate (raw)")
    axes[1].plot(grid, online_s, linewidth=1.7,
                 label="online composed closing rate (0.18 s median)")
    axes[1].plot(grid, position_v, linewidth=1.65,
                 label="offline relative-position slope (0.45 s)")
    axes[1].scatter(
        grid[outliers], online_s[outliers],
        marker="x", s=22, zorder=5,
        label="online-vs-position residual outlier",
    )
    axes[1].set_title(
        _robust_comparison_metrics(position_v, online_s, outliers=outliers),
        fontsize=9,
    )
    axes[1].set_ylabel("closing rate [m/s]")
    _apply_robust_limits(axes[1], online, online_s, position_v)
    axes[1].legend(fontsize=8)

    if "command_thrust" in df.columns:
        axes[2].plot(t, numeric_column(df, "command_thrust"), label="thrust command")
        axes[2].axhline(
            0.73, linestyle="--", linewidth=1, alpha=0.6,
            label="hover reference 0.73",
        )
    axes[2].set_ylabel("thrust [-]")
    axes[2].set_xlabel("elapsed simulation time [s]")
    axes[2].legend(fontsize=8)

    for k, ax in enumerate(axes):
        shade_mission_phases(ax, df, t, label_once=(k == 0))
        ax.grid(True)

    save_current_figure(output_dir, "vertical_descent.png")



def _append_schema_summary(df: pd.DataFrame, lines: list):
    version = df["diagnostics_schema_version"].dropna().astype(str).iloc[0] if "diagnostics_schema_version" in df.columns and df["diagnostics_schema_version"].notna().any() else "legacy"
    lines.insert(3, f"Diagnostics schema: {version}")
    for col,label in (("camera_source_clock","Camera source clock"),("vision_source_clock","Vision source clock"),("command_source_clock","Command source clock")):
        if col in df.columns:
            vals=df[col].dropna().astype(str); vals=vals[~vals.isin(("","nan"))]
            if len(vals):
                unique=', '.join(sorted(vals.unique()))
                lines.insert(4, f"{label}: {unique}")


_original_compute_summary = compute_summary
def compute_summary(df, t, time_column, time_description, expected_platform_frequency_hz):
    text = _original_compute_summary(
        df, t, time_column, time_description, expected_platform_frequency_hz
    )
    lines = text.rstrip().splitlines()
    _append_schema_summary(df, lines)

    model = _coherent_relative_state_model(df)
    if model is not None:
        lines.append("")
        lines.append("Offline relative-state estimation")
        lines.append("---------------------------------")
        lines.append(
            f"Coherent visual/SIM span: {model['t'][-1] - model['t'][0]:.3f} s; "
            f"unique control/vision samples: {model['sample_count']}"
        )
        lines.append(
            "Height source: logged relative_z_m, lightly de-spiked with a "
            "0.18 s rolling median."
        )
        lines.append(
            "Offline closing-rate consistency estimate: local slope of the same "
            "coherent relative_z_m stream (0.45 s regression window)."
        )
        lines.append(
            "The analyser does not independently remap and recombine PX4 and "
            "platform positions for relative motion."
        )

    vehicle = _vehicle_validation_model(
        df,
        float(model["t_abs"][0]) if model is not None else 0.0,
    )
    if vehicle is not None:
        info = vehicle["repair_info"]
        lines.append(
            f"PX4 source-clock increments repaired for validation: "
            f"{info['repaired_count']} / {info['sample_count']} "
            f"(isolated origin switches/backward or oversized steps only)."
        )

    if "camera_source_clock" in df.columns:
        vals = df["camera_source_clock"].dropna().astype(str)
        if len(vals) and len(vals.unique()) > 1:
            lines.append(
                "WARNING: camera_source_clock changed within this log: "
                + ", ".join(sorted(vals.unique()))
            )
    return "\n".join(lines) + "\n"


def make_default_plots(df: pd.DataFrame, t: np.ndarray, output_dir: str | Path, args):
    _TOUCHDOWN_CACHE.clear()
    plot_timebase_diagnostics(df, output_dir)
    plot_source_data_age(df, output_dir)
    plot_vertical_kinematics_validation(df, t, output_dir)
    plot_target_detection_summary(df, t, output_dir)
    plot_detection_boxes_fov(df, t, args.image_width, args.image_height, output_dir, args.max_boxes)
    plot_probe_acceleration(df, t, output_dir)
    plot_gain_schedule(df, t, output_dir)
    plot_vertical_divergence(df, t, output_dir, divergence_setpoint=None)
    plot_vertical_descent(df, t, output_dir)
    plot_closing_rate_spectrum(df, t, output_dir, expected_frequency_hz=args.platform_frequency_hz)
    plot_drone_platform_position_xyz(df, t, output_dir)

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
		default=0.0,
		help="Expected platform frequency for reference/fitting. Default: disabled; the FFT still reports the measured peak.",
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