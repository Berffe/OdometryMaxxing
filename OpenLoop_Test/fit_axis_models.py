"""
Fit control_law.py's per-axis local model from calibration logs.

Each axis is modeled as e[k+1] = a*e[k] + b*u[k] (see control_law.py's
module docstring). Given one or more diagnostics CSVs (or directories
containing them) produced by calibration_node.py, this script recovers
(a, b) per axis by ordinary least squares on consecutive samples, fits
a small intercept term `c` alongside them to absorb any trim mismatch,
and reports goodness-of-fit.

Usage, one operating point:

	python fit_axis_models.py logs/calibration_20260623_140500.csv

Usage, a whole folder of calibration runs (every *.csv inside it):

	python fit_axis_models.py logs/

Mixing files and folders, and several folders, both work too:

	python fit_axis_models.py logs/batch1 logs/batch2 logs/extra_run.csv

Given 2+ files (however they were specified), this additionally checks
the assumed b(area_fraction) ~ sqrt(area_fraction) scaling and prints a
ready-to-use schedule.

R^2 is not enough on its own to trust b. When a is near 1 (as it
usually is for these axes — see the project notes on why), e[k] alone
already predicts e[k+1] well, so R^2 can look high even when b carries
no real signal at all. To catch that, every AxisFit also reports b's
standard error and a t-statistic (b / stderr); |t| < 2 means b is not
reliably distinguishable from zero given this data's noise, regardless
of how good R^2 looks, and gets flagged the same way a low R^2 does.

This script also flags two other things that silently corrupt a fit
and are easy to miss by eye:

	- A single calibration file is only valid evidence about one
	  operating point if area_fraction actually stayed put during the
	  run. If it swept a wide range instead (most commonly because
	  hover_thrust used in calibration_node.py wasn't quite right, so
	  the "neutral" segments between test steps weren't actually
	  neutral, or because of a genuine ground/platform contact event),
	  the fit is describing a blend of operating points, not the one
	  point's area_fraction mean suggests.
	- Possible ground/platform contact: vehicle_vz_m_s pinned near zero
	  for a sustained stretch while command_thrust is clearly varying
	  is not something free flight produces — it means something
	  external (the ground, the platform) is absorbing the thrust
	  command's effect. Any samples after that point describe contact
	  dynamics, not the free-flight model this script fits.

Plots (written to --output-dir, default fit_axis_output):

	- <csv_stem>_area_fraction.png   one per file: area_fraction (and
	  vehicle_z_m on a second axis) over time, to visually confirm the
	  operating point actually held steady, and to see any contact
	  event directly (z stops changing while the test keeps running).
	- b_vs_area_fraction.png         only when 2+ files are given: the
	  fitted b for each axis against area_fraction, with the assumed
	  sqrt(area_fraction) curve overlaid, so the scaling check from the
	  printed table is also something you can actually look at.

This script needs matplotlib/numpy for the analysis and plots, but not
rclpy — it only reads the CSV that diagnostics_writer.py already produces.
"""

import argparse
import csv
import glob
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# (state column, command column, valid-mask column) per axis.
AXIS_COLUMNS = {
	"roll": ("target_offset_x", "command_roll_rad", "target_found"),
	"pitch": ("target_offset_y", "command_pitch_rad", "target_found"),
	"thrust": ("flow_divergence_1_s", "command_thrust", "flow_valid"),
}


class AxisFit:
	def __init__(
		self,
		a: float,
		b: float,
		c: float,
		r_squared: float,
		n_samples: int,
		b_stderr: float = float("nan"),
	):
		self.a = a
		self.b = b
		self.c = c
		self.r_squared = r_squared
		self.n_samples = n_samples
		self.b_stderr = b_stderr

	@property
	def b_t_stat(self) -> float:
		"""
		b / stderr(b). |t| < ~2 means b is not distinguishable from zero
		at roughly 95% confidence given this data's noise -- i.e. the
		fit may have a high R^2 purely because `a` is near 1 (e[k] alone
		predicts e[k+1] well) while b itself carries no real signal.
		See the module docstring section on why R^2 alone is not enough
		to trust b.
		"""
		if not np.isfinite(self.b_stderr) or self.b_stderr <= 0:
			return float("nan")
		return self.b / self.b_stderr

	@property
	def b_is_significant(self) -> bool:
		t = self.b_t_stat
		return np.isfinite(t) and abs(t) >= 2.0

	def __repr__(self) -> str:
		t = self.b_t_stat
		t_str = f"{t:+.2f}" if np.isfinite(t) else "n/a"
		return (
			f"AxisFit(a={self.a:.4f}, b={self.b:.4f} (t={t_str}), c={self.c:.5f}, "
			f"R^2={self.r_squared:.3f}, n={self.n_samples})"
		)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
	with open(path, newline="") as f:
		return list(csv.DictReader(f))


def _safe_float(value: Optional[str]) -> Optional[float]:
	if value is None or value == "":
		return None

	try:
		return float(value)
	except ValueError:
		return None


def build_pairs(
	rows: List[Dict[str, str]],
	state_col: str,
	command_col: str,
	valid_col: str,
	dt_nominal: float,
	dt_tolerance: float,
	restrict_to_axis: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Build (e[k], u[k], e[k+1]) arrays from consecutive valid rows whose
	time gap is within dt_tolerance (fraction) of dt_nominal.

	restrict_to_axis: if given, both endpoints of a pair must have a
	matching `calibration_axis` field. This matters because thrust isn't
	guaranteed open-loop outside its own test phase any more — newer
	calibration_node.py versions actively damp thrust against vz drift
	while roll/pitch are under test, so a row logged during the roll
	phase has a command_thrust driven by feedback, not the open-loop
	step train, and would re-introduce the exact cause/effect
	entanglement this whole logging setup exists to avoid if it leaked
	into the thrust fit. Rows without a calibration_axis field at all
	(older logs) are not restricted — see fit_axis()'s fallback.
	"""
	e_k, u_k, e_k1 = [], [], []

	for i in range(len(rows) - 1):
		row_a, row_b = rows[i], rows[i + 1]

		if restrict_to_axis is not None:
			if row_a.get("calibration_axis") != restrict_to_axis:
				continue
			if row_b.get("calibration_axis") != restrict_to_axis:
				continue

		if row_a.get(valid_col, "0") != "1" or row_b.get(valid_col, "0") != "1":
			continue

		t_a = _safe_float(row_a.get("t_sec"))
		t_b = _safe_float(row_b.get("t_sec"))

		if t_a is None or t_b is None:
			continue

		dt = t_b - t_a

		if dt <= 0.0:
			continue

		if abs(dt - dt_nominal) > dt_tolerance * dt_nominal:
			continue

		e0 = _safe_float(row_a.get(state_col))
		u0 = _safe_float(row_a.get(command_col))
		e1 = _safe_float(row_b.get(state_col))

		if e0 is None or u0 is None or e1 is None:
			continue

		e_k.append(e0)
		u_k.append(u0)
		e_k1.append(e1)

	return np.asarray(e_k), np.asarray(u_k), np.asarray(e_k1)


def fit_axis(
	rows: List[Dict[str, str]],
	axis: str,
	dt_nominal: float,
	dt_tolerance: float,
) -> Optional[AxisFit]:
	state_col, command_col, valid_col = AXIS_COLUMNS[axis]

	has_axis_column = bool(rows) and "calibration_axis" in rows[0] and any(
		row.get("calibration_axis") for row in rows
	)

	e_k, u_k, e_k1 = build_pairs(
		rows, state_col, command_col, valid_col, dt_nominal, dt_tolerance,
		restrict_to_axis=axis if has_axis_column else None,
	)

	if len(e_k) < 5:
		return None

	design = np.column_stack([e_k, u_k, np.ones_like(e_k)])
	coeffs, _, _, _ = np.linalg.lstsq(design, e_k1, rcond=None)
	a, b, c = coeffs

	predicted = design @ coeffs
	residual = e_k1 - predicted
	ss_res = float(np.sum(residual**2))
	ss_tot = float(np.sum((e_k1 - np.mean(e_k1)) ** 2))
	r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

	n = len(e_k)
	p = design.shape[1]  # 3: a, b, c
	b_stderr = float("nan")

	if n > p:
		sigma2 = ss_res / (n - p)
		try:
			xtx_inv = np.linalg.inv(design.T @ design)
			b_stderr = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
		except np.linalg.LinAlgError:
			pass

	return AxisFit(
		a=float(a), b=float(b), c=float(c),
		r_squared=r_squared, n_samples=n, b_stderr=b_stderr,
	)


def area_fraction_stats(rows: List[Dict[str, str]]) -> Optional[Tuple[float, float, float]]:
	values = []

	for row in rows:
		if not row_is_found(row):
			continue

		value = _safe_float(row.get("target_area_fraction"))

		if value is not None:
			values.append(value)

	if not values:
		return None

	return min(values), float(np.mean(values)), max(values)


def row_is_found(row: Dict[str, str]) -> bool:
	return row.get("target_found", "0") == "1"


def detect_possible_contact(
	rows: List[Dict[str, str]],
	vz_threshold: float = 0.015,
	thrust_variation_threshold: float = 0.01,
	window: int = 16,
) -> Optional[Tuple[float, float]]:
	"""
	Heuristic: a sustained stretch where vehicle_vz_m_s stays pinned
	near zero while command_thrust is clearly varying away from its
	mean is not something free flight produces -- if thrust is changing
	but vertical velocity isn't responding at all, something external
	(the ground, the platform) is absorbing it. Returns the (start_t,
	end_t) of the longest such stretch found, or None.

	This is a heuristic, not a certainty -- vz_threshold is a guess at
	the free-flight noise floor and may need adjusting for your sim.
	Treat a hit as "go look at the area_fraction/vehicle_z plot for
	this file", not as an automatic verdict.
	"""
	t = np.array([_safe_float(r.get("t_sec")) for r in rows], dtype=float)
	vz = np.array([_safe_float(r.get("vehicle_vz_m_s")) for r in rows], dtype=float)
	thrust = np.array([_safe_float(r.get("command_thrust")) for r in rows], dtype=float)

	valid = np.isfinite(t) & np.isfinite(vz) & np.isfinite(thrust)

	if valid.sum() < window:
		return None

	thrust_mean = np.nanmean(thrust[valid])

	flagged = np.zeros(len(rows), dtype=bool)

	for i in range(len(rows) - window + 1):
		seg = slice(i, i + window)

		if not np.all(valid[seg]):
			continue

		vz_seg = vz[seg]
		thrust_seg = thrust[seg]

		vz_is_flat = (np.max(vz_seg) - np.min(vz_seg)) < vz_threshold and np.max(np.abs(vz_seg)) < vz_threshold
		thrust_is_moving = np.max(np.abs(thrust_seg - thrust_mean)) > thrust_variation_threshold

		if vz_is_flat and thrust_is_moving:
			flagged[seg] = True

	if not np.any(flagged):
		return None

	# longest contiguous flagged run
	best_len, best_start = 0, None
	run_len, run_start = 0, None

	for i, f in enumerate(flagged):
		if f:
			if run_len == 0:
				run_start = i
			run_len += 1
			if run_len > best_len:
				best_len, best_start = run_len, run_start
		else:
			run_len = 0

	if best_start is None or best_len < window:
		return None

	return float(t[best_start]), float(t[best_start + best_len - 1])


def ensure_output_dir(output_dir: str):
	os.makedirs(output_dir, exist_ok=True)


def save_current_figure(output_dir: str, filename: str):
	path = os.path.join(output_dir, filename)
	plt.tight_layout()
	plt.savefig(path, dpi=160)
	plt.close()
	print(f"  Saved: {path}")


def plot_area_fraction_over_time(rows: List[Dict[str, str]], csv_path: str, output_dir: str):
	"""
	area_fraction vs t_sec for one file, with vehicle_z_m overlaid on a
	second axis. The per-file fit treats the whole file as a single
	operating point — this is the plot that tells you whether that was
	actually true, or whether area_fraction drifted (most likely during
	the thrust step train, which does command real altitude changes —
	see calibration_node.py's safety note), and vehicle_z_m plateauing
	while the run continues is the most direct visual sign of a
	ground/platform contact event.
	"""
	if rows and "target_area_fraction" not in rows[0]:
		print("  area_fraction plot: column not present in this CSV (older log format)")
		return

	t, area_fraction, vehicle_z = [], [], []

	for row in rows:
		if not row_is_found(row):
			continue

		t_value = _safe_float(row.get("t_sec"))
		af_value = _safe_float(row.get("target_area_fraction"))

		if t_value is None or af_value is None:
			continue

		t.append(t_value)
		area_fraction.append(af_value)
		vehicle_z.append(_safe_float(row.get("vehicle_z_m")))

	if len(t) < 2:
		print("  area_fraction plot: not enough valid rows to plot")
		return

	fig, ax1 = plt.subplots(figsize=(9, 4))
	ax1.set_title(f"area_fraction & vehicle_z over time — {os.path.basename(csv_path)}")
	ax1.plot(t, area_fraction, marker=".", linewidth=1.2, color="tab:green", label="area_fraction")
	ax1.axhline(float(np.mean(area_fraction)), linestyle="--", linewidth=1, color="gray",
				label=f"mean area_fraction = {np.mean(area_fraction):.3f}")
	ax1.set_xlabel("time [s]")
	ax1.set_ylabel("area_fraction [-]", color="tab:green")
	ax1.set_ylim(-0.05, 1.05)
	ax1.grid(True)

	if any(z is not None for z in vehicle_z):
		t_z = [tv for tv, z in zip(t, vehicle_z) if z is not None]
		z_vals = [z for z in vehicle_z if z is not None]
		ax2 = ax1.twinx()
		ax2.plot(t_z, z_vals, linewidth=1.0, color="tab:purple", alpha=0.7, label="vehicle_z_m")
		ax2.set_ylabel("vehicle_z_m (NED)", color="tab:purple")

	fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)

	stem = os.path.splitext(os.path.basename(csv_path))[0]
	save_current_figure(output_dir, f"{stem}_area_fraction.png")


def analyze_file(
	path: str, dt_nominal: float, dt_tolerance: float, output_dir: str
) -> Tuple[Dict[str, Optional[AxisFit]], Optional[Tuple[float, float, float]]]:
	rows = read_csv_rows(path)

	print(f"\n{path}  ({len(rows)} rows)")

	has_axis_column = bool(rows) and "calibration_axis" in rows[0] and any(
		row.get("calibration_axis") for row in rows
	)
	if not has_axis_column:
		print(
			"  calibration_axis: column not present (older log format) -- each axis is "
			"fit from every row, not just its own test phase. Fine if thrust was never "
			"actively damped outside its own phase in this log; if you're not sure, "
			"re-run with the current calibration_node.py."
		)

	if rows and "target_area_fraction" not in rows[0]:
		print("  area_fraction: column not present in this CSV (older log format)")
		stats = None
	else:
		stats = area_fraction_stats(rows)
		if stats is not None:
			lo, mean, hi = stats
			print(f"  area_fraction observed: min={lo:.3f}  mean={mean:.3f}  max={hi:.3f}")
			if hi - lo > 0.15:
				print(
					f"  <- WARNING: area_fraction swept a {hi-lo:.3f}-wide range during this "
					f"run; the per-file fit treats it as one operating point at mean={mean:.3f}, "
					f"which is a poor approximation when the range is this wide. Check the "
					f"_area_fraction.png plot for this file."
				)
		else:
			print("  area_fraction: no valid (found) rows in this file")

	contact = detect_possible_contact(rows)
	if contact is not None:
		start_t, end_t = contact
		print(
			f"  <- WARNING: possible ground/platform contact suspected from t={start_t:.1f}s "
			f"to t={end_t:.1f}s (vehicle_vz_m_s pinned near zero while command_thrust was "
			f"clearly varying). Samples in/after that window describe contact dynamics, not "
			f"free flight -- treat any fit using this file with extra suspicion, especially "
			f"thrust. Check the _area_fraction.png plot (vehicle_z_m should plateau there)."
		)

	plot_area_fraction_over_time(rows, path, output_dir)

	fits = {}

	for axis in ("roll", "pitch", "thrust"):
		fit = fit_axis(rows, axis, dt_nominal, dt_tolerance)
		fits[axis] = fit

		if fit is None:
			print(f"  {axis:6s}: not enough valid consecutive samples to fit")
			continue

		flags = []
		if fit.r_squared <= 0.5:
			flags.append("low R^2, model may not fit this axis well")
		if not fit.b_is_significant:
			flags.append(
				"b not significant (|t|<2) -- likely noise, not a real reading of "
				"control effectiveness, even though R^2 looks fine"
			)

		flag_str = "  <- " + "; ".join(flags) if flags else ""
		print(f"  {axis:6s}: {fit}{flag_str}")

	return fits, stats


def collect_axis_points(
	per_file_fits: List[Tuple[str, Dict[str, Optional[AxisFit]], Optional[Tuple[float, float, float]]]],
	axis: str,
) -> List[Tuple[float, float, float]]:
	"""Sorted (mean_area_fraction, a, b) for every file with a valid fit+stats on this axis."""
	points = []

	for path, fits, stats in per_file_fits:
		fit = fits.get(axis)

		if fit is None or stats is None:
			continue

		_, mean_area_fraction, _ = stats
		points.append((mean_area_fraction, fit.a, fit.b))

	points.sort(key=lambda p: p[0])
	return points


def check_scaling_and_print_schedule(
	per_file_fits: List[Tuple[str, Dict[str, Optional[AxisFit]], Optional[Tuple[float, float, float]]]],
	area_fraction_ref: Optional[float],
):
	print("\n--- across operating points ---")

	for axis in ("roll", "pitch", "thrust"):
		points = collect_axis_points(per_file_fits, axis)

		if len(points) < 2:
			continue

		ref = area_fraction_ref if area_fraction_ref is not None else points[0][0]

		print(f"\n{axis}:")
		print(f"  {'area_fraction':>14}  {'a':>8}  {'b':>8}  {'b/sqrt(s/s_ref)':>16}")

		ratios = []
		for area_fraction, a, b in points:
			scale = math.sqrt(max(area_fraction, 1e-9) / max(ref, 1e-9))
			ratio = b / scale if scale > 1e-9 else float("nan")
			ratios.append(ratio)
			print(f"  {area_fraction:14.3f}  {a:8.4f}  {b:8.4f}  {ratio:16.4f}")

		if ratios:
			mean_ratio = float(np.mean(ratios))
			spread = (max(ratios) - min(ratios)) / mean_ratio if mean_ratio != 0 else float("nan")
			print(f"  b_ref candidate (mean of column above): {mean_ratio:.4f}")
			print(f"  relative spread of that column: {spread:.2%}", end="")
			print(
				"  (small -> sqrt(area_fraction) scaling holds reasonably well)"
				if abs(spread) < 0.3
				else "  (large -> the sqrt model is a poor fit; use the raw schedule below instead)"
			)

		print(f"\n  Ready-to-use ScheduledLQR schedule entries for '{axis}' "
			  f"(paste into ControlLaw's *_a/_b_ref or build a custom schedule):")
		for area_fraction, a, b in points:
			print(f"    ({area_fraction:.3f}, [[{a:.4f}]], [[{b:.4f}]], [[state_cost]], [[control_cost]]),")


def plot_b_vs_area_fraction(
	per_file_fits: List[Tuple[str, Dict[str, Optional[AxisFit]], Optional[Tuple[float, float, float]]]],
	area_fraction_ref: Optional[float],
	output_dir: str,
):
	"""
	Fitted b against area_fraction per axis, with the assumed
	b_ref * sqrt(area_fraction / area_fraction_ref) curve overlaid using
	the same b_ref candidate printed by check_scaling_and_print_schedule
	— the visual version of that table, so a poor sqrt fit is something
	you can see, not just read off a spread percentage.
	"""
	axes_with_points = [
		(axis, collect_axis_points(per_file_fits, axis)) for axis in ("roll", "pitch", "thrust")
	]
	axes_with_points = [(axis, points) for axis, points in axes_with_points if len(points) >= 2]

	if not axes_with_points:
		print("\nSkipping b_vs_area_fraction plot. Need 2+ files with a valid fit on at least one axis.")
		return

	fig, axes = plt.subplots(len(axes_with_points), 1, figsize=(8, 3.4 * len(axes_with_points)))

	if len(axes_with_points) == 1:
		axes = [axes]

	for ax, (axis, points) in zip(axes, axes_with_points):
		area_fractions = np.array([p[0] for p in points])
		b_values = np.array([p[2] for p in points])

		ref = area_fraction_ref if area_fraction_ref is not None else area_fractions[0]
		ratios = b_values / np.sqrt(np.maximum(area_fractions, 1e-9) / max(ref, 1e-9))
		b_ref_candidate = float(np.mean(ratios))

		curve_x = np.linspace(min(area_fractions.min(), ref) * 0.9, area_fractions.max() * 1.05, 200)
		curve_y = b_ref_candidate * np.sqrt(np.maximum(curve_x, 1e-9) / max(ref, 1e-9))

		ax.plot(curve_x, curve_y, linestyle="--", color="gray",
				label=f"b_ref * sqrt(s/{ref:.3f})  (b_ref={b_ref_candidate:.4f})")
		ax.scatter(area_fractions, b_values, color="tab:blue", zorder=3, label="fitted b per file")

		ax.set_title(axis)
		ax.set_xlabel("area_fraction [-]")
		ax.set_ylabel("b")
		ax.axhline(0.0, linestyle=":", linewidth=0.8, color="black")
		ax.grid(True)
		ax.legend(loc="best")

	save_current_figure(output_dir, "b_vs_area_fraction.png")


def expand_csv_paths(paths: List[str], pattern: str = "*.csv", recursive: bool = False) -> List[str]:
	"""
	Turn a list of files and/or directories into a flat, deduplicated
	list of CSV file paths. A directory is searched for files matching
	`pattern` (sorted, so calibration_node.py's timestamped filenames
	come out in chronological order for free); pass --recursive to also
	search its subdirectories. A path that doesn't exist is skipped
	with a warning rather than aborting the whole run.
	"""
	expanded = []
	seen = set()

	def add(file_path: str):
		real = os.path.abspath(file_path)
		if real not in seen:
			seen.add(real)
			expanded.append(file_path)

	for path in paths:
		if os.path.isdir(path):
			search_pattern = os.path.join(path, "**", pattern) if recursive else os.path.join(path, pattern)
			matches = sorted(glob.glob(search_pattern, recursive=recursive))

			if not matches:
				print(f"Warning: no files matching '{pattern}' found in directory '{path}'")

			for match in matches:
				add(match)

		elif os.path.isfile(path):
			add(path)

		else:
			print(f"Warning: path not found, skipping: {path}")

	return expanded


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"csv_paths", nargs="+",
		help="one or more calibration CSV files and/or directories containing them",
	)
	parser.add_argument("--dt", type=float, default=0.5, help="nominal control period (s)")
	parser.add_argument(
		"--dt-tolerance", type=float, default=0.2,
		help="fractional tolerance on the time gap between consecutive samples",
	)
	parser.add_argument(
		"--area-fraction-ref", type=float, default=None,
		help="reference area_fraction for the sqrt-scaling check (default: smallest observed)",
	)
	parser.add_argument(
		"--output-dir", default="fit_axis_output",
		help="directory where plots are saved. Default: fit_axis_output.",
	)
	parser.add_argument(
		"--pattern", default="*.csv",
		help="glob pattern used when a directory is given. Default: *.csv",
	)
	parser.add_argument(
		"--recursive", action="store_true",
		help="when a directory is given, also search its subdirectories",
	)
	args = parser.parse_args()

	ensure_output_dir(args.output_dir)

	csv_paths = expand_csv_paths(args.csv_paths, pattern=args.pattern, recursive=args.recursive)

	if not csv_paths:
		print("No CSV files found from the given path(s).")
		return

	if len(csv_paths) > 1:
		print(f"Found {len(csv_paths)} CSV file(s):")
		for path in csv_paths:
			print(f"  {path}")

	per_file_fits = []
	for path in csv_paths:
		fits, stats = analyze_file(path, args.dt, args.dt_tolerance, args.output_dir)
		per_file_fits.append((path, fits, stats))

	if len(per_file_fits) > 1:
		check_scaling_and_print_schedule(per_file_fits, args.area_fraction_ref)
		plot_b_vs_area_fraction(per_file_fits, args.area_fraction_ref, args.output_dir)


if __name__ == "__main__":
	main()