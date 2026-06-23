"""
Fit control_law.py's per-axis local model from calibration logs.

Each axis is modeled as e[k+1] = a*e[k] + b*u[k] (see control_law.py's
module docstring). Given one or more diagnostics CSVs produced by
calibration_node.py, this script recovers (a, b) per axis by ordinary
least squares on consecutive samples, fits a small intercept term `c`
alongside them to absorb any trim mismatch (e.g. if hover_thrust isn't
quite exact), and reports a goodness-of-fit (R^2) so a bad model is
visible rather than silently producing a number.

Usage, one operating point:

	python fit_axis_models.py logs/calibration_20260623_140500.csv

Usage, several operating points (one CSV per altitude/area_fraction),
to additionally check the assumed b(area_fraction) ~ sqrt(area_fraction)
scaling and print a ready-to-use schedule:

	python fit_axis_models.py logs/calib_far.csv logs/calib_mid.csv logs/calib_near.csv

This script does not import rclpy, numpy aside — it only reads the CSV
that diagnostics_writer.py already produces.
"""

import argparse
import csv
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# (state column, command column, valid-mask column) per axis.
AXIS_COLUMNS = {
	"roll": ("target_offset_x", "command_roll_rad", "target_found"),
	"pitch": ("target_offset_y", "command_pitch_rad", "target_found"),
	"thrust": ("flow_divergence_1_s", "command_thrust", "flow_valid"),
}


class AxisFit:
	def __init__(self, a: float, b: float, c: float, r_squared: float, n_samples: int):
		self.a = a
		self.b = b
		self.c = c
		self.r_squared = r_squared
		self.n_samples = n_samples

	def __repr__(self) -> str:
		return (
			f"AxisFit(a={self.a:.4f}, b={self.b:.4f}, c={self.c:.5f}, "
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Build (e[k], u[k], e[k+1]) arrays from consecutive valid rows whose
	time gap is within dt_tolerance (fraction) of dt_nominal.
	"""
	e_k, u_k, e_k1 = [], [], []

	for i in range(len(rows) - 1):
		row_a, row_b = rows[i], rows[i + 1]

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

	e_k, u_k, e_k1 = build_pairs(
		rows, state_col, command_col, valid_col, dt_nominal, dt_tolerance
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

	return AxisFit(a=float(a), b=float(b), c=float(c), r_squared=r_squared, n_samples=len(e_k))


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


def analyze_file(
	path: str, dt_nominal: float, dt_tolerance: float
) -> Tuple[Dict[str, Optional[AxisFit]], Optional[Tuple[float, float, float]]]:
	rows = read_csv_rows(path)

	print(f"\n{path}  ({len(rows)} rows)")

	if rows and "target_area_fraction" not in rows[0]:
		print("  area_fraction: column not present in this CSV (older log format)")
		stats = None
	else:
		stats = area_fraction_stats(rows)
		if stats is not None:
			lo, mean, hi = stats
			print(f"  area_fraction observed: min={lo:.3f}  mean={mean:.3f}  max={hi:.3f}")
		else:
			print("  area_fraction: no valid (found) rows in this file")

	fits = {}

	for axis in ("roll", "pitch", "thrust"):
		fit = fit_axis(rows, axis, dt_nominal, dt_tolerance)
		fits[axis] = fit

		if fit is None:
			print(f"  {axis:6s}: not enough valid consecutive samples to fit")
		else:
			flag = "" if fit.r_squared > 0.5 else "  <- low R^2, model may not fit this axis well"
			print(f"  {axis:6s}: {fit}{flag}")

	return fits, stats


def check_scaling_and_print_schedule(
	per_file_fits: List[Tuple[str, Dict[str, Optional[AxisFit]], Optional[Tuple[float, float, float]]]],
	area_fraction_ref: Optional[float],
):
	print("\n--- across operating points ---")

	for axis in ("roll", "pitch", "thrust"):
		points = []
		for path, fits, stats in per_file_fits:
			fit = fits.get(axis)
			if fit is None or stats is None:
				continue
			_, mean_area_fraction, _ = stats
			points.append((mean_area_fraction, fit.a, fit.b))

		if len(points) < 2:
			continue

		points.sort(key=lambda p: p[0])
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


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("csv_paths", nargs="+", help="one or more calibration CSV files")
	parser.add_argument("--dt", type=float, default=0.5, help="nominal control period (s)")
	parser.add_argument(
		"--dt-tolerance", type=float, default=0.2,
		help="fractional tolerance on the time gap between consecutive samples",
	)
	parser.add_argument(
		"--area-fraction-ref", type=float, default=None,
		help="reference area_fraction for the sqrt-scaling check (default: smallest observed)",
	)
	args = parser.parse_args()

	per_file_fits = []
	for path in args.csv_paths:
		fits, stats = analyze_file(path, args.dt, args.dt_tolerance)
		per_file_fits.append((path, fits, stats))

	if len(per_file_fits) > 1:
		check_scaling_and_print_schedule(per_file_fits, args.area_fraction_ref)


if __name__ == "__main__":
	main()
