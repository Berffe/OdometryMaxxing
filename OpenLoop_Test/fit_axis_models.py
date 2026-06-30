"""
Fit control_law.py's per-axis models from calibration logs.

Models (OLS on consecutive in-phase samples, with a small intercept c that
absorbs trim mismatch):
    scalar roll/pitch/thrust : e[k+1] = a e[k] + b u[k] + c
    state  roll/pitch        : [offset, flow][k+1] = A [offset, flow][k] + B u[k] + c
    visual MIMO (diagnostic) : [flow_x, flow_y] = G [roll, pitch] + c
control_law.py uses the 2-state roll/pitch models and the scalar thrust model;
print_controller_schedule emits exactly those, ready to paste.

Trust checks (R^2 alone is not enough -- when a ~ 1, e[k] predicts e[k+1] well
even if b carries no signal): every fit reports b's standard error and
t = b/stderr; |t| < 2 means b is not distinguishable from zero.

Data-quality selection (this is how you keep failed runs out of the model):
    --min-altitude / --max-altitude  keep only rows in an altitude band
                                     (altitude = -vehicle_z_m); a file that
                                     swept altitude can be narrowed to a good band.
    --exclude SUBSTR ...             skip files whose path matches (e.g. _2m _3m).
    (default)                        files are auto-rejected from the schedule when
                                     area_fraction sweeps too wide, contact is
                                     detected, or too few rows remain; --keep-flagged
                                     includes them anyway, --keep-contact keeps
                                     contact windows.

Usage:
    python fit_axis_models.py logs/                       # a folder of runs
    python fit_axis_models.py logs/ --exclude _2m _3m     # drop known-bad runs
    python fit_axis_models.py logs/ --min-altitude 2.5 --max-altitude 3.5

Plots (--output-dir, default fit_axis_output): per-file area_fraction (+vehicle_z)
over time to confirm a steady operating point and spot contact; b_vs_area_fraction
when 2+ files are kept. Needs numpy/matplotlib, not rclpy.
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

# For roll/pitch, the more useful model is not offset-only but
# [offset, image_velocity]. Prefer normalized image velocity, which has
# the same coordinate convention as target_offset_x/y. Fall back to px/s
# so older CSVs can still be inspected, but do not paste px/s gains into
# a normalized controller without converting the units first.
AXIS_STATE_COLUMNS = {
	"roll": {
		"state": "target_offset_x",
		"velocity_preferred": "flow_mean_x_norm_s",
		"velocity_fallback": "flow_mean_x_px_s",
		"command": "command_roll_rad",
		"valid": "target_found",
	},
	"pitch": {
		"state": "target_offset_y",
		"velocity_preferred": "flow_mean_y_norm_s",
		"velocity_fallback": "flow_mean_y_px_s",
		"command": "command_pitch_rad",
		"valid": "target_found",
	},
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


class AxisStateFit:
	"""
	Discrete local model for roll/pitch:

		[offset[k+1], flow[k+1]]^T = A [offset[k], flow[k]]^T + B u[k] + c

	This is the model to use when the controller state is offset +
	flow_mean. It keeps the old scalar fit available for comparison, but
	roll/pitch control should generally be designed from this one.
	"""

	def __init__(
		self,
		A: np.ndarray,
		B: np.ndarray,
		c: np.ndarray,
		r_squared: np.ndarray,
		n_samples: int,
		velocity_col: str,
		b_stderr: Optional[np.ndarray] = None,
	):
		self.A = np.asarray(A, dtype=float).reshape(2, 2)
		self.B = np.asarray(B, dtype=float).reshape(2)
		self.c = np.asarray(c, dtype=float).reshape(2)
		self.r_squared = np.asarray(r_squared, dtype=float).reshape(2)
		self.n_samples = int(n_samples)
		self.velocity_col = velocity_col
		self.b_stderr = (
			np.asarray(b_stderr, dtype=float).reshape(2)
			if b_stderr is not None
			else np.full(2, np.nan)
		)

	@property
	def b_t_stat(self) -> np.ndarray:
		out = np.full(2, np.nan)
		mask = np.isfinite(self.b_stderr) & (self.b_stderr > 0.0)
		out[mask] = self.B[mask] / self.b_stderr[mask]
		return out

	def __repr__(self) -> str:
		t = self.b_t_stat
		t0 = f"{t[0]:+.2f}" if np.isfinite(t[0]) else "n/a"
		t1 = f"{t[1]:+.2f}" if np.isfinite(t[1]) else "n/a"
		return (
			"AxisStateFit(\n"
			f"    x=[offset, {self.velocity_col}], n={self.n_samples}\n"
			f"    A=[[{self.A[0,0]:+.4f}, {self.A[0,1]:+.4f}], "
			f"[{self.A[1,0]:+.4f}, {self.A[1,1]:+.4f}]]\n"
			f"    B=[{self.B[0]:+.5f}, {self.B[1]:+.5f}]  "
			f"t=[{t0}, {t1}]\n"
			f"    c=[{self.c[0]:+.5f}, {self.c[1]:+.5f}]  "
			f"R^2=[{self.r_squared[0]:.3f}, {self.r_squared[1]:.3f}]\n"
			")"
		)


class VisualMimoFit:
	"""Instantaneous visual-response matrix:

		[flow_x, flow_y]^T = G [roll, pitch]^T + c

	Use this as a sign/coupling diagnostic before designing the controller.
	Prefer actual roll/pitch from VehicleAttitude; fall back to command_roll/
	command_pitch only for older logs.
	"""

	def __init__(
		self,
		G: np.ndarray,
		c: np.ndarray,
		r_squared: np.ndarray,
		n_samples: int,
		input_cols: Tuple[str, str],
		flow_cols: Tuple[str, str],
		g_stderr: Optional[np.ndarray] = None,
		rank: int = 0,
		condition_number: float = float("nan"),
	):
		self.G = np.asarray(G, dtype=float).reshape(2, 2)
		self.c = np.asarray(c, dtype=float).reshape(2)
		self.r_squared = np.asarray(r_squared, dtype=float).reshape(2)
		self.n_samples = int(n_samples)
		self.input_cols = input_cols
		self.flow_cols = flow_cols
		self.g_stderr = (
			np.asarray(g_stderr, dtype=float).reshape(2, 2)
			if g_stderr is not None
			else np.full((2, 2), np.nan)
		)
		self.rank = int(rank)
		self.condition_number = float(condition_number)

	@property
	def g_t_stat(self) -> np.ndarray:
		out = np.full((2, 2), np.nan)
		mask = np.isfinite(self.g_stderr) & (self.g_stderr > 0.0)
		out[mask] = self.G[mask] / self.g_stderr[mask]
		return out

	def __repr__(self) -> str:
		t = self.g_t_stat
		def fmt(x):
			return f"{x:+.2f}" if np.isfinite(x) else "n/a"

		return (
			"VisualMimoFit(\n"
			f"    y=[{self.flow_cols[0]}, {self.flow_cols[1]}], "
			f"u=[{self.input_cols[0]}, {self.input_cols[1]}], n={self.n_samples}\n"
			f"    G=[[{self.G[0,0]:+.5f}, {self.G[0,1]:+.5f}], "
			f"[{self.G[1,0]:+.5f}, {self.G[1,1]:+.5f}]]\n"
			f"       rows are [flow_x, flow_y], cols are [roll, pitch]\n"
			f"    t=[[{fmt(t[0,0])}, {fmt(t[0,1])}], "
			f"[{fmt(t[1,0])}, {fmt(t[1,1])}]]\n"
			f"    c=[{self.c[0]:+.5f}, {self.c[1]:+.5f}]  "
			f"R^2=[{self.r_squared[0]:.3f}, {self.r_squared[1]:.3f}]\n"
			f"    design_rank={self.rank}/3, condition_number={self.condition_number:.2e}\n"
			")"
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


class RowFilter:
	"""
	Row-level data-quality gate applied before any fit.

	Altitude is taken from vehicle_z_m (NED, so altitude_m = -z). Rows
	outside [min_altitude_m, max_altitude_m] are dropped, which is how you
	exclude a bad operating point (e.g. a 2 m / 3 m run that failed) without
	deleting files. drop_contact removes the longest detected ground/contact
	window. Dropping rows is safe for the consecutive-pair fitters: a removed
	row just opens a time gap, which the dt-tolerance check already rejects.
	"""

	def __init__(
		self,
		min_altitude_m: Optional[float] = None,
		max_altitude_m: Optional[float] = None,
		drop_contact: bool = True,
	):
		self.min_altitude_m = min_altitude_m
		self.max_altitude_m = max_altitude_m
		self.drop_contact = drop_contact

	def is_active(self) -> bool:
		return (
			self.min_altitude_m is not None
			or self.max_altitude_m is not None
			or self.drop_contact
		)


def _altitude_m(row: Dict[str, str]) -> Optional[float]:
	"""Altitude above start in meters (NED: altitude = -vehicle_z_m)."""
	z = _safe_float(row.get("vehicle_z_m"))
	return None if z is None else -z


def apply_row_filter(rows: List[Dict[str, str]], row_filter: Optional[RowFilter]) -> List[Dict[str, str]]:
	"""Return the subset of rows passing the altitude band and contact gate."""
	if row_filter is None or not row_filter.is_active():
		return rows

	contact_span = detect_possible_contact(rows) if row_filter.drop_contact else None

	kept = []
	for row in rows:
		if contact_span is not None:
			t = _safe_float(row.get("t_sec"))
			if t is not None and contact_span[0] <= t <= contact_span[1]:
				continue

		if row_filter.min_altitude_m is not None or row_filter.max_altitude_m is not None:
			alt = _altitude_m(row)
			if alt is None:
				continue
			if row_filter.min_altitude_m is not None and alt < row_filter.min_altitude_m:
				continue
			if row_filter.max_altitude_m is not None and alt > row_filter.max_altitude_m:
				continue

		kept.append(row)

	return kept


def assess_file_quality(
	rows: List[Dict[str, str]],
	max_area_fraction_span: float = 0.15,
	min_found_rows: int = 20,
	max_fov_saturated_fraction: float = 0.10,
) -> Tuple[bool, List[str]]:
	"""
	Decide whether a (already filtered) file is trustworthy calibration data.

	Returns (ok, reasons). A file is rejected when its area_fraction still
	sweeps wider than max_area_fraction_span (it is not one operating point),
	when ground/platform contact is detected, when too few found rows remain
	to fit, or when most found rows have target_fov_saturated=True.

	fov_saturated is a DIFFERENT failure than a wide area_fraction span, and
	the latter cannot catch it: once the target's true size exceeds the
	camera's field of view, area_fraction/detection_width/height are clamped
	at the image's own pixel dimensions and stop tracking true range, so a
	saturated run can show a deceptively TIGHT area_fraction span (everything
	pinned at the same clamped value) while the true altitude varies
	substantially underneath -- exactly the case a span-only check would
	wrongly pass as "one clean operating point." Needs the
	target_fov_saturated column (diagnostics_writer.py); older CSVs without
	it skip this check.

	Rejected files are excluded from the cross-file schedule unless
	--keep-flagged is passed.
	"""
	reasons = []

	found = [r for r in rows if row_is_found(r)]
	if len(found) < min_found_rows:
		reasons.append(f"only {len(found)} found rows (< {min_found_rows})")

	stats = area_fraction_stats(rows)
	if stats is not None:
		lo, _, hi = stats
		if hi - lo > max_area_fraction_span:
			reasons.append(
				f"area_fraction span {hi - lo:.3f} > {max_area_fraction_span:.2f} "
				"(not a single operating point)"
			)

	if found and "target_fov_saturated" in found[0]:
		saturated = sum(1 for r in found if r.get("target_fov_saturated", "0") == "1")
		frac = saturated / len(found)
		if frac > max_fov_saturated_fraction:
			reasons.append(
				f"{frac*100:.0f}% of found rows have fov_saturated=True -- the target "
				"exceeds the camera's field of view here; area_fraction/detection box "
				"are a frame-size artifact, not a measurement, regardless of how tight "
				"the area_fraction span looks"
			)

	if detect_possible_contact(rows) is not None:
		reasons.append("ground/platform contact detected")

	return (len(reasons) == 0, reasons)



def _row_time(row: Dict[str, str]) -> Optional[float]:
	return _safe_float(row.get("t_sec"))


def _bool_col(row: Dict[str, str], col: str) -> bool:
	return str(row.get(col, "0")).strip() in ("1", "True", "true")


def _row_passes_identification_quality(row: Dict[str, str], require_flow_quality: bool = False) -> bool:
	"""Reject rows that are valid for control continuity but poor for identification."""
	if row.get("target_is_held", "0") == "1":
		return False
	if row.get("target_fov_saturated", "0") == "1":
		return False
	conf = _safe_float(row.get("target_confidence"))
	if conf is not None and conf < 0.25:
		return False
	if require_flow_quality:
		frac = _safe_float(row.get("flow_affine_inlier_fraction"))
		if frac is not None and frac > 0.0 and frac < 0.35:
			return False
		resid = _safe_float(row.get("flow_affine_residual_rms"))
		if resid is not None and np.isfinite(resid) and resid > 2.0:
			return False
	return True


def _axis_ok(row: Dict[str, str], axis: Optional[str]) -> bool:
	return axis is None or row.get("calibration_axis") == axis


def _find_row_near_time(
	rows: List[Dict[str, str]],
	target_time: float,
	tolerance_sec: float,
	axis: Optional[str],
	valid_col: str,
	require_flow_quality: bool,
) -> Optional[Dict[str, str]]:
	best = None
	best_err = None
	for row in rows:
		if not _axis_ok(row, axis):
			continue
		if row.get(valid_col, "0") != "1":
			continue
		if not _row_passes_identification_quality(row, require_flow_quality=require_flow_quality):
			continue
		t = _row_time(row)
		if t is None:
			continue
		err = abs(t - target_time)
		if err <= tolerance_sec and (best_err is None or err < best_err):
			best = row
			best_err = err
	return best


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
	Build (e[k], u[k], e[k+1]) arrays from rows separated by dt_nominal.

	Rows no longer need to be adjacent in the CSV. This allows calibration_node.py
	to log at 0.1 s while fitting a 0.5 s controller model by pairing samples whose
	timestamps differ by dt_nominal.
	"""
	e_k, u_k, e_k1 = [], [], []
	tol = abs(dt_tolerance * dt_nominal)
	require_flow_quality = (valid_col == "flow_valid")

	for row_a in rows:
		if not _axis_ok(row_a, restrict_to_axis):
			continue
		if row_a.get(valid_col, "0") != "1":
			continue
		if not _row_passes_identification_quality(row_a, require_flow_quality=require_flow_quality):
			continue

		t_a = _row_time(row_a)
		if t_a is None:
			continue
		row_b = _find_row_near_time(
			rows, t_a + dt_nominal, tol, restrict_to_axis, valid_col, require_flow_quality
		)
		if row_b is None:
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


def build_pairs_with_input_delays(
	rows: List[Dict[str, str]],
	state_col: str,
	command_col: str,
	valid_col: str,
	dt_nominal: float,
	dt_tolerance: float,
	restrict_to_axis: Optional[str] = None,
	input_delay_count: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Build e[k], [u[k], u[k-1], ...], e[k+1] for delay-aware ARX fits."""
	e_k, U, e_k1 = [], [], []
	tol = abs(dt_tolerance * dt_nominal)
	require_flow_quality = (valid_col == "flow_valid")

	for row_a in rows:
		if not _axis_ok(row_a, restrict_to_axis):
			continue
		if row_a.get(valid_col, "0") != "1":
			continue
		if not _row_passes_identification_quality(row_a, require_flow_quality=require_flow_quality):
			continue
		t_a = _row_time(row_a)
		if t_a is None:
			continue
		row_b = _find_row_near_time(rows, t_a + dt_nominal, tol, restrict_to_axis, valid_col, require_flow_quality)
		if row_b is None:
			continue

		u_values = []
		ok = True
		for lag in range(input_delay_count + 1):
			row_u = row_a if lag == 0 else _find_row_near_time(
				rows, t_a - lag * dt_nominal, tol, restrict_to_axis, valid_col, require_flow_quality
			)
			if row_u is None:
				ok = False
				break
			u = _safe_float(row_u.get(command_col))
			if u is None:
				ok = False
				break
			u_values.append(u)
		if not ok:
			continue

		e0 = _safe_float(row_a.get(state_col))
		e1 = _safe_float(row_b.get(state_col))
		if e0 is None or e1 is None:
			continue
		e_k.append(e0)
		U.append(u_values)
		e_k1.append(e1)

	return np.asarray(e_k), np.asarray(U), np.asarray(e_k1)

def _has_nonempty_column(rows: List[Dict[str, str]], col: str) -> bool:
	return bool(rows) and col in rows[0] and any(row.get(col, "") != "" for row in rows)




def _column_has_finite_values(rows: List[Dict[str, str]], col: str) -> bool:
	if not (bool(rows) and col in rows[0]):
		return False
	return any(_safe_float(row.get(col)) is not None for row in rows)


def _column_has_variation(
	rows: List[Dict[str, str]],
	col: str,
	min_std: float = 1e-5,
) -> bool:
	values = [
		_safe_float(row.get(col))
		for row in rows
		if _safe_float(row.get(col)) is not None
	]
	if len(values) < 4:
		return False
	return float(np.std(values)) >= min_std


def _velocity_column_for_axis(rows: List[Dict[str, str]], axis: str) -> Optional[str]:
	columns = AXIS_STATE_COLUMNS[axis]
	preferred = columns["velocity_preferred"]
	fallback = columns["velocity_fallback"]

	if _has_nonempty_column(rows, preferred):
		return preferred
	if _has_nonempty_column(rows, fallback):
		return fallback
	return None


def build_state_pairs(
	rows: List[Dict[str, str]],
	axis: str,
	dt_nominal: float,
	dt_tolerance: float,
	restrict_to_axis: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
	"""Build x[k], u[k], x[k+1] for x=[target offset, image velocity]."""
	if axis not in AXIS_STATE_COLUMNS:
		raise ValueError(f"state-pair model is only defined for roll/pitch, not {axis!r}")

	columns = AXIS_STATE_COLUMNS[axis]
	state_col = columns["state"]
	velocity_col = _velocity_column_for_axis(rows, axis)
	command_col = columns["command"]
	valid_col = columns["valid"]
	if velocity_col is None:
		return np.empty((0, 2)), np.empty(0), np.empty((0, 2)), None

	x_k, u_k, x_k1 = [], [], []
	tol = abs(dt_tolerance * dt_nominal)
	for row_a in rows:
		if not _axis_ok(row_a, restrict_to_axis):
			continue
		if row_a.get(valid_col, "0") != "1" or not _row_passes_identification_quality(row_a):
			continue
		t_a = _row_time(row_a)
		if t_a is None:
			continue
		row_b = _find_row_near_time(rows, t_a + dt_nominal, tol, restrict_to_axis, valid_col, False)
		if row_b is None:
			continue

		e0 = _safe_float(row_a.get(state_col))
		v0 = _safe_float(row_a.get(velocity_col))
		u0 = _safe_float(row_a.get(command_col))
		e1 = _safe_float(row_b.get(state_col))
		v1 = _safe_float(row_b.get(velocity_col))
		if None in (e0, v0, u0, e1, v1):
			continue

		x_k.append((e0, v0))
		u_k.append(u0)
		x_k1.append((e1, v1))
	return np.asarray(x_k), np.asarray(u_k), np.asarray(x_k1), velocity_col


def build_state_pairs_with_input_delays(
	rows: List[Dict[str, str]],
	axis: str,
	dt_nominal: float,
	dt_tolerance: float,
	restrict_to_axis: Optional[str] = None,
	input_delay_count: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
	"""Build x[k], [u[k],u[k-1],...], x[k+1] for delay-aware state fits."""
	if axis not in AXIS_STATE_COLUMNS:
		raise ValueError(f"state-pair model is only defined for roll/pitch, not {axis!r}")
	columns = AXIS_STATE_COLUMNS[axis]
	state_col = columns["state"]
	velocity_col = _velocity_column_for_axis(rows, axis)
	command_col = columns["command"]
	valid_col = columns["valid"]
	if velocity_col is None:
		return np.empty((0, 2)), np.empty((0, input_delay_count + 1)), np.empty((0, 2)), None

	x_k, U, x_k1 = [], [], []
	tol = abs(dt_tolerance * dt_nominal)
	for row_a in rows:
		if not _axis_ok(row_a, restrict_to_axis):
			continue
		if row_a.get(valid_col, "0") != "1" or not _row_passes_identification_quality(row_a):
			continue
		t_a = _row_time(row_a)
		if t_a is None:
			continue
		row_b = _find_row_near_time(rows, t_a + dt_nominal, tol, restrict_to_axis, valid_col, False)
		if row_b is None:
			continue

		u_values = []
		ok = True
		for lag in range(input_delay_count + 1):
			row_u = row_a if lag == 0 else _find_row_near_time(rows, t_a - lag * dt_nominal, tol, restrict_to_axis, valid_col, False)
			if row_u is None:
				ok = False
				break
			u = _safe_float(row_u.get(command_col))
			if u is None:
				ok = False
				break
			u_values.append(u)
		if not ok:
			continue

		e0 = _safe_float(row_a.get(state_col))
		v0 = _safe_float(row_a.get(velocity_col))
		e1 = _safe_float(row_b.get(state_col))
		v1 = _safe_float(row_b.get(velocity_col))
		if None in (e0, v0, e1, v1):
			continue
		x_k.append((e0, v0))
		U.append(u_values)
		x_k1.append((e1, v1))
	return np.asarray(x_k), np.asarray(U), np.asarray(x_k1), velocity_col

def fit_axis_state(
	rows: List[Dict[str, str]],
	axis: str,
	dt_nominal: float,
	dt_tolerance: float,
) -> Optional[AxisStateFit]:
	if axis not in AXIS_STATE_COLUMNS:
		return None

	has_axis_column = bool(rows) and "calibration_axis" in rows[0] and any(
		row.get("calibration_axis") for row in rows
	)

	x_k, u_k, x_k1, velocity_col = build_state_pairs(
		rows,
		axis,
		dt_nominal,
		dt_tolerance,
		restrict_to_axis=axis if has_axis_column else None,
	)

	if velocity_col is None or len(u_k) < 8:
		return None

	design = np.column_stack([x_k[:, 0], x_k[:, 1], u_k, np.ones_like(u_k)])
	coeffs, _, _, _ = np.linalg.lstsq(design, x_k1, rcond=None)

	# design @ coeffs predicts columns [offset[k+1], flow[k+1]].
	A = np.array([
		[coeffs[0, 0], coeffs[1, 0]],
		[coeffs[0, 1], coeffs[1, 1]],
	])
	B = np.array([coeffs[2, 0], coeffs[2, 1]])
	c = np.array([coeffs[3, 0], coeffs[3, 1]])

	predicted = design @ coeffs
	residual = x_k1 - predicted
	r_squared = []
	b_stderr = np.full(2, np.nan)
	n = len(u_k)
	p = design.shape[1]

	for j in range(2):
		ss_res = float(np.sum(residual[:, j] ** 2))
		ss_tot = float(np.sum((x_k1[:, j] - np.mean(x_k1[:, j])) ** 2))
		r_squared.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))

		if n > p:
			sigma2 = ss_res / (n - p)
			try:
				xtx_inv = np.linalg.inv(design.T @ design)
				b_stderr[j] = float(np.sqrt(sigma2 * xtx_inv[2, 2]))
			except np.linalg.LinAlgError:
				pass

	return AxisStateFit(
		A=A, B=B, c=c, r_squared=np.asarray(r_squared),
		n_samples=n, velocity_col=velocity_col, b_stderr=b_stderr,
	)


def fit_axis_state_delayed(rows, axis: str, dt_nominal: float, dt_tolerance: float, input_delay_count: int = 2):
	if axis not in AXIS_STATE_COLUMNS or input_delay_count <= 0:
		return None
	has_axis_column = bool(rows) and "calibration_axis" in rows[0] and any(row.get("calibration_axis") for row in rows)
	x_k, U, x_k1, velocity_col = build_state_pairs_with_input_delays(
		rows, axis, dt_nominal, dt_tolerance,
		restrict_to_axis=axis if has_axis_column else None,
		input_delay_count=input_delay_count,
	)
	if velocity_col is None or len(x_k) < 8:
		return None
	design = np.column_stack([x_k[:, 0], x_k[:, 1], U, np.ones(len(x_k))])
	coeffs, _, _, _ = np.linalg.lstsq(design, x_k1, rcond=None)
	predicted = design @ coeffs
	residual = x_k1 - predicted
	r2 = []
	for j in range(2):
		ss_res = float(np.sum(residual[:, j] ** 2))
		ss_tot = float(np.sum((x_k1[:, j] - np.mean(x_k1[:, j])) ** 2))
		r2.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))
	A = np.array([[coeffs[0, 0], coeffs[1, 0]], [coeffs[0, 1], coeffs[1, 1]]])
	B_delays = coeffs[2:2 + input_delay_count + 1, :].T
	c = coeffs[-1, :]
	return {"A": A, "B_delays": B_delays, "B_sum": B_delays.sum(axis=1), "c": c, "r_squared": np.asarray(r2), "n_samples": len(x_k), "velocity_col": velocity_col}


def fit_axis_delayed(rows, axis: str, dt_nominal: float, dt_tolerance: float, input_delay_count: int = 2):
	if input_delay_count <= 0:
		return None
	state_col, command_col, valid_col = AXIS_COLUMNS[axis]
	has_axis_column = bool(rows) and "calibration_axis" in rows[0] and any(row.get("calibration_axis") for row in rows)
	e_k, U, e_k1 = build_pairs_with_input_delays(
		rows, state_col, command_col, valid_col, dt_nominal, dt_tolerance,
		restrict_to_axis=axis if has_axis_column else None,
		input_delay_count=input_delay_count,
	)
	if len(e_k) < 8:
		return None
	design = np.column_stack([e_k, U, np.ones(len(e_k))])
	coeffs, _, _, _ = np.linalg.lstsq(design, e_k1, rcond=None)
	predicted = design @ coeffs
	residual = e_k1 - predicted
	ss_res = float(np.sum(residual ** 2))
	ss_tot = float(np.sum((e_k1 - np.mean(e_k1)) ** 2))
	r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
	return {"a": float(coeffs[0]), "B_delays": np.asarray(coeffs[1:1 + input_delay_count + 1]), "B_sum": float(np.sum(coeffs[1:1 + input_delay_count + 1])), "c": float(coeffs[-1]), "r_squared": r2, "n_samples": len(e_k)}


def _flow_columns_for_mimo(rows: List[Dict[str, str]]) -> Optional[Tuple[str, str]]:
	"""Prefer normalized flow columns; fall back to px/s for older CSVs."""
	if _has_nonempty_column(rows, "flow_mean_x_norm_s") and _has_nonempty_column(rows, "flow_mean_y_norm_s"):
		return "flow_mean_x_norm_s", "flow_mean_y_norm_s"
	if _has_nonempty_column(rows, "flow_mean_x_px_s") and _has_nonempty_column(rows, "flow_mean_y_px_s"):
		return "flow_mean_x_px_s", "flow_mean_y_px_s"
	return None


def _actual_attitude_is_usable(rows: List[Dict[str, str]]) -> bool:
	"""
	True only when the CSV contains real, varying actual attitude samples.

	A previous failure mode produced vehicle_roll_rad/vehicle_pitch_rad columns
	filled with zeros because the DiagnosticsWriter was fixed but no PX4 attitude
	callback was actually received. Treat that as missing data and fall back to
	commanded attitude, while printing a clear diagnostic.
	"""
	if not (
		_has_nonempty_column(rows, "vehicle_roll_rad")
		and _has_nonempty_column(rows, "vehicle_pitch_rad")
	):
		return False

	# If the timestamp column exists, it must contain at least one finite sample.
	if "vehicle_attitude_timestamp_sec" in rows[0]:
		if not _column_has_finite_values(rows, "vehicle_attitude_timestamp_sec"):
			return False

	return (
		_column_has_variation(rows, "vehicle_roll_rad")
		or _column_has_variation(rows, "vehicle_pitch_rad")
	)


def _attitude_input_columns_for_mimo(rows: List[Dict[str, str]]) -> Tuple[str, str]:
	"""Use actual attitude when logged and nonzero; otherwise fall back to commands."""
	if _actual_attitude_is_usable(rows):
		return "vehicle_roll_rad", "vehicle_pitch_rad"
	return "command_roll_rad", "command_pitch_rad"


def fit_visual_mimo(
	rows: List[Dict[str, str]],
	restrict_to_roll_pitch_axes: bool = True,
) -> Optional[VisualMimoFit]:
	"""
	Fit the instantaneous visual response matrix:

		[flow_x, flow_y]^T = G [roll, pitch]^T + c

	This is mainly a sign/coupling diagnostic. It should be fitted on static-
	platform calibration runs first, then reused as the first visual damping
	model. If only roll or only pitch data is present, the design matrix will be
	rank-deficient; the fitted column for the excited axis is still informative,
	but the non-excited column and t-statistics are not trustworthy.
	"""
	flow_cols = _flow_columns_for_mimo(rows)
	if flow_cols is None:
		return None

	input_cols = _attitude_input_columns_for_mimo(rows)
	u_roll_col, u_pitch_col = input_cols
	flow_x_col, flow_y_col = flow_cols

	has_axis_column = bool(rows) and "calibration_axis" in rows[0] and any(
		row.get("calibration_axis") for row in rows
	)

	U, Y = [], []
	for row in rows:
		if restrict_to_roll_pitch_axes and has_axis_column:
			if row.get("calibration_axis") not in ("roll", "pitch"):
				continue

		if row.get("target_found", "0") != "1" or row.get("flow_valid", "0") != "1":
			continue

		u_roll = _safe_float(row.get(u_roll_col))
		u_pitch = _safe_float(row.get(u_pitch_col))
		flow_x = _safe_float(row.get(flow_x_col))
		flow_y = _safe_float(row.get(flow_y_col))

		if None in (u_roll, u_pitch, flow_x, flow_y):
			continue

		U.append((u_roll, u_pitch))
		Y.append((flow_x, flow_y))

	if len(U) < 8:
		return None

	U = np.asarray(U, dtype=float)
	Y = np.asarray(Y, dtype=float)
	design = np.column_stack([U[:, 0], U[:, 1], np.ones(len(U))])
	coeffs, _, rank, singular_values = np.linalg.lstsq(design, Y, rcond=None)

	# coeffs rows are [roll, pitch, bias], columns are [flow_x, flow_y].
	G = np.array([
		[coeffs[0, 0], coeffs[1, 0]],
		[coeffs[0, 1], coeffs[1, 1]],
	])
	c = np.array([coeffs[2, 0], coeffs[2, 1]])

	predicted = design @ coeffs
	residual = Y - predicted
	r_squared = []
	g_stderr = np.full((2, 2), np.nan)
	n = len(U)
	p = design.shape[1]

	condition_number = float("nan")
	if len(singular_values) and np.min(singular_values) > 0.0:
		condition_number = float(np.max(singular_values) / np.min(singular_values))

	for j in range(2):
		ss_res = float(np.sum(residual[:, j] ** 2))
		ss_tot = float(np.sum((Y[:, j] - np.mean(Y[:, j])) ** 2))
		r_squared.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))

		# Standard errors are meaningful only if roll, pitch, and bias are
		# independently excited (rank 3). With roll-only data, rank is usually 2.
		if n > p and rank == p:
			sigma2 = ss_res / (n - p)
			try:
				xtx_inv = np.linalg.inv(design.T @ design)
				# G rows are output [x,y], columns are input [roll,pitch].
				g_stderr[j, 0] = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
				g_stderr[j, 1] = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
			except np.linalg.LinAlgError:
				pass

	return VisualMimoFit(
		G=G,
		c=c,
		r_squared=np.asarray(r_squared),
		n_samples=n,
		input_cols=input_cols,
		flow_cols=flow_cols,
		g_stderr=g_stderr,
		rank=rank,
		condition_number=condition_number,
	)


def fit_visual_mimo_from_paths(csv_paths: List[str]) -> Optional[VisualMimoFit]:
	"""Aggregate all supplied CSVs and fit one roll/pitch visual matrix."""
	all_rows: List[Dict[str, str]] = []
	for path in csv_paths:
		all_rows.extend(read_csv_rows(path))
	return fit_visual_mimo(all_rows, restrict_to_roll_pitch_axes=True)


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


def divergence_motion_report(
	rows: List[Dict[str, str]], min_samples: int = 15, min_vz_std: float = 0.03
) -> Optional[Dict[str, float]]:
	"""
	Check whether optical-flow divergence still tracks vertical motion.

	Physics: for a downward camera approaching a surface, divergence is the
	inverse time-to-contact,  D ~ vz / altitude  (NED: vz > 0 = descending =
	image expanding = D > 0). So D should correlate with vz, and more tightly
	with vz/altitude. The thrust loop's only observable is D, so this is the
	number that says whether thrust control has anything to act on.

	Uses relative_vz_m_s/relative_z_m (vehicle motion relative to the
	platform, see platform_motion.py) when the columns are present and not
	empty, since that -- not the vehicle's own absolute vz -- is what
	divergence actually measures once the platform moves too: on an
	oscillating-platform test, vehicle_vz alone is the wrong variable and can
	show a misleadingly weak correlation even when divergence is tracking
	the true relative motion correctly. Falls back to vehicle_vz_m_s/
	-vehicle_z_m for older logs or a stationary platform, where they're
	equivalent (platform velocity is zero, so relative_vz == vehicle_vz).

	The correlation is expected to collapse in the terminal phase: once the
	target fills the frame, dense flow loses the texture spread needed to see
	the focus of expansion, and the median over a mostly-gradient-free ROI
	washes the signal out. This makes that failure visible per operating point
	instead of surfacing later as an insignificant thrust b.

	Returns None if there is too little vertical motion to judge.
	"""
	D, vz, vz_over_alt = [], [], []
	used_relative = False
	for row in rows:
		if str(row.get("flow_valid", "0")).strip() not in ("1", "True", "true"):
			continue
		d = _safe_float(row.get("flow_raw_divergence_1_s"))
		if d is None:
			d = _safe_float(row.get("flow_divergence_1_s"))

		v = _safe_float(row.get("relative_vz_m_s"))
		if v is not None:
			used_relative = True
			alt_z = _safe_float(row.get("relative_z_m"))
		else:
			v = _safe_float(row.get("vehicle_vz_m_s"))
			alt_z = _safe_float(row.get("vehicle_z_m"))
			alt_z = None if alt_z is None else -alt_z

		if d is None or v is None:
			continue
		D.append(d)
		vz.append(v)
		vz_over_alt.append(v / alt_z if (alt_z is not None and abs(alt_z) > 1e-3) else np.nan)

	D, vz, vz_over_alt = np.array(D), np.array(vz), np.array(vz_over_alt)
	if len(D) < min_samples or np.std(vz) < min_vz_std:
		return None

	corr_vz = float(np.corrcoef(D, vz)[0, 1])
	mask = np.isfinite(vz_over_alt)
	if mask.sum() > min_samples and np.std(vz_over_alt[mask]) > 1e-6 and np.std(D[mask]) > 1e-9:
		corr_vz_over_alt = float(np.corrcoef(D[mask], vz_over_alt[mask])[0, 1])
	else:
		corr_vz_over_alt = float("nan")
	slope = float(np.polyfit(vz, D, 1)[0]) if np.std(vz) > 1e-9 else float("nan")

	return {
		"n": len(D),
		"corr_vz": corr_vz,
		"corr_vz_over_alt": corr_vz_over_alt,
		"slope_D_per_vz": slope,
		"vz_std": float(np.std(vz)),
		"D_std": float(np.std(D)),
		"used_relative_motion": used_relative,
	}


def platform_tracking_report(
	rows: List[Dict[str, str]],
	expected_frequency_hz: Optional[float] = None,
	min_samples: int = 30,
) -> Optional[Dict[str, object]]:
	"""
	For an oscillating-platform test: does the vehicle's measured response
	(relative_vz_m_s, or vehicle_vz_m_s if platform tracking wasn't logged)
	actually show the commanded oscillation frequency, or something else
	(e.g. the thrust loop's own underdamped natural mode, ringing at its own
	frequency rather than tracking the disturbance cycle-by-cycle -- this is
	a real, observed failure mode, not just a hypothetical: a 0.3m/0.2Hz test
	produced a clean, large ~0.043 Hz (~23s period) response with NO energy
	near the commanded 0.2 Hz/5s anywhere in vz, altitude, or divergence, and
	an equivalent stationary-platform run showed no such mode at all -- so it
	was real and platform-triggered, just not a 1:1 echo of the input).

	Reports the dominant spectral peak found (via FFT) alongside the expected
	one, so a mismatch like that is visible immediately rather than requiring
	a manual deep-dive. Needs roughly-evenly-sampled rows; does not resample
	irregular timestamps.

	expected_frequency_hz: the platform's commanded frequency for this test
	(match bee_platform.sdf's z_frequency, mind the Hz-vs-rad/s caveat in
	platform_motion.py). None skips the expected-vs-found comparison.
	"""
	t, sig = [], []
	used_relative = False
	for row in rows:
		ti = _safe_float(row.get("t_sec"))
		v = _safe_float(row.get("relative_vz_m_s"))
		if v is not None:
			used_relative = True
		else:
			v = _safe_float(row.get("vehicle_vz_m_s"))
		if ti is None or v is None:
			continue
		t.append(ti)
		sig.append(v)

	if len(t) < min_samples:
		return None

	t, sig = np.array(t), np.array(sig)
	dt = float(np.median(np.diff(t)))
	if dt <= 1e-6:
		return None

	n = len(sig)
	freqs = np.fft.rfftfreq(n, d=dt)
	spec = np.abs(np.fft.rfft(sig - sig.mean()))

	valid = freqs > (1.0 / (n * dt))  # drop the DC/near-DC bin
	if not valid.any():
		return None
	dominant_idx = np.argmax(np.where(valid, spec, -1))
	dominant_freq = float(freqs[dominant_idx])

	result = {
		"used_relative_motion": used_relative,
		"n": n,
		"dt_sec": dt,
		"nyquist_hz": 0.5 / dt,
		"dominant_frequency_hz": dominant_freq,
		"dominant_period_sec": (1.0 / dominant_freq) if dominant_freq > 1e-9 else float("inf"),
		"signal_std": float(sig.std()),
	}
	if expected_frequency_hz is not None and expected_frequency_hz > 1e-9:
		result["expected_frequency_hz"] = float(expected_frequency_hz)
		result["expected_period_sec"] = 1.0 / expected_frequency_hz
		result["frequency_ratio_found_over_expected"] = dominant_freq / expected_frequency_hz
	return result


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
	path: str, dt_nominal: float, dt_tolerance: float, output_dir: str,
	row_filter: Optional[RowFilter] = None, max_area_fraction_span: float = 0.15,
	platform_frequency_hz: Optional[float] = None,
	input_delay_count: int = 2,
) -> Tuple[Dict[str, Optional[AxisFit]], Optional[Tuple[float, float, float]], bool, List[str]]:
	raw_rows = read_csv_rows(path)
	rows = apply_row_filter(raw_rows, row_filter)

	print(f"\n{path}  ({len(raw_rows)} rows)")
	if len(rows) != len(raw_rows):
		print(f"  row filter kept {len(rows)}/{len(raw_rows)} rows (altitude band / contact removed)")

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

	found_rows = [r for r in rows if row_is_found(r)]
	if found_rows and "target_fov_saturated" in found_rows[0]:
		sat_frac = sum(1 for r in found_rows if r.get("target_fov_saturated", "0") == "1") / len(found_rows)
		print(f"  fov_saturated: {sat_frac*100:.0f}% of found rows")
		if sat_frac > 0.10:
			print(
				"  <- WARNING: target exceeds the camera's field of view here -- "
				"area_fraction/detection box are clamped to the frame's own pixel "
				"size and are not tracking true range, even where the area_fraction "
				"span above looks tight."
			)

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

	plot_area_fraction_over_time(raw_rows, path, output_dir)

	visual_mimo = fit_visual_mimo(rows, restrict_to_roll_pitch_axes=True)
	if visual_mimo is None:
		print("  visual MIMO flow fit: not enough roll/pitch flow+attitude samples")
	else:
		if visual_mimo.input_cols == ("command_roll_rad", "command_pitch_rad"):
			print(
				"  visual MIMO flow fit: using commanded attitude because actual "
				"vehicle_roll_rad/vehicle_pitch_rad are absent, empty, or not varying in this CSV"
			)
		if visual_mimo.flow_cols[0].endswith("_px_s"):
			print(
				"  visual MIMO flow fit: using px/s flow fallback. Prefer new CSVs "
				"with flow_mean_*_norm_s for controller gains."
			)
		if visual_mimo.rank < 3:
			print(
				f"  visual MIMO flow fit: WARNING rank={visual_mimo.rank}/3. "
				"This usually means only roll or only pitch was excited, so the "
				"unexcited column cannot be separated from the bias."
			)
		print(f"  visual MIMO flow fit {visual_mimo}")

	fits = {"visual_mimo": visual_mimo}

	for axis in ("roll", "pitch", "thrust"):
		fit = fit_axis(rows, axis, dt_nominal, dt_tolerance)
		fits[axis] = fit

		if fit is None:
			print(f"  {axis:6s}: not enough valid consecutive samples to fit")
		else:
			flags = []
			if fit.r_squared <= 0.5:
				flags.append("low R^2, model may not fit this axis well")
			if not fit.b_is_significant:
				flags.append(
					"b not significant (|t|<2) -- likely noise, not a real reading of "
					"control effectiveness, even though R^2 looks fine"
				)

			flag_str = "  <- " + "; ".join(flags) if flags else ""
			print(f"  {axis:6s}: scalar {fit}{flag_str}")

		delayed = fit_axis_delayed(rows, axis, dt_nominal, dt_tolerance, input_delay_count=input_delay_count)
		if delayed is not None:
			b_txt = ", ".join(f"B{j}={b:+.5f}" for j, b in enumerate(delayed["B_delays"]))
			print(
				f"          delay-aware scalar: a={delayed['a']:+.4f}, {b_txt}, "
				f"Bsum={delayed['B_sum']:+.5f}, c={delayed['c']:+.5f}, "
				f"R^2={delayed['r_squared']:.3f}, n={delayed['n_samples']}"
			)

		if axis in ("roll", "pitch"):
			state_fit = fit_axis_state(rows, axis, dt_nominal, dt_tolerance)
			fits[f"{axis}_state"] = state_fit
			if state_fit is None:
				print(f"          state [offset, flow_mean]: not enough samples or no flow column")
			else:
				if state_fit.velocity_col.endswith("_px_s"):
					print(
						"          WARNING: using px/s flow fallback. Prefer new CSVs with "
						"flow_mean_*_norm_s before pasting gains into the controller."
					)
				print(f"          state {state_fit}")
				state_delayed = fit_axis_state_delayed(
					rows, axis, dt_nominal, dt_tolerance, input_delay_count=input_delay_count
				)
				if state_delayed is not None:
					B = state_delayed["B_delays"]
					parts = []
					for j in range(B.shape[1]):
						parts.append(f"B{j}=[{B[0,j]:+.5f}, {B[1,j]:+.5f}]")
					bs = state_delayed["B_sum"]
					print(
						"          delay-aware state: " + "; ".join(parts) +
						f"; Bsum=[{bs[0]:+.5f}, {bs[1]:+.5f}], "
						f"R^2=[{state_delayed['r_squared'][0]:.3f}, {state_delayed['r_squared'][1]:.3f}], "
						f"n={state_delayed['n_samples']}"
					)

	div_health = divergence_motion_report(rows)
	if div_health is not None:
		extra = ""
		if div_health["corr_vz_over_alt"] == div_health["corr_vz_over_alt"]:  # not NaN
			extra = f", corr(D, vz/alt)={div_health['corr_vz_over_alt']:+.2f}"
		source = "relative_vz" if div_health["used_relative_motion"] else "vehicle_vz"
		print(
			f"  divergence vs z-motion ({source}): corr(D, vz)={div_health['corr_vz']:+.2f}{extra}  "
			f"(n={div_health['n']}, vz_std={div_health['vz_std']:.2f} m/s)"
		)
		if abs(div_health["corr_vz"]) < 0.6:
			print(
				"    <- WARNING: divergence is weakly coupled to vertical motion here, so the "
				"thrust loop has little real signal at this operating point. Typical once the "
				"target fills the frame (dense flow loses the focus of expansion), but if "
				f"{source}=vehicle_vz, also check this isn't simply the platform moving too -- "
				"see platform_motion.py and --platform-frequency-hz."
			)

	track = platform_tracking_report(rows, expected_frequency_hz=platform_frequency_hz)
	if track is not None:
		source = "relative_vz" if track["used_relative_motion"] else "vehicle_vz (no platform tracking logged)"
		print(
			f"  platform tracking ({source}): dominant response at "
			f"{track['dominant_frequency_hz']:.4f} Hz (period {track['dominant_period_sec']:.1f}s), "
			f"signal_std={track['signal_std']:.3f}, control rate {1.0/track['dt_sec']:.2f} Hz "
			f"(Nyquist {track['nyquist_hz']:.2f} Hz)"
		)
		if "expected_frequency_hz" in track:
			ratio = track["frequency_ratio_found_over_expected"]
			print(
				f"    commanded {track['expected_frequency_hz']:.4f} Hz (period "
				f"{track['expected_period_sec']:.1f}s) -> found/expected ratio={ratio:.2f}"
			)
			if not (0.7 <= ratio <= 1.3):
				print(
					"    <- WARNING: dominant response frequency does not match the commanded "
					"platform frequency. Likely the thrust loop's own underdamped natural mode "
					"ringing rather than a cycle-by-cycle echo of the disturbance (the loop can't "
					"track a disturbance much faster than its own bandwidth), not a units/sign "
					"bug in this report -- but also double-check the Hz-vs-rad/s assumption in "
					"platform_motion.py if this is the first test at a new frequency."
				)

	ok, reasons = assess_file_quality(rows, max_area_fraction_span=max_area_fraction_span)
	# A run with thrust data but a blind divergence observable would only pour
	# noise into the thrust schedule, so flag it as well.
	thrust_rows = sum(1 for r in rows if r.get("calibration_axis") == "thrust")
	if div_health is not None and thrust_rows >= 10 and abs(div_health["corr_vz"]) < 0.4:
		ok = False
		reasons.append(
			f"divergence not tracking vz (corr={div_health['corr_vz']:+.2f}); thrust model "
			"is unidentifiable at this operating point"
		)

	if ok:
		print("  verdict: OK (usable for the schedule)")
	else:
		print("  verdict: EXCLUDED from schedule -- " + "; ".join(reasons)
			  + "  (override with --keep-flagged)")

	return fits, stats, ok, reasons


def collect_axis_points(
	per_file_fits: List[Tuple[str, Dict[str, Optional[AxisFit]], Optional[Tuple[float, float, float]]]],
	axis: str,
) -> List[Tuple[float, float, float]]:
	"""Sorted (mean_area_fraction, a, b) for every file with a valid fit+stats on this axis."""
	points = []

	for path, fits, stats, *_ in per_file_fits:
		fit = fits.get(axis)

		if fit is None or stats is None:
			continue

		_, mean_area_fraction, _ = stats
		points.append((mean_area_fraction, fit.a, fit.b))

	points.sort(key=lambda p: p[0])
	return points


def collect_state_points(per_file_fits, axis: str):
	"""Sorted (mean_area_fraction, AxisStateFit) for files with a valid state fit."""
	points = []
	for _, fits, stats, *_ in per_file_fits:
		fit = fits.get(f"{axis}_state")
		if fit is None or stats is None:
			continue
		points.append((stats[1], fit))
	points.sort(key=lambda p: p[0])
	return points


def print_controller_schedule(per_file_fits):
	"""
	Emit schedule entries in exactly the form control_law.py consumes:
	2-state (A, B) for roll/pitch from the [offset, flow] state fit, and
	scalar (a, b) for thrust. Paste into ROLL_STATE_MODELS / PITCH_STATE_MODELS
	/ THRUST_DIVERGENCE_MODELS. (The 1-state roll/pitch schedule above is only
	a sqrt-scaling diagnostic; the controller uses these 2-state models.)
	"""
	print("\n--- ready-to-paste control_law.py models (kept files only) ---")

	for axis, const in (("roll", "ROLL_STATE_MODELS"), ("pitch", "PITCH_STATE_MODELS")):
		points = collect_state_points(per_file_fits, axis)
		if not points:
			continue
		print(f"\n{const} = (")
		for af, f in points:
			print(
				f"    ({af:.3f}, [[{f.A[0,0]:.4f}, {f.A[0,1]:.4f}], "
				f"[{f.A[1,0]:.4f}, {f.A[1,1]:.4f}]], "
				f"[[{f.B[0]:.5f}], [{f.B[1]:.5f}]]),"
			)
		print(")")

	thrust_points = collect_axis_points(per_file_fits, "thrust")
	if thrust_points:
		print("\nTHRUST_DIVERGENCE_MODELS = (")
		for af, a, b in thrust_points:
			print(f"    ({af:.3f}, [[{a:.4f}]], [[{b:.4f}]]),")
		print(")")


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

		# 1-state schedule: a scaling diagnostic only. control_law.py uses the
		# 2-state roll/pitch models from print_controller_schedule; for thrust
		# the scalar entry below is the one it consumes.
		print(f"\n  1-state {axis} schedule (diagnostic; thrust row is controller-ready):")
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



def split_multi_altitude_csvs(csv_paths: List[str], output_dir: str) -> List[str]:
	"""
	Split a single multi-altitude calibration CSV into temporary per-altitude CSVs.

	calibration_node.py now repeats the full roll/pitch/thrust sequence at several
	altitude knots in one run. The model schedule, however, still needs one file-like
	operating point per area_fraction. Splitting here lets the rest of this fitter
	keep its per-file quality checks and ready-to-paste schedule logic unchanged.
	"""
	split_dir = os.path.join(output_dir, "altitude_splits")
	os.makedirs(split_dir, exist_ok=True)
	out_paths: List[str] = []

	for path in csv_paths:
		rows = read_csv_rows(path)
		if not rows or "calibration_altitude_index" not in rows[0]:
			out_paths.append(path)
			continue

		groups: Dict[str, List[Dict[str, str]]] = {}
		for row in rows:
			idx = row.get("calibration_altitude_index", "")
			if idx == "":
				continue
			groups.setdefault(idx, []).append(row)

		if len(groups) <= 1:
			out_paths.append(path)
			continue

		fieldnames = list(rows[0].keys())
		stem = os.path.splitext(os.path.basename(path))[0]
		for idx, group in sorted(groups.items(), key=lambda kv: float(kv[0])):
			alts = [_safe_float(r.get("calibration_altitude_m")) for r in group]
			alts = [a for a in alts if a is not None]
			alt_label = f"{float(np.mean(alts)):.2f}m" if alts else f"idx{idx}"
			alt_label = alt_label.replace(".", "p")
			out_path = os.path.join(split_dir, f"{stem}_alt{int(float(idx)):02d}_{alt_label}.csv")
			with open(out_path, "w", newline="") as f:
				writer = csv.DictWriter(f, fieldnames=fieldnames)
				writer.writeheader()
				writer.writerows(group)
			out_paths.append(out_path)
		print(f"Split multi-altitude CSV {path} into {len(groups)} per-altitude files in {split_dir}")

	return out_paths


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

	# --- Data-quality selection ---
	parser.add_argument(
		"--min-altitude", type=float, default=None,
		help="drop rows whose altitude (-vehicle_z_m) is below this many meters",
	)
	parser.add_argument(
		"--max-altitude", type=float, default=None,
		help="drop rows whose altitude (-vehicle_z_m) is above this many meters",
	)
	parser.add_argument(
		"--keep-contact", action="store_true",
		help="do NOT drop detected ground/platform contact windows (default: drop them)",
	)
	parser.add_argument(
		"--exclude", nargs="*", default=[],
		help="skip any CSV whose path contains one of these substrings (e.g. _2m _3m)",
	)
	parser.add_argument(
		"--keep-flagged", action="store_true",
		help="include files that fail the quality verdict in the schedule anyway",
	)
	parser.add_argument(
		"--max-area-fraction-span", type=float, default=0.15,
		help="reject a file whose area_fraction sweeps wider than this (not one operating point)",
	)
	parser.add_argument(
		"--input-delay-count", type=int, default=2,
		help="fit extra delayed input columns u[k-1]..u[k-d] for delay-aware diagnostics",
	)
	parser.add_argument(
		"--platform-frequency-hz", type=float, default=None,
		help="commanded oscillating-platform frequency for this test (match bee_platform.sdf's "
		     "z_frequency, mind the Hz-vs-rad/s caveat in platform_motion.py) -- enables the "
		     "found-vs-expected check in platform_tracking_report()",
	)
	args = parser.parse_args()

	ensure_output_dir(args.output_dir)

	csv_paths = expand_csv_paths(args.csv_paths, pattern=args.pattern, recursive=args.recursive)
	csv_paths = split_multi_altitude_csvs(csv_paths, args.output_dir)

	if not csv_paths:
		print("No CSV files found from the given path(s).")
		return

	# Explicit exclusions by path substring (e.g. --exclude _2m _3m).
	if args.exclude:
		kept_paths = [p for p in csv_paths if not any(s in p for s in args.exclude)]
		for p in csv_paths:
			if p not in kept_paths:
				print(f"Excluding (matched --exclude): {p}")
		csv_paths = kept_paths

	if not csv_paths:
		print("All CSV files were excluded.")
		return

	row_filter = RowFilter(
		min_altitude_m=args.min_altitude,
		max_altitude_m=args.max_altitude,
		drop_contact=not args.keep_contact,
	)

	if len(csv_paths) > 1:
		print(f"Found {len(csv_paths)} CSV file(s):")
		for path in csv_paths:
			print(f"  {path}")

	per_file_fits = []
	for path in csv_paths:
		fits, stats, ok, reasons = analyze_file(
			path, args.dt, args.dt_tolerance, args.output_dir,
			row_filter=row_filter, max_area_fraction_span=args.max_area_fraction_span,
			platform_frequency_hz=args.platform_frequency_hz,
			input_delay_count=args.input_delay_count,
		)
		per_file_fits.append((path, fits, stats, ok, reasons))

	# Only trustworthy files feed the cross-file schedule (unless overridden).
	if args.keep_flagged:
		schedule_fits = per_file_fits
	else:
		schedule_fits = [e for e in per_file_fits if e[3]]

	excluded = [e for e in per_file_fits if not e[3]]
	if excluded and not args.keep_flagged:
		print("\nExcluded from schedule (failed quality verdict):")
		for path, _, _, _, reasons in excluded:
			print(f"  {path}: {'; '.join(reasons)}")

	if len(schedule_fits) > 1:
		check_scaling_and_print_schedule(schedule_fits, args.area_fraction_ref)
		plot_b_vs_area_fraction(schedule_fits, args.area_fraction_ref, args.output_dir)
		print_controller_schedule(schedule_fits)
	elif len(schedule_fits) <= 1:
		print("\nNeed 2+ kept files for a schedule; "
			  f"{len(schedule_fits)} passed the quality verdict.")

	# Aggregate MIMO uses the same row filter and only the kept files.
	kept_paths = [e[0] for e in schedule_fits] or [e[0] for e in per_file_fits]
	all_rows: List[Dict[str, str]] = []
	for p in kept_paths:
		all_rows.extend(apply_row_filter(read_csv_rows(p), row_filter))
	aggregate_mimo = fit_visual_mimo(all_rows, restrict_to_roll_pitch_axes=True)
	if aggregate_mimo is not None:
		print("\n--- aggregate visual MIMO flow model over kept CSVs ---")
		if aggregate_mimo.input_cols == ("command_roll_rad", "command_pitch_rad"):
			print(
				"  using commanded attitude because actual vehicle_roll_rad/"
				"vehicle_pitch_rad are absent, empty, or not varying. Re-run calibration_node.py after "
				"the attitude/odometry logging patch for the cleaner model."
			)
		if aggregate_mimo.rank < 3:
			print(
				f"  WARNING rank={aggregate_mimo.rank}/3. You probably need both a "
				"roll CSV and a pitch CSV to identify the full 2x2 matrix."
			)
		print(aggregate_mimo)
		print(
			"  Controller sign convention hint: for a first visual damping test, "
			"use the sign of this G matrix with state [offset_x, offset_y, "
			"flow_x, flow_y], then tune gains conservatively on a static target."
		)


if __name__ == "__main__":
	main()