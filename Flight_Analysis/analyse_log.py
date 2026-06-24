"""
Analyse bee landing diagnostics CSV.

Usage:

	# One CSV -> write all plots directly into results/test9
	python analyse_log.py logs/bee_diagnostics_XXXXXXXX.csv results/test9

	# Whole folder -> write one subfolder per CSV: results/test1, results/test2, ...
	python analyse_log.py logs

Optional:

	python analyse_log.py logs/bee_diagnostics_XXXXXXXX.csv \
		--image-width 640 \
		--image-height 480 \
		--output-dir results/test9 \
		--divergence-setpoint 0.15 \
		--area-fraction-schedule 0.05,0.30,0.60,0.80,0.95,1.0 \
		--max-area-fraction 0.60 \
		--absolute-max-area-fraction 0.95

Generated plots:

	- detection_boxes_fov.png
	- target_position_offsets.png
	- target_quality.png            (target_confidence, target_area_fraction)
	- vehicle_position_xyz.png
	- vehicle_dynamics.png          (vx, vy, vz, yaw)
	- divergence.png
	- flow_mean_velocity.png        (flow_mean_x_px_s, flow_mean_y_px_s)
	- commands.png
	- pipeline_latency.png          (data age per subsystem at logging time)

Plots are skipped (with a printed message, not an error) when their
required columns aren't present in the CSV, so this also works on logs
from before a given column was added.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def read_log(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)

	if df.empty:
		raise ValueError(f"CSV file is empty: {csv_path}")

	return df


def get_time(df: pd.DataFrame) -> np.ndarray:
	"""
	Return normalized time in seconds.

	Priority:
		1. t_sec column
		2. wall_timestamp - first wall_timestamp
		3. sample index
	"""
	if "t_sec" in df.columns:
		t = pd.to_numeric(df["t_sec"], errors="coerce").to_numpy()
		return t - np.nanmin(t)

	if "wall_timestamp" in df.columns:
		t = pd.to_numeric(df["wall_timestamp"], errors="coerce").to_numpy()
		return t - np.nanmin(t)

	return np.arange(len(df), dtype=float)


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

	# Handles 0/1, "true"/"false", "True"/"False".
	if raw.dtype == object:
		cleaned = raw.astype(str).str.lower().str.strip()
		return cleaned.isin(["1", "true", "yes", "y"]).to_numpy(dtype=bool)

	return pd.to_numeric(raw, errors="coerce").fillna(0).to_numpy(dtype=float) > 0.5


def ensure_output_dir(output_dir: str):
	os.makedirs(output_dir, exist_ok=True)


def save_current_figure(output_dir: str, filename: str):
	path = os.path.join(output_dir, filename)
	plt.tight_layout()
	plt.savefig(path, dpi=160)
	plt.close()
	print(f"Saved: {path}")


def plot_detection_boxes_fov(
	df: pd.DataFrame,
	t: np.ndarray,
	image_width: int,
	image_height: int,
	output_dir: str,
	max_boxes: int = 150,
):
	"""
	Reconstruct detection boxes in the camera field of view.

	Uses:
		target_offset_x
		target_offset_y
		target_detection_width_px
		target_detection_height_px

	Coordinate convention:
		offset_x = -1 left, +1 right
		offset_y = -1 top,  +1 bottom
	"""
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

	ax.set_title("Detection boxes in camera field of view")
	ax.set_xlabel("image x [px]")
	ax.set_ylabel("image y [px]")

	ax.set_xlim(0, image_width)
	ax.set_ylim(image_height, 0)
	ax.set_aspect("equal", adjustable="box")

	# Image border.
	ax.add_patch(
		Rectangle(
			(0, 0),
			image_width,
			image_height,
			fill=False,
			linewidth=2,
		)
	)

	# Image center.
	ax.axvline(image_width / 2.0, linestyle="--", linewidth=1)
	ax.axhline(image_height / 2.0, linestyle="--", linewidth=1)

	# Draw boxes through time.
	cmap = plt.get_cmap("viridis")
	center_x_list = []
	center_y_list = []

	for k, idx in enumerate(indices):
		alpha = 0.25 + 0.75 * k / max(len(indices) - 1, 1)
		color = cmap(k / max(len(indices) - 1, 1))

		cx = (0.5 * offset_x[idx] + 0.5) * image_width
		cy = (0.5 * offset_y[idx] + 0.5) * image_height

		x0 = cx - 0.5 * box_w[idx]
		y0 = cy - 0.5 * box_h[idx]

		rect = Rectangle(
			(x0, y0),
			box_w[idx],
			box_h[idx],
			fill=False,
			linewidth=1.2,
			edgecolor=(color[0], color[1], color[2], alpha),
		)

		ax.add_patch(rect)

		center_x_list.append(cx)
		center_y_list.append(cy)

	ax.plot(center_x_list, center_y_list, marker=".", linewidth=1.2, label="detection center")
	ax.legend(loc="best")

	save_current_figure(output_dir, "detection_boxes_fov.png")


def plot_target_position_offsets(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	if "target_offset_x" not in df.columns or "target_offset_y" not in df.columns:
		print("Skipping target offset plot. Missing target_offset_x or target_offset_y.")
		return

	offset_x = numeric_column(df, "target_offset_x")
	offset_y = numeric_column(df, "target_offset_y")
	target_found = bool_column(df, "target_found", default=True)

	plt.figure(figsize=(10, 5))
	plt.title("Target position in image")
	plt.plot(t, offset_x, label="target_offset_x")
	plt.plot(t, offset_y, label="target_offset_y")

	if "target_found" in df.columns:
		not_found = ~target_found
		if np.any(not_found):
			plt.scatter(t[not_found], np.zeros(np.sum(not_found)), marker="x", label="target not found")

	plt.axhline(0.0, linestyle="--", linewidth=1)
	plt.xlabel("time [s]")
	plt.ylabel("normalized image offset [-]")
	plt.grid(True)
	plt.legend()

	save_current_figure(output_dir, "target_position_offsets.png")


def plot_target_quality(
	df: pd.DataFrame,
	t: np.ndarray,
	output_dir: str,
	schedule_points=None,
	max_area_fraction: float = None,
	absolute_max_area_fraction: float = None,
):
	"""
	target_confidence and target_area_fraction over time, with the
	area_fraction schedule breakpoints (and target_acquisition.py's
	large-area thresholds, if given) drawn as reference lines, so you
	can see where the trajectory actually sits relative to the gain
	schedule and the large-area penalty curve.
	"""
	has_confidence = "target_confidence" in df.columns
	has_area_fraction = "target_area_fraction" in df.columns

	if not has_confidence and not has_area_fraction:
		print("Skipping target quality plot. Missing target_confidence and target_area_fraction.")
		return

	target_found = bool_column(df, "target_found", default=True)
	not_found = ~target_found

	rows = int(has_confidence) + int(has_area_fraction)
	fig, axes = plt.subplots(rows, 1, figsize=(10, 3.2 * rows), sharex=True)

	if rows == 1:
		axes = [axes]

	ax_index = 0

	if has_confidence:
		ax = axes[ax_index]
		ax_index += 1

		confidence = numeric_column(df, "target_confidence")
		ax.plot(t, confidence, label="target_confidence", color="tab:blue")

		if np.any(not_found):
			ax.scatter(t[not_found], np.zeros(np.sum(not_found)), marker="x", color="tab:red", label="target not found")

		ax.set_ylabel("confidence [-]")
		ax.set_ylim(-0.05, 1.05)
		ax.grid(True)
		ax.legend(loc="best")

	if has_area_fraction:
		ax = axes[ax_index]
		ax_index += 1

		area_fraction = numeric_column(df, "target_area_fraction")
		ax.plot(t, area_fraction, label="target_area_fraction", color="tab:green")

		if schedule_points:
			for value in schedule_points:
				ax.axhline(value, linestyle=":", linewidth=0.8, color="gray")
			ax.plot([], [], linestyle=":", color="gray", linewidth=0.8, label="schedule points")

		if max_area_fraction is not None:
			ax.axhline(
				max_area_fraction, linestyle="--", linewidth=1, color="tab:orange",
				label=f"max_area_fraction={max_area_fraction:g}",
			)

		if absolute_max_area_fraction is not None:
			ax.axhline(
				absolute_max_area_fraction, linestyle="--", linewidth=1, color="tab:red",
				label=f"absolute_max_area_fraction={absolute_max_area_fraction:g}",
			)

		ax.set_ylabel("area_fraction [-]")
		ax.set_ylim(-0.05, 1.05)
		ax.grid(True)
		ax.legend(loc="best")

	axes[-1].set_xlabel("time [s]")

	save_current_figure(output_dir, "target_quality.png")


def plot_vehicle_position(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	if "vehicle_x_m" not in df.columns or "vehicle_y_m" not in df.columns:
		print("Skipping vehicle position plot. Missing vehicle_x_m or vehicle_y_m.")
		return

	columns = [
		("vehicle_x_m", "vehicle x [m]"),
		("vehicle_y_m", "vehicle y [m]"),
		("vehicle_z_m", "vehicle z [m]"),
	]

	available = [(name, label) for name, label in columns if name in df.columns]

	if not available:
		print("Skipping vehicle position plot. No position columns found.")
		return

	fig, axes = plt.subplots(len(available), 1, figsize=(10, 2.7 * len(available)), sharex=True)

	if len(available) == 1:
		axes = [axes]

	for ax, (column_name, label) in zip(axes, available):
		y = numeric_column(df, column_name)
		ax.plot(t, y, label=label)
		ax.axhline(0.0, linestyle="--", linewidth=1)
		ax.set_ylabel(label)
		ax.grid(True)
		ax.legend(loc="best")

	axes[-1].set_xlabel("time [s]")

	save_current_figure(output_dir, "vehicle_position_xyz.png")


def plot_vehicle_dynamics(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	"""vehicle_vx/vy/vz_m_s and vehicle_yaw_rad: the rest of VehicleState beyond position."""
	columns = [
		("vehicle_vx_m_s", "vehicle vx [m/s]"),
		("vehicle_vy_m_s", "vehicle vy [m/s]"),
		("vehicle_vz_m_s", "vehicle vz [m/s]"),
		("vehicle_yaw_rad", "vehicle yaw [rad]"),
	]

	available = [(name, label) for name, label in columns if name in df.columns]

	if not available:
		print("Skipping vehicle dynamics plot. No velocity/yaw columns found.")
		return

	fig, axes = plt.subplots(len(available), 1, figsize=(10, 2.7 * len(available)), sharex=True)

	if len(available) == 1:
		axes = [axes]

	for ax, (column_name, label) in zip(axes, available):
		y = numeric_column(df, column_name)
		ax.plot(t, y, label=label)
		ax.axhline(0.0, linestyle="--", linewidth=1)
		ax.set_ylabel(label)
		ax.grid(True)
		ax.legend(loc="best")

	axes[-1].set_xlabel("time [s]")

	save_current_figure(output_dir, "vehicle_dynamics.png")


def plot_divergence(df: pd.DataFrame, t: np.ndarray, output_dir: str, divergence_setpoint: float = None):
	divergence_columns = []

	if "flow_divergence_1_s" in df.columns:
		divergence_columns.append(("flow_divergence_1_s", "flow divergence"))

	# Optional future columns, in case you later add D_box to diagnostics.
	optional_columns = [
		("box_divergence_1_s", "box divergence"),
		("box_divergence_filtered_1_s", "box divergence filtered"),
		# Current diagnostics_writer.py uses this name; keep the older variant too
		# so old logs remain readable.
		("flow_raw_divergence_1_s", "raw flow divergence"),
		("flow_divergence_raw_1_s", "raw flow divergence"),
	]

	for column_name, label in optional_columns:
		if column_name in df.columns:
			divergence_columns.append((column_name, label))

	if not divergence_columns:
		print("Skipping divergence plot. Missing flow_divergence_1_s.")
		return

	flow_valid = bool_column(df, "flow_valid", default=True)

	plt.figure(figsize=(10, 5))
	plt.title("Divergence evolution")

	for column_name, label in divergence_columns:
		y = numeric_column(df, column_name)
		plt.plot(t, y, label=label)

	if divergence_setpoint is not None:
		plt.axhline(
			divergence_setpoint, linestyle=":", linewidth=1.4, color="tab:purple",
			label=f"divergence_setpoint={divergence_setpoint:g}",
		)

	invalid = ~flow_valid
	if np.any(invalid):
		plt.scatter(t[invalid], np.zeros(np.sum(invalid)), marker="x", color="tab:red", label="flow invalid")

	plt.axhline(0.0, linestyle="--", linewidth=1)
	plt.xlabel("time [s]")
	plt.ylabel("divergence [1/s]")
	plt.grid(True)
	plt.legend()

	save_current_figure(output_dir, "divergence.png")


def plot_flow_velocity(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	"""flow_mean_x_px_s / flow_mean_y_px_s: the mean optical flow over the target ROI."""
	has_x = "flow_mean_x_px_s" in df.columns
	has_y = "flow_mean_y_px_s" in df.columns

	if not has_x and not has_y:
		print("Skipping flow velocity plot. Missing flow_mean_x_px_s and flow_mean_y_px_s.")
		return

	flow_valid = bool_column(df, "flow_valid", default=True)

	plt.figure(figsize=(10, 5))
	plt.title("Mean optical flow over the target ROI")

	if has_x:
		plt.plot(t, numeric_column(df, "flow_mean_x_px_s"), label="flow_mean_x_px_s")
	if has_y:
		plt.plot(t, numeric_column(df, "flow_mean_y_px_s"), label="flow_mean_y_px_s")

	invalid = ~flow_valid
	if np.any(invalid):
		plt.scatter(t[invalid], np.zeros(np.sum(invalid)), marker="x", color="tab:red", label="flow invalid")

	plt.axhline(0.0, linestyle="--", linewidth=1)
	plt.xlabel("time [s]")
	plt.ylabel("mean flow [px/s]")
	plt.grid(True)
	plt.legend()

	save_current_figure(output_dir, "flow_mean_velocity.png")


def plot_commands(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	command_columns = [
		("command_roll_rad", "roll command [rad]"),
		("command_pitch_rad", "pitch command [rad]"),
		("command_yaw_rad", "yaw command [rad]"),
		("command_thrust", "thrust command [-]"),
	]

	available = [(name, label) for name, label in command_columns if name in df.columns]

	if not available:
		print("Skipping command plot. No command columns found.")
		return

	fig, axes = plt.subplots(len(available), 1, figsize=(10, 2.7 * len(available)), sharex=True)

	if len(available) == 1:
		axes = [axes]

	for ax, (column_name, label) in zip(axes, available):
		y = numeric_column(df, column_name)
		ax.plot(t, y, label=label)
		if column_name == "command_thrust":
			ax.axhline(0.73, linestyle="--", linewidth=1)
		else:
			ax.axhline(0.0, linestyle="--", linewidth=1)
		ax.set_ylabel(label)
		ax.grid(True)
		ax.legend(loc="best")

	axes[-1].set_xlabel("time [s]")

	save_current_figure(output_dir, "commands.png")


def plot_pipeline_latency(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	"""
	How stale each subsystem's measurement was at the moment its row got
	logged: t - <subsystem>_timestamp_sec. Both sides are normalized the
	same way by diagnostics_writer.py, so this is a direct "data age in
	seconds" reading, not just a relative comparison.
	"""
	columns = [
		("target_timestamp_sec", "target age [s]"),
		("flow_timestamp_sec", "flow age [s]"),
		("command_timestamp_sec", "command age [s]"),
		("vehicle_timestamp_sec", "vehicle age [s]"),
	]

	available = [(name, label) for name, label in columns if name in df.columns]

	if not available:
		print("Skipping pipeline latency plot. No *_timestamp_sec columns found.")
		return

	plt.figure(figsize=(10, 5))
	plt.title("Data age at logging time")

	for column_name, label in available:
		subsystem_time = numeric_column(df, column_name)
		age = t - subsystem_time
		plt.plot(t, age, label=label)

	plt.axhline(0.0, linestyle="--", linewidth=1)
	plt.xlabel("time [s]")
	plt.ylabel("age [s]")
	plt.grid(True)
	plt.legend()

	save_current_figure(output_dir, "pipeline_latency.png")


def _looks_like_csv_path(path_like: str) -> bool:
	return Path(path_like).suffix.lower() == ".csv"


def _path_contains_csv(path_like: str) -> bool:
	path = Path(path_like)
	return path.is_dir() and any(child.suffix.lower() == ".csv" for child in path.iterdir())


def _split_inputs_and_output_dir(raw_paths, output_dir_arg: str, default_output_dir: str):
	"""
	Accept both styles:

		python analyse_log.py logs/file.csv results/test9
		python analyse_log.py logs --output-dir results

	If --output-dir was not explicitly changed and the last positional argument
	looks like an output folder rather than a CSV/folder input, use it as the
	output directory. This keeps the command line short while preserving
	--output-dir for unambiguous multi-input calls.
	"""
	paths = [str(path) for path in raw_paths]
	output_dir = output_dir_arg

	# Detect: analyse_log.py <input.csv or input_folder> <output_folder>
	if output_dir_arg == default_output_dir and len(paths) >= 2:
		last = paths[-1]
		last_path = Path(last)

		last_is_input = (
			_looks_like_csv_path(last)
			or _path_contains_csv(last)
		)

		if not last_is_input:
			output_dir = last
			paths = paths[:-1]

	if not paths:
		raise ValueError("No CSV file or log folder was provided.")

	return paths, output_dir


def collect_csv_paths(inputs):
	"""Return sorted CSV paths from one or more files/folders."""
	csv_paths = []

	for item in inputs:
		path = Path(item)

		if path.is_dir():
			csv_paths.extend(sorted(path.glob("*.csv")))
		elif path.is_file() and path.suffix.lower() == ".csv":
			csv_paths.append(path)
		else:
			raise FileNotFoundError(
				f"Input is neither a CSV file nor a folder containing CSV logs: {item}"
			)

	if not csv_paths:
		raise FileNotFoundError("No CSV files found in the provided input(s).")

	return csv_paths


def output_dir_for_csv(csv_path: Path, output_root: Path, total_csv_count: int, index: int, explicit_positional_output: bool):
	"""
	Single CSV + explicit output folder:
		results/test9/<plots>.png

	Folder or multiple CSVs:
		results/test1/<plots>.png
		results/test2/<plots>.png
	"""
	if total_csv_count == 1 and explicit_positional_output:
		return output_root

	if total_csv_count == 1 and output_root.name.lower().startswith("test"):
		return output_root

	return output_root / f"test{index}"


def main():
	parser = argparse.ArgumentParser(
		description="Analyse bee landing diagnostics CSV."
	)

	parser.add_argument(
		"paths",
		nargs="+",
		help=(
			"CSV file(s), folder(s) containing CSV logs, and optionally an output "
			"folder as the final positional argument. Examples: "
			"analyse_log.py logs/file.csv results/test9 ; analyse_log.py logs"
		),
	)

	parser.add_argument(
		"--image-width",
		type=int,
		default=640,
		help="Camera image width in pixels. Default: 640.",
	)

	parser.add_argument(
		"--image-height",
		type=int,
		default=480,
		help="Camera image height in pixels. Default: 480.",
	)

	default_output_dir = "results"
	parser.add_argument(
		"--output-dir",
		default=default_output_dir,
		help="Directory where plots will be saved. Default: results.",
	)

	parser.add_argument(
		"--max-boxes",
		type=int,
		default=150,
		help="Maximum number of detection boxes drawn in field-of-view reconstruction.",
	)

	parser.add_argument(
		"--divergence-setpoint",
		type=float,
		default=0.15,
		help="ControlLaw's divergence_setpoint, drawn as a reference line on the divergence plot. Default: 0.15.",
	)

	parser.add_argument(
		"--area-fraction-schedule",
		default="0.05,0.30,0.60,0.80,0.95,1.0",
		help=(
			"Comma-separated area_fraction values to draw as reference lines "
			"on the target quality plot (match ControlLaw's schedule_points). "
			"Pass an empty string to disable."
		),
	)

	parser.add_argument(
		"--max-area-fraction",
		type=float,
		default=0.60,
		help="TargetAcquisition's max_area_fraction, drawn as a reference line. Default: 0.60.",
	)

	parser.add_argument(
		"--absolute-max-area-fraction",
		type=float,
		default=0.95,
		help="TargetAcquisition's absolute_max_area_fraction, drawn as a reference line. Default: 0.95.",
	)

	args = parser.parse_args()

	# True for commands such as:
	#   py analyse_log.py logs/bee_diagnostics.csv results/test9
	positional_output_requested = (
		args.output_dir == default_output_dir
		and len(args.paths) >= 2
		and not _looks_like_csv_path(args.paths[-1])
		and not _path_contains_csv(args.paths[-1])
	)

	input_paths, output_dir = _split_inputs_and_output_dir(
		raw_paths=args.paths,
		output_dir_arg=args.output_dir,
		default_output_dir=default_output_dir,
	)

	csv_paths = collect_csv_paths(input_paths)
	output_root = Path(output_dir)
	ensure_output_dir(output_root)

	schedule_points = (
		[float(value) for value in args.area_fraction_schedule.split(",") if value.strip() != ""]
		if args.area_fraction_schedule
		else []
	)

	for index, csv_p in enumerate(csv_paths, start=1):
		df = read_log(csv_p)
		t = get_time(df)
		output_complete = output_dir_for_csv(
			csv_path=csv_p,
			output_root=output_root,
			total_csv_count=len(csv_paths),
			index=index,
			explicit_positional_output=positional_output_requested,
		)
		ensure_output_dir(output_complete)

		print(f"Loaded: {csv_p}")
		print(f"Rows: {len(df)}")
		print(f"Time span: {np.nanmin(t):.3f} s to {np.nanmax(t):.3f} s")
		print(f"Output directory: {output_complete}")

		plot_detection_boxes_fov(
			df=df,
			t=t,
			image_width=args.image_width,
			image_height=args.image_height,
			output_dir=output_complete,
			max_boxes=args.max_boxes,
		)

		plot_target_position_offsets(
			df=df,
			t=t,
			output_dir=output_complete,
		)

		plot_target_quality(
			df=df,
			t=t,
			output_dir=output_complete,
			schedule_points=schedule_points,
			max_area_fraction=args.max_area_fraction,
			absolute_max_area_fraction=args.absolute_max_area_fraction,
		)

		plot_vehicle_position(
			df=df,
			t=t,
			output_dir=output_complete,
		)

		plot_vehicle_dynamics(
			df=df,
			t=t,
			output_dir=output_complete,
		)

		plot_divergence(
			df=df,
			t=t,
			output_dir=output_complete,
			divergence_setpoint=args.divergence_setpoint,
		)

		plot_flow_velocity(
			df=df,
			t=t,
			output_dir=output_complete,
		)

		plot_commands(
			df=df,
			t=t,
			output_dir=output_complete,
		)

		plot_pipeline_latency(
			df=df,
			t=t,
			output_dir=output_complete,
		)

		print("Done.")


if __name__ == "__main__":
	main()