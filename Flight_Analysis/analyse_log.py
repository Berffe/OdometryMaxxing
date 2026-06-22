"""
Analyse bee landing diagnostics CSV.

Usage:

	python analyse_log.py logs/bee_diagnostics_XXXXXXXX.csv

Optional:

	python analyse_log.py logs/bee_diagnostics_XXXXXXXX.csv \
		--image-width 640 \
		--image-height 480 \
		--output-dir analysis_output

Generated plots:

	- detection_boxes_fov.png
	- target_position_offsets.png
	- vehicle_position_xy.png
	- divergence.png
	- commands.png
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

	# plt.show()
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

	# plt.show()
	save_current_figure(output_dir, "target_position_offsets.png")

def plot_vehicle_xyz(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	if "vehicle_x_m" not in df.columns or "vehicle_y_m" not in df.columns:
		print("Skipping vehicle XY plot. Missing vehicle_x_m or vehicle_y_m.")
		return

	numeric_columns = [
		("vehicle_x_m", "vehicle x [m]"),
		("vehicle_y_m", "vehicle y [m]"),
		("vehicle_z_m", "vehicle z [m]"),
	]

	available = [(name, label) for name, label in numeric_columns if name in df.columns]

	if not available:
		print("Skipping command plot. No command columns found.")
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

	# plt.show()
	save_current_figure(output_dir, "vehicle_position_xyz.png")


def plot_divergence(df: pd.DataFrame, t: np.ndarray, output_dir: str):
	divergence_columns = []

	if "flow_divergence_1_s" in df.columns:
		divergence_columns.append(("flow_divergence_1_s", "flow divergence"))

	# Optional future columns, in case you later add D_box to diagnostics.
	optional_columns = [
		("box_divergence_1_s", "box divergence"),
		("box_divergence_filtered_1_s", "box divergence filtered"),
		("flow_divergence_raw_1_s", "raw flow divergence"),
	]

	for column_name, label in optional_columns:
		if column_name in df.columns:
			divergence_columns.append((column_name, label))

	if not divergence_columns:
		print("Skipping divergence plot. Missing flow_divergence_1_s.")
		return

	plt.figure(figsize=(10, 5))
	plt.title("Divergence evolution")

	for column_name, label in divergence_columns:
		y = numeric_column(df, column_name)
		plt.plot(t, y, label=label)

	plt.axhline(0.0, linestyle="--", linewidth=1)
	plt.xlabel("time [s]")
	plt.ylabel("divergence [1/s]")
	plt.grid(True)
	plt.legend()

	# plt.show()
	save_current_figure(output_dir, "divergence.png")


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
		ax.axhline(0.0, linestyle="--", linewidth=1)
		ax.set_ylabel(label)
		ax.grid(True)
		ax.legend(loc="best")

	axes[-1].set_xlabel("time [s]")

	# plt.show()
	save_current_figure(output_dir, "commands.png")


def main():
	parser = argparse.ArgumentParser(
		description="Analyse bee landing diagnostics CSV."
	)

	parser.add_argument(
		"csv_path",
		help="Path to diagnostics CSV file.",
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

	parser.add_argument(
		"--output-dir",
		default="analysis_output",
		help="Directory where plots will be saved. Default: analysis_output.",
	)

	parser.add_argument(
		"--max-boxes",
		type=int,
		default=150,
		help="Maximum number of detection boxes drawn in field-of-view reconstruction.",
	)

	args = parser.parse_args()

	HOME_PATH = Path("results")
	ensure_output_dir(HOME_PATH / args.output_dir)

	df = read_log(args.csv_path)
	t = get_time(df)

	print(f"Loaded: {args.csv_path}")
	print(f"Rows: {len(df)}")
	print(f"Time span: {np.nanmin(t):.3f} s to {np.nanmax(t):.3f} s")
	print(f"Output directory: {args.output_dir}")

	plot_detection_boxes_fov(
		df=df,
		t=t,
		image_width=args.image_width,
		image_height=args.image_height,
		output_dir=args.output_dir,
		max_boxes=args.max_boxes,
	)

	plot_target_position_offsets(
		df=df,
		t=t,
		output_dir=args.output_dir,
	)

	plot_vehicle_xyz(
		df=df,
		t=t,
		output_dir=args.output_dir,
	)

	plot_divergence(
		df=df,
		t=t,
		output_dir=args.output_dir,
	)

	plot_commands(
		df=df,
		t=t,
		output_dir=args.output_dir,
	)

	print("Done.")


if __name__ == "__main__":
	main()