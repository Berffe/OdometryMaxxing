"""BEE_LAND 3-panel trajectory video.

Renders the drone/platform trajectory from a run's truth log, with the mission
substate from the controller log annotated on every frame.

The command line mirrors ``analyse_log.py``::

    python3 trajectory_video.py RUN_DIR OUTPUT_DIR
    python3 trajectory_video.py CSV_A CSV_B OUTPUT_DIR

In folder mode the two CSVs are discovered by their ``bee_controller_*`` /
``bee_truth_*`` prefixes and matching run suffix. In explicit mode the two CSVs
are order-independent: each is classified by its columns. ``bee_wind_*`` is not
used here and is ignored if present.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation

DEFAULT_VIDEO_NAME = "trajectory.mp4"


def _candidate_ffmpeg_paths():
	"""Places an ffmpeg binary is plausibly installed, best first.

	matplotlib only looks for ``ffmpeg`` on the PATH. On Windows, winget and
	Chocolatey installs are often not on the PATH of an already-open shell, so
	the usual locations are probed before giving up.
	"""
	for var in ("BEE_FFMPEG", "FFMPEG_PATH", "FFMPEG_BINARY"):
		value = os.environ.get(var)
		if value:
			yield Path(value)

	found = shutil.which("ffmpeg")
	if found:
		yield Path(found)

	local = os.environ.get("LOCALAPPDATA")
	if local:
		yield Path(local) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
		packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
		if packages.is_dir():
			yield from sorted(packages.glob("*FFmpeg*/**/bin/ffmpeg.exe"))

	for fixed in (
		r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
		r"C:\ffmpeg\bin\ffmpeg.exe",
		r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
		"/usr/bin/ffmpeg",
		"/usr/local/bin/ffmpeg",
		"/opt/homebrew/bin/ffmpeg",
	):
		yield Path(fixed)


def ensure_ffmpeg():
	"""Make matplotlib's FFmpeg writer usable, or explain how to install it."""
	if animation.writers.is_available("ffmpeg"):
		return

	for candidate in _candidate_ffmpeg_paths():
		if candidate.is_file():
			matplotlib.rcParams["animation.ffmpeg_path"] = str(candidate)
			if animation.writers.is_available("ffmpeg"):
				print(f"ffmpeg     : {candidate}")
				return

	raise RuntimeError(
		"FFmpeg was not found. The MP4 writer is FFmpeg, and it is a separate "
		"program, not a Python package -- pip install will not provide it.\n"
		"  Windows : winget install Gyan.FFmpeg    (then open a NEW terminal)\n"
		"  Conda   : conda install -c conda-forge ffmpeg\n"
		"  macOS   : brew install ffmpeg\n"
		"  Debian  : sudo apt install ffmpeg\n"
		"Check with 'ffmpeg -version'. If it is installed but not on the PATH, "
		"point this script at it with the BEE_FFMPEG environment variable, e.g. "
		'$env:BEE_FFMPEG = "C:\\ffmpeg\\bin\\ffmpeg.exe"'
	)


LANDING_GEAR = 0.182
DRONE_RADIUS = 0.2

PHASE_LABELS = {
	"prepare": "PREPARE",
	"center": "CENTER",
	"approach_probe": "APPROACH_PROBE",
	"final_probe": "FINAL_PROBE",
	"descend": "DESCEND",
	"infeasible": "INFEASIBLE",
	"probe_hold": "PROBE_HOLD",
	"landed": "LANDED",
	"aborted": "ABORTED",
}


def num(df, name):
	if name not in df.columns:
		return pd.Series(np.nan, index=df.index, dtype=float)
	return pd.to_numeric(df[name], errors="coerce")


def classify_csv(path):
	"""Classify a CSV as controller, truth or wind from its columns alone."""
	cols = set(pd.read_csv(path, nrows=2, low_memory=False).columns)
	if "truth_sim_time_sec" in cols and "truth_drone_position_z_m" in cols:
		return "truth"
	if "wind_sim_time_sec" in cols and "wind_command_x_enu_m_s" in cols:
		return "wind"
	if "flow_sim_timestamp_sec" in cols and "controller_phase" in cols:
		return "controller"
	return "unknown"


def discover_run_pair(run_dir):
	"""Find exactly one complete controller/truth pair inside ``run_dir``.

	The diagnostics files of a run share the same suffix, e.g.::

	    bee_controller_20260817_145438.csv
	    bee_truth_20260817_145438.csv

	Folder mode refuses to guess when more than one complete run is present:
	put one run per folder, or pass the two CSVs explicitly. ``bee_wind_*`` is
	not needed by this script and is ignored.
	"""
	run_dir = Path(run_dir).expanduser()
	if not run_dir.is_dir():
		raise ValueError(f"Run path is not a directory: {run_dir}")

	prefixes = {"controller": "bee_controller_", "truth": "bee_truth_"}
	runs = {}
	for kind, prefix in prefixes.items():
		for path in run_dir.glob(f"{prefix}*.csv"):
			suffix = path.name[len(prefix):]
			runs.setdefault(suffix, {})[kind] = path

	complete = [
		(suffix, files)
		for suffix, files in runs.items()
		if all(kind in files for kind in prefixes)
	]

	if not complete:
		found = {
			kind: len(list(run_dir.glob(f"{prefix}*.csv")))
			for kind, prefix in prefixes.items()
		}
		raise ValueError(
			f"No complete controller/truth pair found in {run_dir}. "
			f"Found controller={found['controller']}, truth={found['truth']} CSV(s)."
		)

	if len(complete) > 1:
		run_ids = ", ".join(sorted(s.removesuffix(".csv") for s, _ in complete))
		raise ValueError(
			f"More than one complete run found in {run_dir}: {run_ids}. "
			"Use one run per folder, or pass the two CSV files explicitly."
		)

	_, files = complete[0]
	return files["controller"], files["truth"]


def resolve_input_paths(paths):
	"""Resolve folder mode or explicit-two-CSV mode.

	Returns ``(controller_csv, truth_csv, output_mp4)``. The output positional
	is a directory unless it ends in ``.mp4``; a directory gets
	``trajectory.mp4`` written inside it, like ``analyse_log.py`` writes its
	plots into the output folder.
	"""
	if len(paths) == 2:
		run_dir, output = paths
		controller_csv, truth_csv = discover_run_pair(run_dir)
	elif len(paths) == 3:
		csv_a, csv_b, output = paths
		by_kind = {}
		for path in (csv_a, csv_b):
			if not Path(path).is_file():
				raise ValueError(f"CSV path does not exist or is not a file: {path}")
			kind = classify_csv(path)
			if kind == "unknown":
				raise ValueError(f"Could not classify {Path(path).name} as controller or truth.")
			if kind in by_kind:
				raise ValueError(f"Received more than one {kind} CSV.")
			by_kind[kind] = Path(path)
		missing = [k for k in ("controller", "truth") if k not in by_kind]
		if missing:
			raise ValueError(
				f"Missing required CSV stream(s): {', '.join(missing)}. "
				f"Got: {', '.join(sorted(by_kind))}."
			)
		controller_csv, truth_csv = by_kind["controller"], by_kind["truth"]
	else:
		raise ValueError("Expected either RUN_DIR OUTPUT, or CSV_A CSV_B OUTPUT.")

	output = Path(output).expanduser()
	output_mp4 = output if output.suffix.lower() == ".mp4" else output / DEFAULT_VIDEO_NAME
	output_mp4.parent.mkdir(parents=True, exist_ok=True)
	return Path(controller_csv), Path(truth_csv), output_mp4


def clean_string(series):
	return series.fillna("").astype(str).str.strip().str.lower()


def load_truth_data(truth_csv, stop_at_contact=False):
	df = pd.read_csv(truth_csv, low_memory=False)

	t = num(df, "truth_sim_time_sec")
	x_d = num(df, "truth_drone_position_x_m")
	y_d = num(df, "truth_drone_position_y_m")
	z_d = num(df, "truth_drone_position_z_m")
	# Use the deck-plane reference point as the animated platform centre.
	# This remains correct when the platform tilts, unlike mixing the platform
	# origin in x/y with the deck-top coordinate only in z.
	platform_x = num(df, "truth_platform_position_x_m")
	platform_y = num(df, "truth_platform_position_y_m")
	platform_z = num(df, "truth_platform_position_z_m")
	deck_x = num(df, "truth_deck_point_x_m")
	deck_y = num(df, "truth_deck_point_y_m")
	deck_z = num(df, "truth_deck_point_z_m")

	x_p = deck_x.where(np.isfinite(deck_x), platform_x)
	y_p = deck_y.where(np.isfinite(deck_y), platform_y)
	z_p = deck_z.where(np.isfinite(deck_z), platform_z)

	# Gazebo drone and platform attitude quaternions (body/platform frame -> world frame).
	qx_d = num(df, "truth_drone_orientation_x")
	qy_d = num(df, "truth_drone_orientation_y")
	qz_d = num(df, "truth_drone_orientation_z")
	qw_d = num(df, "truth_drone_orientation_w")
	qx_p = num(df, "truth_platform_orientation_x")
	qy_p = num(df, "truth_platform_orientation_y")
	qz_p = num(df, "truth_platform_orientation_z")
	qw_p = num(df, "truth_platform_orientation_w")

	stop_idx = len(df) - 1
	if stop_at_contact and "truth_any_contact" in df.columns:
		contact = pd.to_numeric(df["truth_any_contact"], errors="coerce").fillna(0.0) > 0.5
		idx = np.flatnonzero(contact.to_numpy())
		if idx.size:
			stop_idx = int(idx[0])

	mask = (
		np.isfinite(t) & np.isfinite(x_d) & np.isfinite(y_d) & np.isfinite(z_d) &
		np.isfinite(x_p) & np.isfinite(y_p) & np.isfinite(z_p)
	) & (np.arange(len(df)) <= stop_idx)

	data = pd.DataFrame({
		"t_abs": t[mask].to_numpy(float),
		"x_d": x_d[mask].to_numpy(float),
		"y_d": y_d[mask].to_numpy(float),
		"z_d": z_d[mask].to_numpy(float),
		"x_p": x_p[mask].to_numpy(float),
		"y_p": y_p[mask].to_numpy(float),
		"z_p": z_p[mask].to_numpy(float),
		"qx_d": qx_d[mask].to_numpy(float),
		"qy_d": qy_d[mask].to_numpy(float),
		"qz_d": qz_d[mask].to_numpy(float),
		"qw_d": qw_d[mask].to_numpy(float),
		"qx_p": qx_p[mask].to_numpy(float),
		"qy_p": qy_p[mask].to_numpy(float),
		"qz_p": qz_p[mask].to_numpy(float),
		"qw_p": qw_p[mask].to_numpy(float),
	}).reset_index(drop=True)

	# Preserve compatibility with older truth logs: if attitude is unavailable
	# or malformed for a sample, render that sample as level instead of dropping
	# the entire frame from the animation.
	for prefix in ("d", "p"):
		cols = [f"qx_{prefix}", f"qy_{prefix}", f"qz_{prefix}", f"qw_{prefix}"]
		q = np.array(data[cols].to_numpy(float), copy=True)
		q_norm = np.linalg.norm(q, axis=1)
		q_good = np.all(np.isfinite(q), axis=1) & (q_norm > 1e-12)
		q[~q_good] = np.array([0.0, 0.0, 0.0, 1.0])
		q_norm = np.linalg.norm(q, axis=1)
		q = q / q_norm[:, None]
		data[cols] = q

	if len(data) < 2:
		raise ValueError("Not enough valid truth samples.")

	t0 = float(data["t_abs"].iloc[0])
	data["t"] = data["t_abs"] - t0
	return data, t0


def truthy(df, name):
	"""Boolean column tolerant of 1/0, True/False and "true"/"false" logs."""
	if name not in df.columns:
		return pd.Series(False, index=df.index)
	numeric = pd.to_numeric(df[name], errors="coerce")
	text = clean_string(df[name])
	return (numeric.fillna(0.0) > 0.5) | text.isin(["true", "yes"])


def controller_start_time(df, sim_time):
	"""SIM time at which the controller actually takes over the flight.

	Every run begins with the MAVSDK worker lifting the drone to the starting
	altitude. During that leg the mission FSM already reports its initial
	substate, so labelling frames from ``mission_substate`` alone paints the
	whole takeoff as CENTER. The real handover is the first *control* row: a
	row produced by a fresh, valid visual result rather than by an event.

	This is the same rule ``analyse_log.py`` uses for ``t0``, which is why its
	plots start at the handover and ignore the preparation leg entirely.
	Returns ``None`` when no control row can be identified, in which case the
	caller keeps the old behaviour of labelling from the first controller row.
	"""
	event = clean_string(df["event"]) if "event" in df.columns else pd.Series("", index=df.index)
	valid = np.isfinite(sim_time) & truthy(df, "flow_valid")

	mask = valid & (event == "")
	if "vision_sequence" in df.columns:
		mask &= pd.to_numeric(df["vision_sequence"], errors="coerce").notna()

	if not mask.any():
		# Compatibility fallback for logs whose control rows carry an explicit
		# "control" event, and for logs without flow_valid at all.
		mask = valid & event.isin(["", "control"])
	if not mask.any():
		mask = np.isfinite(sim_time) & (event == "")
	if not mask.any():
		return None

	return float(sim_time[mask].min())


def load_controller_phases(controller_csv, truth_t0_abs, truth_t_end_abs):
	"""Return (phase_df, handover_abs).

	``phase_df`` holds the mission labels from the handover onwards.
	``handover_abs`` is the absolute SIM time at which the controller took
	over, or ``None`` if it could not be determined.
	"""
	if controller_csv is None:
		return None, None

	df = pd.read_csv(controller_csv, low_memory=False)

	sim_time = num(df, "flow_sim_timestamp_sec")
	if "command_source_sim_timestamp_sec" in df.columns:
		missing = ~np.isfinite(sim_time)
		sim_time.loc[missing] = num(df.loc[missing], "command_source_sim_timestamp_sec")
	if "contact_truth_sim_timestamp_sec" in df.columns:
		missing = ~np.isfinite(sim_time)
		sim_time.loc[missing] = num(df.loc[missing], "contact_truth_sim_timestamp_sec")

	substate = clean_string(df["mission_substate"]) if "mission_substate" in df.columns else pd.Series("", index=df.index)
	phase = clean_string(df["controller_phase"]) if "controller_phase" in df.columns else pd.Series("", index=df.index)
	event = clean_string(df["event"]) if "event" in df.columns else pd.Series("", index=df.index)

	label = substate.copy()
	empty = label == ""
	label.loc[empty] = phase.loc[empty]
	empty = label == ""
	label.loc[empty] = event.loc[empty]

	handover_abs = controller_start_time(df, sim_time)

	phase_df = pd.DataFrame({"t_abs": sim_time, "phase": label})
	phase_df = phase_df[np.isfinite(phase_df["t_abs"])].copy()
	phase_df = phase_df[(phase_df["t_abs"] >= truth_t0_abs) & (phase_df["t_abs"] <= truth_t_end_abs)].copy()
	phase_df = phase_df[phase_df["phase"] != ""].copy()

	# Mission labels logged before the handover describe an FSM that is not
	# flying the drone yet: drop them so the takeoff leg reads PREPARE.
	if handover_abs is not None:
		phase_df = phase_df[phase_df["t_abs"] >= handover_abs].copy()

	if phase_df.empty:
		return None, handover_abs

	phase_df = phase_df.sort_values("t_abs").drop_duplicates("t_abs", keep="last").reset_index(drop=True)
	phase_df["t"] = phase_df["t_abs"] - truth_t0_abs
	return phase_df, handover_abs


def choose_animation_sampling(data, speed=1.0, video_fps=15.0):
	t = data["t"].to_numpy(float)
	dt = np.diff(t)
	dt = dt[np.isfinite(dt) & (dt > 1e-9)]
	if dt.size == 0:
		raise ValueError("Could not estimate the source sample period.")
	source_dt = float(np.median(dt))
	source_fps = 1.0 / source_dt

	sim_dt_per_video_frame = max(source_dt, float(speed) / float(video_fps))
	step = max(1, int(round(sim_dt_per_video_frame / source_dt)))
	anim = data.iloc[::step].reset_index(drop=True)

	return anim, source_fps, source_dt, sim_dt_per_video_frame, step


def phase_at_time(phase_df, t_rel, prepare_until=None):
	"""Mission label at ``t_rel``, or PREPARE before the controller took over.

	``prepare_until`` is the handover time expressed in plot-relative seconds.
	Frames before it belong to the MAVSDK takeoff leg, which has no mission
	substate of its own.
	"""
	if prepare_until is not None and t_rel < prepare_until:
		return PHASE_LABELS["prepare"]
	if phase_df is None or phase_df.empty:
		return PHASE_LABELS["prepare"] if prepare_until is not None else ""
	times = phase_df["t"].to_numpy(float)
	idx = np.searchsorted(times, t_rel, side="right") - 1
	if idx < 0:
		return PHASE_LABELS["prepare"]
	phase = str(phase_df["phase"].iloc[idx]).strip().lower()
	return PHASE_LABELS.get(phase, phase.upper())


def quaternion_rotation_matrix(qx, qy, qz, qw):
	"""Return the 3x3 active rotation matrix for a normalized quaternion."""
	q = np.asarray([qx, qy, qz, qw], dtype=float)
	norm = float(np.linalg.norm(q))
	if not np.isfinite(norm) or norm <= 1e-12:
		return np.eye(3)
	x, y, z, w = q / norm
	return np.array([
		[1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),       2.0 * (x*z + y*w)],
		[2.0 * (x*y + z*w),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
		[2.0 * (x*z - y*w),       2.0 * (y*z + x*w),       1.0 - 2.0 * (x*x + y*y)],
	])


def platform_geometry(row, platform_radius, theta):
	"""Return the tilted deck outline and its local x/y axes in world coordinates.

	The platform is represented as a circular disk in its local x-y plane.
	Its full Gazebo quaternion rotates that disk into world coordinates. Because
	a circle alone cannot show yaw, two local in-plane radius vectors are also
	returned and drawn in every view.
	"""
	center = np.array([row["x_p"], row["y_p"], row["z_p"]], dtype=float)
	R = quaternion_rotation_matrix(
		row["qx_p"], row["qy_p"], row["qz_p"], row["qw_p"]
	)

	local_circle = np.vstack((
		platform_radius * np.cos(theta),
		platform_radius * np.sin(theta),
		np.zeros_like(theta),
	))
	circle = center[:, None] + R @ local_circle

	local_axes = np.array([
		[platform_radius, 0.0, 0.0],
		[0.0, platform_radius, 0.0],
	]).T
	axis_tips = center[:, None] + R @ local_axes
	return circle, center, axis_tips


def drone_geometry(row, drone_radius, theta):
	"""Return the drone body disk and landing-gear tip in world coordinates.

	The drone is rendered as a circular body disk lying in its local x-y plane,
	rotated by the logged drone quaternion. A single line segment extends along
	the local negative z-axis to indicate the landing-gear length.
	"""
	center = np.array([row["x_d"], row["y_d"], row["z_d"]], dtype=float)
	R = quaternion_rotation_matrix(
		row["qx_d"], row["qy_d"], row["qz_d"], row["qw_d"]
	)

	local_circle = np.vstack((
		drone_radius * np.cos(theta),
		drone_radius * np.sin(theta),
		np.zeros_like(theta),
	))
	circle = center[:, None] + R @ local_circle
	gear_tip = center + R @ np.array([0.0, 0.0, -LANDING_GEAR], dtype=float)
	return circle, center, gear_tip


def draw_fading_2d(ax, x, y, trail_frames, color):
	n = len(x)
	if n < 2:
		return
	start = max(0, n - 1 - trail_frames)
	seg_count = n - 1 - start
	for j, i in enumerate(range(start, n - 1)):
		alpha = (j + 1) / max(1, seg_count)
		ax.plot(x[i:i+2], y[i:i+2], alpha=alpha, linewidth=2.0, color=color)


def draw_fading_3d(ax, x, y, z, trail_frames, color):
	n = len(x)
	if n < 2:
		return
	start = max(0, n - 1 - trail_frames)
	seg_count = n - 1 - start
	for j, i in enumerate(range(start, n - 1)):
		alpha = (j + 1) / max(1, seg_count)
		ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], alpha=alpha, linewidth=2.0, color=color)


def create_mp4(
	truth_csv,
	output_mp4,
	controller_csv=None,
	platform_radius=1.0,
	speed=1.0,
	trail_seconds=6.0,
	video_fps=15.0,
	stop_at_contact=False,
	max_time=None,
	elev=24.0,
	azim=-56.0,
	dpi=110,
):
	ensure_ffmpeg()

	data, truth_t0_abs = load_truth_data(truth_csv, stop_at_contact=stop_at_contact)

	# Same convention as analyse_log.py --max-time: simulated seconds counted
	# from the first plotted sample, not absolute SIM time.
	if max_time is not None:
		data = data[data["t"] <= float(max_time)].reset_index(drop=True)
		if len(data) < 2:
			raise ValueError(
				f"--max-time {max_time} leaves fewer than two truth samples."
			)

	truth_t_end_abs = float(data["t_abs"].iloc[-1])
	phase_df, handover_abs = load_controller_phases(
		controller_csv, truth_t0_abs, truth_t_end_abs
	)

	# Everything before the handover is the MAVSDK worker lifting the drone to
	# the starting altitude. A handover at or before the first truth sample
	# means the run has no preparation leg to label.
	prepare_until = None
	if handover_abs is not None:
		relative = handover_abs - truth_t0_abs
		if relative > 0.0:
			prepare_until = relative

	anim, source_fps, source_dt, sim_dt_per_video_frame, step = choose_animation_sampling(
		data, speed=speed, video_fps=video_fps
	)

	trail_frames = max(1, int(round(trail_seconds / sim_dt_per_video_frame)))

	# Conservative limits that include every possible orientation of both disks
	# and the landing-gear segment.
	drone_extent = DRONE_RADIUS + LANDING_GEAR
	all_x = np.concatenate([
		anim["x_d"] - drone_extent,
		anim["x_d"] + drone_extent,
		anim["x_p"] - platform_radius,
		anim["x_p"] + platform_radius,
	])
	all_y = np.concatenate([
		anim["y_d"] - drone_extent,
		anim["y_d"] + drone_extent,
		anim["y_p"] - platform_radius,
		anim["y_p"] + platform_radius,
	])
	all_z = np.concatenate([
		anim["z_d"] - drone_extent,
		anim["z_d"] + DRONE_RADIUS,
		anim["z_p"] - platform_radius,
		anim["z_p"] + platform_radius,
	])

	x_mid = (float(np.nanmin(all_x)) + float(np.nanmax(all_x))) / 2.0
	y_mid = (float(np.nanmin(all_y)) + float(np.nanmax(all_y))) / 2.0
	z_mid = (float(np.nanmin(all_z)) + float(np.nanmax(all_z))) / 2.0
	half_range = max(
		(float(np.nanmax(all_x)) - float(np.nanmin(all_x))) / 2.0,
		(float(np.nanmax(all_y)) - float(np.nanmin(all_y))) / 2.0,
		(float(np.nanmax(all_z)) - float(np.nanmin(all_z))) / 2.0,
	) + 0.5

	theta = np.linspace(0, 2 * np.pi, 80)

	fig = plt.figure(figsize=(15, 5.2))
	ax3d = fig.add_subplot(1, 3, 1, projection="3d")
	ax_side = fig.add_subplot(1, 3, 2)
	ax_top = fig.add_subplot(1, 3, 3)
	fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.84, wspace=0.28)

	writer = FFMpegWriter(fps=video_fps, metadata={"title": "Drone trajectory animation"}, bitrate=1800)

	with writer.saving(fig, output_mp4, dpi=dpi):
		for k in range(len(anim)):
			ax3d.cla()
			ax_side.cla()
			ax_top.cla()

			row = anim.iloc[k]
			t_rel = float(row["t"])
			phase_text = phase_at_time(phase_df, t_rel, prepare_until)

			xd = anim["x_d"].iloc[:k+1].to_numpy()
			yd = anim["y_d"].iloc[:k+1].to_numpy()
			zd = anim["z_d"].iloc[:k+1].to_numpy()
			xp = anim["x_p"].iloc[:k+1].to_numpy()
			yp = anim["y_p"].iloc[:k+1].to_numpy()
			zp = anim["z_p"].iloc[:k+1].to_numpy()

			drone_circle_xyz, drone_center, gear_tip = drone_geometry(
				row, DRONE_RADIUS, theta
			)
			circle_xyz, platform_center, axis_tips = platform_geometry(
				row, platform_radius, theta
			)

			draw_fading_3d(ax3d, xd, yd, zd, trail_frames, 'b')
			draw_fading_3d(ax3d, xp, yp, zp, trail_frames, 'r')
			ax3d.plot(drone_circle_xyz[0], drone_circle_xyz[1], drone_circle_xyz[2], linewidth=2.0, color='b')
			ax3d.plot(
				[drone_center[0], gear_tip[0]],
				[drone_center[1], gear_tip[1]],
				[drone_center[2], gear_tip[2]],
				linewidth=1.8, color='b',
			)
			ax3d.plot(circle_xyz[0], circle_xyz[1], circle_xyz[2], linewidth=2.0, color='k')
			# Local deck x/y axes make yaw observable as well as roll/pitch tilt.
			ax3d.plot(
				[platform_center[0], axis_tips[0, 0]],
				[platform_center[1], axis_tips[1, 0]],
				[platform_center[2], axis_tips[2, 0]],
				linewidth=1.5, linestyle='--', color='k',
			)
			ax3d.plot(
				[platform_center[0], axis_tips[0, 1]],
				[platform_center[1], axis_tips[1, 1]],
				[platform_center[2], axis_tips[2, 1]],
				linewidth=1.5, linestyle=':', color='k',
			)
			ax3d.scatter([drone_center[0]], [drone_center[1]], [drone_center[2]], s=30)
			ax3d.scatter([gear_tip[0]], [gear_tip[1]], [gear_tip[2]], s=18)
			ax3d.scatter([row["x_p"]], [row["y_p"]], [row["z_p"]], s=25)
			ax3d.set_xlim(x_mid - half_range, x_mid + half_range)
			ax3d.set_ylim(y_mid - half_range, y_mid + half_range)
			ax3d.set_zlim(max(0.0, z_mid - half_range), z_mid + half_range)
			ax3d.set_xlabel("x [m]")
			ax3d.set_ylabel("y [m]")
			ax3d.set_zlabel("z [m]")
			ax3d.set_title("Vue 3D")
			ax3d.view_init(elev=elev, azim=azim)
			ax3d.grid(True)

			draw_fading_2d(ax_side, xd, zd, trail_frames, 'b')
			draw_fading_2d(ax_side, xp, zp, trail_frames, 'r')
			ax_side.plot(drone_circle_xyz[0], drone_circle_xyz[2], linewidth=2.0, color='b')
			ax_side.plot(
				[drone_center[0], gear_tip[0]],
				[drone_center[2], gear_tip[2]],
				linewidth=1.8, color='b',
			)
			# Projection of the same tilted 3D disk onto the x-z side view.
			ax_side.plot(circle_xyz[0], circle_xyz[2], linewidth=3.0, color='k')
			ax_side.plot(
				[platform_center[0], axis_tips[0, 0]],
				[platform_center[2], axis_tips[2, 0]],
				linewidth=1.3, linestyle='--', color='k',
			)
			ax_side.plot(
				[platform_center[0], axis_tips[0, 1]],
				[platform_center[2], axis_tips[2, 1]],
				linewidth=1.3, linestyle=':', color='k',
			)
			ax_side.scatter([drone_center[0]], [drone_center[2]], s=30)
			ax_side.scatter([gear_tip[0]], [gear_tip[2]], s=18)
			ax_side.scatter([row["x_p"]], [row["z_p"]], s=25)
			ax_side.set_xlim(x_mid - half_range, x_mid + half_range)
			ax_side.set_ylim(max(0.0, z_mid - half_range), z_mid + half_range)
			ax_side.set_xlabel("x [m]")
			ax_side.set_ylabel("z [m]")
			ax_side.set_title("Vue de côté (x-z)")
			ax_side.grid(True)

			draw_fading_2d(ax_top, xd, yd, trail_frames, 'b')
			draw_fading_2d(ax_top, xp, yp, trail_frames, 'r')
			ax_top.plot(drone_circle_xyz[0], drone_circle_xyz[1], linewidth=2.0, color='b')
			ax_top.plot(
				[drone_center[0], gear_tip[0]],
				[drone_center[1], gear_tip[1]],
				linewidth=1.8, color='b',
			)
			# Projection of the tilted disk onto x-y. It becomes an ellipse when
			# roll/pitch are non-zero instead of remaining an artificial circle.
			ax_top.plot(circle_xyz[0], circle_xyz[1], linewidth=2.0, color='k')
			ax_top.plot(
				[platform_center[0], axis_tips[0, 0]],
				[platform_center[1], axis_tips[1, 0]],
				linewidth=1.3, linestyle='--', color='k',
			)
			ax_top.plot(
				[platform_center[0], axis_tips[0, 1]],
				[platform_center[1], axis_tips[1, 1]],
				linewidth=1.3, linestyle=':', color='k',
			)
			ax_top.scatter([drone_center[0]], [drone_center[1]], s=30)
			ax_top.scatter([gear_tip[0]], [gear_tip[1]], s=18)
			ax_top.scatter([row["x_p"]], [row["y_p"]], s=25)
			ax_top.set_xlim(x_mid - half_range, x_mid + half_range)
			ax_top.set_ylim(y_mid - half_range, y_mid + half_range)
			ax_top.set_aspect("equal", adjustable="box")
			ax_top.set_xlabel("x [m]")
			ax_top.set_ylabel("y [m]")
			ax_top.set_title("Vue de dessus (x-y)")
			ax_top.grid(True)

			title = f"Trajectoire du drone jusqu'au contact — t = {t_rel:.2f} s"
			if phase_text:
				title += f" | phase: {phase_text}"
			title += f" | vitesse = {speed:.2f}x | MP4 = {video_fps:.1f} fps"
			fig.suptitle(title, fontsize=13)

			writer.grab_frame()

	plt.close(fig)
	return {
		"frames": len(anim),
		"video_fps": video_fps,
		"source_fps": source_fps,
		"source_dt": source_dt,
		"step": step,
		"sim_dt_per_video_frame": sim_dt_per_video_frame,
		"trail_frames": trail_frames,
		"prepare_until": prepare_until,
		"truth_t0_abs": truth_t0_abs,
		"output_mp4": output_mp4,
	}


def parse_args(argv=None):
	parser = argparse.ArgumentParser(
		description=(
			"Render the BEE_LAND 3-panel trajectory video (3D, side, top) from a "
			"run's truth log, annotated with the controller mission substate."
		),
		epilog=(
			"Examples:\n"
			"  python3 trajectory_video.py logs/run results/run\n"
			"  python3 trajectory_video.py logs/run results/run --speed 0.5 --max-time 55\n"
			"  python3 trajectory_video.py bee_truth_RUN.csv bee_controller_RUN.csv results/run"
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument(
		"paths",
		type=Path,
		nargs="+",
		metavar="PATH",
		help=(
			"Either RUN_DIR OUTPUT, or CSV_A CSV_B OUTPUT (order-independent). "
			"OUTPUT is a directory, in which case trajectory.mp4 is written "
			"inside it, or an explicit *.mp4 file."
		),
	)
	parser.add_argument(
		"--platform-radius",
		type=float,
		default=0.5,
		metavar="METRES",
		help="Deck radius drawn in the three views (default: 0.5).",
	)
	parser.add_argument(
		"--speed",
		type=float,
		default=1.0,
		metavar="X",
		help="Playback speed multiplier in SIM time: 0.25, 0.5, 1.5 (default: 1.0).",
	)
	parser.add_argument(
		"--trail-seconds",
		type=float,
		default=6.0,
		metavar="SECONDS",
		help="Length of the fading motion trail (default: 6.0).",
	)
	parser.add_argument(
		"--video-fps",
		type=float,
		default=15.0,
		metavar="FPS",
		help="Output MP4 frame rate (default: 15.0).",
	)
	parser.add_argument(
		"--max-time",
		type=float,
		default=None,
		metavar="SECONDS",
		help=(
			"Stop the video this many simulated seconds after the first truth "
			"sample. Same convention as analyse_log.py --max-time."
		),
	)
	parser.add_argument(
		"--no-stop-at-contact",
		action="store_true",
		help="Keep rendering past the first contact instead of stopping there.",
	)
	parser.add_argument(
		"--elev",
		type=float,
		default=24.0,
		metavar="DEG",
		help="3-D view elevation (default: 24).",
	)
	parser.add_argument(
		"--azim",
		type=float,
		default=-56.0,
		metavar="DEG",
		help="3-D view azimuth (default: -56).",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=110,
		metavar="DPI",
		help="Output resolution (default: 110).",
	)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	controller_csv, truth_csv, output_mp4 = resolve_input_paths(args.paths)

	print(f"controller : {controller_csv}")
	print(f"truth      : {truth_csv}")
	print(f"output     : {output_mp4}")

	result = create_mp4(
		truth_csv=str(truth_csv),
		output_mp4=str(output_mp4),
		controller_csv=str(controller_csv),
		platform_radius=args.platform_radius,
		speed=args.speed,
		trail_seconds=args.trail_seconds,
		video_fps=args.video_fps,
		stop_at_contact= args.no_stop_at_contact,
		max_time=args.max_time,
		elev=args.elev,
		azim=args.azim,
		dpi=args.dpi,
	)

	if result["prepare_until"] is not None:
		print(
			f"PREPARE leg: first {result['prepare_until']:.2f} s of the video "
			"(MAVSDK takeoff, before the controller takes over)"
		)
	else:
		print("PREPARE leg: none (the controller was already flying at the first truth sample)")

	print(
		f"saved {output_mp4} "
		f"({result['frames']} frames at {result['video_fps']:.1f} fps, "
		f"source {result['source_fps']:.1f} Hz, 1 frame per {result['step']} truth samples)"
	)
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except (ValueError, RuntimeError) as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		raise SystemExit(1)