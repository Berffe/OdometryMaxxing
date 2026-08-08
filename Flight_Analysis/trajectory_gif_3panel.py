
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation

LANDING_GEAR = 0.182
DRONE_RADIUS = 0.2

PHASE_LABELS = {
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
		q = data[cols].to_numpy(float)
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


def load_controller_phases(controller_csv, truth_t0_abs, truth_t_end_abs):
	if controller_csv is None:
		return None

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

	phase_df = pd.DataFrame({"t_abs": sim_time, "phase": label})
	phase_df = phase_df[np.isfinite(phase_df["t_abs"])].copy()
	phase_df = phase_df[(phase_df["t_abs"] >= truth_t0_abs) & (phase_df["t_abs"] <= truth_t_end_abs)].copy()
	phase_df = phase_df[phase_df["phase"] != ""].copy()

	if phase_df.empty:
		return None

	phase_df = phase_df.sort_values("t_abs").drop_duplicates("t_abs", keep="last").reset_index(drop=True)
	phase_df["t"] = phase_df["t_abs"] - truth_t0_abs
	return phase_df


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


def phase_at_time(phase_df, t_rel):
	if phase_df is None or phase_df.empty:
		return ""
	times = phase_df["t"].to_numpy(float)
	idx = np.searchsorted(times, t_rel, side="right") - 1
	if idx < 0:
		return ""
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
	elev=24.0,
	azim=-56.0,
	dpi=110,
):
	if not animation.writers.is_available("ffmpeg"):
		raise RuntimeError("FFmpeg is not available in this Python environment.")

	data, truth_t0_abs = load_truth_data(truth_csv, stop_at_contact=stop_at_contact)
	truth_t_end_abs = float(data["t_abs"].iloc[-1])
	phase_df = load_controller_phases(controller_csv, truth_t0_abs, truth_t_end_abs)

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
			phase_text = phase_at_time(phase_df, t_rel)

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
		"truth_t0_abs": truth_t0_abs,
		"output_mp4": output_mp4,
	}


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--truth-csv", type=str, default="/mnt/data/bee_truth_20260720_135806.csv")
	parser.add_argument("--controller-csv", type=str, default="/mnt/data/bee_controller_20260720_135806.csv")
	parser.add_argument("--output-mp4", type=str, default="/mnt/data/drone_platform_trajectory.mp4")
	parser.add_argument("--platform-radius", type=float, default=0.5)
	parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier: 0.25, 0.5, 1.5, etc.")
	parser.add_argument("--trail-seconds", type=float, default=6.0)
	parser.add_argument("--video-fps", type=float, default=15.0, help="Intended output MP4 fps.")
	parser.add_argument("--no-stop-at-contact", action="store_true")
	parser.add_argument("--elev", type=float, default=24.0)
	parser.add_argument("--azim", type=float, default=-56.0)
	parser.add_argument("--dpi", type=int, default=110)
	args = parser.parse_args()

	result = create_mp4(
		truth_csv=args.truth_csv,
		output_mp4=args.output_mp4,
		controller_csv=args.controller_csv,
		platform_radius=args.platform_radius,
		speed=args.speed,
		trail_seconds=args.trail_seconds,
		video_fps=args.video_fps,
		stop_at_contact=not args.no_stop_at_contact,
		elev=args.elev,
		azim=args.azim,
		dpi=args.dpi,
	)
	print(result)
