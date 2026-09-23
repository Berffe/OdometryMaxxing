"""Build campaign-level analysis tables from an OdometryMaxxing campaign.

The campaign harness writes one directory per run containing:

    run.json
    bee_outcome_*.json
    bee_controller_*.csv
    bee_truth_*.csv
    bee_wind_*.csv

This script deliberately treats ``bee_outcome_*.json`` as authoritative for the
terminal outcome.  In particular, INFEASIBLE and ABORTED are never inferred from
trajectory data:

* infeasible -> the commit gate refused the landing opportunity;
* aborted    -> something went wrong with the controller/mission execution.

Touchdown quantities are extracted only from Gazebo truth.  Over the 0.10 s
immediately preceding first physical contact, the script computes the mean
drone-platform relative-velocity components in platform axes and the mean of
the instantaneous relative-speed magnitude.  The existing pad normal closing
rate is retained as a separate diagnostic.

Outputs
-------
campaign_runs.csv
    One row per run.  Contains scenario metadata, authoritative outcome,
    touchdown metrics, CENTER diagnostics and a few transition/gain diagnostics.

campaign_gate_events.csv
    One row per run at the final gate decision/snapshot.  The authoritative
    outcome fields are combined with the logged FINAL_PROBE gate telemetry.
    Gate-related columns are selected by name rather than hard-coded, making the
    table tolerant to telemetry additions.

campaign_issues.csv
    Missing/ambiguous files and parse problems.  An empty file means the scan was
    clean.

Example
-------
    python3 build_campaign_dataset.py ~/BEE_LAND/logs/campaign_20260922
    python3 build_campaign_dataset.py CAMPAIGN --output-dir CAMPAIGN/analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

CONTACT_RATE_AVERAGE_WINDOW_SEC = 0.10

OUTCOME_STATUSES = {
    "landed",
    "infeasible",
    "aborted",
    "crashed",
    "launch_failed",
    "timeout",
}

# Fields removed before hashing run.json into a gate-on/off pairing key.  The
# harness invariant is that the physical scenario is identical across a pair;
# only the gate arm and bookkeeping run id are allowed to differ.
PAIR_PRUNE_KEYS = {
    "gate",
    "gate_enabled",
    "commit_gate_enabled",
    "enable_commit_gate",
    "run_id",
}

GATE_KEYWORDS = (
    "feasible",
    "infeasible",
    "tracking",
    "mismatch",
    "chi",
    "rho",
    "k_min",
    "k_probe",
    "k_floor",
    "k_ceiling",
    "k_touchdown",
    "accel_capacity",
    "window_exists",
    "probe_within_ceiling",
    "peak_accel",
    "probe_peak",
    "near_field",
    "h_crit",
    "h_pred",
)


def _json_scalar(value: Any) -> Any:
    """Return a CSV-safe scalar while preserving structured values as JSON."""
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        if isinstance(value, np.generic):
            return value.item()
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_json(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        out[prefix] = _json_scalar(value)
    else:
        out[prefix] = value
    return out


def _prune_pair_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _prune_pair_fields(child)
            for key, child in value.items()
            if str(key).strip().lower() not in PAIR_PRUNE_KEYS
        }
    if isinstance(value, list):
        return [_prune_pair_fields(child) for child in value]
    return value


def _pair_key(spec: dict[str, Any]) -> str:
    canonical = json.dumps(
        _prune_pair_fields(spec), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _find_one(run_dir: Path, pattern: str) -> tuple[Path | None, str | None]:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return None, f"missing {pattern}"
    if len(matches) > 1:
        # Newest file is usually the complete one if a run restarted in place,
        # but keep an issue row so this never goes unnoticed.
        newest = max(matches, key=lambda p: p.stat().st_mtime)
        return newest, f"multiple {pattern}; used newest {newest.name}"
    return matches[0], None


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _clean_text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype=str)
    return df[column].fillna("").astype(str).str.strip().str.lower()


def _sim_time(df: pd.DataFrame) -> np.ndarray:
    """Construct the best available simulation-time vector row by row."""
    candidates = (
        "truth_sim_time_sec",
        "flow_sim_timestamp_sec",
        "command_source_sim_timestamp_sec",
        "contact_truth_sim_timestamp_sec",
        "wind_sim_time_sec",
        "sim_time_sec",
        "timestamp_sec",
    )
    t = np.full(len(df), np.nan, dtype=float)
    for column in candidates:
        if column not in df.columns:
            continue
        values = _num(df, column).to_numpy(float)
        fill = ~np.isfinite(t) & np.isfinite(values)
        t[fill] = values[fill]
    return t


def _first_true_time(df: pd.DataFrame, column: str, t: np.ndarray) -> float:
    if column not in df.columns:
        return math.nan
    values = _num(df, column).to_numpy(float)
    idx = np.flatnonzero(np.isfinite(t) & np.isfinite(values) & (values > 0.5))
    return float(t[idx[0]]) if idx.size else math.nan


def _precontact_mean(
    truth: pd.DataFrame,
    column: str,
    t: np.ndarray,
    contact_time: float,
    window_sec: float,
) -> float:
    if column not in truth.columns or not np.isfinite(contact_time):
        return math.nan
    values = _num(truth, column).to_numpy(float)
    mask = (
        np.isfinite(t)
        & np.isfinite(values)
        & (t >= contact_time - float(window_sec))
        & (t <= contact_time)
    )
    return float(np.mean(values[mask])) if np.any(mask) else math.nan


def _precontact_relative_velocity(
    truth: pd.DataFrame,
    t: np.ndarray,
    contact_time: float,
    window_sec: float,
) -> dict[str, Any]:
    """Average relative velocity over the final pre-contact window.

    The instantaneous vector is

        v_rel = v_drone - v_platform

    using the Gazebo truth linear velocities already present in the project.
    Each valid sample is rotated from world axes into the instantaneous platform
    axes when the platform quaternion is available.  The reported total speed is
    the mean of ``||v_rel(t)||`` over the window; this is intentionally different
    from taking the norm of the three mean components.  Both are retained so the
    averaging choice remains transparent.
    """
    result = {
        "contact_relative_velocity_x_mean_m_s": math.nan,
        "contact_relative_velocity_y_mean_m_s": math.nan,
        "contact_relative_velocity_z_mean_m_s": math.nan,
        "contact_relative_speed_mean_m_s": math.nan,
        "contact_relative_speed_from_mean_components_m_s": math.nan,
        "contact_velocity_samples": 0,
        "contact_velocity_frame": "unavailable",
    }
    required = [
        f"truth_{entity}_linear_velocity_{axis}_m_s"
        for entity in ("drone", "platform")
        for axis in "xyz"
    ]
    if not np.isfinite(contact_time) or not all(column in truth.columns for column in required):
        return result

    window = (
        np.isfinite(t)
        & (t >= contact_time - float(window_sec))
        & (t <= contact_time)
    )
    indices = np.flatnonzero(window)
    world_vectors: list[np.ndarray] = []
    rotations: list[np.ndarray | None] = []
    speeds: list[float] = []

    for index in indices:
        vd = np.asarray(
            [
                _num(truth.iloc[[index]], f"truth_drone_linear_velocity_{axis}_m_s").iloc[0]
                for axis in "xyz"
            ],
            dtype=float,
        )
        vp = np.asarray(
            [
                _num(truth.iloc[[index]], f"truth_platform_linear_velocity_{axis}_m_s").iloc[0]
                for axis in "xyz"
            ],
            dtype=float,
        )
        if not (np.all(np.isfinite(vd)) and np.all(np.isfinite(vp))):
            continue

        rel_world = vd - vp
        world_vectors.append(rel_world)
        rotations.append(_platform_rotation_world_from_body(truth, int(index)))
        speeds.append(float(np.linalg.norm(rel_world)))

    if not world_vectors:
        return result

    # Never average vectors expressed in mixed coordinate frames.  Use platform
    # axes only when the quaternion exists for every retained sample; otherwise
    # fall back to world-axis relative velocity for the whole 0.10 s window.
    used_platform_frame = all(rotation is not None for rotation in rotations)
    if used_platform_frame:
        components = [rotation.T @ vector for rotation, vector in zip(rotations, world_vectors)]
    else:
        components = world_vectors

    matrix = np.vstack(components)
    mean_components = np.mean(matrix, axis=0)
    result.update(
        {
            "contact_relative_velocity_x_mean_m_s": float(mean_components[0]),
            "contact_relative_velocity_y_mean_m_s": float(mean_components[1]),
            "contact_relative_velocity_z_mean_m_s": float(mean_components[2]),
            "contact_relative_speed_mean_m_s": float(np.mean(speeds)),
            "contact_relative_speed_from_mean_components_m_s": float(np.linalg.norm(mean_components)),
            "contact_velocity_samples": int(len(components)),
            "contact_velocity_frame": "platform_frame" if used_platform_frame else "world_delta_fallback",
        }
    )
    return result


def _nearest_index(t: np.ndarray, when: float) -> int | None:
    good = np.flatnonzero(np.isfinite(t))
    if not good.size or not np.isfinite(when):
        return None
    return int(good[np.argmin(np.abs(t[good] - when))])


def _xyz(df: pd.DataFrame, stem: str, index: int) -> np.ndarray | None:
    values = []
    for axis in "xyz":
        column = f"{stem}_{axis}_m"
        if column not in df.columns:
            return None
        value = pd.to_numeric(pd.Series([df.iloc[index][column]]), errors="coerce").iloc[0]
        if not np.isfinite(value):
            return None
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _platform_rotation_world_from_body(truth: pd.DataFrame, index: int) -> np.ndarray | None:
    names = [
        "truth_platform_orientation_x",
        "truth_platform_orientation_y",
        "truth_platform_orientation_z",
        "truth_platform_orientation_w",
    ]
    if not all(name in truth.columns for name in names):
        return None
    q = np.asarray(
        [pd.to_numeric(pd.Series([truth.iloc[index][name]]), errors="coerce").iloc[0] for name in names],
        dtype=float,
    )
    if not np.all(np.isfinite(q)):
        return None
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return None
    x, y, z, w = q / norm
    # Quaternion is platform/body -> world, matching Gazebo pose convention.
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _to_platform_frame(
    world_point: np.ndarray | None,
    origin_world: np.ndarray | None,
    rotation_world_from_body: np.ndarray | None,
) -> tuple[float, float, float, str]:
    if world_point is None or origin_world is None:
        return math.nan, math.nan, math.nan, "unavailable"
    delta = world_point - origin_world
    if rotation_world_from_body is None:
        local = delta
        method = "world_delta_fallback"
    else:
        local = rotation_world_from_body.T @ delta
        method = "platform_frame"
    return float(local[0]), float(local[1]), float(local[2]), method


def _first_contact_time(truth: pd.DataFrame, t: np.ndarray) -> float:
    # The rising truth flag is preferred because it is the same definition used
    # by the existing per-flight analysis.
    when = _first_true_time(truth, "truth_any_contact", t)
    if np.isfinite(when):
        return when

    # Fall back to the latched scalar timestamp if available.
    for column in (
        "truth_first_any_contact_sim_time_sec",
        "truth_contact_confirmed_sim_time_sec",
        "truth_first_geometric_crossing_sim_time_sec",
    ):
        if column not in truth.columns:
            continue
        values = _num(truth, column).to_numpy(float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            return float(values[0])
    return math.nan


def _contact_pad_point(truth: pd.DataFrame, index: int) -> tuple[np.ndarray | None, str]:
    left_active = False
    right_active = False
    if "truth_left_contact" in truth.columns:
        value = pd.to_numeric(pd.Series([truth.iloc[index]["truth_left_contact"]]), errors="coerce").iloc[0]
        left_active = bool(np.isfinite(value) and value > 0.5)
    if "truth_right_contact" in truth.columns:
        value = pd.to_numeric(pd.Series([truth.iloc[index]["truth_right_contact"]]), errors="coerce").iloc[0]
        right_active = bool(np.isfinite(value) and value > 0.5)

    left = _xyz(truth, "truth_left_pad_position", index)
    right = _xyz(truth, "truth_right_pad_position", index)

    if left_active and right_active and left is not None and right is not None:
        return 0.5 * (left + right), "both"
    if left_active and left is not None:
        return left, "left"
    if right_active and right is not None:
        return right, "right"

    # Contact sensors can be one sample behind the geometric state.  Pick the
    # lowest pad as a deterministic fallback.
    ld = math.nan
    rd = math.nan
    if "truth_left_pad_signed_distance_m" in truth.columns:
        ld = pd.to_numeric(pd.Series([truth.iloc[index]["truth_left_pad_signed_distance_m"]]), errors="coerce").iloc[0]
    if "truth_right_pad_signed_distance_m" in truth.columns:
        rd = pd.to_numeric(pd.Series([truth.iloc[index]["truth_right_pad_signed_distance_m"]]), errors="coerce").iloc[0]
    if left is not None and right is not None and np.isfinite(ld) and np.isfinite(rd):
        return (left, "left_nearest") if ld <= rd else (right, "right_nearest")
    if left is not None:
        return left, "left_fallback"
    if right is not None:
        return right, "right_fallback"
    return None, "unavailable"


def _touchdown_metrics(
    truth: pd.DataFrame,
    contact_window_sec: float,
    platform_radius_m: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "contact_time_sim_sec": math.nan,
        # Normal pad closing rate is preserved as a separate diagnostic.
        "contact_closing_rate_m_s": math.nan,
        "contact_closing_speed_abs_m_s": math.nan,
        # Headline touchdown-speed metric: mean magnitude of drone-platform
        # relative velocity over the final contact window.
        "contact_relative_velocity_x_mean_m_s": math.nan,
        "contact_relative_velocity_y_mean_m_s": math.nan,
        "contact_relative_velocity_z_mean_m_s": math.nan,
        "contact_relative_speed_mean_m_s": math.nan,
        "contact_relative_speed_from_mean_components_m_s": math.nan,
        "contact_speed_abs_m_s": math.nan,  # compatibility alias to relative-speed magnitude
        "contact_velocity_samples": 0,
        "contact_velocity_frame": "unavailable",
        "contact_center_x_platform_m": math.nan,
        "contact_center_y_platform_m": math.nan,
        "contact_center_z_platform_m": math.nan,
        "contact_radial_error_m": math.nan,
        "contact_radial_error_normalized": math.nan,
        "contact_point_x_platform_m": math.nan,
        "contact_point_y_platform_m": math.nan,
        "contact_point_z_platform_m": math.nan,
        "contact_pad": "",
        "contact_position_method": "unavailable",
    }
    if truth.empty:
        return result

    t = _sim_time(truth)
    contact_time = _first_contact_time(truth, t)
    result["contact_time_sim_sec"] = contact_time
    if not np.isfinite(contact_time):
        return result

    closing = _precontact_mean(
        truth,
        "truth_contact_pad_closing_rate_m_s",
        t,
        contact_time,
        contact_window_sec,
    )
    result["contact_closing_rate_m_s"] = closing
    result["contact_closing_speed_abs_m_s"] = abs(closing) if np.isfinite(closing) else math.nan

    velocity = _precontact_relative_velocity(
        truth, t, contact_time, contact_window_sec
    )
    result.update(velocity)
    # ``contact_speed_abs_m_s`` used to mean |normal closing rate| in the first
    # draft of the campaign scripts.  The campaign has not yet been analysed,
    # so make the natural name mean the full relative-speed magnitude from now
    # on while retaining the old quantity explicitly above.
    result["contact_speed_abs_m_s"] = result["contact_relative_speed_mean_m_s"]

    index = _nearest_index(t, contact_time)
    if index is None:
        return result

    origin = _xyz(truth, "truth_deck_point", index)
    if origin is None:
        origin = _xyz(truth, "truth_platform_position", index)
    rotation = _platform_rotation_world_from_body(truth, index)

    drone_center = _xyz(truth, "truth_drone_position", index)
    cx, cy, cz, method = _to_platform_frame(drone_center, origin, rotation)
    result["contact_center_x_platform_m"] = cx
    result["contact_center_y_platform_m"] = cy
    result["contact_center_z_platform_m"] = cz
    result["contact_position_method"] = method
    if np.isfinite(cx) and np.isfinite(cy):
        radius = float(math.hypot(cx, cy))
        result["contact_radial_error_m"] = radius
        if np.isfinite(platform_radius_m) and platform_radius_m > 1e-9:
            result["contact_radial_error_normalized"] = radius / platform_radius_m

    pad_point, pad_label = _contact_pad_point(truth, index)
    px, py, pz, _ = _to_platform_frame(pad_point, origin, rotation)
    result["contact_point_x_platform_m"] = px
    result["contact_point_y_platform_m"] = py
    result["contact_point_z_platform_m"] = pz
    result["contact_pad"] = pad_label
    return result


def _suffix_lookup(flat: dict[str, Any], aliases: Iterable[str], *, prefer: str = "") -> Any:
    aliases_l = [alias.lower() for alias in aliases]
    candidates: list[tuple[int, str, Any]] = []
    for key, value in flat.items():
        key_l = key.lower()
        base = key_l.rsplit(".", 1)[-1]
        for alias in aliases_l:
            exact = key_l == alias or base == alias
            suffix = key_l.endswith("." + alias)
            if exact or suffix:
                score = 0
                if prefer and prefer.lower() in key_l:
                    score -= 10
                score += len(key_l)
                candidates.append((score, key, value))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if np.isfinite(out) else math.nan


def _standard_gate(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"on", "1", "true", "yes", "enabled"}:
        return "on"
    if text in {"off", "0", "false", "no", "disabled"}:
        return "off"
    return text


def _controller_diagnostics(control: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    run: dict[str, Any] = {
        "center_duration_sec": math.nan,
        "center_exit_offset_x": math.nan,
        "center_exit_offset_y": math.nan,
        "center_exit_offset_r": math.nan,
        "center_final_1s_offset_rms": math.nan,
        "post_center_1s_peak_offset_r": math.nan,
        "center_exit_roll_p_scale": math.nan,
        "center_exit_pitch_p_scale": math.nan,
        "center_exit_roll_d_scale": math.nan,
        "center_exit_pitch_d_scale": math.nan,
        "final_probe_exit_roll_p_scale": math.nan,
        "final_probe_exit_pitch_p_scale": math.nan,
        "final_probe_exit_roll_d_scale": math.nan,
        "final_probe_exit_pitch_d_scale": math.nan,
        "center_timed_out_logged": math.nan,
    }
    gate_snapshot: dict[str, Any] = {}
    if control.empty:
        return run, gate_snapshot

    t = _sim_time(control)
    phase = _clean_text(control, "mission_substate")
    if not np.any(phase != ""):
        phase = _clean_text(control, "controller_phase")

    ox = _num(control, "target_offset_x").to_numpy(float)
    oy = _num(control, "target_offset_y").to_numpy(float)
    radius = np.hypot(ox, oy)

    center = phase.to_numpy() == "center"
    center_idx = np.flatnonzero(center & np.isfinite(t))
    if center_idx.size:
        first, last = int(center_idx[0]), int(center_idx[-1])
        run["center_duration_sec"] = max(0.0, float(t[last] - t[first]))
        if np.isfinite(ox[last]):
            run["center_exit_offset_x"] = float(ox[last])
        if np.isfinite(oy[last]):
            run["center_exit_offset_y"] = float(oy[last])
        if np.isfinite(radius[last]):
            run["center_exit_offset_r"] = float(radius[last])

        end_t = float(t[last])
        last_second = center & np.isfinite(t) & (t >= end_t - 1.0) & np.isfinite(radius)
        if np.any(last_second):
            run["center_final_1s_offset_rms"] = float(np.sqrt(np.mean(radius[last_second] ** 2)))
        post = np.isfinite(t) & (t > end_t) & (t <= end_t + 1.0) & np.isfinite(radius)
        if np.any(post):
            run["post_center_1s_peak_offset_r"] = float(np.max(radius[post]))

        def phase_scale(axis: str, kind: str, index: int) -> float:
            independent = f"mission_{axis}_{kind}_scale"
            shared = f"mission_lateral_{kind}_scale"
            for column in (independent, shared):
                if column in control.columns:
                    value = _num(control, column).to_numpy(float)[index]
                    if np.isfinite(value):
                        return float(value)
            return math.nan

        for axis in ("roll", "pitch"):
            for kind in ("p", "d"):
                run[f"center_exit_{axis}_{kind}_scale"] = phase_scale(axis, kind, last)

    # If the current telemetry emits the CENTER completion info as a column,
    # consume it.  Otherwise leave NaN instead of guessing timeout from duration.
    timeout_candidates = [
        c for c in control.columns if "center_timed_out" in c.lower() or "center_timeout" in c.lower()
    ]
    for column in timeout_candidates:
        values = _num(control, column).to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size:
            run["center_timed_out_logged"] = float(finite[-1] > 0.5)
            break

    final_probe = phase.to_numpy() == "final_probe"
    fp_idx = np.flatnonzero(final_probe & np.isfinite(t))
    if fp_idx.size:
        last = int(fp_idx[-1])
        for axis in ("roll", "pitch"):
            for kind in ("p", "d"):
                independent = f"mission_{axis}_{kind}_scale"
                shared = f"mission_lateral_{kind}_scale"
                value = math.nan
                for column in (independent, shared):
                    if column in control.columns:
                        v = _num(control, column).to_numpy(float)[last]
                        if np.isfinite(v):
                            value = float(v)
                            break
                run[f"final_probe_exit_{axis}_{kind}_scale"] = value

    # Decision row: first logged tracking-ready sample is ideal.  The current
    # per-flight analyser explicitly notes that the verdict may be written on a
    # phase-transition row, so do not restrict this search to FINAL_PROBE rows.
    decision_mask = np.zeros(len(control), dtype=bool)
    ready_columns = [c for c in control.columns if c.startswith("mission_tracking") and c.endswith("ready")]
    for column in ready_columns:
        values = _num(control, column).to_numpy(float)
        decision_mask |= np.isfinite(values) & (values > 0.5)
    decision_idx = np.flatnonzero(decision_mask & np.isfinite(t))
    if decision_idx.size:
        index = int(decision_idx[0])
    elif fp_idx.size:
        index = int(fp_idx[-1])
    else:
        index = len(control) - 1

    gate_snapshot["gate_snapshot_time_sim_sec"] = float(t[index]) if np.isfinite(t[index]) else math.nan
    gate_snapshot["gate_snapshot_phase"] = str(phase.iloc[index])

    for column in control.columns:
        column_l = column.lower()
        if not column_l.startswith("mission_"):
            continue
        if not any(keyword in column_l for keyword in GATE_KEYWORDS):
            continue
        value = control.iloc[index][column]
        gate_snapshot[column] = _json_scalar(value)

    # A handful of context channels make a refusal interpretable without opening
    # the full log.
    for column in (
        "target_offset_x",
        "target_offset_y",
        "target_area_fraction",
        "flow_divergence_1_s",
        "flow_lateral_x_norm_s",
        "flow_lateral_y_norm_s",
    ):
        if column in control.columns:
            gate_snapshot[column] = _json_scalar(control.iloc[index][column])

    return run, gate_snapshot


def _platform_radius(spec_flat: dict[str, Any]) -> float:
    # Prefer a platform-scoped radius key.  scenario.py historically uses a
    # radius parameter for the deck geometry.
    candidates = []
    for key, value in spec_flat.items():
        key_l = key.lower()
        if "radius" not in key_l:
            continue
        score = 0 if "platform" in key_l else 10
        if key_l.endswith("radius_m"):
            score -= 2
        candidates.append((score, len(key_l), value))
    for _, _, value in sorted(candidates):
        number = _as_float(value)
        if np.isfinite(number) and number > 0:
            return number
    return math.nan


def _run_metadata(spec: dict[str, Any], outcome: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    flat = _flatten_json(spec)
    outcome_flat = _flatten_json(outcome)

    spec_gate = _suffix_lookup(flat, ("gate", "gate_enabled", "commit_gate_enabled"))
    outcome_gate = outcome.get("commit_gate_enabled")
    gate = _standard_gate(outcome_gate if outcome_gate is not None else spec_gate)

    status = str(outcome.get("status", "")).strip().lower()
    row: dict[str, Any] = {
        "run_id": str(spec.get("run_id") or run_dir.name),
        "run_dir": str(run_dir),
        "pair_key": _pair_key(spec),
        "status": status,
        "status_known": status in OUTCOME_STATUSES,
        "gate": gate,
        "gate_enabled": gate == "on",
        "gate_consistent": (
            True
            if spec_gate is None or outcome_gate is None
            else _standard_gate(spec_gate) == _standard_gate(outcome_gate)
        ),
        "condition": _json_scalar(_suffix_lookup(flat, ("condition",))),
        "seed": _json_scalar(outcome.get("seed", _suffix_lookup(flat, ("seed",)))),
        "repetition": _json_scalar(_suffix_lookup(flat, ("repetition", "rep", "repeat"))),
        "platform_radius_m": _platform_radius(flat),
        "verdict_reached": bool(outcome.get("verdict_reached", False)),
        "refusal_criteria": _json_scalar(outcome.get("refusal_criteria")),
        "refusal_axes": _json_scalar(outcome.get("refusal_axes")),
        "predicted_crossing_height_m": _as_float(outcome.get("predicted_crossing_height_m")),
        "vertical_rho": _as_float(outcome.get("vertical_rho")),
        "abort_phase": _json_scalar(outcome.get("abort_phase")),
        "reason": _json_scalar(outcome.get("reason")),
    }

    for key, value in flat.items():
        row[f"spec_{key.replace('.', '__')}"] = _json_scalar(value)
    for key, value in outcome_flat.items():
        row[f"outcome_{key.replace('.', '__')}"] = _json_scalar(value)
    return row


def build_campaign(campaign_dir: Path, output_dir: Path, contact_window_sec: float) -> None:
    runs_root = campaign_dir / "runs" if (campaign_dir / "runs").is_dir() else campaign_dir
    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir() and (p / "run.json").exists()])
    if not run_dirs:
        raise RuntimeError(f"No run directories with run.json found under {runs_root}")

    run_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        try:
            spec = _read_json(run_dir / "run.json")
        except Exception as exc:  # noqa: BLE001 - issue table is the point here.
            issues.append({"run_dir": str(run_dir), "issue": f"run.json: {exc}"})
            continue

        outcome_path, issue = _find_one(run_dir, "bee_outcome_*.json")
        if issue:
            issues.append({"run_dir": str(run_dir), "issue": issue})
        if outcome_path is None:
            # A harness-level failure may have no controller outcome record. Keep
            # the run in the dataset using an empty outcome rather than silently
            # dropping it.
            outcome: dict[str, Any] = {}
        else:
            try:
                outcome = _read_json(outcome_path)
            except Exception as exc:  # noqa: BLE001
                issues.append({"run_dir": str(run_dir), "issue": f"{outcome_path.name}: {exc}"})
                outcome = {}

        row = _run_metadata(spec, outcome, run_dir)

        truth_path, issue = _find_one(run_dir, "bee_truth_*.csv")
        if issue:
            issues.append({"run_dir": str(run_dir), "issue": issue})
        if truth_path is not None:
            try:
                truth = pd.read_csv(truth_path, low_memory=False)
                row.update(_touchdown_metrics(truth, contact_window_sec, row["platform_radius_m"]))
            except Exception as exc:  # noqa: BLE001
                issues.append({"run_dir": str(run_dir), "issue": f"{truth_path.name}: {exc}"})
                row.update(_touchdown_metrics(pd.DataFrame(), contact_window_sec, row["platform_radius_m"]))
        else:
            row.update(_touchdown_metrics(pd.DataFrame(), contact_window_sec, row["platform_radius_m"]))

        controller_path, issue = _find_one(run_dir, "bee_controller_*.csv")
        if issue:
            issues.append({"run_dir": str(run_dir), "issue": issue})
        gate_snapshot: dict[str, Any] = {}
        if controller_path is not None:
            try:
                control = pd.read_csv(controller_path, low_memory=False)
                controller_run, gate_snapshot = _controller_diagnostics(control)
                row.update(controller_run)
            except Exception as exc:  # noqa: BLE001
                issues.append({"run_dir": str(run_dir), "issue": f"{controller_path.name}: {exc}"})
        else:
            controller_run, _ = _controller_diagnostics(pd.DataFrame())
            row.update(controller_run)

        # Keep file provenance in the run table.  The wind file is not analysed
        # here, but its presence matters for reproducibility.
        wind_path, issue = _find_one(run_dir, "bee_wind_*.csv")
        if issue:
            issues.append({"run_dir": str(run_dir), "issue": issue})
        row["outcome_file"] = outcome_path.name if outcome_path else ""
        row["controller_file"] = controller_path.name if controller_path else ""
        row["truth_file"] = truth_path.name if truth_path else ""
        row["wind_file"] = wind_path.name if wind_path else ""
        row["contact_window_sec"] = float(contact_window_sec)

        run_rows.append(row)

        gate_row = {
            "run_id": row["run_id"],
            "pair_key": row["pair_key"],
            "gate": row["gate"],
            "gate_enabled": row["gate_enabled"],
            "condition": row["condition"],
            "seed": row["seed"],
            "repetition": row["repetition"],
            "platform_radius_m": row["platform_radius_m"],
            "status": row["status"],
            "verdict_reached": row["verdict_reached"],
            "refusal_criteria": row["refusal_criteria"],
            "refusal_axes": row["refusal_axes"],
            "predicted_crossing_height_m": row["predicted_crossing_height_m"],
            "vertical_rho": row["vertical_rho"],
            "abort_phase": row["abort_phase"],
            "reason": row["reason"],
        }
        gate_row.update(gate_snapshot)
        gate_rows.append(gate_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(run_rows)
    gates_df = pd.DataFrame(gate_rows)
    issues_df = pd.DataFrame(issues, columns=["run_dir", "issue"])

    # Stable, useful lead columns; everything else follows alphabetically.
    lead = [
        "run_id",
        "pair_key",
        "condition",
        "seed",
        "repetition",
        "platform_radius_m",
        "gate",
        "status",
        "verdict_reached",
        "refusal_criteria",
        "refusal_axes",
        "abort_phase",
        "reason",
        "contact_time_sim_sec",
        "contact_relative_speed_mean_m_s",
        "contact_relative_velocity_x_mean_m_s",
        "contact_relative_velocity_y_mean_m_s",
        "contact_relative_velocity_z_mean_m_s",
        "contact_relative_speed_from_mean_components_m_s",
        "contact_closing_rate_m_s",
        "contact_closing_speed_abs_m_s",
        "contact_speed_abs_m_s",
        "contact_velocity_samples",
        "contact_velocity_frame",
        "contact_center_x_platform_m",
        "contact_center_y_platform_m",
        "contact_radial_error_m",
        "contact_radial_error_normalized",
    ]
    lead = [c for c in lead if c in runs_df.columns]
    runs_df = runs_df[lead + sorted(c for c in runs_df.columns if c not in lead)]

    runs_path = output_dir / "campaign_runs.csv"
    gates_path = output_dir / "campaign_gate_events.csv"
    issues_path = output_dir / "campaign_issues.csv"
    runs_df.to_csv(runs_path, index=False)
    gates_df.to_csv(gates_path, index=False)
    issues_df.to_csv(issues_path, index=False)

    counts = runs_df["status"].fillna("").replace("", "missing_outcome").value_counts().to_dict()
    print(f"Scanned {len(runs_df)} runs from {runs_root}")
    print("Outcome counts:")
    for key, value in sorted(counts.items()):
        print(f"  {key:>16}: {value}")
    print(f"Wrote {runs_path}")
    print(f"Wrote {gates_path}")
    print(f"Wrote {issues_path} ({len(issues_df)} issue rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path, help="Campaign directory (or its runs/ directory).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: CAMPAIGN_DIR/analysis (or parent/analysis when runs/ is supplied).",
    )
    parser.add_argument(
        "--contact-window",
        type=float,
        default=CONTACT_RATE_AVERAGE_WINDOW_SEC,
        help="Pre-contact averaging window in seconds (default: 0.10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    if not campaign_dir.is_dir():
        raise SystemExit(f"Campaign directory not found: {campaign_dir}")
    if args.contact_window <= 0:
        raise SystemExit("--contact-window must be > 0")

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif campaign_dir.name == "runs":
        output_dir = campaign_dir.parent / "analysis"
    else:
        output_dir = campaign_dir / "analysis"

    build_campaign(campaign_dir, output_dir, float(args.contact_window))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
