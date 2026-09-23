"""Controller-development diagnostics for the first OdometryMaxxing campaign.

This is intentionally separate from the article-level gate comparison.  Its job
is to tell us what to retune after Campaign 1:

* platform-size influence on gate refusals and visual-mismatch refusals;
* paired cases where gate ON refused on visual mismatch while gate OFF landed;
* ABORTED runs and their controller/mission reasons;
* CENTER quality and the immediate post-CENTER excursion;
* optional 6 x 6 abort-rate maps for the two campaign factors.

No acceleration-ground-truth reconstruction is introduced here.  The script
uses only fields already present in campaign_runs.csv / campaign_gate_events.csv.

Example
-------
    python3 analyse_campaign_diagnostics.py \
        campaign/analysis/campaign_runs.csv \
        campaign/analysis/campaign_gate_events.csv

If exactly two six-level scenario factors cannot be inferred safely, pass them:

    --x-factor spec_wind__amplitude --y-factor spec_platform__heave_frequency
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VISUAL_WORDS = ("visual", "mismatch", "tracking", "chi")
EXCLUDED_FACTOR_WORDS = (
    "gate",
    "seed",
    "repetition",
    "repeat",
    "run_id",
    "radius",
    "condition",
    "phase",
)


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _require(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _criteria_tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            decoded = parser(text)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(decoded, (list, tuple, set)):
            return [str(item).strip().lower() for item in decoded if str(item).strip()]
        if isinstance(decoded, str):
            text = decoded
            break
    normalized = text.replace("|", ",").replace(";", ",")
    return [token.strip().strip("[](){}'\"").lower() for token in normalized.split(",") if token.strip()]


def _is_visual_refusal(value: object) -> bool:
    return any(any(word in token for word in VISUAL_WORDS) for token in _criteria_tokens(value))


def _pair_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair_key, group in df.groupby("pair_key", dropna=False):
        on = group[_clean(group["gate"]) == "on"]
        off = group[_clean(group["gate"]) == "off"]
        if on.empty or off.empty:
            continue
        a, b = on.iloc[0], off.iloc[0]
        rows.append(
            {
                "pair_key": pair_key,
                "platform_radius_m": pd.to_numeric(pd.Series([a.get("platform_radius_m")]), errors="coerce").iloc[0],
                "status_on": str(a.get("status", "")).strip().lower(),
                "status_off": str(b.get("status", "")).strip().lower(),
                "criteria_on": a.get("refusal_criteria", ""),
                "visual_refusal_on": _is_visual_refusal(a.get("refusal_criteria", "")),
                "condition": a.get("condition", ""),
                "seed": a.get("seed", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _platform_size_analysis(runs: pd.DataFrame, out: Path) -> None:
    radius = _num(runs, "platform_radius_m")
    runs = runs[np.isfinite(radius)].copy()
    runs["platform_radius_m"] = radius[np.isfinite(radius)].to_numpy(float)
    if runs.empty:
        print("No platform_radius_m values; skipping platform-size analysis")
        return

    gate_on = runs[_clean(runs["gate"]) == "on"].copy()
    gate_on["visual_refusal"] = gate_on["refusal_criteria"].apply(_is_visual_refusal)
    gate_on["is_landed"] = _clean(gate_on["status"]) == "landed"
    gate_on["is_infeasible"] = _clean(gate_on["status"]) == "infeasible"
    gate_on["is_aborted"] = _clean(gate_on["status"]) == "aborted"
    gate_on["is_visual_infeasible"] = gate_on["is_infeasible"] & gate_on["visual_refusal"]

    summary = (
        gate_on.groupby("platform_radius_m", dropna=False)
        .agg(
            n=("run_id", "size"),
            landed=("is_landed", "sum"),
            infeasible=("is_infeasible", "sum"),
            aborted=("is_aborted", "sum"),
            visual_infeasible=("is_visual_infeasible", "sum"),
        )
        .reset_index()
        .sort_values("platform_radius_m")
    )
    for column in ("landed", "infeasible", "aborted", "visual_infeasible"):
        summary[f"{column}_rate"] = summary[column] / summary["n"]
    summary.to_csv(out / "platform_size_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(summary["platform_radius_m"], summary["landed_rate"], marker="o", label="LANDED")
    ax.plot(summary["platform_radius_m"], summary["infeasible_rate"], marker="o", label="INFEASIBLE")
    ax.plot(summary["platform_radius_m"], summary["aborted_rate"], marker="o", label="ABORTED")
    ax.set_xlabel("Platform radius [m]")
    ax.set_ylabel("Fraction of gate-ON runs")
    ax.set_ylim(bottom=0)
    ax.set_title("Outcome versus platform size — gate ON")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "platform_size_outcomes.png", dpi=220)
    plt.close(fig)

    pairs = _pair_table(runs)
    if pairs.empty:
        return
    pairs["visual_refusal_off_landed"] = (
        (pairs["status_on"] == "infeasible")
        & pairs["visual_refusal_on"]
        & (pairs["status_off"] == "landed")
    )
    pairs["valid_pair"] = ~pairs["status_on"].isin({"aborted", "launch_failed", "timeout"}) & ~pairs["status_off"].isin({"aborted", "launch_failed", "timeout"})
    valid = pairs[pairs["valid_pair"] & np.isfinite(pd.to_numeric(pairs["platform_radius_m"], errors="coerce"))].copy()
    if valid.empty:
        return

    paired_summary = (
        valid.groupby("platform_radius_m")
        .agg(
            n_pairs=("pair_key", "size"),
            visual_refusal_off_landed=("visual_refusal_off_landed", "sum"),
            any_visual_refusal_on=("visual_refusal_on", "sum"),
        )
        .reset_index()
        .sort_values("platform_radius_m")
    )
    paired_summary["visual_refusal_off_landed_rate"] = paired_summary["visual_refusal_off_landed"] / paired_summary["n_pairs"]
    paired_summary["any_visual_refusal_on_rate"] = paired_summary["any_visual_refusal_on"] / paired_summary["n_pairs"]
    paired_summary.to_csv(out / "platform_size_visual_refusal_pairs.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(
        paired_summary["platform_radius_m"],
        paired_summary["any_visual_refusal_on_rate"],
        marker="o",
        label="Any visual-mismatch refusal (gate ON)",
    )
    ax.plot(
        paired_summary["platform_radius_m"],
        paired_summary["visual_refusal_off_landed_rate"],
        marker="o",
        label="Visual refusal + paired gate-OFF landing",
    )
    ax.set_xlabel("Platform radius [m]")
    ax.set_ylabel("Fraction of valid ON/OFF pairs")
    ax.set_ylim(bottom=0)
    ax.set_title("Platform size and visual-mismatch refusals")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "platform_size_visual_refusals.png", dpi=220)
    plt.close(fig)


def _refusal_criteria_analysis(runs: pd.DataFrame, out: Path) -> None:
    gate_on_inf = runs[(_clean(runs["gate"]) == "on") & (_clean(runs["status"]) == "infeasible")].copy()
    rows = []
    for _, row in gate_on_inf.iterrows():
        tokens = _criteria_tokens(row.get("refusal_criteria", "")) or ["unspecified"]
        for token in tokens:
            rows.append(
                {
                    "run_id": row.get("run_id", ""),
                    "platform_radius_m": row.get("platform_radius_m", np.nan),
                    "criterion": token,
                }
            )
    exploded = pd.DataFrame(rows)
    exploded.to_csv(out / "refusal_criteria_long.csv", index=False)
    if exploded.empty:
        return
    table = (
        exploded.groupby(["platform_radius_m", "criterion"])
        .size()
        .rename("n")
        .reset_index()
    )
    table.to_csv(out / "refusal_criteria_by_size.csv", index=False)

    counts = exploded["criterion"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(counts) + 1.5)))
    ax.barh(np.arange(len(counts)), counts.to_numpy())
    ax.set_yticks(np.arange(len(counts)), counts.index)
    ax.set_xlabel("Gate-ON INFEASIBLE runs")
    ax.set_title("Refusal criteria")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "refusal_criteria.png", dpi=220)
    plt.close(fig)


def _abort_analysis(runs: pd.DataFrame, out: Path) -> None:
    aborted = runs[_clean(runs["status"]) == "aborted"].copy()
    useful = [
        c
        for c in (
            "run_id",
            "pair_key",
            "gate",
            "condition",
            "seed",
            "repetition",
            "platform_radius_m",
            "abort_phase",
            "reason",
            "center_duration_sec",
            "center_exit_offset_r",
            "center_final_1s_offset_rms",
            "post_center_1s_peak_offset_r",
            "center_exit_roll_p_scale",
            "center_exit_pitch_p_scale",
            "center_exit_roll_d_scale",
            "center_exit_pitch_d_scale",
        )
        if c in aborted.columns
    ]
    spec_cols = [c for c in aborted.columns if c.startswith("spec_")]
    aborted[useful + spec_cols].to_csv(out / "abort_cases.csv", index=False)
    if aborted.empty:
        return

    reason = aborted["reason"].fillna("").replace("", "unspecified").astype(str)
    counts = reason.value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.38 * len(counts) + 1.5)))
    ax.barh(np.arange(len(counts)), counts.to_numpy())
    ax.set_yticks(np.arange(len(counts)), counts.index)
    ax.set_xlabel("ABORTED runs")
    ax.set_title("Controller/mission abort reasons (top 20)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "abort_reasons.png", dpi=220)
    plt.close(fig)


def _center_analysis(runs: pd.DataFrame, out: Path) -> None:
    metrics = [
        "center_duration_sec",
        "center_exit_offset_r",
        "center_final_1s_offset_rms",
        "post_center_1s_peak_offset_r",
    ]
    available = [metric for metric in metrics if metric in runs.columns and _num(runs, metric).notna().any()]
    if not available:
        return

    status = _clean(runs["status"])
    keep_status = [s for s in ("landed", "infeasible", "aborted", "crashed") if np.any(status == s)]
    summaries = []
    for s in keep_status:
        part = runs[status == s]
        for metric in available:
            values = _num(part, metric).dropna().to_numpy(float)
            summaries.append(
                {
                    "status": s,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "p90": float(np.percentile(values, 90)) if len(values) else np.nan,
                }
            )
    pd.DataFrame(summaries).to_csv(out / "center_diagnostics_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, metric in zip(axes.flat, metrics):
        if metric not in available:
            ax.axis("off")
            continue
        groups = []
        labels = []
        for s in keep_status:
            values = _num(runs[status == s], metric).dropna().to_numpy(float)
            if values.size:
                groups.append(values)
                labels.append(s.upper())
        if groups:
            ax.boxplot(groups, showmeans=True)
            ax.set_xticks(np.arange(1, len(labels) + 1), labels)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("CENTER diagnostics by terminal outcome")
    fig.tight_layout()
    fig.savefig(out / "center_diagnostics.png", dpi=220)
    plt.close(fig)

    # A direct diagnostic for the user's current hypothesis: did a run leave
    # CENTER acceptably but then wander during the first second of APPROACH?
    if {"center_exit_offset_r", "post_center_1s_peak_offset_r"}.issubset(runs.columns):
        x = _num(runs, "center_exit_offset_r").to_numpy(float)
        y = _num(runs, "post_center_1s_peak_offset_r").to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        if np.any(good):
            fig, ax = plt.subplots(figsize=(7.5, 5.8))
            for s in keep_status:
                mask = good & (status.to_numpy() == s)
                if np.any(mask):
                    ax.scatter(x[mask], y[mask], s=28, alpha=0.6, label=s.upper())
            lim = max(float(np.nanmax(x[good])), float(np.nanmax(y[good])))
            ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0, label="no post-CENTER growth")
            ax.set_xlabel("Offset magnitude at CENTER exit")
            ax.set_ylabel("Peak offset in first 1 s after CENTER")
            ax.set_title("Immediate off-centre growth after CENTER")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "center_to_approach_offset_growth.png", dpi=220)
            plt.close(fig)


def _factor_candidates(runs: pd.DataFrame) -> list[str]:
    candidates = []
    for column in runs.columns:
        if not column.startswith("spec_"):
            continue
        lower = column.lower()
        if any(word in lower for word in EXCLUDED_FACTOR_WORDS):
            continue
        series = runs[column].dropna()
        if series.empty:
            continue
        # Reject structured JSON fields; a six-entry waveform list is not a
        # six-level campaign factor.
        sample = str(series.iloc[0]).strip()
        if sample.startswith("[") or sample.startswith("{"):
            continue
        if series.astype(str).nunique(dropna=True) == 6:
            candidates.append(column)
    return sorted(candidates)


def _resolve_factor(runs: pd.DataFrame, requested: str | None) -> str | None:
    if not requested:
        return None
    if requested in runs.columns:
        return requested
    matches = [c for c in runs.columns if c.endswith(requested) or c.replace("spec_", "", 1) == requested]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not uniquely resolve factor {requested!r}; matches={matches}")


def _sorted_levels(series: pd.Series) -> list[object]:
    values = series.dropna().unique().tolist()
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().all():
        order = np.argsort(numeric.to_numpy(float))
        return [values[i] for i in order]
    return sorted(values, key=lambda v: str(v))


def _abort_heatmap(runs: pd.DataFrame, xcol: str, ycol: str, out: Path) -> None:
    for gate in ("on", "off"):
        part = runs[_clean(runs["gate"]) == gate].copy()
        x_levels = _sorted_levels(part[xcol])
        y_levels = _sorted_levels(part[ycol])
        if len(x_levels) != 6 or len(y_levels) != 6:
            continue
        matrix = np.full((6, 6), np.nan)
        counts = np.zeros((6, 6), dtype=int)
        for iy, y in enumerate(y_levels):
            for ix, x in enumerate(x_levels):
                cell = part[(part[xcol] == x) & (part[ycol] == y)]
                if cell.empty:
                    continue
                counts[iy, ix] = len(cell)
                matrix[iy, ix] = float(np.mean(_clean(cell["status"]) == "aborted"))

        fig, ax = plt.subplots(figsize=(8, 6.6))
        image = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0.0, vmax=max(0.01, np.nanmax(matrix)))
        ax.set_xticks(range(6), [str(v) for v in x_levels], rotation=35, ha="right")
        ax.set_yticks(range(6), [str(v) for v in y_levels])
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"ABORTED fraction — gate {gate.upper()}")
        for iy in range(6):
            for ix in range(6):
                if np.isfinite(matrix[iy, ix]):
                    ax.text(ix, iy, f"{matrix[iy, ix]:.2f}\n(n={counts[iy, ix]})", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="ABORTED fraction")
        fig.tight_layout()
        fig.savefig(out / f"abort_map_gate_{gate}.png", dpi=220)
        plt.close(fig)


def _gate_snapshot_exports(gates: pd.DataFrame, out: Path) -> None:
    if gates.empty:
        return
    # A compact table focused on visual mismatch.  This is deliberately
    # schema-tolerant: every matching current/future telemetry column is kept.
    columns = [
        c
        for c in gates.columns
        if c in {"run_id", "pair_key", "gate", "status", "platform_radius_m", "refusal_criteria", "refusal_axes"}
        or any(word in c.lower() for word in VISUAL_WORDS)
    ]
    if columns:
        gates[columns].to_csv(out / "visual_mismatch_gate_snapshots.csv", index=False)


def analyse(runs_csv: Path, gates_csv: Path, out: Path, x_factor: str | None, y_factor: str | None) -> None:
    runs = pd.read_csv(runs_csv, low_memory=False)
    gates = pd.read_csv(gates_csv, low_memory=False) if gates_csv.is_file() else pd.DataFrame()
    _require(runs, ("run_id", "pair_key", "gate", "status", "platform_radius_m", "refusal_criteria", "reason"))
    out.mkdir(parents=True, exist_ok=True)

    _platform_size_analysis(runs, out)
    _refusal_criteria_analysis(runs, out)
    _abort_analysis(runs, out)
    _center_analysis(runs, out)
    _gate_snapshot_exports(gates, out)

    xcol = _resolve_factor(runs, x_factor)
    ycol = _resolve_factor(runs, y_factor)
    candidates = _factor_candidates(runs)
    if xcol is None and ycol is None and len(candidates) == 2:
        xcol, ycol = candidates
        print(f"Auto-selected 6-level factors: {xcol}, {ycol}")
    elif xcol is None or ycol is None:
        print("6x6 abort map not generated automatically.")
        print(f"Six-level scalar candidates: {candidates}")
        print("Pass --x-factor and --y-factor to choose explicitly.")
    if xcol is not None and ycol is not None:
        _abort_heatmap(runs, xcol, ycol, out)

    # Machine-readable diagnostic overview.
    overview = (
        runs.groupby(["gate", "status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    overview.to_csv(out / "diagnostic_outcome_counts.csv", index=False)
    print(f"Wrote controller-development diagnostics to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_runs_csv", type=Path)
    parser.add_argument("campaign_gate_events_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--x-factor", default=None, help="Column/suffix for the first 6-level campaign factor.")
    parser.add_argument("--y-factor", default=None, help="Column/suffix for the second 6-level campaign factor.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_csv = args.campaign_runs_csv.expanduser().resolve()
    gates_csv = args.campaign_gate_events_csv.expanduser().resolve()
    if not runs_csv.is_file():
        raise SystemExit(f"File not found: {runs_csv}")
    if not gates_csv.is_file():
        raise SystemExit(f"File not found: {gates_csv}")
    out = args.output_dir.expanduser().resolve() if args.output_dir else runs_csv.parent / "campaign_diagnostics"
    try:
        analyse(runs_csv, gates_csv, out, args.x_factor, args.y_factor)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
