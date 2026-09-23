"""Article-level gate ON/OFF analysis for an OdometryMaxxing campaign.

Input is ``campaign_runs.csv`` produced by ``build_campaign_dataset.py``.
The analysis keeps INFEASIBLE and ABORTED strictly separate:

* INFEASIBLE is a gate refusal and is part of the gate's intended behaviour.
* ABORTED is a controller/mission failure and is reported separately; it is not
  counted as a gate refusal or a bad touchdown.

The two headline observables are:

1. contact speed: mean magnitude of drone-platform relative velocity over the
   final 0.10 s before physical contact;
2. contact position: drone-centre position at first contact in platform axes.

The x/y/z relative-velocity components over the same 0.10 s window are retained
and plotted as diagnostics.  Normal pad closing rate remains available but is no
longer the headline speed metric.

Because the campaign pairs gate ON and OFF with the same physical scenario, the
script also builds a pair table and reports within-scenario touchdown differences
for pairs where both arms landed.

Example
-------
    python3 analyse_gate_effect.py campaign/analysis/campaign_runs.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATUS_ORDER = ["landed", "infeasible", "aborted", "crashed", "launch_failed", "timeout"]


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _require(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"campaign_runs.csv is missing required columns: {', '.join(missing)}")


def _bootstrap_ci(values: np.ndarray, statistic: str, n_boot: int = 10000, seed: int = 20260922) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[draw]
    if statistic == "mean":
        stats = np.mean(samples, axis=1)
    elif statistic == "median":
        stats = np.median(samples, axis=1)
    else:
        raise ValueError(statistic)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _paired_stats(on: np.ndarray, off: np.ndarray, metric: str) -> dict[str, float | str | int]:
    on = np.asarray(on, dtype=float)
    off = np.asarray(off, dtype=float)
    good = np.isfinite(on) & np.isfinite(off)
    on = on[good]
    off = off[good]
    diff = on - off
    mean_lo, mean_hi = _bootstrap_ci(diff, "mean")
    med_lo, med_hi = _bootstrap_ci(diff, "median", seed=20260923)
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else math.nan
    dz = float(np.mean(diff) / sd) if np.isfinite(sd) and sd > 1e-12 else math.nan
    return {
        "metric": metric,
        "n_pairs": int(diff.size),
        "mean_gate_on": float(np.mean(on)) if on.size else math.nan,
        "mean_gate_off": float(np.mean(off)) if off.size else math.nan,
        "median_gate_on": float(np.median(on)) if on.size else math.nan,
        "median_gate_off": float(np.median(off)) if off.size else math.nan,
        "mean_difference_on_minus_off": float(np.mean(diff)) if diff.size else math.nan,
        "mean_difference_ci95_low": mean_lo,
        "mean_difference_ci95_high": mean_hi,
        "median_difference_on_minus_off": float(np.median(diff)) if diff.size else math.nan,
        "median_difference_ci95_low": med_lo,
        "median_difference_ci95_high": med_hi,
        "paired_effect_dz": dz,
    }


def _pair_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair_key, group in df.groupby("pair_key", dropna=False):
        arm = {gate: part.iloc[0] for gate, part in group.groupby("gate") if gate in {"on", "off"}}
        if "on" not in arm or "off" not in arm:
            continue
        on = arm["on"]
        off = arm["off"]
        row = {
            "pair_key": pair_key,
            "condition": on.get("condition", ""),
            "seed": on.get("seed", np.nan),
            "repetition": on.get("repetition", np.nan),
            "platform_radius_m": on.get("platform_radius_m", np.nan),
            "status_on": str(on.get("status", "")),
            "status_off": str(off.get("status", "")),
            "reason_on": on.get("reason", ""),
            "reason_off": off.get("reason", ""),
            "refusal_criteria_on": on.get("refusal_criteria", ""),
            "refusal_axes_on": on.get("refusal_axes", ""),
        }
        for metric in (
            "contact_relative_speed_mean_m_s",
            "contact_relative_velocity_x_mean_m_s",
            "contact_relative_velocity_y_mean_m_s",
            "contact_relative_velocity_z_mean_m_s",
            "contact_relative_speed_from_mean_components_m_s",
            "contact_closing_rate_m_s",
            "contact_closing_speed_abs_m_s",
            "contact_speed_abs_m_s",
            "contact_center_x_platform_m",
            "contact_center_y_platform_m",
            "contact_radial_error_m",
            "contact_radial_error_normalized",
        ):
            a = pd.to_numeric(pd.Series([on.get(metric)]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([off.get(metric)]), errors="coerce").iloc[0]
            row[f"{metric}_on"] = a
            row[f"{metric}_off"] = b
            row[f"{metric}_delta_on_minus_off"] = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
        row["both_landed"] = row["status_on"] == "landed" and row["status_off"] == "landed"
        row["gate_refused_off_landed"] = row["status_on"] == "infeasible" and row["status_off"] == "landed"
        row["gate_refused_off_crashed"] = row["status_on"] == "infeasible" and row["status_off"] == "crashed"
        row["pair_has_abort"] = row["status_on"] == "aborted" or row["status_off"] == "aborted"
        row["pair_has_harness_failure"] = row["status_on"] in {"launch_failed", "timeout"} or row["status_off"] in {"launch_failed", "timeout"}
        rows.append(row)
    return pd.DataFrame(rows)


def _save_outcomes(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    counts = (
        df.groupby(["gate", "status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    counts.to_csv(out / "gate_outcome_counts.csv", index=False)

    pivot = counts.pivot(index="status", columns="gate", values="n").fillna(0)
    ordered = [s for s in STATUS_ORDER if s in pivot.index] + [s for s in pivot.index if s not in STATUS_ORDER]
    pivot = pivot.reindex(ordered)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(pivot.index))
    width = 0.38
    on = pivot["on"].to_numpy(float) if "on" in pivot.columns else np.zeros(len(pivot))
    off = pivot["off"].to_numpy(float) if "off" in pivot.columns else np.zeros(len(pivot))
    ax.bar(x - width / 2, on, width, label="Gate ON")
    ax.bar(x + width / 2, off, width, label="Gate OFF")
    ax.set_xticks(x, [s.replace("_", " ").title() for s in pivot.index], rotation=20, ha="right")
    ax.set_ylabel("Runs")
    ax.set_title("Campaign terminal outcomes — ABORTED kept separate from INFEASIBLE")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "gate_outcomes.png", dpi=220)
    plt.close(fig)
    return counts


def _distribution_plot(df: pd.DataFrame, out: Path) -> None:
    landed = df[_clean(df["status"]) == "landed"].copy()
    data = []
    labels = []
    for gate, label in (("on", "Gate ON"), ("off", "Gate OFF")):
        values = _num(landed[_clean(landed["gate"]) == gate], "contact_relative_speed_mean_m_s").dropna().to_numpy(float)
        if values.size:
            data.append(values)
            labels.append(label)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    rng = np.random.default_rng(20260922)
    for i, values in enumerate(data, start=1):
        jitter = rng.normal(0.0, 0.035, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=16, alpha=0.35)
    ax.set_ylabel("Mean relative-speed magnitude over final 0.10 s [m/s]")
    ax.set_title("Touchdown relative speed for runs that reached physical contact")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "contact_speed_gate.png", dpi=220)
    plt.close(fig)


def _component_distribution_plot(df: pd.DataFrame, out: Path) -> None:
    """Diagnostic ON/OFF distributions for the three mean relative-velocity components."""
    landed = df[_clean(df["status"]) == "landed"].copy()
    if landed.empty:
        return

    metrics = [
        ("contact_relative_velocity_x_mean_m_s", "x"),
        ("contact_relative_velocity_y_mean_m_s", "y"),
        ("contact_relative_velocity_z_mean_m_s", "z"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    rng = np.random.default_rng(20260922)
    for ax, (metric, axis_name) in zip(axes, metrics):
        plotted = []
        labels = []
        for gate, label in (("on", "Gate ON"), ("off", "Gate OFF")):
            values = _num(landed[_clean(landed["gate"]) == gate], metric).dropna().to_numpy(float)
            if values.size:
                plotted.append(values)
                labels.append(label)
        if plotted:
            ax.boxplot(plotted, showmeans=True)
            ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=15)
            for i, values in enumerate(plotted, start=1):
                jitter = rng.normal(0.0, 0.035, size=len(values))
                ax.scatter(np.full(len(values), i) + jitter, values, s=13, alpha=0.30)
        ax.axhline(0.0, linewidth=0.9, linestyle="--")
        ax.set_title(f"Mean relative $v_{axis_name}$")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean component over final 0.10 s [m/s]\n(platform axes when available)")
    fig.suptitle("Pre-contact relative-velocity components — landed runs only")
    fig.tight_layout()
    fig.savefig(out / "contact_velocity_components_gate.png", dpi=220)
    plt.close(fig)


def _position_plot(df: pd.DataFrame, out: Path) -> None:
    landed = df[_clean(df["status"]) == "landed"].copy()
    if landed.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharex=True, sharey=True)
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax, gate, title in zip(axes, ("on", "off"), ("Gate ON", "Gate OFF")):
        part = landed[_clean(landed["gate"]) == gate]
        x = _num(part, "contact_center_x_platform_m").to_numpy(float)
        y = _num(part, "contact_center_y_platform_m").to_numpy(float)
        radius = _num(part, "platform_radius_m").to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(radius) & (radius > 1e-9)
        if np.any(good):
            xn = x[good] / radius[good]
            yn = y[good] / radius[good]
            ax.scatter(xn, yn, s=24, alpha=0.55)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1.2, label="Deck edge")
        ax.axhline(0.0, linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, linewidth=0.8, alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x / platform radius")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y / platform radius")
    fig.suptitle("Touchdown centre position in platform coordinates — landed runs only")
    fig.tight_layout()
    fig.savefig(out / "contact_position_gate.png", dpi=220)
    plt.close(fig)


def _paired_plot(pairs: pd.DataFrame, out: Path) -> None:
    if pairs.empty:
        return
    both = pairs[pairs["both_landed"]].copy()
    if both.empty:
        return
    metrics = [
        ("contact_relative_speed_mean_m_s", "Relative-speed magnitude [m/s]"),
        ("contact_radial_error_normalized", "Radial touchdown error / platform radius"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (metric, ylabel) in zip(axes, metrics):
        on = pd.to_numeric(both[f"{metric}_on"], errors="coerce").to_numpy(float)
        off = pd.to_numeric(both[f"{metric}_off"], errors="coerce").to_numpy(float)
        good = np.isfinite(on) & np.isfinite(off)
        for a, b in zip(on[good], off[good]):
            ax.plot([0, 1], [b, a], alpha=0.20, linewidth=0.8)
        ax.scatter(np.zeros(np.sum(good)), off[good], s=15, alpha=0.45)
        ax.scatter(np.ones(np.sum(good)), on[good], s=15, alpha=0.45)
        ax.set_xticks([0, 1], ["Gate OFF", "Gate ON"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"Paired scenarios, both landed (n={np.sum(good)})")
    fig.tight_layout()
    fig.savefig(out / "paired_touchdown_differences.png", dpi=220)
    plt.close(fig)


def _write_text_summary(df: pd.DataFrame, pairs: pd.DataFrame, stats: pd.DataFrame, out: Path) -> None:
    lines = []
    lines.append("GATE CAMPAIGN SUMMARY")
    lines.append("=" * 72)
    lines.append("Outcome semantics: INFEASIBLE = gate refusal; ABORTED = controller/mission failure.")
    lines.append("")
    for gate in ("on", "off"):
        part = df[_clean(df["gate"]) == gate]
        lines.append(f"Gate {gate.upper()}: n={len(part)}")
        counts = _clean(part["status"]).value_counts()
        for status, n in counts.items():
            lines.append(f"  {status or 'missing':>16}: {n}")
    lines.append("")
    lines.append(f"Complete ON/OFF pairs: {len(pairs)}")
    if not pairs.empty:
        lines.append(f"  both landed:                 {int(pairs['both_landed'].sum())}")
        lines.append(f"  ON infeasible / OFF landed: {int(pairs['gate_refused_off_landed'].sum())}")
        lines.append(f"  ON infeasible / OFF crashed:{int(pairs['gate_refused_off_crashed'].sum())}")
        lines.append(f"  pair contains ABORTED:      {int(pairs['pair_has_abort'].sum())}")
        lines.append(f"  harness-failure pair:       {int(pairs['pair_has_harness_failure'].sum())}")
    lines.append("")
    lines.append("Paired touchdown metrics are computed only for scenarios where BOTH arms landed.")
    for _, row in stats.iterrows():
        lines.append("")
        lines.append(f"{row['metric']} (n={int(row['n_pairs'])})")
        lines.append(f"  mean ON / OFF: {row['mean_gate_on']:.6g} / {row['mean_gate_off']:.6g}")
        lines.append(
            "  mean ON-OFF:  "
            f"{row['mean_difference_on_minus_off']:.6g} "
            f"[95% bootstrap {row['mean_difference_ci95_low']:.6g}, {row['mean_difference_ci95_high']:.6g}]"
        )
        lines.append(
            "  median ON-OFF:"
            f" {row['median_difference_on_minus_off']:.6g} "
            f"[95% bootstrap {row['median_difference_ci95_low']:.6g}, {row['median_difference_ci95_high']:.6g}]"
        )
        lines.append(f"  paired dz:     {row['paired_effect_dz']:.6g}")
    (out / "gate_effect_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyse(runs_csv: Path, out: Path) -> None:
    df = pd.read_csv(runs_csv, low_memory=False)
    _require(
        df,
        (
            "run_id",
            "pair_key",
            "gate",
            "status",
            "contact_relative_speed_mean_m_s",
            "contact_relative_velocity_x_mean_m_s",
            "contact_relative_velocity_y_mean_m_s",
            "contact_relative_velocity_z_mean_m_s",
            "contact_center_x_platform_m",
            "contact_center_y_platform_m",
            "platform_radius_m",
        ),
    )
    df["gate"] = _clean(df["gate"])
    df["status"] = _clean(df["status"])
    out.mkdir(parents=True, exist_ok=True)

    _save_outcomes(df, out)
    pairs = _pair_table(df)
    pairs.to_csv(out / "gate_pair_table.csv", index=False)

    both = pairs[pairs["both_landed"]].copy() if not pairs.empty else pd.DataFrame()
    stats_rows = []
    for metric in (
        "contact_relative_speed_mean_m_s",
        "contact_relative_velocity_x_mean_m_s",
        "contact_relative_velocity_y_mean_m_s",
        "contact_relative_velocity_z_mean_m_s",
        "contact_radial_error_normalized",
    ):
        if not both.empty:
            stats_rows.append(
                _paired_stats(
                    pd.to_numeric(both[f"{metric}_on"], errors="coerce").to_numpy(float),
                    pd.to_numeric(both[f"{metric}_off"], errors="coerce").to_numpy(float),
                    metric,
                )
            )
        else:
            stats_rows.append(_paired_stats(np.array([]), np.array([]), metric))
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(out / "gate_effect_paired_statistics.csv", index=False)

    _distribution_plot(df, out)
    _component_distribution_plot(df, out)
    _position_plot(df, out)
    _paired_plot(pairs, out)
    _write_text_summary(df, pairs, stats, out)

    print(f"Analysed {len(df)} runs and {len(pairs)} complete ON/OFF pairs")
    print(f"Wrote article-level results to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_runs_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_csv = args.campaign_runs_csv.expanduser().resolve()
    if not runs_csv.is_file():
        raise SystemExit(f"File not found: {runs_csv}")
    out = args.output_dir.expanduser().resolve() if args.output_dir else runs_csv.parent / "gate_effect"
    try:
        analyse(runs_csv, out)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
