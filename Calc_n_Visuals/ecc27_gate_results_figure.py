"""ECC-style campaign figure: touchdown velocity + paired refusal consequence."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COMMIT = "#009E73"   # matches phase_machine / standalone figure
REFUSE = "#D55E00"   # matches phase_machine / standalone figure
NEUTRAL = "#4D4D4D"
LIGHT = "#F2F2F2"
TEXT = "#1A1A1A"


def clean(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def num(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)


def style_for_ecc() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def add_box(ax, values, pos, edge, width=0.24, hatch=None):
    if len(values) == 0:
        return
    bp = ax.boxplot(
        [values], positions=[pos], widths=width, patch_artist=True,
        showfliers=False, whis=1.5,
        boxprops={"facecolor": LIGHT, "edgecolor": edge, "linewidth": 0.8, "hatch": hatch},
        whiskerprops={"color": edge, "linewidth": 0.75},
        capprops={"color": edge, "linewidth": 0.75},
        medianprops={"color": edge, "linewidth": 1.15},
    )
    return bp


def add_points(ax, values, pos, color, rng, spread=0.045, marker="o"):

    if len(values) == 0:
        return
    jitter = rng.normal(0.0, spread, size=len(values))
    kwargs = dict(
        s=8, marker=marker, linewidths=0.55, alpha=0.48, zorder=2
    )
    if marker == "x":
        kwargs["color"] = color
    else:
        kwargs["facecolors"] = "none"
        kwargs["edgecolors"] = color
    ax.scatter(np.full(len(values), pos) + jitter, values, **kwargs)


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby("pair_key", dropna=False):
        arms = {gate: part.iloc[0] for gate, part in g.groupby("gate") if gate in {"on", "off"}}
        if "on" not in arms or "off" not in arms:
            continue
        on, off = arms["on"], arms["off"]
        rows.append({
            "pair_key": key,
            "status_on": on["status"],
            "status_off": off["status"],
            "vz_off": pd.to_numeric(pd.Series([off.get("contact_relative_velocity_z_abs_mean_m_s")]), errors="coerce").iloc[0],
            "speed_off": pd.to_numeric(pd.Series([off.get("contact_relative_speed_mean_m_s")]), errors="coerce").iloc[0],
        })
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, output_base: Path) -> None:
    style_for_ecc()
    df = df.copy()
    df["gate"] = clean(df["gate"])
    df["status"] = clean(df["status"])
    landed = df[df["status"] == "landed"].copy()

    metrics_a = [
        ("contact_relative_velocity_x_abs_mean_m_s", r"$|v_x|$"),
        ("contact_relative_velocity_y_abs_mean_m_s", r"$|v_y|$"),
        ("contact_relative_velocity_z_abs_mean_m_s", r"$|v_z|$"),
        ("contact_relative_speed_mean_m_s", r"$\|\mathbf{v}\|$"),
    ]

    pairs = pair_table(df)
    paired_touch = pairs[pairs["status_off"] == "landed"].copy()
    accepted = paired_touch[paired_touch["status_on"] == "landed"]
    refused = paired_touch[paired_touch["status_on"] == "infeasible"]
    refused_off_aborted = int(((pairs["status_on"] == "infeasible") & (pairs["status_off"] == "aborted")).sum())

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.62), gridspec_kw={"width_ratios": [1.17, 0.83]})
    rng = np.random.default_rng(20260924)

    # (a) Landed-run velocity distributions: gate ON vs OFF.
    ax = axes[0]
    base = np.arange(len(metrics_a), dtype=float)
    offset = 0.155
    for j, (metric, _) in enumerate(metrics_a):
        onv = num(landed[landed["gate"] == "on"], metric)
        offv = num(landed[landed["gate"] == "off"], metric)
        add_box(ax, onv, base[j] - offset, COMMIT)
        add_points(ax, onv, base[j] - offset, COMMIT, rng, marker="o")
        add_box(ax, offv, base[j] + offset, NEUTRAL, hatch="///")
        add_points(ax, offv, base[j] + offset, NEUTRAL, rng, marker="x")
    ax.set_xticks(base, [label for _, label in metrics_a])
    ax.set_ylabel(r"Mean over final 0.10 s [m s$^{-1}$]")
    ax.set_ylim(0.0, 0.205)
    ax.set_title("(a) Touchdown velocity, landed runs", loc="left", pad=3.5, fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    n_on = int(((landed["gate"] == "on")).sum())
    n_off = int(((landed["gate"] == "off")).sum())
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=COMMIT,
                   markersize=4.2, label=f"Gate ON (n={n_on})"),
            Line2D([0], [0], marker="x", linestyle="none", markerfacecolor="none", markeredgecolor=NEUTRAL,
                   markersize=4.2, label=f"Gate OFF (n={n_off})"),
        ],
        loc="upper left", frameon=False, ncol=1, handletextpad=0.4, borderaxespad=0.25,
    )

    # (b) Counterfactual: Gate-OFF touchdown grouped by what Gate-ON decided.
    ax = axes[1]
    metrics_b = [("vz_off", r"$|v_z|$"), ("speed_off", r"$\|\mathbf{v}\|$")]
    base = np.arange(len(metrics_b), dtype=float)
    for j, (metric, _) in enumerate(metrics_b):
        a = pd.to_numeric(accepted[metric], errors="coerce").dropna().to_numpy(float)
        r = pd.to_numeric(refused[metric], errors="coerce").dropna().to_numpy(float)
        add_box(ax, a, base[j] - offset, COMMIT)
        add_points(ax, a, base[j] - offset, COMMIT, rng, marker="o")
        add_box(ax, r, base[j] + offset, REFUSE, hatch="///")
        add_points(ax, r, base[j] + offset, REFUSE, rng, marker="x")
    ax.set_xticks(base, [label for _, label in metrics_b])
    ax.set_ylabel(r"Gate-OFF mean over final 0.10 s [m s$^{-1}$]")
    ax.set_ylim(0.0, 0.155)
    ax.set_title("(b) Paired consequence of the ON decision", loc="left", pad=3.5, fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=COMMIT,
                   markersize=4.2, label=f"ON landed (n={len(accepted)})"),
            Line2D([0], [0], marker="x", linestyle="none", markerfacecolor="none", markeredgecolor=REFUSE,
                   markersize=4.2, label=f"ON refused (n={len(refused)})"),
        ],
        loc="upper left", frameon=False, handletextpad=0.4, borderaxespad=0.25,
    )
    if refused_off_aborted:
        ax.text(
            0.98, 0.03,
            f"{refused_off_aborted} refused pairs: OFF aborted\n(no touchdown sample)",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=6.4, color=TEXT,
        )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    fig.subplots_adjust(left=0.085, right=0.995, top=0.91, bottom=0.19, wspace=0.28)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=400)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("campaign_runs_csv", type=Path)
    p.add_argument("--output", type=Path, default=Path("ecc27_gate_results"))
    args = p.parse_args()
    df = pd.read_csv(args.campaign_runs_csv, low_memory=False)
    make_figure(df, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
