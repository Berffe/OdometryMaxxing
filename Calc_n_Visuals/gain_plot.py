"""Figure 1 - admissible gain region against relative height.

Two panels sharing the height axis: a calm platform whose probed floor leaves
the region non-empty at gear height, and a severe one whose floor crosses the
ceiling above it. The contribution is legible from the figure alone.

Both ceilings are straight lines through the origin, and so is the schedule:
with k = k_exp exp(-w t) and h = h0 exp(-w t), the scheduled gain is
k = (k_exp / h0) h until it clamps. That is why it stays inside the wedge.
"""

import numpy as np
import matplotlib.pyplot as plt

import ieee_figure_style as style

# --- Configuration. Replace with the values used in the campaign. ----------
T_SAMPLE_S = 0.077          # controller sampling interval
H_GEAR_M = 0.18             # landing-gear height
H0_M = 3.0                  # height at which the schedule starts
K_EXPLORE = 6.5             # initial exploratory gain
CEILING_MARGIN = 0.8        # contraction factor s < 1

D_MAX_CALM = 1.0            # probed floor, calm platform
D_MAX_SEVERE = 6.0          # probed floor, severe platform

H_MAX_M = 0.5
K_MAX_PLOT = 8.0

# Okabe-Ito: distinguishable in colour and separated in greyscale value.
C_STABILITY = "#333333"
C_NO_OSCILLATION = "#0072B2"
C_STABLE = "#B2006B"
C_FLOOR = "#D55E00"
C_SCHEDULE = "#009E73"
C_WEDGE = "#DCE9F2"


def ceilings(h):
    """Stability ceiling 2h/T and the stricter no-oscillation ceiling h/T."""
    return 2.0 * h / T_SAMPLE_S, h / T_SAMPLE_S


def scheduled_gain(h, k_floor):
    """k = (k_exp / h0) h, clamped from below at the asymptote."""
    return np.maximum(K_EXPLORE * h / H0_M, k_floor)


def draw_panel(ax, k_min, title, show_ylabel, show_schedule=True):
    h = np.linspace(0, H_MAX_M, 400)
    k_stability, k_no_osc = ceilings(h)
    k_floor_line = np.full_like(h, k_min)

    admissible = np.minimum(k_no_osc, K_MAX_PLOT)
    ax.fill_between(h, k_floor_line, admissible,
                    where=admissible > k_floor_line,
                    color=C_WEDGE, linewidth=0, zorder=0)

    ax.plot(h, k_stability, color=C_STABILITY, linestyle="-", zorder=3)
    ax.plot(h, k_no_osc, color=C_NO_OSCILLATION, linestyle="--", zorder=3)
    ax.plot(h, k_floor_line, color=C_FLOOR, linestyle="-", zorder=3)

    # Only drawn where a descent is actually committed to.
    if show_schedule:
        k_floor_asymptote = max(CEILING_MARGIN * H_GEAR_M / T_SAMPLE_S, k_min)
        ax.plot(h, scheduled_gain(h, k_floor_asymptote),
                color=C_SCHEDULE, linestyle="-.", linewidth=0.9, zorder=4)

    ax.axvline(H_GEAR_M, color="0.45", linestyle=":", linewidth=0.8, zorder=2)

    # The decision: is the region still non-empty at gear height?
    k_gear = H_GEAR_M / T_SAMPLE_S
    feasible = k_min <= k_gear
    ax.plot(H_GEAR_M, k_gear, marker="o", markersize=3.5,
            color=C_NO_OSCILLATION, zorder=5)
    ax.plot(H_GEAR_M, k_min, marker="o" if feasible else "x", markersize=3.5,
            color=C_FLOOR, zorder=5)
    if feasible:
        ax.annotate("", xy=(H_GEAR_M, k_gear), xytext=(H_GEAR_M, k_min),
                    arrowprops=dict(arrowstyle="<->", color="0.25",
                                    linewidth=0.7, shrinkA=2, shrinkB=2),
                    zorder=6)

    ax.set_xlim(0, H_MAX_M)
    ax.set_ylim(0, K_MAX_PLOT)
    ax.set_xlabel(r"Relative height $h$ [m]")
    if show_ylabel:
        ax.set_ylabel(r"Vertical gain $K_z$ [m/s]")
    else:
        ax.tick_params(labelleft=False)

    ax.grid(True, axis="y", color="0.9", linewidth=0.4)
    ax.set_axisbelow(True)
    # ax.text(0.04, 0.93, title, transform=ax.transAxes,
    #         fontsize=style.BASE_FONT_PT - 1, va="top")


def main() -> None:
    style.use_ieee_style()
    width = style.COLUMN_WIDTH_IN
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.46),
                             sharey=True)

    draw_panel(axes[0], D_MAX_CALM, "(a) afforded", show_ylabel=True, show_schedule=False)
    draw_panel(axes[1], D_MAX_SEVERE, "(b) refused", show_ylabel=False,
               show_schedule=False)

    # Inline labels instead of a legend: at this width a legend box costs
    # more area than the curves it explains. Positions are chosen to sit in
    # gaps between curves; re-check them if the configuration changes.
    small = style.BASE_FONT_PT - 1.5
    axes[0].text(0.205, 7.15, r"$K^{\max}$", color=C_STABILITY, fontsize=small)
    axes[0].text(0.315, 2.0, "no osc.", color=C_NO_OSCILLATION, fontsize=small)
    axes[0].text(0.25, 6.1, "stable w/ osc.", color=C_STABLE, fontsize=small)
    axes[0].text(0.315, 0.35, r"$K^{\min}$", color=C_FLOOR, fontsize=small)
    # axes[0].text(0.275, 2.15, "schedule", color=C_SCHEDULE, fontsize=small)
    axes[0].text(0.196, 1.22, r"$\rho\leq1$", color="0.25", fontsize=small)
    axes[1].text(0.195, 0.35, r"$h_{\rm gear}$", color="0.35", fontsize=small)
    axes[1].text(0.245, 6.25, "region empty", color=C_FLOOR, fontsize=small)

    fig.get_layout_engine().set(wspace=0.02)
    plt.show()
    style.save(fig, "gain_window.pdf")


if __name__ == "__main__":
    main()