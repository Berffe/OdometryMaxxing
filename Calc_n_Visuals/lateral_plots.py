"""Lateral optical-flow analysis for the FINAL_PROBE roll controller.

The landing target has already been centered during CENTER. In FINAL_PROBE,
the roll controller regulates only the lateral translational optical flow

    lambda = (v_platform - v_drone) / h

toward zero. With the small-angle approximation a_drone = g * phi and
phi_cmd = k_D * lambda, the acceleration-equivalent gain is K = g * k_D.

The continuous relative-velocity dynamics are exactly discretized with a
zero-order hold. The script produces only:
    1. lateral optical-flow residual frequency response;
    2. drone/platform synchronization a_d / a_p;
    3. discrete closed-loop root locus.

The pitch-axis analysis is directly analogous after the appropriate sign
convention is applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LateralConfig:
    # Frozen height during FINAL_PROBE.
    height_m: float = 0.18

    # Roll D gains: phi_cmd = k_D * lambda.
    # Units: rad / (1/s) = rad*s.
    roll_kd_values: tuple[float, ...] = (0.10, 0.30, 0.80)

    # Effective visual/control update interval.
    sample_time_s: float = 0.06
    gravity_m_s2: float = 9.81

    # Lateral platform oscillation.
    platform_frequency_hz: float = 0.4
    platform_amplitude_m: float = 0.10

    # Frequency grid.
    f_min_hz: float = 0.01
    n_frequency_points: int = 2500

    # <= 0 selects 1.15 * max(configured gains, critical roll gain).
    root_locus_gain_max: float = 0.0
    root_locus_points: int = 1500


def frequency_grid(cfg: LateralConfig) -> np.ndarray:
    f_nyquist = 0.5 / cfg.sample_time_s
    return np.logspace(
        math.log10(cfg.f_min_hz),
        math.log10(0.98 * f_nyquist),
        cfg.n_frequency_points,
    )


def lateral_flow_plant_response(
    cfg: LateralConfig,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Return P_lambda(z) from residual acceleration to lateral optical flow.

    Frozen-height model:
        lambda[k+1] = lambda[k] + T/h * (a_p[k] - a_d[k])

    Therefore:
        P_lambda(z) = (T/h) / (z - 1)
    """
    T = cfg.sample_time_s
    h = cfg.height_m
    z = np.exp(1j * 2.0 * math.pi * frequencies_hz * T)
    return (T / h) / (z - 1.0)


def acceleration_gain(cfg: LateralConfig, roll_kd: float) -> float:
    """Convert the roll-angle D gain to acceleration-equivalent gain K."""
    return cfg.gravity_m_s2 * float(roll_kd)


def critical_roll_gain(cfg: LateralConfig) -> float:
    """Discrete stability ceiling k_D,crit = 2h/(gT)."""
    return 2.0 * cfg.height_m / (
        cfg.gravity_m_s2 * cfg.sample_time_s
    )


def closed_loop_pole(cfg: LateralConfig, roll_kd: float) -> complex:
    """Pole of a_d/a_p for the D-only sampled lateral optical-flow loop."""
    mu = (
        cfg.gravity_m_s2
        * float(roll_kd)
        * cfg.sample_time_s
        / cfg.height_m
    )
    return complex(1.0 - mu, 0.0)


def compute_for_gain(cfg: LateralConfig, roll_kd: float) -> dict:
    frequencies_hz = frequency_grid(cfg)
    P_lambda = lateral_flow_plant_response(cfg, frequencies_hz)

    K = acceleration_gain(cfg, roll_kd)
    L = K * P_lambda
    pole = closed_loop_pole(cfg, roll_kd)

    return {
        "f": frequencies_hz,
        "kd": float(roll_kd),
        "K": K,
        "mu": K * cfg.sample_time_s / cfg.height_m,
        "T_platform": L / (1.0 + L),       # a_d / a_p
        "H_lambda_ap": P_lambda / (1.0 + L),  # lambda / a_p
        "pole": pole,
        "stable": bool(abs(pole) < 1.0 - 1e-10),
    }


def complex_at_frequency(f: np.ndarray, H: np.ndarray, f0: float) -> complex:
    """Interpolate a complex response on the logarithmic frequency axis."""
    x, x0 = np.log(f), math.log(f0)
    return complex(
        np.interp(x0, x, np.real(H)),
        np.interp(x0, x, np.imag(H)),
    )


def platform_acceleration_amplitude(cfg: LateralConfig) -> float:
    omega = 2.0 * math.pi * cfg.platform_frequency_hz
    return cfg.platform_amplitude_m * omega * omega


def magnitude_db(H: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))


def phase_deg(H: np.ndarray) -> np.ndarray:
    phase = np.rad2deg(np.angle(H))
    return (phase + 180.0) % 360.0 - 180.0


def curve_style(response: dict) -> tuple[str, str]:
    suffix = "" if response["stable"] else " (instable)"
    linestyle = "-" if response["stable"] else "--"
    return f"k_D={response['kd']:g}{suffix}", linestyle


def mark_platform_frequency(ax, cfg: LateralConfig) -> None:
    ax.axvline(
        cfg.platform_frequency_hz,
        linestyle=":",
        linewidth=1.3,
        label="fréquence de la plateforme",
    )


# --------------------------------------------------------------------------- #
# 1. Lateral optical-flow residual
# --------------------------------------------------------------------------- #


def plot_lateral_flow_residual(
    cfg: LateralConfig,
    responses: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(
        "Résiduel de flux optique latéral causé par les oscillations "
        "de la plateforme\n"
        r"$e_\lambda=\lambda-\lambda_{\mathrm{ref}},\quad "
        r"\lambda_{\mathrm{ref}}=0$"
    )

    for response in responses:
        f = response["f"]
        H_lambda_ap = response["H_lambda_ap"]
        H_lambda_xp = -(2.0 * math.pi * f) ** 2 * H_lambda_ap
        label, linestyle = curve_style(response)

        axes[0, 0].loglog(
            f,
            np.maximum(np.abs(H_lambda_xp), 1e-12),
            linestyle=linestyle,
            label=label,
        )
        axes[1, 0].semilogx(
            f,
            phase_deg(H_lambda_xp),
            linestyle=linestyle,
            label=label,
        )
        axes[0, 1].loglog(
            f,
            np.maximum(np.abs(H_lambda_ap), 1e-12),
            linestyle=linestyle,
            label=label,
        )
        axes[1, 1].semilogx(
            f,
            phase_deg(H_lambda_ap),
            linestyle=linestyle,
            label=label,
        )

        f0 = cfg.platform_frequency_hz
        H_xp_0 = complex_at_frequency(f, H_lambda_xp, f0)
        H_ap_0 = complex_at_frequency(f, H_lambda_ap, f0)
        residual_amp = abs(H_xp_0) * cfg.platform_amplitude_m

        axes[0, 0].scatter([f0], [abs(H_xp_0)], s=35)
        axes[0, 1].scatter([f0], [abs(H_ap_0)], s=35)
        axes[1, 0].scatter(
            [f0], phase_deg(np.array([H_xp_0])), s=35,
        )
        axes[1, 1].scatter(
            [f0], phase_deg(np.array([H_ap_0])), s=35,
        )

        for ax, value in (
            (axes[0, 0], H_xp_0),
            (axes[0, 1], H_ap_0),
        ):
            ax.annotate(
                f"{residual_amp:.3f} s$^{{-1}}$",
                xy=(f0, abs(value)),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
            )

    for ax in axes.flat:
        mark_platform_frequency(ax, cfg)
        ax.grid(True, which="both")
        ax.legend(loc="best")

    axes[0, 0].set_title(r"Entrée position : $e_\lambda/x_p$")
    axes[0, 0].set_ylabel(
        r"$|e_\lambda/x_p|$ [s$^{-1}$/m]"
    )
    axes[1, 0].set_ylabel("phase [deg]")
    axes[1, 0].set_xlabel("fréquence [Hz]")

    axes[0, 1].set_title(r"Entrée accélération : $e_\lambda/a_p$")
    axes[0, 1].set_ylabel(r"$|e_\lambda/a_p|$ [s/m]")
    axes[1, 1].set_ylabel("phase [deg]")
    axes[1, 1].set_xlabel("fréquence [Hz]")

    for ax in axes[1, :]:
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axhline(-90.0, linestyle=":", linewidth=1)
        ax.axhline(90.0, linestyle=":", linewidth=1)

    fig.tight_layout()


# --------------------------------------------------------------------------- #
# 2. Drone/platform synchronization
# --------------------------------------------------------------------------- #


def plot_platform_synchronization(
    cfg: LateralConfig,
    responses: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        "Synchronisation latérale drone–plateforme\n"
        r"$T_p=a_d/a_p=v_d/v_p=x_d/x_p=L/(1+L)$"
    )

    for response in responses:
        f = response["f"]
        T_platform = response["T_platform"]
        label, linestyle = curve_style(response)

        axes[0].semilogx(
            f,
            magnitude_db(T_platform),
            linestyle=linestyle,
            label=label,
        )
        axes[1].semilogx(
            f,
            phase_deg(T_platform),
            linestyle=linestyle,
            label=label,
        )

        f0 = cfg.platform_frequency_hz
        T0 = complex_at_frequency(f, T_platform, f0)
        mag0_db = magnitude_db(np.array([T0]))[0]
        phase0 = phase_deg(np.array([T0]))[0]

        axes[0].scatter([f0], [mag0_db], s=35)
        axes[1].scatter([f0], [phase0], s=35)
        axes[0].annotate(
            f"{abs(T0):.3f}",
            xy=(f0, mag0_db),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
        axes[1].annotate(
            f"{phase0:.1f}°",
            xy=(f0, phase0),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )

    for ax in axes:
        mark_platform_frequency(ax, cfg)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.grid(True, which="both")
        ax.legend(loc="best")

    axes[0].set_ylabel(r"$|a_d/a_p|$ [dB]")
    axes[1].set_ylabel("phase [deg]")
    axes[1].set_xlabel("fréquence [Hz]")
    fig.tight_layout()


# --------------------------------------------------------------------------- #
# 3. Discrete root locus
# --------------------------------------------------------------------------- #


def root_locus_data(cfg: LateralConfig):
    kd_critical = critical_roll_gain(cfg)
    gain_max = cfg.root_locus_gain_max
    if gain_max <= 0.0:
        gain_max = 1.15 * max(max(cfg.roll_kd_values), kd_critical)

    gains = np.linspace(0.0, gain_max, cfg.root_locus_points)
    poles = np.array([closed_loop_pole(cfg, kd) for kd in gains])
    return gains, poles, kd_critical


def plot_root_locus(cfg: LateralConfig) -> None:
    _, poles, kd_critical = root_locus_data(cfg)
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    fig.suptitle(
        "Lieu discret du pôle en boucle fermée de "
        r"$T_p=a_d/a_p$ lorsque $k_D$ varie"
    )

    ax.plot(np.real(poles), np.imag(poles), ".", ms=2)
    ax.scatter(
        [1.0], [0.0], marker="x", s=80, label=r"pôle pour $k_D=0$",
    )

    for kd in cfg.roll_kd_values:
        pole = closed_loop_pole(cfg, kd)
        state = "" if abs(pole) < 1.0 - 1e-10 else " (instable)"
        ax.scatter(
            [np.real(pole)],
            [np.imag(pole)],
            s=38,
            label=fr"$k_D={kd:g}${state}",
        )

    theta = np.linspace(0.0, 2.0 * math.pi, 600)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        linestyle="--",
        linewidth=1,
        label="cercle unité",
    )
    ax.scatter(
        [-1.0],
        [0.0],
        marker="x",
        s=90,
        label=r"$k_{D,crit}=2h/(gT)$",
    )
    ax.annotate(
        f"kD,crit={kd_critical:.3g}",
        xy=(-1.0, 0.0),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
    )

    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_title("Modèle discret ZOH du contrôleur D latéral")
    ax.set_xlabel(r"Re$(z)$")
    ax.set_ylabel(r"Im$(z)$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()


def print_summary(cfg: LateralConfig, responses: list[dict]) -> None:
    kd_critical = critical_roll_gain(cfg)
    a_platform = platform_acceleration_amplitude(cfg)

    print("\n=== Lateral optical-flow analysis: roll axis ===")
    print(
        f"h={cfg.height_m:.4f} m, T={cfg.sample_time_s:.4f} s, "
        f"g={cfg.gravity_m_s2:.4f} m/s², "
        f"kDcrit={kd_critical:.4f} rad*s"
    )
    print(
        f"platform: A={cfg.platform_amplitude_m:.4f} m, "
        f"f={cfg.platform_frequency_hz:.4f} Hz, "
        f"|a_p|={a_platform:.4f} m/s²\n"
    )

    for response in responses:
        T0 = complex_at_frequency(
            response["f"],
            response["T_platform"],
            cfg.platform_frequency_hz,
        )
        H0 = complex_at_frequency(
            response["f"],
            response["H_lambda_ap"],
            cfg.platform_frequency_hz,
        )
        state = "stable" if response["stable"] else "UNSTABLE"
        print(
            f"kD={response['kd']:g}: {state}, "
            f"K=g*kD={response['K']:.4f} m/s, "
            f"mu={response['mu']:.4f}, "
            f"pole={response['pole'].real:.4f}, "
            f"|a_d/a_p|={abs(T0):.4f}, "
            f"phase={np.angle(T0, deg=True):.2f} deg, "
            f"|e_lambda|={abs(H0) * a_platform:.4f} 1/s"
        )


def run_analysis(cfg: LateralConfig) -> None:
    responses = [compute_for_gain(cfg, kd) for kd in cfg.roll_kd_values]
    print_summary(cfg, responses)
    plot_lateral_flow_residual(cfg, responses)
    plot_platform_synchronization(cfg, responses)
    plot_root_locus(cfg)
    plt.show()


if __name__ == "__main__":
    configuration = LateralConfig(
        height_m=0.18,
        roll_kd_values=(0.10, 0.30, 0.80),
        sample_time_s=0.06,
        platform_frequency_hz=0.4,
        platform_amplitude_m=0.10,
    )
    run_analysis(configuration)
