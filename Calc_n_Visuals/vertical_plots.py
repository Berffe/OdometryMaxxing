"""Vertical optical-flow analysis for a sampled divergence controller.

The continuous relative dynamics are exactly discretized with a zero-order
hold. The script produces only:
	1. divergence-residual frequency response;
	2. drone/platform synchronization a_d / a_p;
	3. discrete closed-loop root locus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FrequencyConfig:
	height_m: float = 2.0
	divergence_op_1_s: float = 0.0
	gain_values: tuple[float, ...] = (7.5, 10.0)
	sample_time_s: float = 0.05

	platform_frequency_hz: float = 0.4
	platform_amplitude_m: float = 0.10

	f_min_hz: float = 0.01
	n_frequency_points: int = 2500

	# <= 0 selects 1.15 * max(configured gains, de Croon critical gain).
	root_locus_gain_max: float = 0.0
	root_locus_points: int = 1500


def model_matrices(cfg: FrequencyConfig):
	"""Frozen continuous model and its exact ZOH discretization.

	x = [delta_h, delta_c]^T,  c = -h_dot,  D = c/h.
	The disturbance channel is a_p-a_d and a_d = K*delta_D.
	"""
	h = max(float(cfg.height_m), 1e-9)
	D0 = float(cfg.divergence_op_1_s)
	T = max(float(cfg.sample_time_s), 1e-9)

	A = np.array([[0.0, -1.0], [0.0, 0.0]])
	B = np.array([[0.0], [1.0]])
	F = np.array([[1.0, -T], [0.0, 1.0]])
	G = np.array([[-0.5 * T * T], [T]])
	C = np.array([[-D0 / h, 1.0 / h]])
	return A, B, F, G, C


def frequency_grid(cfg: FrequencyConfig) -> np.ndarray:
	f_nyquist = 0.5 / cfg.sample_time_s
	return np.logspace(
		math.log10(cfg.f_min_hz),
		math.log10(0.98 * f_nyquist),
		cfg.n_frequency_points,
	)


def discrete_response(
	F: np.ndarray,
	G: np.ndarray,
	C: np.ndarray,
	frequencies_hz: np.ndarray,
	sample_time_s: float,
) -> np.ndarray:
	"""Evaluate C(zI-F)^(-1)G at z=exp(j 2*pi*f*T)."""
	identity = np.eye(F.shape[0])
	response = np.empty(frequencies_hz.shape, dtype=complex)

	for i, frequency_hz in enumerate(frequencies_hz):
		z = np.exp(1j * 2.0 * math.pi * frequency_hz * sample_time_s)
		response[i] = (C @ np.linalg.solve(z * identity - F, G))[0, 0]

	return response


def discrete_transfer_poles(cfg: FrequencyConfig, gain_k: float) -> np.ndarray:
	"""Poles of T_p(z)=K P_D(z)/(1+K P_D(z))."""
	h = cfg.height_m
	D0 = cfg.divergence_op_1_s
	T = cfg.sample_time_s
	K = float(gain_k)

	# At D0=0, the neutral-height mode cancels from a_d/a_p.
	if abs(D0) < 1e-12:
		return np.array([1.0 - K * T / h], dtype=complex)

	a = 1.0 + 0.5 * D0 * T
	b = 1.0 - 0.5 * D0 * T
	return np.roots(
		[
			1.0,
			-2.0 + K * T * a / h,
			1.0 - K * T * b / h,
		]
	).astype(complex)


def compute_for_gain(cfg: FrequencyConfig, gain_k: float) -> dict:
	frequencies_hz = frequency_grid(cfg)
	_, _, F, G, C = model_matrices(cfg)

	# P_D maps residual relative acceleration (a_p-a_d) to divergence.
	P_D = discrete_response(F, G, C, frequencies_hz, cfg.sample_time_s)
	L = float(gain_k) * P_D

	poles = discrete_transfer_poles(cfg, gain_k)
	stable = bool(np.all(np.abs(poles) < 1.0 - 1e-10))

	return {
		"f": frequencies_hz,
		"K": float(gain_k),
		"T_platform": L / (1.0 + L),        # a_d/a_p
		"H_eD_ap": P_D / (1.0 + L),         # e_D/a_p
		"poles": poles,
		"stable": stable,
	}


def complex_at_frequency(f: np.ndarray, H: np.ndarray, f0: float) -> complex:
	"""Interpolate on the logarithmic frequency axis."""
	x, x0 = np.log(f), math.log(f0)
	return complex(
		np.interp(x0, x, np.real(H)),
		np.interp(x0, x, np.imag(H)),
	)


def platform_acceleration_amplitude(cfg: FrequencyConfig) -> float:
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
	return f"K={response['K']:g}{suffix}", linestyle


def mark_platform_frequency(ax, cfg: FrequencyConfig) -> None:
	ax.axvline(
		cfg.platform_frequency_hz,
		linestyle=":",
		linewidth=1.3,
		label="fréquence de la plateforme",
	)


# --------------------------------------------------------------------------- #
# 1. Divergence residual
# --------------------------------------------------------------------------- #


def plot_divergence_residual(cfg: FrequencyConfig, responses: list[dict]) -> None:
	fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
	fig.suptitle(
		"Résiduel de divergence causé par les oscillations de la plateforme\n"
		r"$e_D=D_{\mathrm{obs}}-D_{\mathrm{ref}}$"
	)

	for response in responses:
		f = response["f"]
		H_eD_ap = response["H_eD_ap"]
		H_eD_zp = -(2.0 * math.pi * f) ** 2 * H_eD_ap
		label, linestyle = curve_style(response)

		axes[0, 0].loglog(
			f, np.maximum(np.abs(H_eD_zp), 1e-12),
			linestyle=linestyle, label=label,
		)
		axes[1, 0].semilogx(
			f, phase_deg(H_eD_zp), linestyle=linestyle, label=label,
		)
		axes[0, 1].loglog(
			f, np.maximum(np.abs(H_eD_ap), 1e-12),
			linestyle=linestyle, label=label,
		)
		axes[1, 1].semilogx(
			f, phase_deg(H_eD_ap), linestyle=linestyle, label=label,
		)

		f0 = cfg.platform_frequency_hz
		H_zp_0 = complex_at_frequency(f, H_eD_zp, f0)
		H_ap_0 = complex_at_frequency(f, H_eD_ap, f0)
		residual_amp = abs(H_zp_0) * cfg.platform_amplitude_m

		for ax, value in (
			(axes[0, 0], H_zp_0),
			(axes[0, 1], H_ap_0),
		):
			ax.scatter([f0], [abs(value)], s=35)
			ax.annotate(
				f"{residual_amp:.3f} s$^{{-1}}$",
				xy=(f0, abs(value)),
				xytext=(8, 8),
				textcoords="offset points",
				fontsize=8,
			)

		axes[1, 0].scatter([f0], phase_deg(np.array([H_zp_0])), s=35)
		axes[1, 1].scatter([f0], phase_deg(np.array([H_ap_0])), s=35)

	for ax in axes.flat:
		mark_platform_frequency(ax, cfg)
		ax.grid(True, which="both")
		ax.legend(loc="best")

	axes[0, 0].set_title(r"Entrée position : $e_D/z_p$")
	axes[0, 0].set_ylabel(r"$|e_D/z_p|$ [s$^{-1}$/m]")
	axes[1, 0].set_ylabel("phase [deg]")
	axes[1, 0].set_xlabel("fréquence [Hz]")

	axes[0, 1].set_title(r"Entrée accélération : $e_D/a_p$")
	axes[0, 1].set_ylabel(r"$|e_D/a_p|$ [s/m]")
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
	cfg: FrequencyConfig,
	responses: list[dict],
) -> None:
	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
	fig.suptitle(
		"Synchronisation drone–plateforme\n"
		r"$T_p=a_d/a_p=z_d/z_p=L/(1+L)$"
	)

	for response in responses:
		f = response["f"]
		T_platform = response["T_platform"]
		label, linestyle = curve_style(response)

		axes[0].semilogx(
			f, magnitude_db(T_platform), linestyle=linestyle, label=label,
		)
		axes[1].semilogx(
			f, phase_deg(T_platform), linestyle=linestyle, label=label,
		)

		f0 = cfg.platform_frequency_hz
		T0 = complex_at_frequency(f, T_platform, f0)
		mag0_db = magnitude_db(np.array([T0]))[0]
		phase0 = phase_deg(np.array([T0]))[0]

		axes[0].scatter([f0], [mag0_db], s=35)
		axes[1].scatter([f0], [phase0], s=35)
		axes[0].annotate(
			f"{abs(T0):.3f}", xy=(f0, mag0_db), xytext=(8, 8),
			textcoords="offset points", fontsize=8,
		)
		axes[1].annotate(
			f"{phase0:.1f}°", xy=(f0, phase0), xytext=(8, 8),
			textcoords="offset points", fontsize=8,
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


def root_locus_data(cfg: FrequencyConfig):
	critical_gain = 2.0 * cfg.height_m / cfg.sample_time_s
	gain_max = cfg.root_locus_gain_max
	if gain_max <= 0.0:
		gain_max = 1.15 * max(max(cfg.gain_values), critical_gain)

	gains = np.linspace(0.0, gain_max, cfg.root_locus_points)
	poles = [discrete_transfer_poles(cfg, K) for K in gains]
	return gains, np.asarray(poles), critical_gain


def plot_root_locus(cfg: FrequencyConfig) -> None:
	_, poles, critical_gain = root_locus_data(cfg)
	fig, ax = plt.subplots(figsize=(7.5, 7.0))
	fig.suptitle(
		"Lieu discret des pôles en boucle fermée de "
		r"$T_p=a_d/a_p$ lorsque $K$ varie"
	)

	# A point cloud avoids numerical branch-switching artifacts.
	ax.plot(np.real(poles).ravel(), np.imag(poles).ravel(), ".", ms=2)

	poles_0 = discrete_transfer_poles(cfg, 0.0)
	ax.scatter(
		np.real(poles_0), np.imag(poles_0), marker="x", s=80,
		label="pôles pour K=0",
	)

	for K in cfg.gain_values:
		poles_K = discrete_transfer_poles(cfg, K)
		state = "" if np.all(np.abs(poles_K) < 1.0 - 1e-10) else " (instable)"
		ax.scatter(
			np.real(poles_K), np.imag(poles_K), s=38,
			label=f"K={K:g}{state}",
		)

	theta = np.linspace(0.0, 2.0 * math.pi, 600)
	ax.plot(
		np.cos(theta), np.sin(theta), linestyle="--", linewidth=1,
		label="cercle unité",
	)
	ax.scatter(
		[-1.0], [0.0], marker="x", s=90, label=r"$K_{crit}=2h/T$",
	)
	ax.annotate(
		f"Kcrit={critical_gain:.3g}", xy=(-1.0, 0.0), xytext=(8, 8),
		textcoords="offset points", fontsize=8,
	)

	ax.axhline(0.0, linewidth=0.8)
	ax.axvline(0.0, linewidth=0.8)
	ax.set_title("Modèle discret ZOH utilisé pour la commande")
	ax.set_xlabel(r"Re$(z)$")
	ax.set_ylabel(r"Im$(z)$")
	ax.set_aspect("equal", adjustable="datalim")
	ax.grid(True)
	ax.legend(loc="best")
	fig.tight_layout()


def print_summary(cfg: FrequencyConfig, responses: list[dict]) -> None:
	Kcrit = 2.0 * cfg.height_m / cfg.sample_time_s
	a_platform = platform_acceleration_amplitude(cfg)

	print("\n=== Vertical optical-flow analysis ===")
	print(
		f"h={cfg.height_m:.4f} m, D0={cfg.divergence_op_1_s:.4f} 1/s, "
		f"T={cfg.sample_time_s:.4f} s, Kcrit={Kcrit:.4f}"
	)
	print(
		f"platform: A={cfg.platform_amplitude_m:.4f} m, "
		f"f={cfg.platform_frequency_hz:.4f} Hz, "
		f"|a_p|={a_platform:.4f} m/s²\n"
	)

	for response in responses:
		T0 = complex_at_frequency(
			response["f"], response["T_platform"], cfg.platform_frequency_hz,
		)
		H0 = complex_at_frequency(
			response["f"], response["H_eD_ap"], cfg.platform_frequency_hz,
		)
		state = "stable" if response["stable"] else "UNSTABLE"
		print(
			f"K={response['K']:g}: {state}, "
			f"max|pole|={np.max(np.abs(response['poles'])):.4f}, "
			f"|a_d/a_p|={abs(T0):.4f}, "
			f"phase={np.angle(T0, deg=True):.2f} deg, "
			f"|e_D|={abs(H0) * a_platform:.4f} 1/s"
		)


def run_analysis(cfg: FrequencyConfig) -> None:
	responses = [compute_for_gain(cfg, K) for K in cfg.gain_values]
	print_summary(cfg, responses)
	plot_divergence_residual(cfg, responses)
	plot_platform_synchronization(cfg, responses)
	plot_root_locus(cfg)
	plt.show()


if __name__ == "__main__":
	configuration = FrequencyConfig(
		height_m=0.5,
		divergence_op_1_s=0.0,
		gain_values=(1.5, 6.5, 20.0),
		sample_time_s=0.06,
		platform_frequency_hz=0.4,
		platform_amplitude_m=0.10,
	)
	run_analysis(configuration)
