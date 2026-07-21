"""
de_croon_frequency_response.py

Frequency-response analysis for the de Croon discrete optical-divergence model.

Purpose:
	1. Open-loop Bode + margins:
		L(z) = K z^-nd P(z)

	2. Nyquist:
		stability-margin sanity check, not the main performance plot.

	3. Closed-loop divergence tracking:
		D_obs / D_ref = K P(z) / (1 + K z^-nd P(z))

	4. Sensitivity:
		S(z) = 1 / (1 + K z^-nd P(z))

	5. Platform disturbance rejection:
		a_platform -> relative height
		a_platform -> relative velocity
		a_platform -> observed divergence

No plots are saved. All figures are shown at the very end with plt.show().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class FrequencyConfig:
	# Frozen operating point
	height_m: float = 2.0
	divergence_op_1_s: float = 0.0

	# Compare several proportional divergence gains
	gain_values: tuple[float, ...] = (7.5, 10.0)

	# Discrete sample/dead-time used in the de Croon model.
	# Use the effective visual/control delay, not necessarily the ROS publish period.
	sample_time_s: float = 0.05

	# Extra integer-sample delay in the feedback path.
	# 0 = pure de Croon sampled/ZOH model.
	delay_samples: int = 0

	# Platform oscillation to evaluate disturbance rejection
	platform_frequency_hz: float = 0.4
	platform_amplitude_m: float = 0.10

	# Frequency grid
	f_min_hz: float = 0.01
	n_points: int = 2500


# --------------------------------------------------------------------------- #
# de Croon frozen discrete model
# --------------------------------------------------------------------------- #

def decroon_discrete_matrices(
	height_m: float,
	divergence_op_1_s: float,
	sample_time_s: float,
):
	"""Frozen-height, linearized de Croon vertical model.

	State convention:
		x = [delta_h, delta_c]^T

	where:
		h = relative height above the platform, positive upward gap
		c = closing velocity, positive when the drone approaches the platform
		D = c / h = observed optical divergence, positive during closing

	Dynamics:
		h_dot = -c
		c_dot = u + a_platform

	where u is the acceleration-like thrust command in the closing direction.

	Discrete ZOH:
		x[k+1] = F x[k] + G u[k]

	Linearized output:
		D = c / h

	around c0 = D0 * h0:

		delta_D = (-D0 / h0) delta_h + (1 / h0) delta_c
	"""
	h = max(1e-6, float(height_m))
	D0 = float(divergence_op_1_s)
	T = max(1e-6, float(sample_time_s))

	F = np.array(
		[
			[1.0, -T],
			[0.0, 1.0],
		],
		dtype=float,
	)

	G = np.array(
		[
			[-0.5 * T * T],
			[T],
		],
		dtype=float,
	)

	C_div = np.array(
		[
			[-D0 / h, 1.0 / h],
		],
		dtype=float,
	)

	C_h = np.array([[1.0, 0.0]], dtype=float)
	C_c = np.array([[0.0, 1.0]], dtype=float)

	return F, G, C_div, C_h, C_c


def discrete_ss_response(
	F: np.ndarray,
	G: np.ndarray,
	C: np.ndarray,
	freq_hz: np.ndarray,
	sample_time_s: float,
) -> np.ndarray:
	"""Evaluate C (zI - F)^-1 G at z = exp(j omega T)."""
	T = float(sample_time_s)
	I = np.eye(F.shape[0])
	H = np.empty_like(freq_hz, dtype=complex)

	for i, f in enumerate(freq_hz):
		z = np.exp(1j * 2.0 * math.pi * f * T)
		H[i] = (C @ np.linalg.solve(z * I - F, G))[0, 0]

	return H


def make_frequency_grid(cfg: FrequencyConfig) -> np.ndarray:
	f_nyquist = 0.5 / cfg.sample_time_s
	f_max = 0.98 * f_nyquist

	return np.logspace(
		math.log10(cfg.f_min_hz),
		math.log10(f_max),
		cfg.n_points,
	)


# --------------------------------------------------------------------------- #
# Margins
# --------------------------------------------------------------------------- #

def _crossings(x: np.ndarray, y: np.ndarray, target: float) -> list[float]:
	out = []
	yy = y - target

	for i in range(len(x) - 1):
		y0 = yy[i]
		y1 = yy[i + 1]

		if not (np.isfinite(y0) and np.isfinite(y1)):
			continue

		if y0 == 0.0:
			out.append(float(x[i]))
		elif y0 * y1 < 0.0:
			alpha = abs(y0) / (abs(y0) + abs(y1))
			out.append(float(x[i] + alpha * (x[i + 1] - x[i])))

	return out


def estimate_margins(freq_hz: np.ndarray, L: np.ndarray) -> dict:
	mag = np.abs(L)
	mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
	phase_deg = np.rad2deg(np.unwrap(np.angle(L)))

	gain_crossings = _crossings(freq_hz, mag_db, 0.0)
	phase_crossings = _crossings(freq_hz, phase_deg, -180.0)

	gain_cross_hz = gain_crossings[0] if gain_crossings else None
	phase_margin_deg = None
	if gain_cross_hz is not None:
		phase_at_gc = float(np.interp(gain_cross_hz, freq_hz, phase_deg))
		phase_margin_deg = 180.0 + phase_at_gc

	phase_cross_hz = phase_crossings[0] if phase_crossings else None
	gain_margin_db = None
	if phase_cross_hz is not None:
		mag_at_pc = float(np.interp(phase_cross_hz, freq_hz, mag))
		if mag_at_pc > 1e-12:
			gain_margin_db = -20.0 * math.log10(mag_at_pc)

	return {
		"gain_cross_hz": gain_cross_hz,
		"phase_margin_deg": phase_margin_deg,
		"phase_cross_hz": phase_cross_hz,
		"gain_margin_db": gain_margin_db,
	}


# --------------------------------------------------------------------------- #
# Frequency response computation
# --------------------------------------------------------------------------- #

def compute_for_gain(cfg: FrequencyConfig, gain_k: float) -> dict:
	f = make_frequency_grid(cfg)
	Ts = cfg.sample_time_s

	F, G, C_div, C_h, C_c = decroon_discrete_matrices(
		height_m=cfg.height_m,
		divergence_op_1_s=cfg.divergence_op_1_s,
		sample_time_s=Ts,
	)

	P_div = discrete_ss_response(F, G, C_div, f, Ts)
	P_h = discrete_ss_response(F, G, C_h, f, Ts)
	P_c = discrete_ss_response(F, G, C_c, f, Ts)

	z = np.exp(1j * 2.0 * math.pi * f * Ts)
	delay = z ** (-int(cfg.delay_samples))

	# Open loop
	L = float(gain_k) * delay * P_div

	# Closed-loop tracking: divergence setpoint -> observed divergence
	T_div_ref_to_div = (float(gain_k) * P_div) / (1.0 + L)

	# Sensitivity
	S = 1.0 / (1.0 + L)

	# Disturbance response.
	# Platform acceleration enters in the same acceleration channel as u,
	# but as an external disturbance, with feedback fighting it.
	H_ap_to_div = P_div / (1.0 + L)
	H_ap_to_h = P_h / (1.0 + L)
	H_ap_to_c = P_c / (1.0 + L)

	return {
		"freq_hz": f,
		"K": float(gain_k),
		"P_div": P_div,
		"L": L,
		"T_div_ref_to_div": T_div_ref_to_div,
		"S": S,
		"H_ap_to_div": H_ap_to_div,
		"H_ap_to_h": H_ap_to_h,
		"H_ap_to_c": H_ap_to_c,
		"margins": estimate_margins(f, L),
	}


def complex_at_frequency(freq_hz: np.ndarray, H: np.ndarray, f0: float) -> complex:
	real = np.interp(f0, freq_hz, np.real(H))
	imag = np.interp(f0, freq_hz, np.imag(H))
	return complex(real, imag)


def platform_accel_amplitude(cfg: FrequencyConfig) -> float:
	w = 2.0 * math.pi * cfg.platform_frequency_hz
	return cfg.platform_amplitude_m * w * w


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #

def _fmt_optional(value: Optional[float], fmt: str) -> str:
	if value is None or not np.isfinite(value):
		return "not found"
	return format(value, fmt)


def print_summary(cfg: FrequencyConfig, responses: list[dict]) -> None:
	print("")
	print("=== de Croon discrete frequency-response analysis ===")
	print(f"height h                         = {cfg.height_m:.4f} m")
	print(f"operating divergence D0           = {cfg.divergence_op_1_s:.4f} 1/s")
	print(f"sample time Ts                    = {cfg.sample_time_s:.4f} s")
	print(f"extra feedback delay              = {cfg.delay_samples:d} samples")
	print(f"de Croon Kcrit = 2h/Ts            = {2.0 * cfg.height_m / cfg.sample_time_s:.4f}")
	print(f"platform frequency                = {cfg.platform_frequency_hz:.4f} Hz")
	print(f"platform amplitude                = {cfg.platform_amplitude_m:.4f} m")
	print(f"platform acceleration amplitude   = {platform_accel_amplitude(cfg):.4f} m/s²")
	print("")

	for r in responses:
		K = r["K"]
		m = r["margins"]
		f = r["freq_hz"]
		a_amp = platform_accel_amplitude(cfg)

		Hh = complex_at_frequency(f, r["H_ap_to_h"], cfg.platform_frequency_hz)
		Hc = complex_at_frequency(f, r["H_ap_to_c"], cfg.platform_frequency_hz)
		HD = complex_at_frequency(f, r["H_ap_to_div"], cfg.platform_frequency_hz)
		TD = complex_at_frequency(f, r["T_div_ref_to_div"], cfg.platform_frequency_hz)
		S = complex_at_frequency(f, r["S"], cfg.platform_frequency_hz)

		print(f"--- K = {K:.4f} ---")
		print(f"hcrit = K*Ts/2                  = {K * cfg.sample_time_s / 2.0:.4f} m")
		print(f"gain crossover                  = {_fmt_optional(m['gain_cross_hz'], '.4f')} Hz")
		print(f"phase margin                    = {_fmt_optional(m['phase_margin_deg'], '.2f')} deg")
		print(f"phase crossover                 = {_fmt_optional(m['phase_cross_hz'], '.4f')} Hz")
		print(f"gain margin                     = {_fmt_optional(m['gain_margin_db'], '.2f')} dB")
		print(f"|Dobs/Dref| at fp               = {abs(TD):.4f}")
		print(f"phase(Dobs/Dref) at fp          = {np.rad2deg(np.angle(TD)):.2f} deg")
		print(f"|S| at fp                       = {abs(S):.4f}")
		print(f"residual relative-z amp at fp   = {abs(Hh) * a_amp:.4f} m")
		print(f"residual closing-vel amp at fp  = {abs(Hc) * a_amp:.4f} m/s")
		print(f"divergence disturbance amp fp   = {abs(HD) * a_amp:.4f} 1/s")
		print("")


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

def _mag_db(H: np.ndarray) -> np.ndarray:
	return 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))


def _phase_deg(H: np.ndarray) -> np.ndarray:
	return np.rad2deg(np.unwrap(np.angle(H)))


def _mark_platform_frequency(ax, cfg: FrequencyConfig):
	ax.axvline(cfg.platform_frequency_hz, linestyle=":", linewidth=1.3, label="fréquence de la plateforme")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_open_loop_bode(cfg: FrequencyConfig, responses: list[dict]) -> None:
	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
	fig.suptitle("Open-loop Bode: L(z) = K z^-nd P(z)")

	for r in responses:
		f = r["freq_hz"]
		K = r["K"]
		L = r["L"]

		axes[0].semilogx(f, _mag_db(L), label=f"K={K:g}")
		axes[1].semilogx(f, _phase_deg(L), label=f"K={K:g}")

		gc = r["margins"]["gain_cross_hz"]
		if gc is not None:
			axes[0].axvline(gc, linestyle="--", linewidth=0.8, alpha=0.55)

	_mark_platform_frequency(axes[0], cfg)
	_mark_platform_frequency(axes[1], cfg)

	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("|L| [dB]")
	axes[0].grid(True, which="both")
	axes[0].legend(loc="best")

	axes[1].axhline(-180.0, linestyle="--", linewidth=1)
	axes[1].set_ylabel("phase [deg]")
	axes[1].set_xlabel("frequency [Hz]")
	axes[1].grid(True, which="both")
	axes[1].legend(loc="best")

	plt.tight_layout()


def plot_closed_loop_tracking(cfg: FrequencyConfig, responses: list[dict]) -> None:
	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
	fig.suptitle(r'Synchronisation drone-plateforme: $D_{obs} / D_{ref}$')

	for r in responses:
		f = r["freq_hz"]
		K = r["K"]
		TD = r["T_div_ref_to_div"]

		axes[0].semilogx(f, _mag_db(TD), label=f"K={K:g}")
		axes[1].semilogx(f, _phase_deg(TD), label=f"K={K:g}")

		f0 = cfg.platform_frequency_hz
		TD0 = complex_at_frequency(f, TD, f0)
		axes[0].scatter([f0], [_mag_db(np.array([TD0]))[0]], s=32)
		axes[1].scatter([f0], [np.rad2deg(np.angle(TD0))], s=32)

	_mark_platform_frequency(axes[0], cfg)
	_mark_platform_frequency(axes[1], cfg)

	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel(r"$|D_{obs} / D_{ref}|$ [dB]")
	axes[0].grid(True, which="both")
	axes[0].legend(loc="best")

	axes[1].axhline(0.0, linestyle="--", linewidth=1)
	axes[1].set_ylabel("phase [deg]")
	axes[1].set_xlabel("fréquence [Hz]")
	axes[1].grid(True, which="both")
	axes[1].legend(loc="best")

	plt.tight_layout()


def plot_sensitivity(cfg: FrequencyConfig, responses: list[dict]) -> None:
	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
	fig.suptitle("Sensitivity: S(z) = 1 / (1 + L(z))")

	for r in responses:
		f = r["freq_hz"]
		K = r["K"]
		S = r["S"]

		axes[0].semilogx(f, _mag_db(S), label=f"K={K:g}")
		axes[1].semilogx(f, _phase_deg(S), label=f"K={K:g}")

		f0 = cfg.platform_frequency_hz
		S0 = complex_at_frequency(f, S, f0)
		axes[0].scatter([f0], [_mag_db(np.array([S0]))[0]], s=32)
		axes[1].scatter([f0], [np.rad2deg(np.angle(S0))], s=32)

	_mark_platform_frequency(axes[0], cfg)
	_mark_platform_frequency(axes[1], cfg)

	axes[0].axhline(0.0, linestyle="--", linewidth=1)
	axes[0].set_ylabel("|S| [dB]")
	axes[0].grid(True, which="both")
	axes[0].legend(loc="best")

	axes[1].set_ylabel("phase [deg]")
	axes[1].set_xlabel("frequency [Hz]")
	axes[1].grid(True, which="both")
	axes[1].legend(loc="best")

	plt.tight_layout()


def plot_platform_disturbance_response(cfg: FrequencyConfig, responses: list[dict]) -> None:
	a_amp = platform_accel_amplitude(cfg)

	fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
	fig.suptitle(
		"Closed-loop platform disturbance response\n"
		f"A={cfg.platform_amplitude_m:.3f} m, fp={cfg.platform_frequency_hz:.3f} Hz, "
		f"ap_amp={a_amp:.3f} m/s²"
	)

	for r in responses:
		f = r["freq_hz"]
		K = r["K"]

		# These are transfer magnitudes from platform acceleration to response.
		Hh = r["H_ap_to_h"]
		Hc = r["H_ap_to_c"]
		HD = r["H_ap_to_div"]

		axes[0].loglog(f, np.maximum(np.abs(Hh), 1e-12), label=f"K={K:g}")
		axes[1].loglog(f, np.maximum(np.abs(Hc), 1e-12), label=f"K={K:g}")
		axes[2].loglog(f, np.maximum(np.abs(HD), 1e-12), label=f"K={K:g}")

		f0 = cfg.platform_frequency_hz
		Hh0 = complex_at_frequency(f, Hh, f0)
		Hc0 = complex_at_frequency(f, Hc, f0)
		HD0 = complex_at_frequency(f, HD, f0)

		axes[0].scatter([f0], [abs(Hh0)], s=32)
		axes[1].scatter([f0], [abs(Hc0)], s=32)
		axes[2].scatter([f0], [abs(HD0)], s=32)

		axes[0].annotate(
			f"{abs(Hh0) * a_amp:.3f} m",
			xy=(f0, abs(Hh0)),
			xytext=(8, 8),
			textcoords="offset points",
			fontsize=8,
		)

		axes[1].annotate(
			f"{abs(Hc0) * a_amp:.3f} m/s",
			xy=(f0, abs(Hc0)),
			xytext=(8, 8),
			textcoords="offset points",
			fontsize=8,
		)

		axes[2].annotate(
			f"{abs(HD0) * a_amp:.3f} 1/s",
			xy=(f0, abs(HD0)),
			xytext=(8, 8),
			textcoords="offset points",
			fontsize=8,
		)

	for ax in axes:
		_mark_platform_frequency(ax, cfg)
		ax.grid(True, which="both")
		ax.legend(loc="best")

	axes[0].set_ylabel("|h_rel / a_p| [m/(m/s²)]")
	axes[1].set_ylabel("|c_rel / a_p| [(m/s)/(m/s²)]")
	axes[2].set_ylabel("|D_obs / a_p| [(1/s)/(m/s²)]")
	axes[2].set_xlabel("frequency [Hz]")

	plt.tight_layout()


def plot_nyquist(cfg: FrequencyConfig, responses: list[dict]) -> None:
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	fig.suptitle("Nyquist of open loop L(z)")

	ax_full, ax_zoom = axes

	for r in responses:
		L = r["L"]
		K = r["K"]

		ax_full.plot(np.real(L), np.imag(L), label=f"K={K:g}")
		ax_full.plot(np.real(L), -np.imag(L), linestyle="--", alpha=0.45)

		ax_zoom.plot(np.real(L), np.imag(L), label=f"K={K:g}")
		ax_zoom.plot(np.real(L), -np.imag(L), linestyle="--", alpha=0.45)

	for ax in axes:
		ax.scatter([-1.0], [0.0], marker="x", s=90, label="-1 point")
		ax.axhline(0.0, linewidth=1)
		ax.axvline(0.0, linewidth=1)
		ax.set_xlabel("Re{L}")
		ax.set_ylabel("Im{L}")
		ax.grid(True)
		ax.legend(loc="best")
		ax.set_aspect("equal", adjustable="box")

	ax_full.set_title("Full Nyquist")
	ax_zoom.set_title("Zoom around stability-critical region")
	ax_zoom.set_xlim(-2.5, 1.0)
	ax_zoom.set_ylim(-2.0, 2.0)

	plt.tight_layout()

def _wrap_phase_deg(phi: np.ndarray) -> np.ndarray:
	"""Wrap phase to [-180, 180] deg for easier residual/input interpretation."""
	return (phi + 180.0) % 360.0 - 180.0


def plot_residual_divergence_over_platform_input(cfg: FrequencyConfig, responses: list[dict]) -> None:
	"""Plot residual divergence caused by platform oscillation.

	This is the most direct moving-platform limitation plot:

		e_D = D_obs - D_ref

	For a platform-disturbance test, D_ref is not oscillating, so the dynamic
	residual is the observed divergence created by the moving platform.

	We show both:
		e_D / z_p   : residual divergence per meter of platform motion
		e_D / a_p   : residual divergence per platform acceleration

	Because a_p = -omega^2 z_p, the z_p-input phase includes an extra 180 deg
	sign flip relative to acceleration input.
	"""
	fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
	fig.suptitle(
		"Résiduel de la divergence causé par les oscillations de la plateforme \n"
		r"$e_D = D_{\mathrm{obs}} - D_{\mathrm{ref}}$"
	)

	for r in responses:
		f = r["freq_hz"]
		K = r["K"]

		omega = 2.0 * math.pi * f

		# Existing disturbance transfer:
		# platform acceleration -> residual divergence.
		H_eD_ap = r["H_ap_to_div"]

		# Platform position -> residual divergence.
		# Since a_p = -omega^2 z_p.
		H_eD_zp = -(omega ** 2) * H_eD_ap

		label = f"K={K:g}"

		# Magnitude: e_D / z_p
		axes[0, 0].loglog(f, np.maximum(np.abs(H_eD_zp), 1e-12), label=label)

		# Phase: e_D relative to z_p
		axes[1, 0].semilogx(f, _wrap_phase_deg(np.rad2deg(np.angle(H_eD_zp))), label=label)

		# Magnitude: e_D / a_p
		axes[0, 1].loglog(f, np.maximum(np.abs(H_eD_ap), 1e-12), label=label)

		# Phase: e_D relative to a_p
		axes[1, 1].semilogx(f, _wrap_phase_deg(np.rad2deg(np.angle(H_eD_ap))), label=label)

		# Mark actual platform frequency and annotate actual residual amplitude.
		f0 = cfg.platform_frequency_hz
		A_p = cfg.platform_amplitude_m
		a_amp = platform_accel_amplitude(cfg)

		H_zp_0 = complex_at_frequency(f, H_eD_zp, f0)
		H_ap_0 = complex_at_frequency(f, H_eD_ap, f0)

		res_from_zp = abs(H_zp_0) * A_p
		res_from_ap = abs(H_ap_0) * a_amp

		axes[0, 0].scatter([f0], [abs(H_zp_0)], s=35)
		axes[1, 0].scatter([f0], [_wrap_phase_deg(np.array([np.rad2deg(np.angle(H_zp_0))]))[0]], s=35)

		axes[0, 1].scatter([f0], [abs(H_ap_0)], s=35)
		axes[1, 1].scatter([f0], [_wrap_phase_deg(np.array([np.rad2deg(np.angle(H_ap_0))]))[0]], s=35)

		axes[0, 0].annotate(
			f"{res_from_zp:.3f} 1/s",
			xy=(f0, abs(H_zp_0)),
			xytext=(8, 8),
			textcoords="offset points",
			fontsize=8,
		)

		axes[0, 1].annotate(
			f"{res_from_ap:.3f} 1/s",
			xy=(f0, abs(H_ap_0)),
			xytext=(8, 8),
			textcoords="offset points",
			fontsize=8,
		)

	for ax in axes.flat:
		_mark_platform_frequency(ax, cfg)
		ax.grid(True, which="both")
		ax.legend(loc="best")

	axes[0, 0].set_title(r"Input position: $e_D / z_p$")
	axes[0, 0].set_ylabel(r"$|e_D/z_p|$  [(1/s)/m]")

	axes[1, 0].set_ylabel("phase [deg]")
	axes[1, 0].set_xlabel("fréquence [Hz]")

	axes[0, 1].set_title(r"Input accélération: $e_D / a_p$")
	axes[0, 1].set_ylabel(r"$|e_D/a_p|$  [(1/s)/(m/s²)]")

	axes[1, 1].set_ylabel("phase [deg]")
	axes[1, 1].set_xlabel("fréquence [Hz]")

	for ax in axes[1, :]:
		ax.axhline(0.0, linestyle="--", linewidth=1)
		ax.axhline(-90.0, linestyle=":", linewidth=1)
		ax.axhline(90.0, linestyle=":", linewidth=1)

	plt.tight_layout()

def plot_drone_reaction_over_platform_input(
	cfg: FrequencyConfig,
	responses: list[dict],
) -> None:
	"""Plot how strongly the drone follows the platform.

	The main transfer is:

		T_platform = a_drone / a_platform
		           = L / (1 + L)
		           = 1 - S

	In the ideal linear model, this is also equivalent to:

		z_drone / z_platform
		v_drone / v_platform

	for sinusoidal perturbations and zero initial-condition effects.
	"""
	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

	fig.suptitle(
		"Synchronisation drone–plateforme\n"
		r"$T_p = a_d/a_p = z_d/z_p = 1-S$"
	)

	for r in responses:
		f = r["freq_hz"]
		K = r["K"]

		# Platform-to-drone complementary sensitivity.
		T_platform = 1.0 - r["S"]

		label = f"K={K:g}"

		axes[0].semilogx(
			f,
			_mag_db(T_platform),
			label=label,
		)

		axes[1].semilogx(
			f,
			_wrap_phase_deg(
				np.rad2deg(np.angle(T_platform))
			),
			label=label,
		)

		# Mark the tested platform frequency.
		f0 = cfg.platform_frequency_hz
		T0 = complex_at_frequency(f, T_platform, f0)

		mag0_db = _mag_db(np.array([T0]))[0]
		phase0_deg = _wrap_phase_deg(
			np.array([np.rad2deg(np.angle(T0))])
		)[0]

		axes[0].scatter([f0], [mag0_db], s=35)
		axes[1].scatter([f0], [phase0_deg], s=35)

		axes[0].annotate(
			f"{abs(T0):.3f}",
			xy=(f0, mag0_db),
			xytext=(8, 8),
			textcoords="offset points",
			fontsize=8,
		)

		axes[1].annotate(
			f"{phase0_deg:.1f}°",
			xy=(f0, phase0_deg),
			xytext=(8, 8),
			textcoords="offset points",
			fontsize=8,
		)

	for ax in axes:
		_mark_platform_frequency(ax, cfg)
		ax.grid(True, which="both")
		ax.legend(loc="best")

	# Ideal synchronization reference.
	axes[0].axhline(
		0.0,
		linestyle="--",
		linewidth=1,
		label="suivi parfait",
	)

	axes[1].axhline(
		0.0,
		linestyle="--",
		linewidth=1,
	)

	axes[0].set_ylabel(
		r"$|a_d/a_p|$ [dB]"
	)

	axes[1].set_ylabel(
		"phase [deg]"
	)

	axes[1].set_xlabel(
		"fréquence [Hz]"
	)

	plt.tight_layout()

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run_analysis(cfg: FrequencyConfig) -> None:
	responses = [compute_for_gain(cfg, K) for K in cfg.gain_values]
	plt.rcParams['text.usetex'] = True

	print_summary(cfg, responses)

	# plot_open_loop_bode(cfg, responses)
	# plot_closed_loop_tracking(cfg, responses)
	plot_residual_divergence_over_platform_input(cfg, responses)
	plot_drone_reaction_over_platform_input(cfg, responses)
	# plot_sensitivity(cfg, responses)
	# plot_platform_disturbance_response(cfg, responses)
	# plot_nyquist(cfg, responses)

	# Show everything only at the end.
	plt.show()


if __name__ == "__main__":
	cfg = FrequencyConfig(
		# Choose the frozen height you want to inspect.
		# Repeat the analysis for several heights to see how the descent changes.
		height_m=0.18,

		# Use 0.0 for hover/probe interpretation.
		# Use your descent D* value, e.g. 0.30, for descent frozen-point analysis.
		divergence_op_1_s=2,

		# Compare your last two tested gains.
		gain_values=(1.5, 6.5, 20),

		# Effective discrete time/dead-time.
		# This should represent the visual-control sample/effective correction time.
		sample_time_s=0.06,

		# Add delay samples here to see how quickly margins degrade.
		delay_samples=0,

		# Last test: 0.4 Hz, 0.1 m platform.
		platform_frequency_hz=0.4,
		platform_amplitude_m=0.10,
	)

	run_analysis(cfg)