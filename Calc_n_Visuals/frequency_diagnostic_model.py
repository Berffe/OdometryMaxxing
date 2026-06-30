"""
Discrete sampled-data frequency diagnostic model for the visual landing controller.

This script uses python-control for the system representation and feedback algebra.
It models the vertical/divergence channel as a continuous drone plant controlled by
an explicitly discrete visual controller with ZOH-discretized command-to-motion
behavior.

Example:
	python frequency_diagnostic_model.py --sample-periods 0.5 0.2 0.1 0.05 --h0 2.0

Model summary
-------------
Continuous drone plant:
	P_z(s) = K_T / (s^2 (tau_T s + 1))

Discrete optical-flow linearization:
	D[k] ~= -(1/h0) * ( d/dt h[k] + D_star h[k] )

where d/dt is approximated by a backward finite difference, consistent with
sampled image data.

Closed-loop algebra:
	D_true = P_uD u + P_pD z_platform
	y      = H_meas D_true
	u      = C y

Therefore:
	y/zp      = H_meas P_pD / (1 - H_meas P_uD C)
	u/zp      = C y/zp
	D_true/zp = P_pD + P_uD u/zp

For standard return-ratio plots, define:
	L = - H_meas P_uD C
so the closed-loop denominator is 1 + L.

Notes
-----
This is a first diagnostic model. It intentionally ignores saturation and slew
limits. It does include sampling, ZOH discretization, visual delay, divergence
smoothing, raw/filtered divergence blend, lead compensation, and integral action.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import control as ct


@dataclass
class ModelParams:
	# Operating point / visual linearization
	h0: float = 2.0                     # nominal relative height [m]
	divergence_setpoint: float = 0.0    # D_star [1/s]

	# Continuous vertical plant: thrust_delta -> drone vertical position
	# P_z(s) = K_T / (s^2 (tau_T s + 1))
	K_T: float = 1.0                    # position gain from normalized thrust to vertical acceleration-like response
	tau_T: float = 0.12                 # actuator/PX4 vertical lag [s]

	# Visual delay and divergence smoothing
	tau_vis: float = 0.001               # camera + transport + optical-flow computation delay [s]
	pade_order: int = 3                 # order for Padé delay approximation before discretization
	divergence_smoothing_alpha: float = 0.60  # d_f[k] = alpha d_f[k-1] + (1-alpha) d_raw[k]
	raw_divergence_weight: float = 0.15        # d_control = w*d_raw + (1-w)*d_filtered

	# Simplified discrete divergence controller
	# Positive controller_gain means: positive divergence error -> positive thrust correction.
	controller_gain: float = 0.80
	integral_gain: float = 0.018
	divergence_lead_time: float = 0.25
	divergence_lead_filter_alpha: float = 0.80

	# Optional extra pure digital delay in whole control path, in samples.
	# Use this if logs reveal the command applied by PX4 is one or more ticks late.
	extra_control_delay_samples: int = 0

	# Frequency sweep
	f_min_hz: float = 0.02
	f_max_hz: float = 8.0
	n_freq: int = 900


def const_tf(value: float, Ts: float):
	"""Return a discrete-time constant transfer function with sampling time Ts."""
	return ct.tf([float(value)], [1.0], Ts)


def zinv_tf(Ts: float):
	"""Return z^-1 as a discrete-time transfer function."""
	return ct.tf([1.0], [1.0, 0.0], Ts)


def backward_difference_tf(Ts: float):
	"""Return D(z) = (1 - z^-1)/Ts = (z - 1)/(Ts z)."""
	return ct.tf([1.0 / Ts, -1.0 / Ts], [1.0, 0.0], Ts)


def first_order_exponential_smoother_tf(alpha: float, Ts: float):
	"""
	H(z) = (1-alpha)/(1-alpha z^-1)
		= ((1-alpha) z)/(z-alpha)
	"""
	alpha = float(np.clip(alpha, 0.0, 0.999999))
	return ct.tf([1.0 - alpha, 0.0], [1.0, -alpha], Ts)


def forward_euler_integrator_tf(Ts: float):
	"""
	I(z) = Ts/(1 - z^-1) = Ts z/(z - 1).

	This matches the implementation pattern integral += error * dt.
	"""
	return ct.tf([Ts, 0.0], [1.0, -1.0], Ts)


def filtered_derivative_tf(alpha: float, Ts: float):
	"""
	Transfer from error e[k] to filtered derivative r[k]:
		raw_rate[k] = (e[k] - e[k-1]) / Ts
		r[k] = alpha r[k-1] + (1-alpha) raw_rate[k]

	H(z) = ((1-alpha)/Ts) * (1 - z^-1)/(1 - alpha z^-1)
		= ((1-alpha)/Ts) * (z - 1)/(z - alpha)
	"""
	alpha = float(np.clip(alpha, 0.0, 0.999999))
	gain = (1.0 - alpha) / Ts
	return ct.tf([gain, -gain], [1.0, -alpha], Ts)


def discrete_visual_linearization_tf(params: ModelParams, Ts: float):
	"""
	Discrete approximation of:
		G_D(s) = -(s + D_star)/h0

	using backward difference for s. This avoids trying to discretize an
	improper continuous derivative block directly.
	"""
	d_dt = backward_difference_tf(Ts)
	D_star = const_tf(params.divergence_setpoint, Ts)
	return -(d_dt + D_star) / params.h0


def visual_delay_tf(params: ModelParams, Ts: float):
	"""
	Approximate visual delay as a Padé transfer function discretized at Ts.

	For very small delay or zero delay, returns unity.
	"""
	if params.tau_vis <= 0.0:
		return const_tf(1.0, Ts)

	order = max(1, int(params.pade_order))
	num, den = ct.pade(params.tau_vis, order)
	delay_c = ct.tf(num, den)

	# Tustin preserves the frequency-domain character of the Padé approximation
	# better than treating the delay block as a held-input physical plant.
	return ct.sample_system(delay_c, Ts, method="tustin")


def measurement_filter_tf(params: ModelParams, Ts: float):
	"""
	Discrete measurement path from true divergence to controller divergence input.

	d_filtered[k] = alpha d_filtered[k-1] + (1-alpha) d_raw[k]
	d_used[k] = w*d_raw[k] + (1-w)*d_filtered[k]
	plus visual delay.
	"""
	w = float(np.clip(params.raw_divergence_weight, 0.0, 1.0))
	H_smooth = first_order_exponential_smoother_tf(params.divergence_smoothing_alpha, Ts)
	H_blend = const_tf(w, Ts) + (1.0 - w) * H_smooth
	H_delay = visual_delay_tf(params, Ts)

	if params.extra_control_delay_samples > 0:
		H_delay = H_delay * (zinv_tf(Ts) ** int(params.extra_control_delay_samples))

	return H_delay * H_blend


def divergence_controller_tf(params: ModelParams, Ts: float):
	"""
	Simplified discrete controller from measured divergence error to thrust delta.

	The fast path approximates the current lead-compensated divergence branch:
		lead_error = e + T_lead * filtered((e[k] - e[k-1]) / Ts)
		u_fast = Kp * lead_error

	The slow path approximates:
		integral += e * Ts
		u_int = Ki * integral

	Sign convention:
		positive measured divergence error -> positive thrust correction
		when controller_gain > 0.
	"""
	one = const_tf(1.0, Ts)
	lead = one + params.divergence_lead_time * filtered_derivative_tf(
		params.divergence_lead_filter_alpha, Ts
	)
	integral = forward_euler_integrator_tf(Ts)
	return params.controller_gain * lead + params.integral_gain * integral


def build_discrete_system(params: ModelParams, Ts: float) -> dict:
	"""Build all main discrete transfer functions for one sample period."""
	s = ct.TransferFunction.s

	# Continuous plant from thrust correction to vertical position.
	Pz_c = params.K_T / (s**2 * (params.tau_T * s + 1.0))
	Pz_d = ct.sample_system(Pz_c, Ts, method="zoh")

	# Discrete optical-flow linearization.
	G_D_d = discrete_visual_linearization_tf(params, Ts)

	# True divergence paths.
	P_uD = G_D_d * Pz_d       # thrust command -> true divergence
	P_pD = -G_D_d             # platform vertical position -> true divergence

	# Measurement path and controller.
	H_meas = measurement_filter_tf(params, Ts)
	C = divergence_controller_tf(params, Ts)

	# Use L = -H*P_uD*C so denominator is 1 + L.
	L = -H_meas * P_uD * C
	S = 1.0 / (1.0 + L)

	# Closed-loop transfer functions from platform position to outputs.
	T_zp_to_y = H_meas * P_pD * S
	T_zp_to_u = C * T_zp_to_y
	T_zp_to_D = P_pD + P_uD * T_zp_to_u

	return {
		"Ts": Ts,
		"Pz_d": Pz_d,
		"G_D_d": G_D_d,
		"H_meas": H_meas,
		"C": C,
		"L": L,
		"S": S,
		"T_zp_to_y": T_zp_to_y,
		"T_zp_to_u": T_zp_to_u,
		"T_zp_to_D": T_zp_to_D,
	}


def frequency_response_mag_phase(sys, omega: np.ndarray):
	"""Compatibility wrapper around python-control frequency_response."""
	response = ct.frequency_response(sys, omega)
	if hasattr(response, "magnitude"):
		mag = np.asarray(response.magnitude).squeeze()
		phase = np.asarray(response.phase).squeeze()
		omega_out = np.asarray(response.omega).squeeze()
	else:  # older python-control versions
		mag, phase, omega_out = response
		mag = np.asarray(mag).squeeze()
		phase = np.asarray(phase).squeeze()
		omega_out = np.asarray(omega_out).squeeze()
	return mag, phase, omega_out


def db(x: np.ndarray, floor: float = 1e-15):
	return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def phase_deg_unwrapped(phase_rad: np.ndarray):
	return np.rad2deg(np.unwrap(phase_rad))


def make_frequency_grid(params: ModelParams):
	f = np.logspace(
		math.log10(params.f_min_hz),
		math.log10(params.f_max_hz),
		int(params.n_freq),
	)
	return f, 2.0 * np.pi * f


def safe_margin(sys):
	"""Return gain/phase margin tuple, replacing failures with NaNs."""
	try:
		gm, pm, wcg, wcp = ct.margin(sys)
		return gm, pm, wcg, wcp
	except Exception:
		return np.nan, np.nan, np.nan, np.nan


def plot_overlay_bode(
	systems: Sequence[dict],
	key: str,
	params: ModelParams,
	ylabel: str,
	title: str,
	output_path: Path,
):
	f, omega = make_frequency_grid(params)

	fig_mag, ax_mag = plt.subplots(figsize=(9, 5))
	fig_phase, ax_phase = plt.subplots(figsize=(9, 5))

	for item in systems:
		Ts = item["Ts"]
		nyquist_hz = 0.5 / Ts
		mask = f < 0.92 * nyquist_hz
		if not np.any(mask):
			continue

		mag, phase, _ = frequency_response_mag_phase(item[key], omega[mask])
		label = f"Ts={Ts:g}s ({1.0/Ts:g} Hz)"
		ax_mag.semilogx(f[mask], db(mag), label=label)
		ax_phase.semilogx(f[mask], phase_deg_unwrapped(phase), label=label)

	ax_mag.set_title(title + " — magnitude")
	ax_mag.set_xlabel("frequency [Hz]")
	ax_mag.set_ylabel(ylabel)
	ax_mag.grid(True, which="both")
	ax_mag.legend()

	ax_phase.set_title(title + " — phase")
	ax_phase.set_xlabel("frequency [Hz]")
	ax_phase.set_ylabel("phase [deg]")
	ax_phase.grid(True, which="both")
	ax_phase.legend()

	fig_mag.tight_layout()
	fig_phase.tight_layout()
	fig_mag.savefig(output_path.with_name(output_path.stem + "_magnitude.png"), dpi=180)
	fig_phase.savefig(output_path.with_name(output_path.stem + "_phase.png"), dpi=180)
	plt.close(fig_mag)
	plt.close(fig_phase)


def plot_nyquist_return_ratio(systems: Sequence[dict], params: ModelParams, output_path: Path):
	f, omega = make_frequency_grid(params)
	fig, ax = plt.subplots(figsize=(7, 7))

	for item in systems:
		Ts = item["Ts"]
		nyquist_hz = 0.5 / Ts
		mask = f < 0.92 * nyquist_hz
		if not np.any(mask):
			continue

		mag, phase, _ = frequency_response_mag_phase(item["L"], omega[mask])
		resp = mag * np.exp(1j * phase)
		ax.plot(resp.real, resp.imag, label=f"Ts={Ts:g}s ({1.0/Ts:g} Hz)")

	ax.plot([-1.0], [0.0], marker="x", markersize=10, label="-1 point")
	ax.axhline(0.0, linewidth=0.8)
	ax.axvline(0.0, linewidth=0.8)
	ax.set_aspect("equal", adjustable="box")
	ax.set_title("Nyquist of return ratio L(z)")
	ax.set_xlabel("Re{L}")
	ax.set_ylabel("Im{L}")
	ax.grid(True)
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def print_summary_table(systems: Sequence[dict]):
	print("\nDiscrete bandwidth diagnostic summary")
	print("=" * 78)
	print(
		f"{'Ts [s]':>8} {'fs [Hz]':>8} {'Nyq [Hz]':>9} "
		f"{'PM [deg]':>10} {'wcp [Hz]':>10} {'GM [dB]':>10} {'wcg [Hz]':>10}"
	)
	print("-" * 78)

	for item in systems:
		Ts = item["Ts"]
		gm, pm, wcg, wcp = safe_margin(item["L"])
		gm_db = np.nan if gm is None or gm <= 0 or not np.isfinite(gm) else 20.0 * np.log10(gm)
		wcg_hz = np.nan if wcg is None or not np.isfinite(wcg) else wcg / (2.0 * np.pi)
		wcp_hz = np.nan if wcp is None or not np.isfinite(wcp) else wcp / (2.0 * np.pi)

		print(
			f"{Ts:8.3f} {1.0/Ts:8.2f} {0.5/Ts:9.2f} "
			f"{pm:10.2f} {wcp_hz:10.3f} {gm_db:10.2f} {wcg_hz:10.3f}"
		)
	print("=" * 78)
	print("Notes:")
	print("  L is defined so the closed-loop denominator is 1 + L.")
	print("  Frequencies near Nyquist are shown only for diagnosis, not as usable bandwidth.")
	print("  Treat useful control bandwidth as comfortably below Nyquist, usually fs/5 or less.")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Discrete python-control frequency diagnostic model for visual divergence landing."
	)
	parser.add_argument("--sample-periods", type=float, nargs="+", default=[0.5, 0.2, 0.1, 0.05])
	parser.add_argument("--output-dir", type=Path, default=Path("results"))

	parser.add_argument("--h0", type=float, default=ModelParams.h0)
	parser.add_argument("--divergence-setpoint", type=float, default=ModelParams.divergence_setpoint)
	parser.add_argument("--K-T", dest="K_T", type=float, default=ModelParams.K_T)
	parser.add_argument("--tau-T", dest="tau_T", type=float, default=ModelParams.tau_T)
	parser.add_argument("--tau-vis", type=float, default=ModelParams.tau_vis)
	parser.add_argument("--pade-order", type=int, default=ModelParams.pade_order)
	parser.add_argument("--divergence-smoothing-alpha", type=float, default=ModelParams.divergence_smoothing_alpha)
	parser.add_argument("--raw-divergence-weight", type=float, default=ModelParams.raw_divergence_weight)
	parser.add_argument("--controller-gain", type=float, default=ModelParams.controller_gain)
	parser.add_argument("--integral-gain", type=float, default=ModelParams.integral_gain)
	parser.add_argument("--divergence-lead-time", type=float, default=ModelParams.divergence_lead_time)
	parser.add_argument("--divergence-lead-filter-alpha", type=float, default=ModelParams.divergence_lead_filter_alpha)
	parser.add_argument("--extra-control-delay-samples", type=int, default=ModelParams.extra_control_delay_samples)
	parser.add_argument("--f-min-hz", type=float, default=ModelParams.f_min_hz)
	parser.add_argument("--f-max-hz", type=float, default=ModelParams.f_max_hz)
	parser.add_argument("--n-freq", type=int, default=ModelParams.n_freq)
	parser.add_argument("--show", action="store_true", help="Show plots interactively after saving.")

	return parser.parse_args()


def main():
	args = parse_args()
	params = ModelParams(
		h0=args.h0,
		divergence_setpoint=args.divergence_setpoint,
		K_T=args.K_T,
		tau_T=args.tau_T,
		tau_vis=args.tau_vis,
		pade_order=args.pade_order,
		divergence_smoothing_alpha=args.divergence_smoothing_alpha,
		raw_divergence_weight=args.raw_divergence_weight,
		controller_gain=args.controller_gain,
		integral_gain=args.integral_gain,
		divergence_lead_time=args.divergence_lead_time,
		divergence_lead_filter_alpha=args.divergence_lead_filter_alpha,
		extra_control_delay_samples=args.extra_control_delay_samples,
		f_min_hz=args.f_min_hz,
		f_max_hz=args.f_max_hz,
		n_freq=args.n_freq,
	)

	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	systems = [build_discrete_system(params, Ts) for Ts in args.sample_periods]

	print_summary_table(systems)

	plot_overlay_bode(
		systems,
		key="L",
		params=params,
		ylabel="|L| [dB]",
		title="Open-loop return ratio L(z)",
		output_path=output_dir / "open_loop_return_ratio.png",
	)
	plot_overlay_bode(
		systems,
		key="S",
		params=params,
		ylabel="|S| [dB]",
		title="Sensitivity S(z)=1/(1+L)",
		output_path=output_dir / "sensitivity.png",
	)
	plot_overlay_bode(
		systems,
		key="T_zp_to_D",
		params=params,
		ylabel="|D_true / z_platform| [dB]",
		title="Platform vertical motion to true divergence",
		output_path=output_dir / "platform_to_true_divergence.png",
	)
	plot_overlay_bode(
		systems,
		key="T_zp_to_y",
		params=params,
		ylabel="|D_meas / z_platform| [dB]",
		title="Platform vertical motion to measured/used divergence",
		output_path=output_dir / "platform_to_measured_divergence.png",
	)
	plot_overlay_bode(
		systems,
		key="T_zp_to_u",
		params=params,
		ylabel="|thrust_delta / z_platform| [dB]",
		title="Platform vertical motion to thrust correction",
		output_path=output_dir / "platform_to_thrust.png",
	)
	plot_nyquist_return_ratio(
		systems,
		params=params,
		output_path=output_dir / "nyquist_return_ratio.png",
	)

	print(f"\nSaved plots to: {output_dir.resolve()}")

	if args.show:
		plt.show()


if __name__ == "__main__":
	main()
