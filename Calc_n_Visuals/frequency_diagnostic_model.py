"""
Root-locus analysis of the simplified lateral (roll/pitch) visual-servo loop.

Self-contained: numpy + matplotlib only, no external control-theory package.
For this system the closed loop reduces to a plain quadratic, so a real root
solve is exact -- there is nothing a bigger library buys here.

------------------------------------------------------------------------------
MODEL
------------------------------------------------------------------------------
State: the normalized image offset x (dimensionless, same [-1,1] convention as
target_offset_x/y -- see state.py) and its rate xdot, approximated by the
optical-flow measurement flow_x/y (see optical_flow.py -- both are scaled by
the same half-width/half-height factor, so xdot and flow share one convention;
verified in flight logs to agree in SIGN >97% of the time, i.e. flow really is
tracking d(offset)/dt, not something else).

PLANT (roll/pitch command -> offset). For small tilt angles, horizontal
acceleration is
    a ~= g * u                                        [m/s^2],  u = commanded tilt [rad]
and a given PHYSICAL lateral acceleration maps to a normalized-OFFSET
acceleration that scales as 1/height (the same physical displacement is a
smaller image-plane fraction the further away the target is). Lumping this
into one plant gain:
    xddot = Kx * u ,       Kx = g / H_ref              [1/s^2 per rad]
where H_ref is a representative operating height.

THIS IS A DELIBERATE SIMPLIFICATION, matching the "simplified lateral model"
asked for -- know what it leaves out before trusting the plot for large gain
pushes:
  - Kx is actually height-dependent (shrinks as the vehicle descends -- this
    project already handles the ANALOGOUS vertical-axis problem properly via
    mission_routine.py's k(t) height schedule; nothing this simple exists yet
    for the lateral axis).
  - PX4's own inner attitude-tracking loop is assumed INSTANT (u applied with
    no lag). A real inner loop adds phase lag, which will make the true
    system less stable at high gain than this plot predicts.
  - control_law.py's command shaping (soft_limit saturation, slew-rate limit,
    low-pass filter) is entirely NONLINEAR and is not part of a linear root
    locus. This plot describes SMALL-SIGNAL behavior near x=0, not what
    happens during a large, saturating excursion (e.g. early CENTER).
  - Optical-flow measurement noise/lag (empirically shown in this project to
    make raising kd behave WORSE than this idealized model predicts) is not
    modeled. Treat any "raise kd" conclusion from this plot with real
    skepticism and prefer validating against a log, the way this project
    already does.

CONTROL LAW (see control_law.py): u = -(kp*x + kd*xdot)

CLOSED-LOOP characteristic equation:
    xddot + Kx*kd*xdot + Kx*kp*x = 0
    s^2 + (Kx*kd)*s + (Kx*kp) = 0
    wn = sqrt(Kx*kp) ,   zeta = (kd/2) * sqrt(Kx/kp)

------------------------------------------------------------------------------
THREE VIEWS ARE PLOTTED
------------------------------------------------------------------------------
1. CLASSICAL root locus: kp = K*kp0, kd = K*kd0 for a swept scalar K -- i.e.
   push the CURRENT (kp0, kd0) up/down together, preserving their RATIO. This
   holds the open-loop PD zero fixed at s=-kp0/kd0 (the textbook root-locus
   setup) but does NOT hold closed-loop zeta constant -- zeta scales as
   sqrt(K) here (only a kp*K, kd*sqrt(K) scaling holds zeta fixed while
   raising bandwidth, which is a DIFFERENT, deliberately-chosen move this
   project has used elsewhere; don't conflate the two).
2. INDEPENDENT kp sweep, kd held at kd0 -- how do poles move if you push
   proportional gain alone?
3. INDEPENDENT kd sweep, kp held at kp0 -- how do poles move if you push
   derivative gain alone?

Constant-damping-ratio rays (zeta = cos(angle from the negative real axis))
and the current operating point are marked on every plot.

Run directly: `python lateral_root_locus.py`. Edit the CONFIG block for your
axis (roll/pitch) and current gains before running.
"""

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIG -- edit to match the axis and gains you're analyzing
# ==============================================================================
G = 9.80665                      # m/s^2
H_REF = 5.0                      # m, representative operating height for Kx.
                                  # Lower H_REF (closer to touchdown) -> larger
                                  # Kx -> everything below shifts as if gain
                                  # were higher. Re-run at a couple of H_REF
                                  # values bracketing your mission's altitude
                                  # range rather than trusting one number.
KX = G / H_REF                   # lumped plant gain [1/s^2 per rad]

AXIS_LABEL = "roll"              # just a plot label
KP0 = 0.44                       # current proportional gain
KD0 = 0.155                      # current derivative gain

K_MAX = 6.0                      # classical-root-locus gain sweep: K in [0, K_MAX]
KP_SWEEP = np.linspace(1e-3, 1.5, 600)   # independent kp sweep (kd = KD0 fixed)
KD_SWEEP = np.linspace(1e-3, 0.6, 600)   # independent kd sweep (kp = KP0 fixed)

ZETA_RAYS = (0.1, 0.2, 0.3, 0.5, 0.707, 1.0)   # damping-ratio reference rays


# ==============================================================================
# Model
# ==============================================================================
def closed_loop_poles(kp, kd, Kx=KX):
	"""Roots of s^2 + Kx*kd*s + Kx*kp = 0, for scalar or array-like kp/kd.
	Returns (pole1, pole2) as complex arrays, broadcasting kp/kd together."""
	kp = np.atleast_1d(np.asarray(kp, dtype=float))
	kd = np.atleast_1d(np.asarray(kd, dtype=float))
	kp, kd = np.broadcast_arrays(kp, kd)

	a = np.ones_like(kp)
	b = Kx * kd
	c = Kx * kp
	disc = (b**2 - 4.0 * a * c).astype(complex)
	sqrt_disc = np.sqrt(disc)
	p1 = (-b + sqrt_disc) / (2.0 * a)
	p2 = (-b - sqrt_disc) / (2.0 * a)
	return p1, p2


def natural_freq_and_damping(kp, kd, Kx=KX):
	"""wn [rad/s], zeta [-] for the same closed loop. zeta is only the
	standard textbook quantity where kp>0; guarded against kp<=0 (unstable
	regardless of kd -- a zero or negative proportional gain gives no
	restoring force at all)."""
	kp = np.atleast_1d(np.asarray(kp, dtype=float))
	kd = np.atleast_1d(np.asarray(kd, dtype=float))
	wn = np.sqrt(np.maximum(Kx * kp, 0.0))
	with np.errstate(divide="ignore", invalid="ignore"):
		zeta = np.where(kp > 0, 0.5 * kd * np.sqrt(Kx / np.maximum(kp, 1e-12)), np.nan)
	return wn, zeta


# ==============================================================================
# Plot helpers
# ==============================================================================
def _add_zeta_rays(ax, reach):
	"""Constant-damping-ratio reference rays: a 2nd-order pole sits at angle
	arccos(zeta) from the NEGATIVE real axis, for any wn -- so each zeta is a
	straight ray from the origin, independent of gain."""
	for z in ZETA_RAYS:
		theta = np.arccos(np.clip(z, -1.0, 1.0))       # angle from -real axis
		x = -reach * np.cos(theta)
		y = reach * np.sin(theta)
		ax.plot([0, x], [0, y], color="0.75", linewidth=0.8, zorder=0)
		ax.plot([0, x], [0, -y], color="0.75", linewidth=0.8, zorder=0)
		ax.annotate(f"\u03b6={z:g}", (x, y), fontsize=7, color="0.45",
		            textcoords="offset points", xytext=(2, 2))


def _style_pole_axes(ax, title, reach):
	ax.axhline(0, color="0.3", linewidth=0.8, zorder=1)
	ax.axvline(0, color="tab:red", linewidth=1.3, alpha=0.6, zorder=1,
	           label="stability boundary (Re=0)")
	ax.axvspan(0, reach, color="tab:red", alpha=0.05, zorder=0)
	_add_zeta_rays(ax, reach)
	ax.set_xlim(-reach, reach * 0.3)
	ax.set_ylim(-reach, reach)
	ax.set_aspect("equal")
	ax.set_xlabel("Re(s)  [1/s]")
	ax.set_ylabel("Im(s)  [1/s]")
	ax.set_title(title, fontsize=10)
	ax.grid(True, alpha=0.3)


def plot_classical_root_locus(ax, kp0=KP0, kd0=KD0, Kx=KX, k_max=K_MAX):
	"""kp=K*kp0, kd=K*kd0 for K in [0, k_max] -- fixed open-loop PD zero at
	s=-kp0/kd0, the textbook root-locus sweep."""
	K = np.linspace(1e-4, k_max, 2000)
	p1, p2 = closed_loop_poles(K * kp0, K * kd0, Kx)
	reach = 1.3 * max(np.max(np.abs(p1)), np.max(np.abs(p2)), abs(kp0 / kd0))

	_style_pole_axes(ax, f"Classical root locus ({AXIS_LABEL}): K scales (kp0,kd0) together", reach)
	ax.plot(p1.real, p1.imag, color="tab:blue", linewidth=1.6, label="locus")
	ax.plot(p2.real, p2.imag, color="tab:blue", linewidth=1.6)

	p1_now, p2_now = closed_loop_poles(kp0, kd0, Kx)
	ax.plot([p1_now[0].real, p2_now[0].real], [p1_now[0].imag, p2_now[0].imag],
	        "o", color="black", markersize=7, zorder=5, label=f"current (K=1): kp={kp0:g}, kd={kd0:g}")
	ax.legend(loc="upper left", fontsize=7)


def plot_kp_sweep(ax, kp_sweep=KP_SWEEP, kd0=KD0, Kx=KX, kp0=KP0):
	p1, p2 = closed_loop_poles(kp_sweep, kd0, Kx)
	reach = 1.3 * max(np.max(np.abs(p1)), np.max(np.abs(p2)))
	_style_pole_axes(ax, f"kp sweep ({AXIS_LABEL}), kd fixed at {kd0:g}", reach)
	sc = ax.scatter(p1.real, p1.imag, c=kp_sweep, cmap="viridis", s=6, label="locus")
	ax.scatter(p2.real, p2.imag, c=kp_sweep, cmap="viridis", s=6)
	cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label("kp")

	p1_now, p2_now = closed_loop_poles(kp0, kd0, Kx)
	ax.plot([p1_now[0].real, p2_now[0].real], [p1_now[0].imag, p2_now[0].imag],
	        "o", color="red", markersize=8, zorder=5, label=f"current kp={kp0:g}")
	ax.legend(loc="upper left", fontsize=7)


def plot_kd_sweep(ax, kd_sweep=KD_SWEEP, kp0=KP0, Kx=KX, kd0=KD0):
	p1, p2 = closed_loop_poles(kp0, kd_sweep, Kx)
	reach = 1.3 * max(np.max(np.abs(p1)), np.max(np.abs(p2)))
	_style_pole_axes(ax, f"kd sweep ({AXIS_LABEL}), kp fixed at {kp0:g}", reach)
	sc = ax.scatter(p1.real, p1.imag, c=kd_sweep, cmap="plasma", s=6, label="locus")
	ax.scatter(p2.real, p2.imag, c=kd_sweep, cmap="plasma", s=6)
	cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label("kd")

	p1_now, p2_now = closed_loop_poles(kp0, kd0, Kx)
	ax.plot([p1_now[0].real, p2_now[0].real], [p1_now[0].imag, p2_now[0].imag],
	        "o", color="red", markersize=8, zorder=5, label=f"current kd={kd0:g}")
	ax.legend(loc="upper left", fontsize=7)


def print_operating_point(kp0=KP0, kd0=KD0, Kx=KX):
	wn, zeta = natural_freq_and_damping(kp0, kd0, Kx)
	p1, p2 = closed_loop_poles(kp0, kd0, Kx)
	print(f"=== {AXIS_LABEL} operating point ===")
	print(f"Kx = {Kx:.4f} 1/s^2 per rad  (H_ref={H_REF:g} m)")
	print(f"kp = {kp0:g}   kd = {kd0:g}")
	print(f"wn = {wn[0]:.4f} rad/s   ({wn[0]/(2*np.pi):.4f} Hz)")
	print(f"zeta = {zeta[0]:.4f}")
	print(f"poles = {p1[0]:.4f} , {p2[0]:.4f}")
	if zeta[0] < 1.0:
		wd = wn[0] * np.sqrt(1 - zeta[0]**2)
		print(f"damped oscillation frequency wd = {wd:.4f} rad/s ({wd/(2*np.pi):.4f} Hz)")
	print()


if __name__ == "__main__":
	print_operating_point()

	fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
	plot_classical_root_locus(axes[0])
	plot_kp_sweep(axes[1])
	plot_kd_sweep(axes[2])
	fig.suptitle(
		f"Lateral PD loop root locus -- {AXIS_LABEL} axis "
		f"(simplified model: xddot = Kx*u, u=-(kp*x+kd*xdot), Kx=g/H_ref={KX:.3f})",
		fontsize=11,
	)
	fig.tight_layout(rect=[0, 0, 1, 0.94])
	fig.savefig("lateral_root_locus.png", dpi=160)
	print("Saved: lateral_root_locus.png")
	plt.show()