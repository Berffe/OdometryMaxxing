import numpy as np
import matplotlib.pyplot as plt

# z range
z = np.linspace(0, 7, 500)

# Gains
k_min = 1.5 * np.ones_like(z)
k_z = 10 * z

# Critical height
h_crit = 0.2

# Intersection point: 1.5 = 10*z
z_intersection = 1.5 / 10
k_intersection = 1.5

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -----------------------
# Full plot
# -----------------------
axes[0].plot(z, k_min, label="Gain minimal (Herissé)")
axes[0].plot(z, k_z, label="Gain maximal (de Croon)")

axes[0].axvline(h_crit, linestyle="--", label="Hauteur train d'atterrissage")
axes[0].scatter(z_intersection, k_intersection, zorder=5)

axes[0].set_xlabel("coordonés z [m]")
axes[0].set_ylabel("Gain k")
axes[0].set_title("Évolution gain")
axes[0].set_xlim(0, 7)
axes[0].set_ylim(0, 75)
axes[0].grid(True)
axes[0].legend()

# -----------------------
# Zoomed plot
# -----------------------
axes[1].plot(z, k_min, label="Gain minimal (Herissé)")
axes[1].plot(z, k_z, label="Gain maximal (de Croon)")

axes[1].axvline(h_crit, linestyle="--", label="Hauteur train d'atterrissage")
axes[1].scatter(z_intersection, k_intersection, zorder=5)

axes[1].text(
    z_intersection + 0.015,
    k_intersection + 0.25,
    f"Intersection\nz = {z_intersection:.2f} m"
)

axes[1].set_xlabel("coordonés z [m]")
axes[1].set_ylabel("Gain k")
axes[1].set_title("Zoom: z de 0 à 0.5 m")
axes[1].set_xlim(0, 0.5)
axes[1].set_ylim(0, 6)
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

# -----------------------
# PAR RAPPORT AU TEMPS
# -----------------------
# Parameters
omega = 0.5
z0 = 3.0

# Time range
t = np.linspace(0, 12, 500)

# Exponential descent
z_t = z0 * np.exp(-omega * t)

# Gains as functions of time
k_min_t = 1.5 * np.ones_like(t)
k_z_t = 10 * z_t

# Important times
t_hcrit = np.log(z0 / h_crit) / omega
t_intersection = np.log(z0 / z_intersection) / omega

fig_time, ax = plt.subplots(figsize=(8, 5))

ax.plot(t, k_min_t, label="Gain minimal (Herissé)")
ax.plot(t, k_z_t, label="Gain maximal (de Croon)")

ax.axvline(
    t_hcrit,
    linestyle="--",
    label="Passage hauteur train d'atterrissage"
)

ax.axvline(
    t_intersection,
    linestyle=":",
    label="Intersection des gains"
)

ax.scatter(t_intersection, k_intersection, zorder=5)

ax.text(
    t_intersection + 0.15,
    k_intersection + 0.5,
    f"Intersection\nt = {t_intersection:.2f} s"
)

ax.set_xlabel("Temps [s]")
ax.set_ylabel("Gain k")
ax.set_title("Évolution du gain en fonction du temps")
ax.set_xlim(0, 12)
ax.set_ylim(0, 35)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()