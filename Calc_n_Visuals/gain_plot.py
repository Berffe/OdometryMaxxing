import numpy as np
import matplotlib.pyplot as plt
import math

# z range
z = np.linspace(0, 7, 500)

# Gains
k_min_s = 1
k_min = k_min_s * np.ones_like(z)
k_z = 26 * z

# Critical height
h_crit = 0.18

# Intersection point: 1 = 26*z
z_intersection_stable = h_crit
k_intersection_stable = h_crit*26

# Intersection point: 1 = 13*z
z_intersection_osci = h_crit
k_intersection_osci = h_crit*13

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -----------------------
# Full plot
# -----------------------
axes[0].plot(z, k_min, label="Gain minimal", color='b')
axes[0].plot(z, k_z/2, label="Gain maximal - sans oscillations", color='g')
axes[0].plot(z, k_z, label="Gain maximal - stabilité", color='r')

axes[0].axvline(h_crit, linestyle="--", label="Hauteur train d'atterrissage")
# axes[0].scatter(z_intersection_stable, k_intersection, zorder=5)

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
axes[1].plot(z, k_min, label="Gain minimal", color='b')
axes[1].plot(z, k_z/2, label="Gain maximal - sans oscillations", color='g')
axes[1].plot(z, k_z, label="Gain maximal - stabilité", color='r')

axes[1].axvline(h_crit, linestyle="--", label="Hauteur train d'atterrissage")
axes[1].scatter(h_crit, k_intersection_osci, zorder=5, color='g')
axes[1].scatter(h_crit, k_intersection_stable, zorder=5, color='r')

axes[1].text(
    z_intersection_stable + 0.015,
    k_intersection_stable - 0.5,
    f"Gain max stable\nk_z= {k_intersection_stable:.2f}"
)

axes[1].text(
    z_intersection_osci + 0.015,
    k_intersection_osci - 0.5,
    f"Gain max sans oscillations\nk_z = {k_intersection_osci:.2f}"
)

axes[1].set_xlabel("coordonés z [m]")
axes[1].set_ylabel("Gain k")
axes[1].set_title("Zoom: z de 0 à 0.5 m")
axes[1].set_xlim(0, 0.5)
axes[1].set_ylim(0, 6)
axes[1].grid(True)
# axes[1].legend()

plt.tight_layout()

# -----------------------
# PAR RAPPORT AU TEMPS
# -----------------------
# Parameters
omega = 0.5
z0 = 3.0
k_exp = 6.5

# Time range
t = np.linspace(0, 12, 500)

# Exponential descent
z_t = z0 * np.exp(-omega * t)

# Gains as functions of time
k_min_t = 1 * np.ones_like(t)
k_z_t = 26 * z_t
k_floor = max(0.8*k_intersection_osci, k_min_s)
k_z_prog = lambda time : np.maximum(k_exp*np.exp(-omega*time), k_floor)

# Important times
t_hcrit = np.log(z0 / h_crit) / omega
# t_intersection_stable = np.log(z0 / z_intersection_stable) / omega
# t_intersection_osci = np.log(z0 / z_intersection_stable) / omega

fig_time, ax = plt.subplots(figsize=(8, 5))

ax.plot(t, k_min_t, label="Gain minimal", color='b')
ax.plot(t, k_z_t/2, label="Gain maximal - sans oscillations", color='g')
ax.plot(t, k_z_t, label="Gain maximal - stable", color='r')
ax.plot(t, k_z_prog(t), label="Gain programmé - k_z(t)", linestyle='--', color='c')

ax.axvline(
    t_hcrit,
    linestyle="--",
    label="Passage hauteur train d'atterrissage"
)

ax.scatter(t_hcrit, k_intersection_osci, zorder=5, color='g')
ax.scatter(t_hcrit, k_intersection_stable, zorder=5, color='r')

ax.text(
    t_hcrit + 0.15,
    k_intersection_stable + 0.5,
    f"Gain max stabilité \nk_z= {k_intersection_stable:.2f} s"
)

ax.text(
    t_hcrit + 0.15,
    k_intersection_osci - 0.1,
    f"Gain max sans oscillations\nk_z = {k_intersection_osci:.2f} s"
)

ax.set_xlabel("Temps [s]")
ax.set_ylabel("Gain k")
ax.set_title("Évolution du gain en fonction du temps")
ax.set_xlim(0, 8)
ax.set_ylim(0, 35)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()