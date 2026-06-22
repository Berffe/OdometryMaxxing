import numpy as np
import matplotlib.pyplot as plt

# Define constants
A = 25
B = 0.03
C = -0.03

# Define the function h(x, y)
# def h(x, y):
#     return A * np.sin(B * x) * np.cos(C * y)
def h(x, y):
    return - A * C * np.sin(B * x) * np.sin(C * y)

# Generate x and y values centered around the origin to capture the peaks
x = np.linspace(-200, 200, 400)
y = np.linspace(-200, 200, 400)
X, Y = np.meshgrid(x, y)
Z = h(X, Y)

# Selected Max and Min points close to the origin
x_max, y_max = np.pi/ (B * 2), np.pi/ (C * 2)
x_min, y_min = np.pi/ (B * 2), -np.pi/ (C * 2)

z_max = h(x_max, y_max)
z_min = h(x_min, y_min)

# Create the plot
plt.figure(figsize=(10, 8))

# Plot contours with a color map
contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour, label='dh/dy (x, y)')

# Add contour lines for better visibility
lines = plt.contour(X, Y, Z, levels=10, colors='black', alpha=0.3)
plt.clabel(lines, inline=True, fontsize=8)

# Mark and annotate the Global Maximum
plt.scatter(x_max, y_max, color='red', edgecolors='black', s=100, zorder=5, label=f'Máximo global ({z_max:.1f})')
plt.annotate(f'Max: {z_max:.1f}\n({x_max:.1f}, {y_max:.1f})', 
             xy=(x_max, y_max), xytext=(x_max + 10, y_max + 10),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="red", alpha=0.8))

# Mark and annotate the Global Minimum
plt.scatter(x_min, y_min, color='blue', edgecolors='black', s=100, zorder=5, label=f'Mínimo Global ({z_min:.1f})')
plt.annotate(f'Min: {z_min:.1f}\n({x_min:.1f}, {y_min:.1f})', 
             xy=(x_min, y_min), xytext=(x_min - 45, y_min - 20),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="blue", alpha=0.8))

# Labels and Styling
plt.title(r'$ \frac{\partial}{\partial y} h(x, y) = 25 \cdot 0.03 \cdot \cos(0.03x) \cdot \cos(-0.03y)$', fontsize=14, pad=15)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend(loc='upper right')
plt.grid(alpha=0.2)

# Show the plot
plt.show()