"""
Procedural albedo texture for flower_disc.obj.

Multi-scale design so optical-flow texture stays dense across the WHOLE
disc, not just the rim, at every camera distance:
  - coarse petal-intensity modulation (large scale, visible far away)
  - phyllotaxis (Fibonacci golden-angle spiral) floret blobs across the
    whole disc -- the same dense dot pattern used in photogrammetry /
    digital-image-correlation targets, specifically because it gives
    dense, well-conditioned correspondences for optical flow
  - fine correlated grain noise (a few-px correlation length -- pure
    per-pixel white noise has no spatial structure smaller than one
    pixel for a flow algorithm to lock onto) for texture at the closest
    range, sub-floret scale.
Color stays within one saturated hue band so the existing HSV-based
saliency detector (broad hue range, min_saturation/min_value floors)
keeps working unmodified -- see check_hsv_floors().
"""

import numpy as np
from PIL import Image


def generate_flower_texture(size: int = 1024, petal_count: int = 10, n_florets: int = 1400, seed: int = 0):
	rng = np.random.default_rng(seed)
	yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
	gx = (xx - size / 2) / (size / 2)
	gy = (yy - size / 2) / (size / 2)
	r = np.sqrt(gx**2 + gy**2)
	theta = np.arctan2(gy, gx)
	disc = r <= 1.0

	base_hue, base_sat, base_val = 340.0 / 360.0, 0.75, 0.62

	# Coarse petal modulation (large scale, visible far away).
	petal_mod = 1.0 + 0.30 * np.cos(petal_count * theta) * np.clip(r, 0, 1)

	# Phyllotaxis floret field: dense dot pattern, golden-angle spiral.
	golden_angle = np.pi * (3.0 - np.sqrt(5.0))
	floret_field = np.zeros((size, size), dtype=np.float64)
	c = 1.0 / np.sqrt(max(n_florets - 1, 1))
	for n in range(n_florets):
		rn = c * np.sqrt(n)
		if rn > 1.05:
			break
		th = n * golden_angle
		cx, cy = rn * np.cos(th), rn * np.sin(th)
		px, py = (cx * 0.5 + 0.5) * size, (cy * 0.5 + 0.5) * size
		spacing = c / (2.0 * np.sqrt(max(n, 1)))
		radius_px = max(2.5, min(0.85 * spacing * size / 2.0, size * 0.022))
		rad_i = int(np.ceil(radius_px))
		x0, x1 = max(0, int(px - rad_i - 1)), min(size, int(px + rad_i + 2))
		y0, y1 = max(0, int(py - rad_i - 1)), min(size, int(py + rad_i + 2))
		if x1 <= x0 or y1 <= y0:
			continue
		sub_x = xx[y0:y1, x0:x1]
		sub_y = yy[y0:y1, x0:x1]
		d = np.sqrt((sub_x - px) ** 2 + (sub_y - py) ** 2)
		# Smooth circular bump, exactly zero outside its own footprint (d>radius)
		# -- a true dot, not a square: clip(...) alone would plateau at a
		# constant value for all d>radius, stamping the whole square bounding
		# box rather than fading out at the circle's edge.
		within = d <= radius_px
		safe_ratio = np.clip(1.0 - d / radius_px, 0.0, 1.0)
		bump = np.where(within, safe_ratio ** 1.4, 0.0)
		current = floret_field[y0:y1, x0:x1]
		floret_field[y0:y1, x0:x1] = np.where(bump > current, bump, current)

	# Fine correlated grain: low-res noise, bilinearly upsampled so it has a
	# multi-pixel correlation length (a trackable structure, unlike i.i.d. noise).
	low = rng.normal(0, 1, size=(max(size // 16, 8), max(size // 16, 8)))
	low_u8 = ((low - low.min()) / (np.ptp(low) + 1e-9) * 255).astype(np.uint8)
	grain = np.asarray(
		Image.fromarray(low_u8).resize((size, size), Image.BILINEAR), dtype=np.float64
	) / 255.0

	# Contrast depth is the key parameter for optical-flow trackability: dense
	# structure alone (the floret/grain pattern) is not enough if the resulting
	# pixel intensity barely moves -- Farneback needs real grayscale gradient.
	val = base_val * petal_mod * (0.55 + 0.85 * floret_field) * (0.85 + 0.30 * grain)
	val = np.clip(val, 0.20, 1.0)  # floor stays above the detector's min_value=45/255=0.176
	sat = np.clip(base_sat * (1.0 - 0.15 * floret_field), 0.30, 1.0)  # floor above min_saturation=60/255=0.235
	hue = (base_hue + 0.01 * np.cos(3 * theta) + 0.006 * (grain - 0.5)) % 1.0

	rgb = _hsv_to_rgb_array(hue, sat, val)

	# Outside the disc (never sampled by this mesh) -- muted vignette tint,
	# avoids a harsh black/transparent seam if ever previewed in a 3D viewer.
	outside_val = np.clip(base_val * 0.55, 0.2, 1.0)
	rgb[~disc] = (np.array([outside_val, outside_val * 0.75, outside_val * 0.8]) * 255).astype(np.uint8)

	return rgb, {"floret_field": floret_field, "petal_mod": petal_mod, "disc": disc, "r": r, "theta": theta}


def generate_flower_normal_map(aux: dict, floret_height: float = 0.45, strength: float = 1.6) -> np.ndarray:
	"""
	Tangent-space normal map derived from the floret dots' height field, so
	shading lines up with the painted dot pattern instead of being an
	unrelated flat/placeholder map.

	Bumps come ONLY from floret_field, not the petal modulation: petal_mod is
	a function of theta=atan2(y,x), which has a 1/r gradient singularity at
	the disc center -- differentiating it for a height field produces a
	huge spurious gradient right at the center (a bright pinwheel artifact).
	floret_field is built from explicit Euclidean distances to floret
	centers and has no such singularity, so it's the only height contributor.

	normal = normalize([-dh/dx, -dh/dy, 1]), encoded as ((n+1)/2*255) in RGB
	(standard tangent-space encoding, what gz-sim's PBR normal_map expects).
	This is a rendering/observability bonus on top of (not a substitute for)
	the diffuse albedo contrast that actually fixes optical-flow tracking.
	"""
	floret_field, disc = aux["floret_field"], aux["disc"]
	size = floret_field.shape[0]

	height = floret_height * floret_field
	dx = 2.0 / size
	gy, gx = np.gradient(height, dx)

	nx = -gx * strength
	ny = -gy * strength
	nz = np.ones_like(height)
	norm = np.sqrt(nx**2 + ny**2 + nz**2)
	nx, ny, nz = nx / norm, ny / norm, nz / norm

	rgb = np.zeros((size, size, 3), dtype=np.uint8)
	rgb[..., 0] = np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
	rgb[..., 1] = np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
	rgb[..., 2] = np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)

	rgb[~disc] = (128, 128, 255)  # flat/neutral normal outside the mesh's UV footprint
	return rgb


def _hsv_to_rgb_array(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
	h6 = hue * 6.0
	i = np.floor(h6).astype(int) % 6
	f = h6 - np.floor(h6)
	p = val * (1 - sat)
	q = val * (1 - f * sat)
	t = val * (1 - (1 - f) * sat)
	r_ch = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [val, q, p, p, t, val])
	g_ch = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, val, val, q, p, p])
	b_ch = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, val, val, q])
	rgb = np.zeros(hue.shape + (3,), dtype=np.uint8)
	rgb[..., 0] = np.clip(r_ch * 255, 0, 255).astype(np.uint8)
	rgb[..., 1] = np.clip(g_ch * 255, 0, 255).astype(np.uint8)
	rgb[..., 2] = np.clip(b_ch * 255, 0, 255).astype(np.uint8)
	return rgb


def check_hsv_floors(rgb: np.ndarray, disc: np.ndarray, min_saturation: int = 60, min_value: int = 45):
	import cv2

	hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
	s_in, v_in = hsv[..., 1][disc], hsv[..., 2][disc]
	return {
		"s_min": int(s_in.min()), "s_mean": float(s_in.mean()),
		"v_min": int(v_in.min()), "v_mean": float(v_in.mean()),
		"s_ok": bool(s_in.min() >= min_saturation),
		"v_ok": bool(v_in.min() >= min_value),
	}


if __name__ == "__main__":
	rgb, aux = generate_flower_texture(size=1024, n_florets=2200, seed=0)
	Image.fromarray(rgb).save("flower_top.png")
	print("albedo:", check_hsv_floors(rgb, aux["disc"]))

	normal_rgb = generate_flower_normal_map(aux)
	Image.fromarray(normal_rgb).save("flower_top_normal.png")
	print("normal map saved: flower_top_normal.png")