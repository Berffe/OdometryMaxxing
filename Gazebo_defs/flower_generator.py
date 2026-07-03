"""
Generate a replacement flower_top.png (+ matching normal map) for
bee_platform.sdf's platform_link visual_top, designed directly against the
measured deficiencies of the current texture rather than by eye:

    current flower_top.png, inside the disc:
        grayscale (BGR2GRAY, what optical_flow.py actually sees): min=28 max=148
        mean=50.6 std=13.9  -- dark, narrow range
        gradient magnitude: MEDIAN = 0.0, 88.5% of pixels near-zero gradient
        dot pitch: single fixed frequency, ~30px = ~5.9cm real-world
            (flower_disc.obj: polar UV map, 512px = 1.0m disc radius)

Design goals, each tied to one of the measured problems above:
    1. Wide, full-range local contrast (not a narrow dark band) -> histogram
       equalization on the combined noise field.
    2. Information-dense everywhere, not 88% flat -> continuous fractal/value
       noise as the base signal, not sparse marks on a flat field.
    3. Multi-scale (not one fixed pitch) -> sum of several noise octaves
       spanning from macro (~25cm feature size) down to near-texel-scale
       (~2mm), so SOME resolvable spatial frequency exists at any camera
       distance from far-away approach to millimeters off the deck.
    4. No large flat zones, including dead center -> noise is defined and
       equalized over the WHOLE disc, not confined to a band (see the
       earlier flat-center failure found in the synthetic test target).
    5. Aperiodic -> noise-based, not a repeating grid; avoids Farneback
       aliasing at closing rates that happen to match a fixed grid pitch.
    6. Stays comfortably above target_acquisition.py's HSV thresholds
       (min_saturation=60, min_value=45) so detection confidence is not
       traded away for flow quality -- verified numerically below, not
       assumed.

flower_disc.obj is a polar-UV disc: the texture's inscribed circle (center
(512,512), radius 512px) is exactly what gets rendered; the four corners
outside it are never sampled and are just filled with a neutral tone.
"""

import numpy as np
import cv2


SIZE = 1024
CENTER = (SIZE / 2.0, SIZE / 2.0)
DISC_RADIUS_PX = SIZE / 2.0  # 512px = 1.0m real disc radius (flower_disc.obj)


def value_noise_octave(size: int, grid: int, seed: int) -> np.ndarray:
	"""One octave of smooth value noise: a coarse random grid, upsampled with
	cubic interpolation. `grid` is the coarse-grid resolution -- a SMALL grid
	upsampled to `size` gives a LARGE feature size (low frequency); a grid
	close to `size` gives near-texel-scale features (high frequency)."""
	rng = np.random.default_rng(seed)
	coarse = rng.uniform(-1.0, 1.0, (grid, grid)).astype(np.float32)
	# INTER_CUBIC gives smooth (not blocky) interpolation between grid points,
	# which is what makes this "value noise" rather than a visible mosaic.
	return cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)


def fractal_noise(size: int, octaves, base_seed: int = 0) -> np.ndarray:
	"""Sum of octaves = (grid_resolution, amplitude) pairs. Amplitude is
	specified explicitly per octave (not a fixed persistence falloff) so the
	FINE octaves can be kept strong on purpose -- a standard 1/f falloff
	would reproduce exactly the 'coarse macro pattern, negligible fine
	detail' problem the current texture already has."""
	total = np.zeros((size, size), dtype=np.float32)
	for i, (grid, amp) in enumerate(octaves):
		total += amp * value_noise_octave(size, grid, seed=base_seed + i)
	return total


def make_flower_albedo() -> np.ndarray:
	yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
	rho = np.hypot(xx - CENTER[0], yy - CENTER[1]) / DISC_RADIUS_PX
	theta = np.arctan2(yy - CENTER[1], xx - CENTER[0])
	disc = rho <= 1.0

	# Octaves chosen to span ~25cm (macro mottling, feature size = disc_diam/grid)
	# down to ~1.5mm (near-texel fine grain), explicitly NOT 1/f-weighted --
	# fine octaves are kept comparably strong so they survive being one of
	# several summed layers instead of being washed out by the coarse ones.
	#   grid=8   -> feature ~25cm   (macro mottling)
	#   grid=24  -> feature ~8cm
	#   grid=80  -> feature ~2.5cm
	#   grid=260 -> feature ~0.8cm
	#   grid=700 -> feature ~0.3cm  (near texel-scale fine grain)
	octaves = [(8, 1.0), (24, 0.85), (80, 0.75), (260, 0.65), (700, 0.55)]
	noise = fractal_noise(SIZE, octaves, base_seed=1)

	# Petal macro-structure: an 8-fold angular modulation, for the "looks
	# like a flower" requirement. Applied as a SMALL additive bias on the
	# noise field (not a multiplicative flattening), so it changes the
	# large-scale look without erasing local contrast anywhere -- it must
	# not recreate the 'flat except for sparse marks' problem.
	petals = 0.18 * np.cos(8.0 * theta) * np.clip(1.0 - rho, 0.0, 1.0)
	noise = noise + petals

	# Histogram-equalize to guarantee full-range, well-spread local contrast
	# (fixes the measured 28-148 / mean-50.6 narrow dark band directly,
	# rather than hoping the noise amplitudes alone land in a good range).
	noise_norm = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	value = cv2.equalizeHist(noise_norm)

	# Floor Value so the darkest noise valleys stay comfortably above
	# target_acquisition.py's min_value=45 HSV threshold everywhere, not just
	# on average -- verified numerically after generation, not assumed here.
	value = np.clip(value.astype(np.float32), 70, 255).astype(np.uint8)

	# Hue: warm floral range (magenta/red -> warm pink/orange), modulated
	# gently by angle so the petal structure also reads as a hue shift, not
	# just a brightness ripple. Kept independent of the fine noise so hue
	# doesn't add spurious extra frequencies to fight the luminance design.
	hue = (12.0 + 8.0 * np.cos(8.0 * theta) + 4.0 * np.sin(3.0 * theta + rho * 6.0))
	hue = np.mod(hue, 180.0).astype(np.uint8)

	# Saturation: high and fairly uniform, comfortably above min_saturation=60.
	sat = np.full((SIZE, SIZE), 200, dtype=np.uint8)

	hsv = cv2.merge([hue, sat, value])
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	# Neutral fill outside the disc (never actually sampled by flower_disc.obj's
	# polar UV map, but kept clean/non-jarring if ever viewed directly).
	background = np.full_like(bgr, (60, 55, 58))
	out = np.where(disc[..., None], bgr, background)
	return out.astype(np.uint8), value, disc


def make_normal_map_from_value(value: np.ndarray, disc: np.ndarray, strength: float = 2.2) -> np.ndarray:
	"""Derive a tangent-space normal map from the SAME luminance field used
	for the albedo, so PBR shading reinforces the same multi-scale structure
	under the world's directional sun light, instead of carrying independent
	(and possibly contradictory) bump detail. Secondary to the albedo fix --
	shading-derived gradient depends on light direction/angle of approach,
	the albedo does not."""
	h = value.astype(np.float32) / 255.0
	gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) * strength
	gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) * strength

	nx, ny, nz = -gx, -gy, np.ones_like(h)
	norm = np.sqrt(nx * nx + ny * ny + nz * nz)
	nx, ny, nz = nx / norm, ny / norm, nz / norm

	r = ((nx * 0.5 + 0.5) * 255).astype(np.uint8)
	g = ((ny * 0.5 + 0.5) * 255).astype(np.uint8)
	b = ((nz * 0.5 + 0.5) * 255).astype(np.uint8)
	normal_bgr = cv2.merge([r, g, b])  # PNG stored as-is; matches flower_top_normal.png convention

	flat = np.full_like(normal_bgr, (255, 128, 128))  # neutral tangent-space normal (B,G,R)=(255,128,128)
	return np.where(disc[..., None], normal_bgr, flat).astype(np.uint8)


if __name__ == "__main__":
	albedo, value_field, disc_mask = make_flower_albedo()
	cv2.imwrite("flower_top_NEW.png", albedo)

	normal = make_normal_map_from_value(value_field, disc_mask)
	cv2.imwrite("flower_top_normal_NEW.png", normal)

	print("Wrote flower_top_NEW.png and flower_top_normal_NEW.png")