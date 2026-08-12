"""
Shared bench comparing the ways a single flow field can be reduced to lambda.

WHAT THIS COMPARES
------------------
Every estimator here consumes the SAME Farneback field from ONE
``OpticalFlowEstimator.update()`` call. Nothing re-runs Farneback. The question
is not "is Farneback the right dense-flow method" -- that was settled long ago
and the box-size cross-check that used to live in ``_optical_flow_debug.py``
went with it. The open question is narrower and still live:

    given one flow field, which REDUCTION recovers lambda most faithfully?

That matters because ``lambda`` is the only vertical measurement the controller
has. Its bias sets ``D*`` tracking error, and its noise propagates straight into
``PlatformProbe.peak_accel``, which is the number every feasibility gate rests
on. A reduction that is unbiased but noisy and one that is smooth but attenuated
fail the mission in completely different ways, and the CSV alone cannot tell
them apart.

CONVENTION
----------
All estimators return the PHYSICAL EXPANSION RATE

    lambda = -h_dot / h        [1/s]

NOT the full 2-D flow divergence ``2*lambda``. This is the single easiest thing
to get wrong when adding a reduction: the finite-difference field already
half-traces, and the legacy unconstrained fit does not. Both are normalised here
so the panel compares like with like.

THE ESTIMATORS
--------------
``production``
    The constrained 4-parameter fit, robust/trimmed, exactly as flown.
    Read back from ``FlowResult.raw_divergence`` -- not recomputed, so the panel
    cannot silently disagree with the vehicle.

``production_filtered``
    ``FlowResult.divergence``: the above after the EMA. Plotted to make filter
    lag visible, since that lag is real dead time in the vertical loop.

``constrained_plain``
    Same 4-parameter model, robust trimming OFF. Isolates what the trimming buys
    (or costs) on a given scene.

``legacy_unconstrained``
    The historical reduction: two independent 6-parameter affine fits returning
    ``a1 + b2``, halved here to reach lambda. Kept as a REFERENCE, not a
    candidate -- it is what the estimator used to do, and having it on screen is
    what makes "the constrained fit is better" a measurement rather than a
    claim.

``field_median``
    Median of the per-pixel finite-difference field. This is the production
    DEGENERATE FALLBACK, so its behaviour is not academic: it is what the
    vehicle actually flies when the affine solve is rank-deficient.

``field_mean`` / ``field_trimmed``
    The same field under two other reductions. The mean is the outlier-sensitive
    bound; the 10-90 trimmed mean sits between the two and is the obvious
    candidate if the median ever proves too coarse.

READING THE PANEL
-----------------
Ground truth is known analytically because the scene is synthetic. Per estimator
the bench reports instantaneous value, signed error, running bias (mean error)
and RMS error. Bias and RMS are the two that matter and they are independent:
a reduction can track the shape of lambda perfectly and still sit 20% low.
"""

from collections import deque
from typing import Optional

import cv2
import numpy as np

from bee_control.vision.optical_flow import OpticalFlowEstimator

#: Estimator key -> (display label, BGR colour, short role tag).
#:
#: Order is the drawing and table order: production first, reference last.
ESTIMATORS = (
	("production",           "4-param constrained (FLOWN)", (80, 255, 80),   "prod"),
	("production_filtered",  "  + EMA filter",              (40, 200, 40),   "prod"),
	("constrained_plain",    "4-param, no trimming",        (255, 200, 60),  "alt"),
	("field_median",         "field median (FALLBACK)",     (80, 180, 255),  "prod"),
	("field_trimmed",        "field trimmed 10-90",         (200, 140, 255), "alt"),
	("field_mean",           "field mean",                  (140, 140, 140), "alt"),
	("legacy_unconstrained", "6-param legacy (a1+b2)/2",    (80, 80, 255),   "ref"),
)

ESTIMATORS_PLOTTED = (
	("production",           "4-param constrained (FLOWN)", (80, 255, 80),   "prod"),
	("production_filtered",  "  + EMA filter",              (40, 200, 40),   "prod"),
	("legacy_unconstrained", "6-param legacy (a1+b2)/2",    (80, 80, 255),   "ref"),
)

ESTIMATOR_KEYS = tuple(key for key, _, _, _ in ESTIMATORS)


def _finite(value: float) -> float:
	value = float(value)
	return value if np.isfinite(value) else float("nan")


def _legacy_unconstrained_lambda(
	flow_px_s: np.ndarray,
	pixel_scale: float,
) -> float:
	"""Two independent 6-parameter affine fits; returns lambda, not 2*lambda.

	    u = a0 + a1*x + a2*y
	    v = b0 + b1*x + b2*y

	The historical estimator returned ``a1 + b2``, which for isotropic
	fronto-parallel expansion is the FULL 2-D divergence. Halved here so it is
	directly comparable with every other entry in this module.

	Deliberately kept simple: no border ring, no stride, no weighting, no
	trimming. Adding those would make it a different method and destroy its
	value as a fixed historical reference point.
	"""
	if flow_px_s is None or flow_px_s.ndim != 3:
		return float("nan")

	roi_height, roi_width = flow_px_s.shape[:2]
	if roi_width < 3 or roi_height < 3:
		return float("nan")

	spacing = 1.0 / max(1e-6, float(pixel_scale))

	ys, xs = np.mgrid[0:roi_height, 0:roi_width]
	x = (xs.astype(np.float64) - 0.5 * (roi_width - 1)) * spacing
	y = (ys.astype(np.float64) - 0.5 * (roi_height - 1)) * spacing

	u = flow_px_s[:, :, 0].astype(np.float64)
	v = flow_px_s[:, :, 1].astype(np.float64)

	good = np.isfinite(u) & np.isfinite(v)
	if int(np.count_nonzero(good)) < 6:
		return float("nan")

	design = np.stack(
		[np.ones(int(np.count_nonzero(good))), x[good], y[good]],
		axis=1,
	)

	try:
		coef_u, *_ = np.linalg.lstsq(design, u[good], rcond=None)
		coef_v, *_ = np.linalg.lstsq(design, v[good], rcond=None)
	except np.linalg.LinAlgError:
		return float("nan")

	# coef_u = [a0, a1, a2]; coef_v = [b0, b1, b2]  ->  (a1 + b2) / 2
	return 0.5 * (float(coef_u[1]) + float(coef_v[2]))


def _trimmed_mean(field: np.ndarray, low: float = 10.0, high: float = 90.0) -> float:
	flat = field[np.isfinite(field)]
	if flat.size == 0:
		return float("nan")

	lo, hi = np.percentile(flat, [low, high])
	kept = flat[(flat >= lo) & (flat <= hi)]
	if kept.size == 0:
		return float(np.mean(flat))

	return float(np.mean(kept))


def evaluate_estimators(
	flow_estimator: OpticalFlowEstimator,
	flow_result,
	debug: dict,
) -> dict:
	"""Reduce one already-computed flow field by every method in ESTIMATORS.

	``flow_estimator`` is only used to reach the constrained fit for the
	no-trimming variant. Calling a private method is deliberate: re-implementing
	the 4-parameter solve here would let the bench drift away from the flown
	code, which is the one thing it must never do.
	"""
	values = {key: float("nan") for key in ESTIMATOR_KEYS}

	if not debug:
		return values

	values["production"] = _finite(getattr(flow_result, "raw_divergence", float("nan")))
	values["production_filtered"] = _finite(getattr(flow_result, "divergence", float("nan")))

	field = debug.get("divergence_field")
	if field is not None and getattr(field, "size", 0) > 0:
		finite_field = field[np.isfinite(field)]
		if finite_field.size:
			values["field_median"] = float(np.median(finite_field))
			values["field_mean"] = float(np.mean(finite_field))
			values["field_trimmed"] = _trimmed_mean(field)

	flow_px_s = debug.get("flow_px_s")
	if flow_px_s is None:
		return values

	timing = flow_estimator.last_timing_data() or {}
	pixel_scale = float(timing.get("fit_pixel_scale", 1.0) or 1.0)
	image_width = int(timing.get("image_width_px", 0) or 0)
	image_height = int(timing.get("image_height_px", 0) or 0)

	values["legacy_unconstrained"] = _finite(
		_legacy_unconstrained_lambda(flow_px_s, pixel_scale)
	)

	try:
		plain, _, _ = flow_estimator._fit_divergence_affine(
			flow_px_s=flow_px_s,
			image_width=image_width,
			image_height=image_height,
			gradient_magnitude=None,
			robust=False,
			pixel_scale=pixel_scale,
			timing={},
		)
		values["constrained_plain"] = _finite(plain)
	except Exception:
		values["constrained_plain"] = float("nan")

	return values


class EstimatorComparison:
	"""Accumulates per-estimator error statistics against known ground truth."""

	def __init__(self, history_length: int = 240):
		self._history_length = int(history_length)
		self.reset()

	def reset(self):
		self.truth_history = deque(maxlen=self._history_length)
		self.history = {
			key: deque(maxlen=self._history_length) for key in ESTIMATOR_KEYS
		}
		self._sum_error = {key: 0.0 for key in ESTIMATOR_KEYS}
		self._sum_sq_error = {key: 0.0 for key in ESTIMATOR_KEYS}
		self._count = {key: 0 for key in ESTIMATOR_KEYS}
		self.latest = {key: float("nan") for key in ESTIMATOR_KEYS}
		self.latest_truth = float("nan")
		self.latest_fit_quality = float("nan")
		self._sum_fit_quality = 0.0
		self._fit_quality_count = 0

	def update(self, values: dict, truth: float, fit_quality: float = float("nan")):
		"""Record one frame. ``truth`` may be NaN when it is not known."""
		self.latest = dict(values)
		self.latest_truth = float(truth)
		self.truth_history.append(float(truth))

		self.latest_fit_quality = float(fit_quality)
		if np.isfinite(fit_quality):
			self._sum_fit_quality += float(fit_quality)
			self._fit_quality_count += 1

		for key in ESTIMATOR_KEYS:
			value = float(values.get(key, float("nan")))
			self.history[key].append(value)

			if not (np.isfinite(value) and np.isfinite(truth)):
				continue

			error = value - truth
			self._sum_error[key] += error
			self._sum_sq_error[key] += error * error
			self._count[key] += 1

	def bias(self, key: str) -> float:
		n = self._count[key]
		return self._sum_error[key] / n if n else float("nan")

	def rms(self, key: str) -> float:
		n = self._count[key]
		return float(np.sqrt(self._sum_sq_error[key] / n)) if n else float("nan")

	def samples(self, key: str) -> int:
		return self._count[key]

	def mean_fit_quality(self) -> float:
		n = self._fit_quality_count
		return self._sum_fit_quality / n if n else float("nan")


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

_BG = (24, 24, 24)
_FG = (235, 235, 235)
_DIM = (150, 150, 150)
_TRUTH_COLOR = (255, 255, 255)


def _put(canvas, text, org, scale=0.42, color=_FG, thickness=1):
	cv2.putText(
		canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
		thickness, cv2.LINE_AA,
	)


def _fmt(value: float, width: int = 8, decimals: int = 4) -> str:
	if value is None or not np.isfinite(value):
		return "--".rjust(width)
	return f"{value:+.{decimals}f}".rjust(width)


def make_comparison_table(
	comparison: EstimatorComparison,
	width: int = 620,
	height: int = 280,
	title: str = "Divergence reduction comparison",
	subtitle: str = "",
) -> np.ndarray:
	"""Table: value, error, running bias and RMS error per estimator.

	Column positions are derived from ``width`` rather than hardcoded, and the
	two running-statistic columns are dropped below ~520 px. Both harnesses ask
	for a different width, and a fixed layout silently clipped the rightmost
	columns in the narrower one -- which is worse than not showing them, since
	a half-drawn number still looks like a number.
	"""
	canvas = np.full((height, width, 3), _BG, dtype=np.uint8)

	wide = width >= 520
	right = width - 10

	# Right-aligned numeric columns, packed from the right edge.
	col_w = 74 if wide else 84
	x_rms = right - col_w if wide else None
	x_bias = (x_rms - col_w) if wide else None
	x_err = (x_bias - col_w) if wide else (right - col_w)
	x_lambda = x_err - col_w

	_put(canvas, title, (10, 20), 0.48, _FG, 1)

	truth = comparison.latest_truth
	truth_text = (
		f"truth lambda = {truth:+.4f} 1/s"
		if np.isfinite(truth)
		else "truth lambda = unknown"
	)
	_put(canvas, truth_text, (10, 40), 0.42, _TRUTH_COLOR, 1)

	# fit_quality belongs next to the numbers it explains. A low value does not
	# mean a bad estimate in every mode: pure translation has no expansion for
	# the model to explain, so quality is legitimately near zero there while the
	# lambda reported is still correct (and correctly near zero).
	quality = comparison.mean_fit_quality()
	if np.isfinite(quality):
		_put(
			canvas,
			f"mean fit_q {quality:.3f}",
			(x_lambda, 40), 0.40,
			(90, 220, 90) if quality > 0.5 else (120, 170, 255),
		)

	if subtitle:
		_put(canvas, subtitle, (10, 56), 0.36, _DIM)

	header_y = 74
	_put(canvas, "estimator", (10, header_y), 0.38, _DIM)
	_put(canvas, "lambda".rjust(9), (x_lambda, header_y), 0.38, _DIM)
	_put(canvas, "err".rjust(9), (x_err, header_y), 0.38, _DIM)
	if wide:
		_put(canvas, "bias".rjust(9), (x_bias, header_y), 0.38, _DIM)
		_put(canvas, "rms".rjust(9), (x_rms, header_y), 0.38, _DIM)

	cv2.line(canvas, (10, header_y + 6), (right, header_y + 6), _DIM, 1)

	row_y = header_y + 24
	row_step = max(18, min(23, (height - header_y - 46) // len(ESTIMATORS)))
	label_scale = 0.38 if wide else 0.34

	for key, label, color, role in ESTIMATORS:
		value = comparison.latest.get(key, float("nan"))
		error = (
			value - truth
			if np.isfinite(value) and np.isfinite(truth)
			else float("nan")
		)

		cv2.rectangle(canvas, (10, row_y - 8), (20, row_y - 1), color, -1)
		_put(canvas, label, (26, row_y), label_scale, _FG if role == "prod" else _DIM)
		_put(canvas, _fmt(value), (x_lambda, row_y), 0.38, color)
		_put(canvas, _fmt(error), (x_err, row_y), 0.38, color)
		if wide:
			_put(canvas, _fmt(comparison.bias(key)), (x_bias, row_y), 0.38, color)
			_put(canvas, _fmt(comparison.rms(key)), (x_rms, row_y), 0.38, color)

		row_y += row_step

	footer = (
		"FLOWN = production   FALLBACK = used when the affine solve degenerates"
		if wide
		else "FLOWN = production   FALLBACK = degenerate-solve path"
	)
	_put(canvas, footer, (10, height - 22), 0.33, _DIM)
	_put(
		canvas,
		"values are lambda = -hdot/h [1/s], not the 2-D divergence",
		(10, height - 8), 0.33, _DIM,
	)

	return canvas


def make_comparison_plot(
	comparison: EstimatorComparison,
	width: int = 620,
	height: int = 280,
	title: str = "lambda vs ground truth",
) -> np.ndarray:
	"""Time series of every estimator with ground truth overlaid."""
	canvas = np.full((height, width, 3), _BG, dtype=np.uint8)
	_put(canvas, title, (12, 20), 0.50, _FG, 1)

	left, right = 48, width - 12
	top, bottom = 34, height - 26

	series = [
		np.array(comparison.history[key], dtype=np.float64)
		for key in ESTIMATOR_KEYS
	]
	truth = np.array(comparison.truth_history, dtype=np.float64)

	pool = [s[np.isfinite(s)] for s in series if s.size]
	pool.append(truth[np.isfinite(truth)])
	pool = [p for p in pool if p.size]

	if not pool:
		_put(canvas, "no samples yet", (left, (top + bottom) // 2), 0.45, _DIM)
		return canvas

	stacked = np.concatenate(pool)
	lo = float(np.min(stacked))
	hi = float(np.max(stacked))
	if hi - lo < 1e-6:
		lo, hi = lo - 0.05, hi + 0.05

	pad = 0.1 * (hi - lo)
	lo -= pad
	hi += pad

	n = max(len(truth), max((len(s) for s in series), default=0))
	if n < 2:
		_put(canvas, "collecting...", (left, (top + bottom) // 2), 0.45, _DIM)
		return canvas

	def to_xy(index, value):
		x = left + int(round((right - left) * index / max(1, n - 1)))
		y = bottom - int(round((bottom - top) * (value - lo) / (hi - lo)))
		return x, max(top, min(bottom, y))

	# Axes and the lambda = 0 line.
	cv2.rectangle(canvas, (left, top), (right, bottom), (60, 60, 60), 1)
	if lo < 0.0 < hi:
		_, zero_y = to_xy(0, 0.0)
		cv2.line(canvas, (left, zero_y), (right, zero_y), (70, 70, 70), 1)

	_put(canvas, f"{hi:+.3f}", (6, top + 8), 0.34, _DIM)
	_put(canvas, f"{lo:+.3f}", (6, bottom), 0.34, _DIM)

	def draw(values, color, thickness):
		points = [
			to_xy(i, v) for i, v in enumerate(values) if np.isfinite(v)
		]
		if len(points) >= 2:
			cv2.polylines(
				canvas, [np.array(points, dtype=np.int32)],
				False, color, thickness, cv2.LINE_AA,
			)

	# Ground truth underneath everything, thick and white.
	draw(truth, _TRUTH_COLOR, 2)

	for (key, _, color, role), values in zip(ESTIMATORS_PLOTTED, series):
		draw(values, color, 2 if role == "prod" else 1)

	_put(canvas, "white = ground truth", (left + 6, bottom + 18), 0.36, _DIM)

	return canvas
