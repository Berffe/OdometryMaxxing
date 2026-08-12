"""
Optical flow estimation, decoupled from ROS.

Production path:

    update(frame_bgr, timestamp, target=None) -> FlowResult

When a valid TargetEstimate is provided, the dense optical flow and the
scalar expansion rate are computed only inside the target bounding box.

The control-facing expansion rate is obtained from a constrained four-parameter
least-squares fit to the flow field (see _fit_divergence_affine):

    u(x, y) = t_x + lambda*x - r*y
    v(x, y) = t_y + r*x + lambda*y

The fit is performed directly in original-pixel coordinates and velocities, so
``lambda`` has units of 1/s and is invariant to the ROI origin and image size.
For an ideal fronto-parallel approach it is the physically useful quantity

    lambda = -h_dot / h,

not the full 2-D flow divergence ``du/dx + dv/dy = 2*lambda``.  The historical
FlowResult field is still named ``divergence`` for interface compatibility, but
from this revision onward it stores ``lambda``.  The constrained model uses all
valid flow vectors while excluding shear/anisotropic scale modes that are not
part of the intended fronto-parallel landing kinematics.

The fit is WEIGHTED by each pixel's reference-frame image-gradient
magnitude (see _gradient_magnitude / _weighted_affine_least_squares), so
a texture-poor patch is down-weighted directly rather than relying on a
textured rim elsewhere in the ROI to out-vote it. That reliance was a real
gap: once TargetEstimate.fov_saturated (the target's true size exceeds
the camera's FOV -- see state.py), the rim is by definition outside the
frame, and only the interior remains to fit against. The ROI is also, at
that point, at its largest of the whole descent -- handled by shrinking the
Farneback search problem itself (see the constructor's downsample_target_px
docstring) rather than the earlier design's separate, MORE expensive
close-range parameter set: a continuous downsample tied to ROI size, not a
binary switch tied to fov_saturated.

The fit additionally reports fit_quality (a weighted R^2 -- see
_fit_divergence_affine), so a degraded-but-still-numeric divergence
estimate can be told apart from a well-supported one. As of this writing
it is DIAGNOSIS-ONLY: logged (diagnostics_writer.py's flow_fit_quality
column) but not yet read by control_law.py or mission_routine.py.

Debug path (from the repo root):

    python -m bee_control.vision.optical_flow

That imports bee_control/tests/_optical_flow_debug.py and starts the visual
debug test. The dense flow field is still not part of FlowResult.
"""

from typing import Optional, Tuple

import time

import cv2
import numpy as np

from bee_control.core.state import FlowResult, TargetEstimate
ROI = Tuple[int, int, int, int]  # x0, y0, x1, y1; x1/y1 are exclusive


class OpticalFlowEstimator:
    #: Every key ``last_timing_data()`` can emit, WITHOUT any log prefix.
    #:
    #: The estimator owns this list because the estimator is what produces the
    #: values: previously the same names were re-declared inside
    #: ``DiagnosticsWriter._fieldnames()``, and any stage timing added here but
    #: forgotten there was silently discarded at write time.  (That is exactly
    #: what happened to ``prederotation_diagnostic_active``.)
    #:
    #: Not every key appears on every frame -- the early-return paths in
    #: ``update()`` stop partway through -- so a consumer must treat a missing
    #: key as "stage not reached", not as an error.
    TIMING_FIELDS = (
        # Whole-call cost and validity.
        "total_wall_ms", "total_cpu_ms", "valid", "dt_sec",
        # Per-stage wall costs, in pipeline order.
        "grayscale_ms", "roi_setup_ms", "downsample_resize_ms",
        "farneback_ms", "flow_scaling_upsample_ms", "derotation_ms",
        "mean_flow_ms", "gradient_ms", "divergence_field_debug_ms",
        "affine_fit_ms", "prederotation_fit_ms", "divergence_filter_ms",
        "result_and_state_ms",
        # Operating point actually used for this frame.
        "image_width_px", "image_height_px",
        "roi_width_px", "roi_height_px",
        "working_width_px", "working_height_px", "working_flow_vectors",
        "downsample_scale", "fit_pixel_scale",
        "derotation_active", "prederotation_diagnostic_active",
        # Affine-fit internals (re-prefixed from _fit_divergence_affine).
        "affine_setup_ms", "affine_initial_solve_ms",
        "affine_residual_quantile_ms", "affine_refit_ms",
        "affine_input_points", "affine_sampled_points", "affine_fit_stride",
        "affine_finite_points", "affine_used_points",
        "affine_border_exclusion_px", "affine_border_excluded_points",
        "affine_border_fraction", "affine_border_min_px", "affine_border_max_px",
        "affine_points_used", "affine_fit_quality",
        # Static algorithm configuration, echoed per frame so a log is
        # self-describing without needing the source revision.
        "farneback_pyr_scale", "farneback_levels", "farneback_winsize",
        "farneback_iterations", "farneback_poly_n", "farneback_poly_sigma",
        "divergence_smoothing_alpha",
    )

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 21,
        iterations: int = 5,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        # --- ROI-adaptive downsampling (replaces the old binary close-range
        # Farneback parameter set) ---
        #
        # The old design used a SECOND, more expensive Farneback parameter set
        # (winsize_close=35, levels_close=4, pyr_scale_close=0.6) switched on
        # fov_saturated, reasoning that a bigger/closer ROI needs a bigger
        # search window. That made the single most expensive regime of the
        # whole descent (full-frame ROI, closest to touchdown) also the most
        # computationally expensive -- exactly backwards for a latency budget
        # that a de Croon stability gate depends on (see bee_node.py's
        # STABILITY_DT_SEC / VISION_PROCESSING_LATENCY_BUDGET_SEC).
        #
        # The replacement: shrink the ROI itself before Farneback runs, by a
        # factor tied to the ROI's own size, so the array Farneback actually
        # searches stays close to a FIXED working size regardless of how big
        # the target has grown -- a bigger ROI gets MORE downsampling, not a
        # bigger search window. This is justified by the same "we don't need
        # per-pixel resolution once the ROI is a single expanding surface"
        # reasoning the old close-range branch already used, just applied to
        # the search cost directly instead of to the window size.
        #
        # Mechanics (see update()): prev_roi/gray_roi are resized down by
        # `scale` before calcOpticalFlowFarneback (a single, cheap cv2.resize,
        # INTER_AREA -- the correct anti-aliasing choice for shrinking);
        # Farneback then runs with the SAME parameters (pyr_scale/levels/
        # winsize/iterations/poly_n/poly_sigma above) in both regimes, since
        # after downsampling a big-ROI close-range frame and a small-ROI
        # far-field frame present Farneback with a similarly-sized problem.
        # The resulting flow field is amplitude-corrected (divided by
        # `scale` -- a downsampled-pixel of apparent motion is 1/scale
        # original pixels of real motion).
        #
        # From there, WITHOUT de-rotation active, the affine divergence fit
        # runs DIRECTLY on this still-downsampled field -- it does not need
        # reconstructing to full ROI resolution first. _fit_divergence_affine
        # takes a pixel_scale argument for exactly this: it widens the
        # coordinate spacing it fits against by 1/scale to match, so the
        # fitted slope (the divergence) comes out in the same physical units
        # regardless of how densely the array it's fitting was sampled. This
        # matters because profiling showed the fit itself (the weighted
        # lstsq solve plus the trim-and-refit pass) -- not Farneback -- was
        # the dominant cost once Farneback alone had already been shrunk by
        # downsampling: reconstructing the field to full resolution before
        # fitting left the fit solving over the same point count as before,
        # undoing most of the savings for the one piece of update() that
        # actually cost the most. Fitting on the small field directly closes
        # that gap (measured ~2x further reduction on top of the earlier
        # downsampling-with-upsample version, in addition to whatever
        # derotation being off already saved).
        #
        # WITH de-rotation active, this shortcut is skipped: Derotator
        # samples the rotational model at full-image pixel coordinates (see
        # derotation.py), so update() upsamples the field back to full ROI
        # resolution first in that case, exactly as before -- correctness
        # for a currently-disabled feature was not worth the risk of a
        # subtle scale bug to save time on a path that isn't running.
        #
        # downsample_target_px: the working array's target max dimension.
        #     scale = clip(downsample_target_px / max(roi_w, roi_h),
        #                   downsample_min_scale, 1.0)
        #     A ROI already <= this size is left alone (scale clips to 1.0):
        #     the far-field regime, with its naturally compact ROI, is
        #     unaffected. Only a ROI bigger than this target gets shrunk, and
        #     it shrinks MORE the bigger it is -- exactly the "big ROI, more
        #     downsample" behavior wanted, and it degrades gracefully (a
        #     continuous function of ROI size) rather than the old binary
        #     fov_saturated switch.
        # downsample_min_scale: a floor so an extreme close-in ROI (already
        #     the full 120x80 frame at this project's camera resolution)
        #     can't be shrunk into too few pixels for the affine fit's
        #     min_points_for_affine_fit guard to have real texture to chew on.
        downsample_target_px: int = 96,
        downsample_min_scale: float = 0.5,
        require_target_roi: bool = True,
        roi_margin_fraction: float = 0.05,
        min_roi_size_px: int = 32,
        divergence_smoothing: float = 0.3,
        min_points_for_affine_fit: int = 30,
        affine_inlier_quantile: float = 0.85,
        affine_fit_stride: int = 2,
        # Adaptive Farneback-border exclusion for the physical 4x4 fit only.
        # Farneback itself still runs on the complete working ROI. The border
        # is computed in WORKING-grid pixels from the smaller working dimension:
        #
        #   b = clip(round(fraction * min(W_work, H_work)), min_px, max_px)
        #
        # Defaults give b=5 at 32 px and b=15 at 96 px, preserving roughly the
        # same central-area fraction across the operating range. Set fraction=0
        # and min_px=0 to recover the no-exclusion baseline.
        affine_border_fraction: float = 5.0 / 32.0,
        affine_border_min_px: int = 5,
        affine_border_max_px: int = 15,
        store_debug: bool = False,
        # Optional ego-rotation removal. Pass a derotation.Derotator to enable
        # it; leave None for the legacy (no de-rotation) behavior. When set,
        # update() also needs a per-frame body_rates vector to actually
        # subtract anything -- without it the flow is passed through unchanged.
        # DISABLED BY DEFAULT as of the light optical-flow pass (see
        # bee_node.py): re-enable once the downsampled flow field has been
        # re-validated against the derotation acceptance test in
        # derotation.py's docstring (downsampling changes the flow field's
        # spatial resolution, which the rotational-field model samples at
        # full resolution).
        derotator=None,
        # Diagnostic-only comparison between divergence before and after
        # de-rotation. Disabled by default because it requires a second affine
        # fit on the raw flow field and does not feed the controller.
        compute_prederotation_diagnostic: bool = False,
    ):
        self._prev_gray = None
        self._prev_bgr = None
        self._prev_timestamp = None

        self._pyr_scale = pyr_scale
        self._levels = levels
        self._winsize = winsize
        self._iterations = iterations
        self._poly_n = poly_n
        self._poly_sigma = poly_sigma

        self._downsample_target_px = max(3, int(downsample_target_px))
        self._downsample_min_scale = max(1e-3, min(1.0, float(downsample_min_scale)))

        self._require_target_roi = bool(require_target_roi)
        self._roi_margin_fraction = float(roi_margin_fraction)
        self._min_roi_size_px = int(min_roi_size_px)

        self._divergence_smoothing = float(divergence_smoothing)
        self._filtered_divergence = 0.0
        self._has_filtered_divergence = False

        # Affine-fit divergence (see module docstring). min_points is a
        # degenerate-case guard (a typical ROI has >>1000 flow vectors);
        # inlier_quantile keeps the best fraction by residual on a single
        # trim-and-refit pass, so a cluster of unreliable (textureless or
        # specular) flow vectors can't dominate the global fit.
        self._min_points_for_affine_fit = int(min_points_for_affine_fit)
        self._affine_inlier_quantile = float(affine_inlier_quantile)
        # Uniform spatial decimation for the affine fit only. Farneback and
        # mean-flow calculations still use the complete working field. A stride
        # of 2 samples one vector from each 2x2 block, retaining full-ROI
        # coverage while reducing the robust fit workload by about 4x.
        self._affine_fit_stride = max(1, int(affine_fit_stride))
        # Adaptive exclusion is expressed in the WORKING Farneback grid and
        # applied before affine-fit stride/weighting/robust trimming.
        self._affine_border_fraction = max(0.0, float(affine_border_fraction))
        self._affine_border_min_px = max(0, int(affine_border_min_px))
        self._affine_border_max_px = max(
            self._affine_border_min_px, int(affine_border_max_px)
        )

        self._store_debug = bool(store_debug)
        self._last_debug = {}
        self._last_timing = {}

        # Ego-rotation removal (see derotation.py). None -> disabled.
        self._derotator = derotator

        # When enabled, run an additional non-robust affine fit on the raw
        # (pre-de-rotation) flow for offline diagnostics. This value is never
        # used by the control-facing divergence estimate.
        self._compute_prederotation_diagnostic = bool(
            compute_prederotation_diagnostic
        )

    def update(
        self,
        frame_bgr,
        timestamp: float,
        target: Optional[TargetEstimate] = None,
        # Mean body angular rate (rad/s, FRD [p, q, r]) over the interval
        # between the previous frame and this one -- see bee_node.on_camera and
        # derotation.AngularRateBuffer. Only used when a derotator was supplied
        # to the constructor; None (or no derotator) => no de-rotation.
        body_rates=None,
    ) -> FlowResult:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        timing = {
            "farneback_pyr_scale": float(self._pyr_scale),
            "farneback_levels": int(self._levels),
            "farneback_winsize": int(self._winsize),
            "farneback_iterations": int(self._iterations),
            "farneback_poly_n": int(self._poly_n),
            "farneback_poly_sigma": float(self._poly_sigma),
            "divergence_smoothing_alpha": float(self._divergence_smoothing),
        }

        def finish_timing():
            timing["total_wall_ms"] = 1000.0 * (time.perf_counter() - wall_start)
            timing["total_cpu_ms"] = 1000.0 * (time.process_time() - cpu_start)
            self._last_timing = dict(timing)

        def invalid_result(message: str, *, previous_frame=None, current_frame=None, roi=None):
            stage_start = time.perf_counter()
            result = FlowResult(timestamp=timestamp, valid=False)
            self._save_debug(
                result=result,
                previous_frame=previous_frame,
                current_frame=current_frame,
                roi=roi,
                message=message,
            )
            timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
            timing["valid"] = 0
            finish_timing()
            return result

        if frame_bgr is None:
            return invalid_result("No frame", current_frame=None)

        stage_start = time.perf_counter()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        image_height, image_width = gray.shape[:2]
        timing["grayscale_ms"] = 1000.0 * (time.perf_counter() - stage_start)
        timing["image_width_px"] = int(image_width)
        timing["image_height_px"] = int(image_height)

        if self._prev_gray is None or self._prev_timestamp is None:
            stage_start = time.perf_counter()
            self._prev_gray = gray
            self._prev_bgr = frame_bgr.copy()
            self._prev_timestamp = timestamp
            timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
            timing["valid"] = 0
            result = FlowResult(timestamp=timestamp, valid=False)
            self._save_debug(
                result=result,
                current_frame=frame_bgr,
                message="Waiting for previous frame",
            )
            finish_timing()
            return result

        dt = float(timestamp - self._prev_timestamp)
        timing["dt_sec"] = dt

        if dt <= 1e-6:
            stage_start = time.perf_counter()
            self._prev_gray = gray
            self._prev_bgr = frame_bgr.copy()
            self._prev_timestamp = timestamp
            timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
            timing["valid"] = 0
            result = FlowResult(timestamp=timestamp, valid=False)
            self._save_debug(
                result=result,
                current_frame=frame_bgr,
                message="Invalid dt",
            )
            finish_timing()
            return result

        stage_start = time.perf_counter()
        previous_frame = self._prev_bgr.copy() if self._prev_bgr is not None else None
        roi = self._target_roi_from_estimate(
            target=target,
            image_width=image_width,
            image_height=image_height,
        )
        timing["roi_setup_ms"] = 1000.0 * (time.perf_counter() - stage_start)

        if roi is None and self._require_target_roi:
            stage_start = time.perf_counter()
            self._prev_gray = gray
            self._prev_bgr = frame_bgr.copy()
            self._prev_timestamp = timestamp
            timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
            timing["valid"] = 0
            result = FlowResult(timestamp=timestamp, valid=False)
            self._save_debug(
                result=result,
                previous_frame=previous_frame,
                current_frame=frame_bgr,
                roi=None,
                message="No valid target ROI",
            )
            finish_timing()
            return result

        if roi is None:
            roi = (0, 0, image_width, image_height)

        x0, y0, x1, y1 = roi
        prev_roi = self._prev_gray[y0:y1, x0:x1]
        gray_roi = gray[y0:y1, x0:x1]
        roi_height, roi_width = prev_roi.shape[:2]
        timing["roi_width_px"] = int(roi_width)
        timing["roi_height_px"] = int(roi_height)

        if roi_width < 3 or roi_height < 3:
            stage_start = time.perf_counter()
            self._prev_gray = gray
            self._prev_bgr = frame_bgr.copy()
            self._prev_timestamp = timestamp
            timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
            timing["valid"] = 0
            result = FlowResult(timestamp=timestamp, valid=False)
            self._save_debug(
                result=result,
                previous_frame=previous_frame,
                current_frame=frame_bgr,
                roi=roi,
                message="ROI too small",
            )
            finish_timing()
            return result

        # ROI-adaptive downsampling: time resizing separately from Farneback.
        stage_start = time.perf_counter()
        scale = self._downsample_scale_for_roi(roi_width, roi_height)
        if scale < 1.0:
            small_w = max(3, int(round(roi_width * scale)))
            small_h = max(3, int(round(roi_height * scale)))
            prev_small = cv2.resize(
                prev_roi, (small_w, small_h), interpolation=cv2.INTER_AREA
            )
            gray_small = cv2.resize(
                gray_roi, (small_w, small_h), interpolation=cv2.INTER_AREA
            )
        else:
            prev_small = prev_roi
            gray_small = gray_roi
        timing["downsample_resize_ms"] = 1000.0 * (time.perf_counter() - stage_start)
        timing["downsample_scale"] = float(scale)
        timing["working_width_px"] = int(prev_small.shape[1])
        timing["working_height_px"] = int(prev_small.shape[0])
        timing["working_flow_vectors"] = int(prev_small.shape[0] * prev_small.shape[1])

        stage_start = time.perf_counter()
        flow_small_per_frame = cv2.calcOpticalFlowFarneback(
            prev_small,
            gray_small,
            None,
            self._pyr_scale,
            self._levels,
            self._winsize,
            self._iterations,
            self._poly_n,
            self._poly_sigma,
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        timing["farneback_ms"] = 1000.0 * (time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        if scale < 1.0:
            flow_small_per_frame = flow_small_per_frame / scale

        derotation_active = self._derotator is not None and body_rates is not None
        # Keep the compact working grid even when derotation is enabled. The
        # rotational model is sampled at the corresponding normalized full-image
        # coordinates instead of upsampling the measured flow.
        flow_px_per_frame = flow_small_per_frame
        fit_pixel_scale = scale
        gradient_source = prev_small
        flow_px_s = flow_px_per_frame / dt
        timing["flow_scaling_upsample_ms"] = 1000.0 * (time.perf_counter() - stage_start)
        timing["fit_pixel_scale"] = float(fit_pixel_scale)
        timing["derotation_active"] = int(bool(derotation_active))

        stage_start = time.perf_counter()
        raw_flow_px_s = flow_px_s
        if derotation_active:
            flow_px_s = self._derotator.derotate_working_grid(
                flow_px_s, body_rates, roi=(x0, y0, x1, y1),
                pixel_scale=fit_pixel_scale,
            )
        timing["derotation_ms"] = 1000.0 * (time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        raw_mean_flow_x = float(np.mean(raw_flow_px_s[:, :, 0]))
        raw_mean_flow_y = float(np.mean(raw_flow_px_s[:, :, 1]))
        mean_flow_x = float(np.mean(flow_px_s[:, :, 0]))
        mean_flow_y = float(np.mean(flow_px_s[:, :, 1]))
        mean_flow_x_norm = mean_flow_x / max(0.5 * image_width, 1.0)
        mean_flow_y_norm = mean_flow_y / max(0.5 * image_height, 1.0)
        timing["mean_flow_ms"] = 1000.0 * (time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        gradient_magnitude = self._gradient_magnitude(gradient_source)
        timing["gradient_ms"] = 1000.0 * (time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        divergence_field = (
            self._estimate_divergence_field(
                flow_px_s=flow_px_s,
                image_width=image_width,
                image_height=image_height,
                pixel_scale=fit_pixel_scale,
            )
            if self._store_debug
            else None
        )
        timing["divergence_field_debug_ms"] = 1000.0 * (
            time.perf_counter() - stage_start
        )

        affine_timing = {}
        stage_start = time.perf_counter()
        raw_divergence, n_inliers, fit_quality = self._fit_divergence_affine(
            flow_px_s=flow_px_s,
            image_width=image_width,
            image_height=image_height,
            gradient_magnitude=gradient_magnitude,
            pixel_scale=fit_pixel_scale,
            timing=affine_timing,
        )
        timing["affine_fit_ms"] = 1000.0 * (time.perf_counter() - stage_start)
        for key, value in affine_timing.items():
            timing[f"affine_{key}"] = value
        timing["affine_points_used"] = int(n_inliers)
        timing["affine_fit_quality"] = float(fit_quality)

        stage_start = time.perf_counter()
        filtered_divergence = self._filter_divergence(raw_divergence)
        timing["divergence_filter_ms"] = 1000.0 * (time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        prederotation_diagnostic_active = (
            derotation_active and self._compute_prederotation_diagnostic
        )
        if prederotation_diagnostic_active:
            divergence_prederotation, _, _ = self._fit_divergence_affine(
                flow_px_s=raw_flow_px_s,
                image_width=image_width,
                image_height=image_height,
                gradient_magnitude=gradient_magnitude,
                robust=False,
                pixel_scale=fit_pixel_scale,
            )
        else:
            # Do not report the post-de-rotation result as though it had been
            # measured before de-rotation. NaN clearly marks the diagnostic as
            # unavailable while preserving the existing FlowResult interface.
            divergence_prederotation = float("nan")
        timing["prederotation_diagnostic_active"] = int(
            prederotation_diagnostic_active
        )
        timing["prederotation_fit_ms"] = 1000.0 * (time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        result = FlowResult(
            timestamp=timestamp,
            valid=True,
            mean_flow_x=mean_flow_x,
            mean_flow_y=mean_flow_y,
            mean_flow_x_norm=float(mean_flow_x_norm),
            mean_flow_y_norm=float(mean_flow_y_norm),
            divergence=float(filtered_divergence),
            raw_divergence=float(raw_divergence),
            fit_quality=float(fit_quality),
            roi_x0=int(x0),
            roi_y0=int(y0),
            roi_x1=int(x1),
            roi_y1=int(y1),
            derotated=bool(derotation_active),
            mean_flow_x_raw=float(raw_mean_flow_x),
            mean_flow_y_raw=float(raw_mean_flow_y),
            divergence_prederotation=float(divergence_prederotation),
        )
        self._save_debug(
            result=result,
            previous_frame=previous_frame,
            current_frame=frame_bgr,
            flow_px_s=flow_px_s,
            divergence_field=divergence_field,
            raw_divergence=raw_divergence,
            filtered_divergence=filtered_divergence,
            roi=roi,
            message="",
        )
        self._prev_gray = gray
        self._prev_bgr = frame_bgr.copy()
        self._prev_timestamp = timestamp
        timing["result_and_state_ms"] = 1000.0 * (time.perf_counter() - stage_start)
        timing["valid"] = 1
        finish_timing()
        return result

    def _fit_divergence_affine(
        self,
        flow_px_s: np.ndarray,
        image_width: int,
        image_height: int,
        gradient_magnitude: Optional[np.ndarray] = None,
        robust: bool = True,
        pixel_scale: float = 1.0,
        timing: Optional[dict] = None,
    ) -> Tuple[float, int, float]:
        """Estimate the physical expansion rate ``lambda = -h_dot/h``.

        The previous implementation fitted two independent affine fields,

            u = a0 + a1*x + a2*y
            v = b0 + b1*x + b2*y,

        and returned ``a1 + b2``.  For isotropic fronto-parallel expansion that
        quantity is the FULL 2-D flow divergence ``2*lambda``.  From this
        revision onward the estimator fits the physically constrained model

            u = t_x + lambda*x - r*y
            v = t_y + r*x + lambda*y,

        whose four unknowns are ``[t_x, t_y, lambda, r]``.  ``lambda`` is then
        returned directly.  The legacy FlowResult field remains named
        ``divergence`` only for API/log compatibility.

        The solve is intentionally performed in ORIGINAL PIXEL coordinates:
        ``flow_px_s`` has already been amplitude-corrected by the caller after
        ROI downsampling, and adjacent samples represent ``1/pixel_scale``
        original pixels (times the fit stride).  Using pixel coordinates makes
        the common expansion/rotation model exact for square image pixels and
        avoids any aspect-ratio artefact from separately normalising x/y by
        image width/height.  Translation absorbs the arbitrary ROI-local origin.

        Weighting, spatial decimation, robust residual trimming and fit_quality
        keep the same semantics as before. An adaptive border ring is
        removed from the WORKING Farneback grid
        before spatial decimation; this tests whether boundary vectors with
        incomplete Farneback support are responsible for lambda attenuation.
        Setting ``affine_border_fraction=0`` and
        ``affine_border_min_px=0`` restores the previous fit
        selection exactly. The degenerate fallback uses the
        finite-difference field, which already returns half of the full 2-D
        divergence and therefore has the same physical ``lambda`` convention.
        """
        fit_wall_start = time.perf_counter()
        if timing is None:
            timing = {}

        roi_height, roi_width = flow_px_s.shape[:2]
        timing["input_points"] = int(roi_height * roi_width)

        # ------------------------------------------------------------------
        # Adaptive Farneback-border exclusion.
        #
        # Farneback still sees and estimates the COMPLETE working ROI. We only
        # discard the outer ring when selecting vectors for the 4x4 lambda fit.
        # The border is defined in WORKING-grid pixels and scales with the
        # smaller working dimension so far-field ROIs retain enough vectors
        # while near-field/full-frame ROIs reject a wider unreliable boundary.
        # The exclusion is applied before the existing affine-fit stride.
        working_min_dim = min(roi_height, roi_width)
        requested_border = int(round(
            self._affine_border_fraction * float(working_min_dim)
        ))
        requested_border = max(
            self._affine_border_min_px,
            min(self._affine_border_max_px, requested_border),
        )
        max_safe_border = max(0, (working_min_dim - 3) // 2)
        border = min(requested_border, max_safe_border)

        if border > 0:
            flow_for_fit = flow_px_s[
                border:roi_height - border,
                border:roi_width - border,
            ]
            if (
                gradient_magnitude is not None
                and gradient_magnitude.shape == (roi_height, roi_width)
            ):
                gradient_for_fit = gradient_magnitude[
                    border:roi_height - border,
                    border:roi_width - border,
                ]
            else:
                gradient_for_fit = None
        else:
            flow_for_fit = flow_px_s
            gradient_for_fit = (
                gradient_magnitude
                if gradient_magnitude is not None
                and gradient_magnitude.shape == (roi_height, roi_width)
                else None
            )

        cropped_height, cropped_width = flow_for_fit.shape[:2]
        timing["border_exclusion_px"] = int(border)
        timing["border_excluded_points"] = int(
            roi_height * roi_width - cropped_height * cropped_width
        )
        timing["border_fraction"] = float(self._affine_border_fraction)
        timing["border_min_px"] = int(self._affine_border_min_px)
        timing["border_max_px"] = int(self._affine_border_max_px)

        # Fit-only spatial decimation. No averaging/resizing is introduced here:
        # regular slicing keeps the measured vector values while the coordinate
        # spacing below is widened by the exact same stride.
        fit_stride = max(1, int(self._affine_fit_stride))
        if fit_stride > 1:
            flow_fit = flow_for_fit[::fit_stride, ::fit_stride]
            gradient_fit = (
                gradient_for_fit[::fit_stride, ::fit_stride]
                if gradient_for_fit is not None
                else None
            )
        else:
            flow_fit = flow_for_fit
            gradient_fit = gradient_for_fit

        fit_height, fit_width = flow_fit.shape[:2]
        timing["fit_stride"] = fit_stride
        timing["sampled_points"] = int(fit_height * fit_width)
        if fit_width < 3 or fit_height < 3:
            timing.update({
                "setup_ms": 1000.0 * (time.perf_counter() - fit_wall_start),
                "initial_solve_ms": 0.0,
                "residual_quantile_ms": 0.0,
                "refit_ms": 0.0,
                "finite_points": 0,
                "used_points": 0,
            })
            return 0.0, 0, 0.0

        # Keep the fit in original-pixel units. The caller has already divided
        # the downsampled Farneback vectors by ``pixel_scale``, so their values
        # are original px/s. Only the spatial coordinate spacing must be widened.
        u_flat = flow_fit[:, :, 0].ravel().astype(np.float64)
        v_flat = flow_fit[:, :, 1].ravel().astype(np.float64)

        s = max(1e-6, float(pixel_scale))
        spacing_px = fit_stride / s
        rows, cols = np.mgrid[0:fit_height, 0:fit_width]
        x = (cols * spacing_px).ravel().astype(np.float64)
        y = (rows * spacing_px).ravel().astype(np.float64)

        if gradient_fit is not None and gradient_fit.shape == (fit_height, fit_width):
            weight_flat = gradient_fit.ravel().astype(np.float64)
        else:
            weight_flat = np.ones_like(u_flat)

        finite = np.isfinite(u_flat) & np.isfinite(v_flat) & np.isfinite(weight_flat)
        n_finite = int(np.count_nonzero(finite))
        timing["finite_points"] = n_finite
        if n_finite < self._min_points_for_affine_fit:
            field = self._estimate_divergence_field(
                flow_px_s, image_width, image_height, pixel_scale=s
            )
            timing.update({
                "setup_ms": 1000.0 * (time.perf_counter() - fit_wall_start),
                "initial_solve_ms": 0.0,
                "residual_quantile_ms": 0.0,
                "refit_ms": 0.0,
                "used_points": n_finite,
            })
            return self._scalar_from_divergence_field(field), n_finite, 0.0

        x, y, u_flat, v_flat, weight_flat = (
            x[finite], y[finite], u_flat[finite], v_flat[finite], weight_flat[finite]
        )
        timing["setup_ms"] = 1000.0 * (time.perf_counter() - fit_wall_start)

        stage_start = time.perf_counter()
        coeffs, expansion_rate, fit_quality = self._weighted_affine_least_squares(
            x, y, u_flat, v_flat, weight_flat
        )
        timing["initial_solve_ms"] = 1000.0 * (time.perf_counter() - stage_start)

        if not robust:
            timing["residual_quantile_ms"] = 0.0
            timing["refit_ms"] = 0.0
            timing["used_points"] = n_finite
            return float(expansion_rate), n_finite, float(fit_quality)

        stage_start = time.perf_counter()
        tx, ty, lam, rot = coeffs
        u_pred = tx + lam * x - rot * y
        v_pred = ty + rot * x + lam * y
        residual = (u_flat - u_pred) ** 2 + (v_flat - v_pred) ** 2
        threshold = np.quantile(residual, self._affine_inlier_quantile)
        inliers = residual <= threshold
        timing["residual_quantile_ms"] = 1000.0 * (
            time.perf_counter() - stage_start
        )

        stage_start = time.perf_counter()
        n_inliers = int(np.count_nonzero(inliers))
        if n_inliers >= self._min_points_for_affine_fit:
            _, expansion_rate, fit_quality = self._weighted_affine_least_squares(
                x[inliers], y[inliers], u_flat[inliers], v_flat[inliers],
                weight_flat[inliers]
            )
            n_used = n_inliers
        else:
            n_used = n_finite
        timing["refit_ms"] = 1000.0 * (time.perf_counter() - stage_start)
        timing["used_points"] = n_used

        return float(expansion_rate), n_used, float(fit_quality)

    @staticmethod
    def _weighted_affine_least_squares(
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        weight: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Weighted 4x4 physical flow solve.

        Unknown vector::

            p = [t_x, t_y, lambda, r]^T

        with

            u = t_x + lambda*x - r*y
            v = t_y + r*x + lambda*y.

        Rather than building a ``(2N)x4`` matrix, the 4x4 normal equations are
        formed from weighted scalar sums. This keeps the solve cheaper than the
        previous pair of 3x3 affine solves and avoids allocating a stacked design
        matrix on every frame. ``lambda`` and ``r`` both have units 1/s when x/y
        are pixels and u/v are px/s.
        """
        w = np.clip(weight, 0.0, None).astype(np.float64, copy=False)
        w_mean = float(np.mean(w)) if np.any(w > 0.0) else 0.0
        if w_mean <= 1e-12:
            w = np.ones_like(w)
            w_mean = 1.0
        w = w / w_mean

        # Weighted sums for X^T W X and X^T W b. The lambda/rotation cross-term
        # cancels identically: x*(-y) + y*x == 0 for every flow vector.
        s0 = float(np.sum(w))
        sx = float(np.sum(w * x))
        sy = float(np.sum(w * y))
        srr = float(np.sum(w * (x * x + y * y)))

        su = float(np.sum(w * u))
        sv = float(np.sum(w * v))
        slam = float(np.sum(w * (x * u + y * v)))
        srot = float(np.sum(w * (-y * u + x * v)))

        AtA = np.array([
            [s0, 0.0, sx, -sy],
            [0.0, s0, sy, sx],
            [sx, sy, srr, 0.0],
            [-sy, sx, 0.0, srr],
        ], dtype=np.float64)
        Atb = np.array([su, sv, slam, srot], dtype=np.float64)

        try:
            coeffs = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            # Rare degenerate fallback: build the explicit weighted system only
            # when the compact normal equations cannot be solved.
            zeros = np.zeros_like(x)
            ones = np.ones_like(x)
            design_u = np.column_stack([ones, zeros, x, -y])
            design_v = np.column_stack([zeros, ones, y, x])
            sw = np.sqrt(w)
            design = np.vstack([design_u * sw[:, None], design_v * sw[:, None]])
            target = np.concatenate([u * sw, v * sw])
            coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)

        tx, ty, lam, rot = coeffs
        pred_u = tx + lam * x - rot * y
        pred_v = ty + rot * x + lam * y
        resid_u = u - pred_u
        resid_v = v - pred_v
        ss_res = float(np.sum(w * (resid_u ** 2 + resid_v ** 2)))

        u_mean_w = float(np.sum(w * u) / np.sum(w))
        v_mean_w = float(np.sum(w * v) / np.sum(w))
        ss_tot = float(
            np.sum(w * (u - u_mean_w) ** 2)
            + np.sum(w * (v - v_mean_w) ** 2)
        )
        fit_quality = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        return coeffs, float(lam), float(fit_quality)

    @staticmethod
    def _gradient_magnitude(gray_roi: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude of a grayscale ROI -- the structural-
        reliability weight for the affine divergence fit (see
        _fit_divergence_affine). A flat/texture-poor patch has near-zero
        gradient here and is the classic aperture-problem case: Farneback's
        own polynomial-expansion estimate is least trustworthy exactly
        where this is smallest."""
        gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def _downsample_scale_for_roi(self, roi_width: int, roi_height: int) -> float:
        """scale = clip(downsample_target_px / max(roi_w, roi_h), min_scale, 1.0)

        1.0 (no downsampling) for any ROI already at or below the target size
        -- the far-field regime is unaffected. Shrinks continuously, more for
        a bigger ROI, once it exceeds the target -- replaces the old binary
        fov_saturated switch to a bigger search window with a continuous
        switch to a smaller search problem. See constructor docstring.
        """
        largest_dim = max(int(roi_width), int(roi_height), 1)
        scale = self._downsample_target_px / float(largest_dim)
        return max(self._downsample_min_scale, min(1.0, scale))

    def reset(self):
        self._prev_gray = None
        self._prev_bgr = None
        self._prev_timestamp = None

        self._filtered_divergence = 0.0
        self._has_filtered_divergence = False

        self._last_debug = {}
        self._last_timing = {}

    def last_debug_data(self) -> dict:
        return dict(self._last_debug)

    def last_timing_data(self) -> dict:
        """Return the immutable scalar timing snapshot from the latest update."""
        return dict(self._last_timing)

    def _target_roi_from_estimate(
        self,
        target: Optional[TargetEstimate],
        image_width: int,
        image_height: int,
    ) -> Optional[ROI]:
        if target is None or not target.found:
            return None

        detection_width = float(getattr(target, "detection_width", 0.0))
        detection_height = float(getattr(target, "detection_height", 0.0))

        if detection_width <= 1.0 or detection_height <= 1.0:
            return None

        cx = (0.5 * float(target.offset_x) + 0.5) * image_width
        cy = (0.5 * float(target.offset_y) + 0.5) * image_height

        margin_x = self._roi_margin_fraction * detection_width
        margin_y = self._roi_margin_fraction * detection_height

        roi_width = max(
            detection_width + 2.0 * margin_x,
            float(self._min_roi_size_px),
        )

        roi_height = max(
            detection_height + 2.0 * margin_y,
            float(self._min_roi_size_px),
        )

        x0 = int(round(cx - 0.5 * roi_width))
        y0 = int(round(cy - 0.5 * roi_height))
        x1 = int(round(cx + 0.5 * roi_width))
        y1 = int(round(cy + 0.5 * roi_height))

        x0 = max(0, min(image_width - 1, x0))
        y0 = max(0, min(image_height - 1, y0))
        x1 = max(x0 + 1, min(image_width, x1))
        y1 = max(y0 + 1, min(image_height, y1))

        if x1 - x0 < 3 or y1 - y0 < 3:
            return None

        return (x0, y0, x1, y1)

    def _estimate_divergence_field(
        self,
        flow_px_s: np.ndarray,
        image_width: int,
        image_height: int,
        pixel_scale: float = 1.0,
    ) -> np.ndarray:
        """Per-pixel fallback/debug estimate of physical ``lambda`` [1/s].

        Work directly in original-pixel units, matching the constrained 4x4
        fit. ``flow_px_s`` is already amplitude-corrected to original px/s;
        adjacent working-grid samples are ``1/pixel_scale`` original pixels
        apart. For isotropic expansion, ``du/dx = dv/dy = lambda``, hence the
        half-trace below returns ``lambda`` exactly rather than the full 2-D
        divergence ``2*lambda``.
        """
        roi_height, roi_width = flow_px_s.shape[:2]

        if roi_width < 3 or roi_height < 3:
            return np.zeros((roi_height, roi_width), dtype=np.float32)

        s = max(1e-6, float(pixel_scale))
        spacing_px = 1.0 / s
        du_dx = np.gradient(flow_px_s[:, :, 0], spacing_px, axis=1)
        dv_dy = np.gradient(flow_px_s[:, :, 1], spacing_px, axis=0)

        return 0.5 * (du_dx + dv_dy)

    @staticmethod
    def _scalar_from_divergence_field(divergence_field: np.ndarray) -> float:
        if divergence_field is None or divergence_field.size == 0:
            return 0.0

        return float(np.median(divergence_field))

    def _filter_divergence(self, divergence: float) -> float:
        alpha = max(0.0, min(1.0, self._divergence_smoothing))

        if not self._has_filtered_divergence:
            self._filtered_divergence = float(divergence)
            self._has_filtered_divergence = True
        else:
            self._filtered_divergence = (
                alpha * self._filtered_divergence
                + (1.0 - alpha) * float(divergence)
            )

        return self._filtered_divergence

    def _save_debug(
        self,
        result: FlowResult,
        previous_frame=None,
        current_frame=None,
        flow_px_s: Optional[np.ndarray] = None,
        divergence_field: Optional[np.ndarray] = None,
        raw_divergence: float = 0.0,
        filtered_divergence: float = 0.0,
        roi: Optional[ROI] = None,
        message: str = "",
    ):
        if not self._store_debug:
            self._last_debug = {}
            return

        self._last_debug = {
            "result": result,
            "previous_frame": previous_frame,
            "current_frame": current_frame,
            "flow_px_s": flow_px_s,
            "divergence_field": divergence_field,
            "raw_divergence": float(raw_divergence),
            "filtered_divergence": float(filtered_divergence),
            "roi": roi,
            "message": message,
        }


if __name__ == "__main__":
    # The debug harness lives in bee_control/tests/, not beside this module.
    #
    # Run it with `python -m bee_control.vision.optical_flow` from the repo
    # root. A bare `python optical_flow.py` cannot work and never could: this
    # module's own top-level `from bee_control.core.state import ...` already
    # requires the package to be importable, and that runs long before this
    # block. -m is the supported invocation.
    #
    # NOTE: setup.py excludes tests/ from the installed package, so this is a
    # run-from-source tool only.
    from bee_control.tests._optical_flow_debug import test

    test()