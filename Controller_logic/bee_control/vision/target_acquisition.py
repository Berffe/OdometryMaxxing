"""
Lightweight, NN-free target acquisition, decoupled from ROS.

    update(frame_bgr, timestamp) -> TargetEstimate

Pipeline: blur -> HSV colorfulness mask -> STATIC OCCLUSION REMOVAL ->
morphological cleanup -> contour selection -> centroid -> normalized offsets +
bounding box. The box is returned so optical_flow can restrict divergence to
the target ROI; area_fraction is the controller's scheduling variable.

area_fraction is CONTOUR area over VISIBLE frame area
-----------------------------------------------------
Two things about that name are easy to get wrong:

  * It is ``cv2.contourArea``, not the bounding box. A disc inscribed in a
    square frame is pi/4 = 0.785 of it, so a circular flower that perfectly
    fills the view reports 0.785, never 1.0.
  * The denominator is the number of pixels the camera can actually see the
    world through: frame area MINUS the static occlusion (see below). A view
    containing nothing but flower and landing gear therefore reports ~1.0,
    which is what the approach schedule assumes.

Static occlusion (the landing gear)
-----------------------------------
The legs sit in the camera's field of view at FIXED image coordinates. They are
gray, so the HSV colorfulness mask already rejects them -- but they still
occupied the denominator, which capped area_fraction well below what the
scheduler expected, and they severed the flower region at the left and right
borders so it could never touch them. That made fov_saturated (all four
borders) structurally unreachable until the flower overran the legs entirely.

Both are fixed by declaring the occlusion once, in NORMALIZED coordinates, and
(a) removing it from the candidate mask, (b) removing it from the area
denominator, and (c) measuring border contact against the VISIBLE region rather
than the array bounds. With no occlusion declared every one of those reduces
exactly to the previous behavior.

Use ``TargetAcquisition.calibrate_occlusion(frames)`` to re-derive the rectangles
for a different camera mount; DEFAULT_LEG_OCCLUSION is only correct for the
mount it was measured on.

Two robustness behaviors:
  - Large detections are down-weighted, never rejected (_large_area_penalty):
    a near-full-frame flower at touchdown is the success condition, not an
    outlier.
  - A short temporal hold bridges single-frame dropouts (_held_target_or_lost):
    within loss_grace_period_sec the last good estimate is reused with confidence
    decayed toward zero, instead of immediately reporting found=False.

TargetEstimate.fov_saturated flags when the box touches all four borders OF THE
VISIBLE REGION: the target's true size meets or exceeds the camera's usable
field of view, not just fills it. Past that point area_fraction/detection_width/
height are a frame-size artifact, not a measurement -- see state.py.

Run `python -m bee_control.vision.target_acquisition` from the repo root to
launch the visual debug test; the harness lives in
bee_control/tests/_target_acquisition_debug.py.
"""

from typing import Optional, Sequence, Tuple
from dataclasses import replace

import cv2
import numpy as np

from bee_control.core.state import FlowResult, TargetEstimate
HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]

#: An occluded rectangle in NORMALIZED frame coordinates, ``(x0, y0, x1, y1)``
#: with 0.0 = left/top and 1.0 = right/bottom. Normalized so one declaration
#: survives a camera resolution change.
OcclusionRect = Tuple[float, float, float, float]

#: The landing gear as seen by the down-facing camera, measured off the
#: 2026-08-19 100x100 flight frames: the two struts run down the outer 8% of
#: each side for the upper ~80% of the frame, each with a small bracket bulging
#: inward near mid-height.
#:
#: These are specific to THIS camera mount. Re-derive with
#: ``TargetAcquisition.calibrate_occlusion()`` if the mount, lens or camera
#: pose changes; a stale occlusion mask silently biases area_fraction, which is
#: the approach scheduler's range coordinate.
DEFAULT_LEG_OCCLUSION: Tuple[OcclusionRect, ...] = (
    (0.00, 0.00, 0.08, 0.80),   # left strut
    (0.92, 0.00, 1.00, 0.81),   # right strut
    (0.08, 0.33, 0.12, 0.47),   # left bracket
    (0.87, 0.33, 0.92, 0.47),   # right bracket
)


class TargetAcquisition:
    def __init__(
        self,
        hsv_ranges: Optional[Sequence[HSVRange]] = None,
        min_area_px: float = 30.0,
        max_area_fraction: float = 0.94,
        absolute_max_area_fraction: float = 0.95,
        min_large_area_penalty: float = 0.5,
        blur_kernel_size: int = 3,
        morph_kernel_size: int = 3,
        morph_open_iterations: int = 1,
        morph_close_kernel_size: Optional[int] = None,
        morph_close_iterations: int = 2,
        occlusion_rects: Optional[Sequence[OcclusionRect]] = DEFAULT_LEG_OCCLUSION,
        min_saturation: int = 60,
        min_value: int = 45,
        canny_low: int = 60,
        canny_high: int = 140,
        loss_grace_period_sec: float = 0.15,
        fov_saturation_margin_px: int = 2,
    ):
        self._hsv_ranges = list(hsv_ranges) if hsv_ranges is not None else None

        self._min_area_px = float(min_area_px)
        self._max_area_fraction = float(max_area_fraction)
        self._absolute_max_area_fraction = float(absolute_max_area_fraction)
        self._min_large_area_penalty = float(min_large_area_penalty)

        self._blur_kernel_size = self._make_odd(blur_kernel_size)

        # Cleanup is split into its two halves because they do opposite jobs and
        # want opposite tuning. OPEN erases speckle and must stay small, or it
        # eats the thin petal tips that carry the target's true extent. CLOSE
        # bridges holes punched by dark, low-saturation petal shadows.
        #
        # Measured on the 2026-08-19 flight frames at the native 100x100 camera
        # resolution: the RAW mask holds at most 1.4% interior holes, and OPEN
        # 3x1 + CLOSE 3x2 already drives that to 0.00% on EVERY sampled frame.
        # There is nothing left for a stronger CLOSE to fix, and it is not free:
        # 5x2 inflates area_fraction by +1.5% and 5x3 by +2.9%, straight into
        # the approach scheduler's range coordinate. Hence the defaults below.
        # The knobs exist so a different flower texture can be handled without
        # a code change -- not because the current values need raising.
        self._morph_kernel_size = self._make_odd(morph_kernel_size)
        self._morph_open_iterations = max(0, int(morph_open_iterations))
        self._morph_close_kernel_size = self._make_odd(
            morph_kernel_size
            if morph_close_kernel_size is None
            else morph_close_kernel_size
        )
        self._morph_close_iterations = max(0, int(morph_close_iterations))

        # Static occlusion, resolved lazily per frame size and cached: the
        # rasterized mask plus the per-row/per-column visible extents that
        # _is_fov_saturated needs. See _occlusion_for().
        self._occlusion_rects: Tuple[OcclusionRect, ...] = (
            tuple(tuple(float(v) for v in r) for r in occlusion_rects)
            if occlusion_rects
            else ()
        )
        self._occlusion_cache: dict = {}

        self._min_saturation = int(min_saturation)
        self._min_value = int(min_value)

        self._canny_low = int(canny_low)
        self._canny_high = int(canny_high)

        # Temporal hold: lets a brief detection dropout reuse the last
        # known-good TargetEstimate instead of immediately going to
        # found=False. See _held_target_or_lost().
        self._loss_grace_period_sec = float(loss_grace_period_sec)
        self._last_found_target: Optional[TargetEstimate] = None
        self._last_found_time: Optional[float] = None
        self._fov_saturation_margin_px = max(0, int(fov_saturation_margin_px))

    def update(
        self,
        frame_bgr,
        flow_result: Optional[FlowResult] = None,
        timestamp: Optional[float] = None,
    ) -> TargetEstimate:
        """
        Production function used by bee_node.py.

        Returns only TargetEstimate:

            timestamp
            found
            offset_x
            offset_y
            confidence
            detection_width
            detection_height
            area_fraction
        """
        timestamp = self._resolve_timestamp(flow_result, timestamp)

        if frame_bgr is None:
            return TargetEstimate(timestamp=timestamp, found=False)

        height, width = frame_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return TargetEstimate(timestamp=timestamp, found=False)

        masks = self._build_masks(frame_bgr)
        contour = self._select_best_contour(masks["clean_mask"], width, height)

        return self._resolve_target(contour, width, height, timestamp)

    def process_debug(
        self,
        frame_bgr,
        flow_result: Optional[FlowResult] = None,
        timestamp: Optional[float] = None,
    ) -> dict:
        """
        Debug helper used by target_acquisition_debug.py.

        This function is not used by the controller. It exposes the
        intermediate masks and selected contour for visualization.
        """
        timestamp = self._resolve_timestamp(flow_result, timestamp)

        if frame_bgr is None:
            return {
                "frame": None,
                "blurred": None,
                "hsv_mask": None,
                "edges": None,
                "occlusion_mask": None,
                "combined_mask": None,
                "clean_mask": None,
                "contour": None,
                "target": TargetEstimate(timestamp=timestamp, found=False),
            }

        height, width = frame_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return {
                "frame": frame_bgr,
                "blurred": frame_bgr.copy(),
                "hsv_mask": None,
                "edges": None,
                "occlusion_mask": None,
                "combined_mask": None,
                "clean_mask": None,
                "contour": None,
                "target": TargetEstimate(timestamp=timestamp, found=False),
            }

        masks = self._build_masks(frame_bgr)
        contour = self._select_best_contour(masks["clean_mask"], width, height)

        target = self._resolve_target(contour, width, height, timestamp)

        return {
            "frame": frame_bgr,
            "blurred": masks["blurred"],
            "hsv_mask": masks["hsv_mask"],
            "edges": masks["edges"],
            "occlusion_mask": masks["occlusion_mask"],
            "combined_mask": masks["combined_mask"],
            "clean_mask": masks["clean_mask"],
            "contour": contour,
            "target": target,
        }

    def reset(self):
        """Clear temporal memory (the held last-known-good target)."""
        self._last_found_target = None
        self._last_found_time = None

    def _resolve_target(
        self,
        contour,
        width: int,
        height: int,
        timestamp: float,
    ) -> TargetEstimate:
        """
        Turn a selected contour (or lack of one) into a TargetEstimate,
        updating/consulting the temporal hold as needed.
        """
        if contour is None:
            return self._held_target_or_lost(timestamp)

        target = self._target_from_contour(contour, width, height, timestamp)
        self._last_found_target = target
        self._last_found_time = timestamp

        return target

    def _held_target_or_lost(self, timestamp: float) -> TargetEstimate:
        """
        Bridge brief detection dropouts. Within loss_grace_period_sec of the
        last good detection, reuse it with confidence linearly decayed toward
        zero (decay = 1 - elapsed/grace), so downstream consumers can tell the
        estimate is aging rather than freshly detected. Otherwise report lost.
        """
        if (
            self._loss_grace_period_sec > 0.0
            and self._last_found_target is not None
            and self._last_found_time is not None
        ):
            elapsed = timestamp - self._last_found_time

            if 0.0 <= elapsed <= self._loss_grace_period_sec:
                decay = 1.0 - (elapsed / self._loss_grace_period_sec)

                return replace(
                    self._last_found_target,
                    timestamp=timestamp,
                    confidence=self._last_found_target.confidence * decay,
                )

        return TargetEstimate(timestamp=timestamp, found=False)

    def _occlusion_for(self, width: int, height: int) -> dict:
        """Rasterize the declared occlusion for this frame size, once.

        Returns the mask plus everything derived from it that the rest of the
        class needs, so no consumer re-derives geometry per frame:

            ``mask``            uint8, 255 where the camera cannot see the world
            ``visible_px``      the area_fraction denominator
            ``first_col``/``last_col``   per ROW, the visible column extent
            ``first_row``/``last_row``   per COLUMN, the visible row extent

        A fully occluded row or column would leave its extent undefined; those
        entries are filled with the array bounds so the border tests below
        degrade to the un-occluded behavior instead of raising.
        """
        key = (int(width), int(height))
        cached = self._occlusion_cache.get(key)
        if cached is not None:
            return cached

        mask = np.zeros((height, width), dtype=np.uint8)

        for x0, y0, x1, y1 in self._occlusion_rects:
            # Normalized -> pixel, clamped, and always at least one pixel wide
            # so a thin declared strip cannot silently rasterize to nothing.
            px0 = int(np.clip(round(x0 * width), 0, width - 1))
            px1 = int(np.clip(round(x1 * width), 0, width))
            py0 = int(np.clip(round(y0 * height), 0, height - 1))
            py1 = int(np.clip(round(y1 * height), 0, height))
            px1 = max(px1, px0 + 1)
            py1 = max(py1, py0 + 1)
            mask[py0:py1, px0:px1] = 255

        visible = mask == 0
        visible_px = int(visible.sum())

        cols = np.arange(width)
        rows = np.arange(height)

        # np.where on an all-False row would return an empty array; using
        # min/max over a masked index array keeps it branchless.
        col_idx = np.where(visible, cols[None, :], width)
        first_col = col_idx.min(axis=1)
        col_idx = np.where(visible, cols[None, :], -1)
        last_col = col_idx.max(axis=1)

        row_idx = np.where(visible, rows[:, None], height)
        first_row = row_idx.min(axis=0)
        row_idx = np.where(visible, rows[:, None], -1)
        last_row = row_idx.max(axis=0)

        # Fully occluded line -> fall back to the array bound.
        first_col = np.where(first_col >= width, 0, first_col)
        last_col = np.where(last_col < 0, width - 1, last_col)
        first_row = np.where(first_row >= height, 0, first_row)
        last_row = np.where(last_row < 0, height - 1, last_row)

        resolved = {
            "mask": mask,
            "visible_px": visible_px if visible_px > 0 else width * height,
            "first_col": first_col,
            "last_col": last_col,
            "first_row": first_row,
            "last_row": last_row,
        }
        self._occlusion_cache[key] = resolved
        return resolved

    @staticmethod
    def calibrate_occlusion(
        frames: Sequence,
        min_saturation: int = 60,
        min_value: int = 45,
        blur_kernel_size: int = 3,
    ) -> np.ndarray:
        """Derive an occlusion mask from frames in which the target FILLS the view.

        The landing gear is the only thing that is simultaneously static in image
        coordinates and never colorful, so on frames where the flower covers
        everything else, "never passed the HSV mask in ANY frame" isolates it.

        Pass the last second or two before touchdown. Returns a uint8 mask (255 =
        occluded) to inspect and convert into normalized rectangles; it is
        deliberately NOT wired in automatically, because a calibration run with a
        partially visible target would quietly mask off real flower.
        """
        accumulated = None

        for frame in frames:
            blurred = cv2.GaussianBlur(
                frame,
                (TargetAcquisition._make_odd(blur_kernel_size),) * 2,
                0,
            )
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            colorful = cv2.inRange(
                hsv,
                np.array((0, int(min_saturation), int(min_value)), dtype=np.uint8),
                np.array((179, 255, 255), dtype=np.uint8),
            )
            accumulated = (
                colorful.astype(np.uint32)
                if accumulated is None
                else accumulated + colorful
            )

        if accumulated is None:
            return np.zeros((0, 0), dtype=np.uint8)

        return ((accumulated == 0).astype(np.uint8)) * 255

    def _build_masks(self, frame_bgr) -> dict:
        blurred = cv2.GaussianBlur(
            frame_bgr,
            (self._blur_kernel_size, self._blur_kernel_size),
            0,
        )

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        if self._hsv_ranges:
            hsv_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in self._hsv_ranges:
                lower_arr = np.array(lower, dtype=np.uint8)
                upper_arr = np.array(upper, dtype=np.uint8)

                hsv_mask = cv2.bitwise_or(
                    hsv_mask,
                    cv2.inRange(hsv, lower_arr, upper_arr),
                )
        else:
            hsv_mask = cv2.inRange(
                hsv,
                np.array((0, self._min_saturation, self._min_value), dtype=np.uint8),
                np.array((179, 255, 255), dtype=np.uint8),
            )

        # Target identity is based on colorfulness rather than generic image
        # structure.  The broad HSV mask accepts any hue, but requires enough
        # saturation and brightness to reject gray shadows, landing-gear edges,
        # and most background texture.  Canny is intentionally not part of the
        # production candidate mask: otherwise any strong edge can become a
        # competing contour even when it has no flower-like colorfulness.
        edges = np.zeros(hsv.shape[:2], dtype=np.uint8)
        combined_mask = hsv_mask.copy()

        # Static occlusion (landing gear). The gray struts already fail the HSV
        # test, so this rarely deletes a lit pixel -- it is here for the frames
        # where a specular highlight or reflected flower color on the strut
        # WOULD pass, and to keep one definition of "not the world" shared with
        # the area denominator and the border test.
        height_px, width_px = hsv.shape[:2]
        occlusion = self._occlusion_for(width_px, height_px)

        if self._occlusion_rects:
            combined_mask = cv2.bitwise_and(
                combined_mask,
                cv2.bitwise_not(occlusion["mask"]),
            )

        open_kernel = np.ones(
            (self._morph_kernel_size, self._morph_kernel_size),
            dtype=np.uint8,
        )
        close_kernel = np.ones(
            (self._morph_close_kernel_size, self._morph_close_kernel_size),
            dtype=np.uint8,
        )

        clean_mask = combined_mask

        if self._morph_open_iterations > 0:
            clean_mask = cv2.morphologyEx(
                clean_mask,
                cv2.MORPH_OPEN,
                open_kernel,
                iterations=self._morph_open_iterations,
            )

        if self._morph_close_iterations > 0:
            clean_mask = cv2.morphologyEx(
                clean_mask,
                cv2.MORPH_CLOSE,
                close_kernel,
                iterations=self._morph_close_iterations,
            )

        # CLOSE dilates before it erodes, so it can spill back over a strut and
        # re-annex occluded pixels. Re-apply the occlusion afterwards, or the
        # contour reaches a border the camera cannot actually see through.
        if self._occlusion_rects:
            clean_mask = cv2.bitwise_and(
                clean_mask,
                cv2.bitwise_not(occlusion["mask"]),
            )

        return {
            "blurred": blurred,
            "hsv_mask": hsv_mask,
            "edges": edges,
            "occlusion_mask": occlusion["mask"],
            "combined_mask": combined_mask,
            "clean_mask": clean_mask,
        }

    def _select_best_contour(self, mask: np.ndarray, width: int, height: int):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        # Score on the same visible-area denominator the reported
        # area_fraction uses, so the saturation point of area_score and the
        # large-area penalty knee mean the same thing in both places.
        image_area = float(self._occlusion_for(width, height)["visible_px"])

        best_contour = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area_px:
                continue

            area_fraction = area / image_area

            perimeter = cv2.arcLength(contour, closed=True)
            if perimeter <= 1e-6:
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue

            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            compactness = 4.0 * np.pi * area / (perimeter * perimeter)
            compactness = max(0.0, min(1.0, compactness))

            nx = (cx - 0.5 * width) / (0.5 * width)
            ny = (cy - 0.5 * height) / (0.5 * height)

            center_distance = min(1.0, (nx * nx + ny * ny) ** 0.5)
            center_score = 1.0 - center_distance

            area_score = min(1.0, area_fraction / 0.05)
            large_area_penalty = self._large_area_penalty(area_fraction)

            score = (
                0.50 * area_score
                + 0.25 * compactness
                + 0.25 * center_score
            )

            score *= large_area_penalty

            if score > best_score:
                best_score = score
                best_contour = contour

        return best_contour

    def _target_from_contour(
        self,
        contour,
        width: int,
        height: int,
        timestamp: float,
    ) -> TargetEstimate:
        moments = cv2.moments(contour)

        if abs(moments["m00"]) < 1e-6:
            return TargetEstimate(timestamp=timestamp, found=False)

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        offset_x = (cx - 0.5 * width) / (0.5 * width)
        offset_y = (cy - 0.5 * height) / (0.5 * height)

        area = cv2.contourArea(contour)

        # Denominator is the VISIBLE frame area: flower + landing gear filling
        # the view reports ~1.0, not ~0.87. offset_x/offset_y and the detection
        # box stay in full-frame pixels -- they are image coordinates for the
        # control loop, not a coverage measure.
        area_fraction = area / float(self._occlusion_for(width, height)["visible_px"])

        _, _, detection_width, detection_height = cv2.boundingRect(contour)

        confidence = self._estimate_confidence(contour, area_fraction)
        fov_saturated = self._is_fov_saturated(contour, width, height)

        return TargetEstimate(
            timestamp=timestamp,
            found=True,
            offset_x=float(offset_x),
            offset_y=float(offset_y),
            confidence=float(confidence),
            detection_width=float(detection_width),
            detection_height=float(detection_height),
            area_fraction=float(area_fraction),
            fov_saturated=fov_saturated,
        )

    def _large_area_penalty(self, area_fraction: float) -> float:
        """
        Down-weight (never reject) contours covering a large frame fraction; a
        full-frame flower at touchdown is the success condition. The penalty
        ramps linearly 1.0 -> min_large_area_penalty as area_fraction goes from
        max_area_fraction to absolute_max_area_fraction, then holds at that floor.
        """
        if area_fraction <= self._max_area_fraction:
            return 1.0

        span = max(
            self._absolute_max_area_fraction - self._max_area_fraction,
            1e-6,
        )

        ratio = (area_fraction - self._max_area_fraction) / span
        ratio = max(0.0, min(1.0, ratio))

        return 1.0 - (1.0 - self._min_large_area_penalty) * ratio

    def _is_fov_saturated(self, contour, width: int, height: int) -> bool:
        """
        True if the contour's bounding box touches all four borders of the
        VISIBLE region (within fov_saturation_margin_px) -- the true target's
        projected size meets or exceeds the camera's usable field of view, not
        just fills it. cv2.boundingRect cannot report a box larger than the
        image array, so area_fraction/detection_width/height stop tracking true
        range past this point, regardless of how much closer the target
        actually gets.

        Measuring against the VISIBLE region rather than the array bounds is
        what keeps this flag reachable at all. The landing gear covers the outer
        columns, so a contour can never reach x <= margin while the struts are
        declared -- testing against the raw array would make saturation
        structurally impossible and silently pin the flag to False for the whole
        flight. Each side is compared against the visible extent over the rows
        (or columns) the box actually spans, not a frame-wide extent, because
        the struts stop partway down and a frame-wide bound would be the array
        bound again.

        With no occlusion declared, every limit below collapses to 0 / width-1 /
        height-1 and this is exactly the previous test.
        """
        x, y, w, h = cv2.boundingRect(contour)
        m = self._fov_saturation_margin_px

        if w <= 0 or h <= 0:
            return False

        occlusion = self._occlusion_for(width, height)

        rows = slice(max(0, y), min(height, y + h))
        cols = slice(max(0, x), min(width, x + w))

        left_limit = int(occlusion["first_col"][rows].min())
        right_limit = int(occlusion["last_col"][rows].max())
        top_limit = int(occlusion["first_row"][cols].min())
        bottom_limit = int(occlusion["last_row"][cols].max())

        return (
            x <= left_limit + m
            and y <= top_limit + m
            and (x + w) >= right_limit + 1 - m
            and (y + h) >= bottom_limit + 1 - m
        )

    @staticmethod
    def _estimate_confidence(contour, area_fraction: float) -> float:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        if perimeter <= 1e-6:
            return 0.0

        compactness = 4.0 * np.pi * area / (perimeter * perimeter)
        compactness = max(0.0, min(1.0, compactness))

        area_score = min(1.0, area_fraction / 0.05)

        confidence = 0.70 * area_score + 0.30 * compactness
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _resolve_timestamp(
        flow_result: Optional[FlowResult],
        timestamp: Optional[float],
    ) -> float:
        if timestamp is not None:
            return float(timestamp)

        if flow_result is not None:
            return float(flow_result.timestamp)

        return 0.0

    @staticmethod
    def _make_odd(value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1


if __name__ == "__main__":
    # The debug harness lives in bee_control/tests/, not beside this module.
    #
    # Run it with `python -m bee_control.vision.target_acquisition` from the
    # repo root. A bare `python target_acquisition.py` cannot work and never
    # could: this module's own top-level `from bee_control.core.state import
    # ...` already requires the package to be importable, and that runs long
    # before this block. -m is the supported invocation.
    #
    # NOTE: setup.py excludes tests/ from the installed package, so this is a
    # run-from-source tool only.
    from bee_control.tests._target_acquisition_debug import test

    test()