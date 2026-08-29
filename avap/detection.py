"""Phase 2 mask generation and detection-tool measurements.

The ROI mask is supplied by the alignment/coordinate-mapping stage.  This
module owns the one HSV/morphology mask path required by Design Law L6 and
keeps the ROI denominator beside the resulting foreground mask.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from avap.alignment import Pose, transform_points
from avap.constants import HSV_CHANNEL_SCALES
from avap.recipe import Rule


class DetectionInputError(ValueError):
    """Raised when an image or binary-mask input violates the tool contract."""


@dataclass(frozen=True)
class DetectionMask:
    """Final foreground mask and the ROI pixels that define its denominator."""

    foreground: np.ndarray
    roi: np.ndarray


@dataclass(frozen=True)
class CoverageMeasurement:
    coverage: float
    continuity: float | None
    foreground_pixels: int
    roi_pixels: int
    largest_component_pixels: int


@dataclass(frozen=True)
class CoverageResult:
    measurement: CoverageMeasurement
    passed: bool
    failed_params: tuple[str, ...]


@dataclass(frozen=True)
class BlobMeasurement:
    """One 8-connected component, measured against the ROI that contains it."""

    pixels: int
    area: float
    circularity: float
    solidity: float
    aspect_ratio: float


@dataclass(frozen=True)
class BlobRejection:
    """The single threshold that removed one measured blob before counting."""

    blob: BlobMeasurement
    param: str
    measured: float
    threshold: float
    operator: str


@dataclass(frozen=True)
class BlobResult:
    kept: tuple[BlobMeasurement, ...]
    rejected: tuple[BlobRejection, ...]
    passed: bool
    failed_params: tuple[str, ...]


# Shape thresholds that remove a blob, in the order they are tested.  Every
# non-count key of PARAM_SPECS["blob"] must appear here or it would load fine
# and then never be read - a dead parameter (L1).  A test enforces both
# directions of that correspondence.
BLOB_FILTERS: tuple[tuple[str, str, str], ...] = (
    ("area_min", "area", "<"),
    ("area_max", "area", ">"),
    ("circularity_min", "circularity", "<"),
    ("circularity_max", "circularity", ">"),
    ("solidity_min", "solidity", "<"),
    ("aspect_ratio_min", "aspect_ratio", "<"),
    ("aspect_ratio_max", "aspect_ratio", ">"),
)


def _binary_mask(value: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape != shape:
        actual = getattr(value, "shape", None)
        raise DetectionInputError(f"{name}: 2D mask shape {shape} required - {actual!r}")
    if value.dtype != np.bool_ and value.dtype != np.uint8:
        raise DetectionInputError(f"{name}: bool or uint8 mask required - {value.dtype}")
    unique = np.unique(value)
    if not all(item in (0, 1, 255) for item in unique):
        raise DetectionInputError(f"{name}: binary values 0/1/255 required - {unique.tolist()}")
    return np.where(value != 0, 255, 0).astype(np.uint8)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise DetectionInputError(f"{name}: mapping required") from exc


def _hsv_bounds(detect: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_lower = np.asarray(detect.get("lower"))
    raw_upper = np.asarray(detect.get("upper"))
    if (raw_lower.shape != (3,) or raw_upper.shape != (3,)
            or raw_lower.dtype == np.bool_ or raw_upper.dtype == np.bool_
            or not np.issubdtype(raw_lower.dtype, np.number)
            or not np.issubdtype(raw_upper.dtype, np.number)):
        raise DetectionInputError("detect.lower/upper: three numeric 0..1 values required")
    lower = raw_lower.astype(np.float64)
    upper = raw_upper.astype(np.float64)
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or (lower < 0.0).any() or (lower > 1.0).any()
            or (upper < 0.0).any() or (upper > 1.0).any()):
        raise DetectionInputError("detect.lower/upper: three finite 0..1 values required")
    if (lower[1:] > upper[1:]).any():
        raise DetectionInputError("detect.lower: S/V cannot exceed detect.upper")
    return lower, upper


def make_roi_mask(
    rect_golden: Sequence[float],
    pose: Pose,
    golden_size: tuple[int, int],
) -> np.ndarray:
    """Map a normalized golden rectangle through a pose into a filled ROI."""
    values = np.asarray(rect_golden)
    width, height = golden_size
    if (values.shape != (4,) or values.dtype == np.bool_
            or not np.issubdtype(values.dtype, np.number)):
        raise DetectionInputError("rect_golden: four numeric values required")
    values = values.astype(np.float64)
    if (not np.isfinite(values).all() or width <= 0 or height <= 0):
        raise DetectionInputError("rect_golden/golden_size: finite positive values required")
    x, y, rect_width, rect_height = values
    x0, y0 = int(round(x * width)), int(round(y * height))
    x1 = int(round((x + rect_width) * width)) - 1
    y1 = int(round((y + rect_height) * height)) - 1
    if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height):
        raise DetectionInputError("rect_golden: rectangle is outside the golden frame")

    corners = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
    )
    polygon = np.rint(transform_points(corners, pose, golden_size)).astype(np.int32)
    if (np.any(polygon[:, 0] < 0) or np.any(polygon[:, 0] >= width)
            or np.any(polygon[:, 1] < 0) or np.any(polygon[:, 1] >= height)):
        raise DetectionInputError("mapped ROI leaves the inspection frame")
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi, [polygon], 255)
    return roi


def _morph_int(
    morph: Mapping[str, Any],
    key: str,
    default: int,
    lo: int,
    hi: int,
    *,
    odd: bool = False,
) -> int:
    value = morph.get(key, default)
    if (not isinstance(value, int) or isinstance(value, bool)
            or not lo <= value <= hi or (odd and value % 2 == 0)):
        qualifier = " odd" if odd else ""
        raise DetectionInputError(
            f"detect.morph.{key}:{qualifier} integer {lo}..{hi} required - {value!r}"
        )
    return value


def make_mask(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    detect: Mapping[str, Any] | Sequence[tuple[str, Any]],
) -> DetectionMask:
    """Create the sole HSV detection mask, bounded by the ROI.

    Hue bounds may wrap through zero (for example 0.95..0.05).  The stage order
    is fixed by docs/DESIGN.md section 3: HSV -> OPEN -> ROI cut -> CLOSE ->
    ROI cut.  OPEN runs before the cut so the ROI border cannot erode a coating
    that continues past it; CLOSE runs after the cut so material lying outside
    the ROI cannot bridge a gap inside it.  The second cut restores the
    foreground-inside-ROI invariant that CLOSE can break, because the bridge it
    draws between two fragments may run outside the ROI.
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8
            or image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0):
        actual = (getattr(image_bgr, "shape", None), getattr(image_bgr, "dtype", None))
        raise DetectionInputError(f"image_bgr: non-empty HxWx3 uint8 required - {actual!r}")

    shape = image_bgr.shape[:2]
    roi = _binary_mask(roi_mask, "roi_mask", shape)
    if not np.any(roi):
        raise DetectionInputError("roi_mask: at least one ROI pixel required")

    config = _mapping(detect, "detect")
    if config.get("space", "hsv") != "hsv":
        raise DetectionInputError(f"detect.space: only hsv supported - {config.get('space')!r}")
    lower, upper = _hsv_bounds(config)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    scale = np.asarray(HSV_CHANNEL_SCALES)
    lower_u8 = np.ceil(lower * scale).astype(np.uint8)
    upper_u8 = np.floor(upper * scale).astype(np.uint8)
    empty_channels = lower_u8 > upper_u8
    if lower[0] > upper[0]:
        empty_channels[0] = False
    if empty_channels.any():
        names = "/".join(
            name for name, empty in zip("HSV", empty_channels) if empty
        )
        raise DetectionInputError(
            f"detect.lower/upper: empty quantized HSV band - {names}"
        )
    if lower[0] <= upper[0]:
        foreground = cv2.inRange(hsv, lower_u8, upper_u8)
    else:
        high_upper = upper_u8.copy()
        high_upper[0] = int(HSV_CHANNEL_SCALES[0])
        low_lower = lower_u8.copy()
        low_lower[0] = 0
        foreground = cv2.bitwise_or(
            cv2.inRange(hsv, lower_u8, high_upper),
            cv2.inRange(hsv, low_lower, upper_u8),
        )

    kernel = None
    open_iter = close_iter = 0
    morph_value = config.get("morph")
    if morph_value:
        morph = _mapping(morph_value, "detect.morph")
        shapes = {
            "ellipse": cv2.MORPH_ELLIPSE,
            "rect": cv2.MORPH_RECT,
            "cross": cv2.MORPH_CROSS,
        }
        kernel_name = str(morph.get("kernel", "ellipse"))
        if kernel_name not in shapes:
            raise DetectionInputError(f"detect.morph.kernel: unsupported - {kernel_name!r}")
        size = _morph_int(morph, "size", 5, 1, 99, odd=True)
        kernel = cv2.getStructuringElement(shapes[kernel_name], (size, size))
        open_iter = _morph_int(morph, "open_iter", 1, 0, 10)
        close_iter = _morph_int(morph, "close_iter", 1, 0, 10)

    if open_iter:
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, kernel, iterations=open_iter
        )
    foreground = cv2.bitwise_and(foreground, roi)
    if close_iter:
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_CLOSE, kernel, iterations=close_iter
        )
        foreground = cv2.bitwise_and(foreground, roi)

    return DetectionMask(foreground=foreground, roi=roi)


def _validated_pair(mask: DetectionMask) -> tuple[np.ndarray, np.ndarray]:
    """Return (roi, foreground) as 0/255 masks, or raise on a broken pair.

    Shared by every measurement tool so they cannot drift apart on what counts
    as a usable mask - the ROI denominator and the foreground-inside-ROI
    invariant have to mean the same thing to all of them.
    """
    if not isinstance(mask, DetectionMask):
        raise DetectionInputError("mask: DetectionMask required")
    shape = getattr(mask.roi, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise DetectionInputError("mask.roi: 2D mask required")
    roi = _binary_mask(mask.roi, "mask.roi", shape)
    foreground = _binary_mask(mask.foreground, "mask.foreground", shape)
    if not np.any(roi):
        raise DetectionInputError("mask.roi: at least one ROI pixel required")
    if np.any((foreground != 0) & (roi == 0)):
        raise DetectionInputError("mask.foreground: pixels outside ROI are forbidden")
    return roi, foreground


def measure_coverage(mask: DetectionMask) -> CoverageMeasurement:
    """Measure foreground/ROI and largest-component/foreground fractions."""
    roi, foreground = _validated_pair(mask)

    roi_pixels = int(np.count_nonzero(roi))
    foreground_pixels = int(np.count_nonzero(foreground))
    coverage = foreground_pixels / roi_pixels
    if foreground_pixels == 0:
        return CoverageMeasurement(coverage, None, 0, roi_pixels, 0)

    _labels, _map, stats, _centroids = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return CoverageMeasurement(
        coverage=coverage,
        continuity=largest / foreground_pixels,
        foreground_pixels=foreground_pixels,
        roi_pixels=roi_pixels,
        largest_component_pixels=largest,
    )


def evaluate_coverage(mask: DetectionMask, rule: Rule) -> CoverageResult:
    """Evaluate one loaded coverage rule with inclusive threshold boundaries."""
    if not isinstance(rule, Rule) or rule.tool != "coverage":
        actual = getattr(rule, "tool", None)
        raise DetectionInputError(f"rule: coverage Rule required - {actual!r}")
    params = dict(rule.params)
    measurement = measure_coverage(mask)
    failed: list[str] = []
    if measurement.coverage < float(params["min"]):
        failed.append("min")
    if "max" in params and measurement.coverage > float(params["max"]):
        failed.append("max")
    if "continuity_min" in params and (
        measurement.continuity is None
        or measurement.continuity < float(params["continuity_min"])
    ):
        failed.append("continuity_min")
    return CoverageResult(measurement, not failed, tuple(failed))


def measure_blobs(mask: DetectionMask) -> tuple[BlobMeasurement, ...]:
    """Measure every 8-connected component of the foreground inside the ROI.

    Per docs/DESIGN.md section 6.1 a blob *is* the connected component; contours
    only supply shape descriptors for a component already fixed.  An empty
    foreground yields an empty tuple - zero components is a real measurement of
    ``count=0``, not a missing one, which is why this differs from coverage's
    ``continuity=None``.

    Three degenerate cases the section does not spell out are settled here and
    fixed by tests:

    * ``circularity`` exceeds 1.0 for small components (a 3x3 square measures
      1.767) and is undefined for a single pixel, whose outer perimeter is 0.
      Both resolve to 1.0 - the clamp for the first, an explicit branch for the
      second, since ``arcLength`` returns a plain float and dividing by it
      raises rather than yielding an infinity the clamp could absorb.
    * ``solidity`` uses the *pixel count* of the filled convex hull, not
      ``cv2.contourArea``.  The contour runs through pixel centres, so its area
      undercounts the hull: a filled 3x3 square would score 9/4 = 2.25.  Pixel
      counting keeps numerator and denominator in the same unit, so the ratio
      stays within 1.0 and a hollow coating still falls below it.
    * ``minAreaRect`` reports a zero side for thin or tiny components (a 1x10
      line measures (0, 9)), which would make the ratio infinite or undefined.
      Sides are read as pixel extents - side + 1 - so a 20x5 rectangle measures
      exactly 4.0 and a single pixel 1.0, and the ratio stays >= 1 and finite.
    """
    roi, foreground = _validated_pair(mask)
    roi_pixels = int(np.count_nonzero(roi))
    # An empty foreground needs no special case: labelling reports background
    # only, so the loop below runs zero times and yields ().
    total, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    blobs = []
    for label in range(1, total):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        pixels = int(stats[label, cv2.CC_STAT_AREA])
        # Measurements are translation invariant, so crop to the component's own
        # bounding box instead of materialising a full-frame mask per component.
        window = labels[top : top + height, left : left + width]
        component = np.where(window == label, 255, 0).astype(np.uint8)
        # RETR_EXTERNAL on an already 8-connected component yields exactly one
        # contour, and it is the outer one - which is what keeps holes out of the
        # perimeter.  Unpacking rather than picking the largest keeps that fact
        # load-bearing instead of papering over a wrong retrieval mode.
        (contour,), _hierarchy = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        perimeter = cv2.arcLength(contour, True)
        circularity = (
            min(4.0 * np.pi * pixels / perimeter**2, 1.0) if perimeter > 0 else 1.0
        )

        hull = cv2.convexHull(contour)
        hull_mask = np.zeros_like(component)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        hull_pixels = int(np.count_nonzero(hull_mask))
        solidity = pixels / hull_pixels if hull_pixels else 1.0

        (_centre, (rect_w, rect_h), _angle) = cv2.minAreaRect(contour)
        long_side, short_side = max(rect_w, rect_h) + 1.0, min(rect_w, rect_h) + 1.0

        blobs.append(
            BlobMeasurement(
                pixels=pixels,
                area=pixels / roi_pixels,
                circularity=float(circularity),
                solidity=float(solidity),
                aspect_ratio=long_side / short_side,
            )
        )
    return tuple(blobs)


def _first_rejection(
    blob: BlobMeasurement, params: Mapping[str, Any]
) -> BlobRejection | None:
    """Find the first threshold that removes this blob, in BLOB_FILTERS order."""
    for name, field, operator in BLOB_FILTERS:
        if name not in params:
            continue
        measured = float(getattr(blob, field))
        threshold = float(params[name])
        removed = measured < threshold if operator == "<" else measured > threshold
        if removed:
            return BlobRejection(blob, name, measured, threshold, operator)
    return None


def evaluate_blob(mask: DetectionMask, rule: Rule) -> BlobResult:
    """Measure, remove blobs failing a shape threshold, then judge the survivors.

    The order is fixed by docs/DESIGN.md section 6.1: shape parameters perform a
    real removal, and only what survives reaches ``count_min``/``count_max``.
    Each removed blob carries the single threshold that removed it - the first
    in ``BLOB_FILTERS`` order - so ``len(rejected)`` is the number of blobs
    removed, not the number of failed comparisons.
    """
    if not isinstance(rule, Rule) or rule.tool != "blob":
        actual = getattr(rule, "tool", None)
        raise DetectionInputError(f"rule: blob Rule required - {actual!r}")
    params = dict(rule.params)

    kept: list[BlobMeasurement] = []
    rejected: list[BlobRejection] = []
    for blob in measure_blobs(mask):
        rejection = _first_rejection(blob, params)
        if rejection is None:
            kept.append(blob)
        else:
            rejected.append(rejection)

    count = len(kept)
    failed: list[str] = []
    if count < int(params["count_min"]):
        failed.append("count_min")
    if count > int(params["count_max"]):
        failed.append("count_max")
    return BlobResult(tuple(kept), tuple(rejected), not failed, tuple(failed))
