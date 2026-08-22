"""Two-anchor rigid alignment for AVAP Phase 1.

The aligner owns one narrow job: locate exactly two golden-frame anchors,
recover a fixed-scale rigid pose, and fail loudly as UNKNOWN when any gate is
not satisfied. Detection and verdict logic belong to Phase 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from avap.recipe import Alignment, Anchor


class AlignmentConfigError(ValueError):
    """The recipe/golden pair cannot construct a valid aligner."""


class AlignStatus(str, Enum):
    OK = "OK"
    UNKNOWN = "UNKNOWN"


class AlignFailCode(str, Enum):
    IMAGE_SIZE_MISMATCH = "IMAGE_SIZE_MISMATCH"
    NUMERIC_NONFINITE = "NUMERIC_NONFINITE"
    ANCHOR_SCORE_LOW = "ANCHOR_SCORE_LOW"
    SCALE_OUT_OF_RANGE = "SCALE_OUT_OF_RANGE"
    SHIFT_OUT_OF_RANGE = "SHIFT_OUT_OF_RANGE"
    ROTATION_OUT_OF_RANGE = "ROTATION_OUT_OF_RANGE"


@dataclass(frozen=True)
class Pose:
    """Rotation about the golden-frame centre, then translation in pixels."""

    tx: float
    ty: float
    theta_deg: float


@dataclass(frozen=True)
class AnchorMatch:
    anchor_id: str
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class AlignmentResult:
    status: AlignStatus
    pose: Pose | None
    anchor_matches: tuple[AnchorMatch, ...]
    fail_code: AlignFailCode | None
    scale_ratio: float | None


def _rect_pixels(
    rect: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map a normalized rectangle by its edges, avoiding width drift."""
    width, height = size
    x, y, w, h = rect
    x0 = int(round(x * width))
    y0 = int(round(y * height))
    x1 = int(round((x + w) * width))
    y1 = int(round((y + h) * height))
    return x0, y0, x1 - x0, y1 - y0


def _gray(image: np.ndarray) -> np.ndarray:
    if (np.issubdtype(image.dtype, np.floating)
            and not np.isfinite(image).all()):
        raise AlignmentConfigError("이미지에 NaN 또는 inf가 있음")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise AlignmentConfigError(f"지원하지 않는 이미지 shape: {image.shape}")


def _parabolic_offset(left: float, centre: float, right: float) -> float:
    if not all(math.isfinite(value) for value in (left, centre, right)):
        return 0.0
    denominator = left - 2.0 * centre + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def _peak(response: np.ndarray) -> tuple[float, float, float]:
    if not np.isfinite(response).all():
        return math.nan, math.nan, math.nan
    _min_value, score, _min_location, location = cv2.minMaxLoc(response)
    x, y = location
    dx = 0.0
    dy = 0.0
    if 0 < x < response.shape[1] - 1:
        dx = _parabolic_offset(
            float(response[y, x - 1]),
            float(response[y, x]),
            float(response[y, x + 1]),
        )
    if 0 < y < response.shape[0] - 1:
        dy = _parabolic_offset(
            float(response[y - 1, x]),
            float(response[y, x]),
            float(response[y + 1, x]),
        )
    return float(x) + dx, float(y) + dy, float(score)


def _match_patch(window: np.ndarray, patch: np.ndarray) -> tuple[float, float, float]:
    """Full-resolution NCC in an already limited search window."""
    if (window.shape[0] < patch.shape[0]
            or window.shape[1] < patch.shape[1]):
        raise AlignmentConfigError(
            f"탐색창 {window.shape[::-1]}이 패치 {patch.shape[::-1]}보다 작음"
        )

    response = cv2.matchTemplate(
        window, patch, cv2.TM_CCOEFF_NORMED
    )
    return _peak(response)


def transform_points(
    points: np.ndarray,
    pose: Pose,
    golden_size: tuple[int, int],
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Apply or invert AVAP's centre-based rigid pose to Nx2 points."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"points는 Nx2 배열이어야 함: {values.shape}")
    width, height = golden_size
    centre = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    radians = math.radians(pose.theta_deg)
    cosine, sine = math.cos(radians), math.sin(radians)
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    translation = np.array([pose.tx, pose.ty], dtype=np.float64)
    if inverse:
        return (values - centre - translation) @ rotation + centre
    return (values - centre) @ rotation.T + centre + translation


def normalized_shift_frac(pose: Pose, golden_size: tuple[int, int]) -> float:
    """Per-axis normalized centre translation, scaled into a 0..1 fraction."""
    width, height = golden_size
    return math.hypot(pose.tx / width, pose.ty / height) / math.sqrt(2.0)


def _unknown(
    matches: tuple[AnchorMatch, ...],
    code: AlignFailCode,
    scale_ratio: float | None = None,
) -> AlignmentResult:
    return AlignmentResult(
        status=AlignStatus.UNKNOWN,
        pose=None,
        anchor_matches=matches,
        fail_code=code,
        scale_ratio=scale_ratio,
    )


class Aligner:
    """Immutable golden templates plus per-image two-anchor alignment."""

    def __init__(self, alignment: Alignment, golden_bgr: np.ndarray):
        self.alignment = alignment
        height, width = golden_bgr.shape[:2]
        self.size = (width, height)
        if len(alignment.anchors) != 2:
            raise AlignmentConfigError("2점 강체 정렬에는 앵커가 정확히 2개 필요함")

        golden_gray = _gray(golden_bgr)
        templates: list[tuple[Anchor, np.ndarray, tuple[int, int, int, int]]] = []
        golden_centres: list[tuple[float, float]] = []
        for anchor in alignment.anchors:
            ox, oy, ow, oh = _rect_pixels(anchor.origin, self.size)
            sx, sy, sw, sh = _rect_pixels(anchor.search, self.size)
            patch = golden_gray[oy:oy + oh, ox:ox + ow].copy()
            patch_std = float(patch.std()) if patch.size else math.nan
            if not math.isfinite(patch_std) or patch_std < 1e-6:
                raise AlignmentConfigError(f"앵커 '{anchor.id}' 패치가 비었거나 무채움")
            if sw < ow or sh < oh:
                raise AlignmentConfigError(f"앵커 '{anchor.id}' 탐색창이 패치보다 작음")
            templates.append((anchor, patch, (sx, sy, sw, sh)))
            golden_centres.append((ox + ow / 2.0, oy + oh / 2.0))
        if not math.isfinite(math.dist(*golden_centres)) or math.dist(*golden_centres) == 0.0:
            raise AlignmentConfigError("두 앵커 중심이 같음")
        self._templates = tuple(templates)
        self._golden_centres = tuple(golden_centres)

    def align(self, image_bgr: np.ndarray) -> AlignmentResult:
        width, height = self.size
        if image_bgr.shape[:2] != (height, width):
            return _unknown((), AlignFailCode.IMAGE_SIZE_MISMATCH)
        if (np.issubdtype(image_bgr.dtype, np.floating)
                and not np.isfinite(image_bgr).all()):
            return _unknown((), AlignFailCode.NUMERIC_NONFINITE)
        image_gray = _gray(image_bgr)

        matches: list[AnchorMatch] = []
        for anchor, patch, (sx, sy, sw, sh) in self._templates:
            window = image_gray[sy:sy + sh, sx:sx + sw]
            local_x, local_y, score = _match_patch(window, patch)
            if not all(math.isfinite(value) for value in (local_x, local_y, score)):
                return _unknown(tuple(matches), AlignFailCode.NUMERIC_NONFINITE)
            x = sx + local_x + patch.shape[1] / 2.0
            y = sy + local_y + patch.shape[0] / 2.0
            matches.append(AnchorMatch(anchor.id, x, y, score))

        frozen_matches = tuple(matches)
        if any(not math.isfinite(match.score) or match.score < anchor.min_score
               for match, anchor in zip(matches, alignment_anchors(self.alignment))):
            return _unknown(frozen_matches, AlignFailCode.ANCHOR_SCORE_LOW)

        golden_points = np.asarray(self._golden_centres, dtype=np.float64)
        matched_points = np.asarray([(m.x, m.y) for m in matches], dtype=np.float64)
        golden_vector = golden_points[1] - golden_points[0]
        matched_vector = matched_points[1] - matched_points[0]
        golden_distance = float(np.linalg.norm(golden_vector))
        matched_distance = float(np.linalg.norm(matched_vector))
        scale_ratio = matched_distance / golden_distance
        if not math.isfinite(scale_ratio):
            return _unknown(
                frozen_matches, AlignFailCode.NUMERIC_NONFINITE, scale_ratio
            )
        if abs(scale_ratio - 1.0) > self.alignment.scale_tol:
            return _unknown(
                frozen_matches, AlignFailCode.SCALE_OUT_OF_RANGE, scale_ratio
            )

        golden_angle = math.atan2(golden_vector[1], golden_vector[0])
        matched_angle = math.atan2(matched_vector[1], matched_vector[0])
        theta_deg = math.degrees(golden_angle - matched_angle)
        theta_deg = (theta_deg + 180.0) % 360.0 - 180.0
        rotation_only = Pose(0.0, 0.0, theta_deg)
        rotated = transform_points(golden_points, rotation_only, self.size)
        translation = np.mean(matched_points - rotated, axis=0)
        pose = Pose(float(translation[0]), float(translation[1]), theta_deg)
        if not all(math.isfinite(value) for value in (
                pose.tx, pose.ty, pose.theta_deg)):
            return _unknown(
                frozen_matches, AlignFailCode.NUMERIC_NONFINITE, scale_ratio
            )

        shift_frac = normalized_shift_frac(pose, self.size)
        if not math.isfinite(shift_frac):
            return _unknown(
                frozen_matches, AlignFailCode.NUMERIC_NONFINITE, scale_ratio
            )
        if shift_frac > self.alignment.max_shift_frac:
            return _unknown(
                frozen_matches, AlignFailCode.SHIFT_OUT_OF_RANGE, scale_ratio
            )
        if abs(pose.theta_deg) > self.alignment.max_rotation_deg:
            return _unknown(
                frozen_matches, AlignFailCode.ROTATION_OUT_OF_RANGE, scale_ratio
            )
        return AlignmentResult(
            status=AlignStatus.OK,
            pose=pose,
            anchor_matches=frozen_matches,
            fail_code=None,
            scale_ratio=scale_ratio,
        )


def alignment_anchors(alignment: Alignment) -> tuple[Anchor, Anchor]:
    """Typed two-anchor view after recipe validation."""
    anchors = alignment.anchors
    if len(anchors) != 2:
        raise AlignmentConfigError("앵커는 정확히 2개여야 함")
    return anchors[0], anchors[1]
