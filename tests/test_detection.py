"""Phase 2 single-mask contract and coverage-tool regression tests."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from avap.detection import (
    CoverageMeasurement,
    DetectionInputError,
    DetectionMask,
    evaluate_coverage,
    make_mask,
    make_roi_mask,
    measure_coverage,
)
from avap.alignment import Aligner, AlignStatus, Pose
from avap.recipe import PARAM_SPECS, Rule, load_recipe
from avap.synth import apply_pose, draw_golden


REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "recipes" / "sample_synth.json"


def _mask(foreground: np.ndarray, roi: np.ndarray | None = None) -> DetectionMask:
    foreground = np.asarray(foreground, dtype=np.uint8) * 255
    if roi is None:
        roi = np.ones(foreground.shape, dtype=np.uint8)
    return DetectionMask(foreground, np.asarray(roi, dtype=np.uint8) * 255)


def _rule(**params: float) -> Rule:
    return Rule("coverage", tuple(sorted(params.items())))


def test_make_mask_closes_an_interior_hole_and_stays_inside_the_roi():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[2:5, 2:5] = 255
    image[3, 3] = 0
    roi = np.zeros((7, 7), dtype=np.uint8)
    roi[:, :4] = 255
    detect = {
        "space": "hsv",
        "lower": [0.0, 0.0, 0.9],
        "upper": [1.0, 0.1, 1.0],
        "morph": {"kernel": "rect", "size": 3, "open_iter": 0, "close_iter": 1},
    }

    result = make_mask(image, roi, detect)

    assert result.foreground.dtype == np.uint8
    assert np.count_nonzero(result.foreground) == 6
    assert np.all(result.foreground[:, 4:] == 0)
    assert result.foreground[3, 3] == 255  # close filled the one-pixel hole


def test_material_outside_the_roi_cannot_close_a_gap_inside_it():
    # CLOSE가 ROI 교집합 앞에 있으면 ROI 바깥 물질이 경계를 넘어 안쪽 도포와 이어지고,
    # 그 다리가 ROI 내부를 채워 continuity가 잡으려던 끊김을 지운다.
    # 계약(OPEN -> ROI 컷 -> CLOSE -> ROI 컷)은 바깥 물질을 CLOSE 전에 제거한다.
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    roi = np.zeros((200, 200), dtype=np.uint8)
    roi[80:120, 80:120] = 255
    image[95:105, 68:80] = 255   # ROI 바깥에만 있는 물질 (x < 80)
    image[95:105, 84:120] = 255  # ROI 안 도포
    # 둘 사이 간격 x=80..83 은 ROI '안쪽'이라, 다리가 놓이면 내부가 메워진다.
    detect = {
        "space": "hsv",
        "lower": [0.0, 0.0, 0.5],
        "upper": [1.0, 0.2, 1.0],
        "morph": {"kernel": "rect", "size": 9, "open_iter": 0, "close_iter": 1},
    }

    foreground = make_mask(image, roi, detect).foreground

    assert not np.any(foreground[:, 80:84]), "ROI 바깥 물질이 내부 간격을 메웠다"
    assert np.count_nonzero(foreground) == 10 * 36  # ROI 안 도포 그대로


def test_closing_cannot_grow_past_a_concave_roi_edge():
    # 마지막 ROI 컷이 막는 유일한 경로. 볼록 ROI 는 자기 경계 밖으로 자랄 수 없어
    # (무작위 400회 유출 0건) 이 성질이 무발동이므로, 실제로 물게 하려면 오목 ROI 가 필요하다.
    # L자 ROI 의 두 팔에 조각을 하나씩 두면 CLOSE 가 잇는 다리가 오목 코너 바깥을 지난다.
    roi = np.zeros((120, 120), dtype=np.uint8)
    roi[30:90, 30:60] = 255          # 세로팔
    roi[30:50, 30:95] = 255          # 가로팔 - 코너 바깥은 x>=60 & y>=50
    bright = np.zeros((120, 120), dtype=np.uint8)
    bright[44:50, 70:91] = 255       # 가로팔 안, 코너 위
    bright[52:71, 52:60] = 255       # 세로팔 안, 코너 아래
    image = np.repeat(bright[:, :, None], 3, axis=2)
    detect = {
        "space": "hsv",
        "lower": [0.0, 0.0, 0.5],
        "upper": [1.0, 0.2, 1.0],
        "morph": {"kernel": "ellipse", "size": 9, "open_iter": 0, "close_iter": 3},
    }

    foreground = make_mask(image, roi, detect).foreground

    assert not np.any(cv2.bitwise_and(foreground, cv2.bitwise_not(roi))), (
        "CLOSE 가 오목 ROI 경계를 넘었다 - 마지막 ROI 컷 누락"
    )


def test_make_mask_supports_hue_wrap():
    hues = [170, 171, 179, 0, 8, 9]
    hsv = np.array([[[hue, 255, 255] for hue in hues]], dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detect = {"lower": [0.95, 0.9, 0.9], "upper": [0.05, 1.0, 1.0]}

    result = make_mask(image, np.ones((1, 6), dtype=bool), detect)

    np.testing.assert_array_equal(
        result.foreground != 0,
        [[False, True, True, True, True, False]],
    )


@pytest.mark.parametrize(
    ("hsv_pixel", "lower", "upper"),
    [
        ([0, 0, 102], [0.0, 0.0, 102 / 255], [1.0, 1.0, 102 / 255]),
        ([89, 255, 255], [89 / 179, 0.0, 0.0], [89 / 179, 1.0, 1.0]),
    ],
    ids=["v-grid-aligned-single-bin", "h-grid-aligned-single-bin"],
)
def test_make_mask_accepts_grid_aligned_single_bin_bands(hsv_pixel, lower, upper):
    hsv = np.array([[hsv_pixel]], dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = make_mask(
        image,
        np.ones((1, 1), dtype=bool),
        {"lower": lower, "upper": upper},
    )

    assert result.foreground[0, 0] == 255


@pytest.mark.parametrize(
    ("lower", "upper", "empty_channel"),
    [
        ([0.0, 0.0, 0.5], [1.0, 1.0, 0.5], "V"),
        ([0.5, 0.0, 0.0], [0.5, 1.0, 1.0], "H"),
        ([0.95, 0.5, 0.0], [0.05, 0.5, 1.0], "S"),
    ],
    ids=["v-off-grid-zero-width", "h-off-grid-zero-width", "hue-wrap-with-empty-s"],
)
def test_make_mask_rejects_empty_quantized_hsv_bands(lower, upper, empty_channel):
    image = np.zeros((1, 1, 3), dtype=np.uint8)

    with pytest.raises(
        DetectionInputError,
        match=rf"empty quantized HSV band - {empty_channel}$",
    ):
        make_mask(
            image,
            np.ones((1, 1), dtype=bool),
            {"lower": lower, "upper": upper},
        )


def test_even_morph_kernel_is_rejected_before_opencv_can_shift_the_mask():
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    detect = {
        "lower": [0.0, 0.0, 0.0],
        "upper": [1.0, 1.0, 1.0],
        "morph": {"size": 2},
    }
    with pytest.raises(DetectionInputError, match="odd integer"):
        make_mask(image, np.ones((3, 3), dtype=bool), detect)


def test_coverage_denominator_is_roi_and_continuity_is_largest_over_foreground():
    roi = np.zeros((4, 5), dtype=np.uint8)
    roi[:, :4] = 1  # 16 ROI pixels, not the 20-pixel bounding frame
    foreground = np.zeros_like(roi)
    foreground[0:2, 0:2] = 1  # component of 4
    foreground[3, 2:4] = 1    # component of 2

    measured = measure_coverage(_mask(foreground, roi))

    assert measured == CoverageMeasurement(
        coverage=6 / 16,
        continuity=4 / 6,
        foreground_pixels=6,
        roi_pixels=16,
        largest_component_pixels=4,
    )


def test_diagonal_pixels_use_eight_connectivity():
    foreground = np.eye(3, dtype=np.uint8)
    measured = measure_coverage(_mask(foreground))
    assert measured.continuity == 1.0
    assert measured.largest_component_pixels == 3


def test_empty_foreground_keeps_continuity_undefined_and_fails_required_check():
    mask = _mask(np.zeros((2, 3), dtype=np.uint8))

    measured = measure_coverage(mask)
    result = evaluate_coverage(mask, _rule(min=0.0, continuity_min=0.1))

    assert measured.continuity is None
    assert measured.coverage == 0.0
    assert not result.passed
    assert result.failed_params == ("continuity_min",)


def test_every_coverage_parameter_can_flip_the_result_at_its_boundary():
    foreground = np.array(
        [[1, 1, 0, 0], [1, 0, 0, 1]],
        dtype=np.uint8,
    )
    mask = _mask(foreground)  # coverage 1/2, continuity 3/4
    exact = _rule(min=0.5, max=0.5, continuity_min=0.75)
    assert evaluate_coverage(mask, exact).passed

    outside_by_param = {
        "min": math.nextafter(0.5, 1.0),
        "max": math.nextafter(0.5, 0.0),
        "continuity_min": math.nextafter(0.75, 1.0),
    }
    assert set(outside_by_param) == set(PARAM_SPECS["coverage"])
    for param, outside in outside_by_param.items():
        values = {"min": 0.5, "max": 0.5, "continuity_min": 0.75}
        values[param] = outside
        result = evaluate_coverage(mask, _rule(**values))
        assert not result.passed
        assert result.failed_params == (param,)


@pytest.mark.parametrize(
    "scenario, passed, failed_params",
    [
        ("ok", True, ()),
        ("ng_missing", False, ("min", "continuity_min")),
        ("ng_broken", False, ("continuity_min",)),
    ],
)
def test_sample_recipe_coverage_rule_separates_synthetic_scenarios(
    scenario, passed, failed_params
):
    recipe = load_recipe(SAMPLE)
    roi_config = recipe.rois[0]
    image = draw_golden(scenario)
    height, width = image.shape[:2]
    roi_mask = make_roi_mask(roi_config.rect_golden, Pose(0.0, 0.0, 0.0), (width, height))
    rule = next(rule for rule in roi_config.rules if rule.tool == "coverage")

    result = evaluate_coverage(
        make_mask(image, roi_mask, roi_config.detect),
        rule,
    )

    assert result.passed is passed
    assert result.failed_params == failed_params


@pytest.mark.parametrize(
    "scenario, expected",
    [("ok", True), ("ng_missing", False), ("ng_broken", False)],
)
def test_coverage_verdict_is_invariant_across_recovered_poses(scenario, expected):
    recipe = load_recipe(SAMPLE)
    roi_config = recipe.rois[0]
    golden = draw_golden("ok")
    source = draw_golden(scenario)
    aligner = Aligner(recipe.alignment, golden)
    rule = next(rule for rule in roi_config.rules if rule.tool == "coverage")
    size = recipe.golden_size
    rng = np.random.default_rng(20260823)

    for tx, ty, theta in zip(
        rng.uniform(-20.0, 20.0, 6),
        rng.uniform(-20.0, 20.0, 6),
        rng.uniform(-1.5, 1.5, 6),
    ):
        image = apply_pose(source, float(tx), float(ty), float(theta))
        aligned = aligner.align(image)
        assert aligned.status is AlignStatus.OK and aligned.pose is not None
        roi_mask = make_roi_mask(roi_config.rect_golden, aligned.pose, size)
        result = evaluate_coverage(make_mask(image, roi_mask, roi_config.detect), rule)
        assert result.passed is expected


@pytest.mark.parametrize(
    "mask, message",
    [
        (DetectionMask(np.ones((2, 2), dtype=np.uint8) * 127,
                       np.ones((2, 2), dtype=np.uint8) * 255), "binary values"),
        (DetectionMask(np.ones((2, 2), dtype=np.uint8) * 255,
                       np.zeros((2, 2), dtype=np.uint8)), "at least one ROI"),
        (DetectionMask(np.array([[255, 0]], dtype=np.uint8),
                       np.array([[0, 255]], dtype=np.uint8)), "outside ROI"),
    ],
)
def test_invalid_binary_mask_contract_is_loud(mask, message):
    with pytest.raises(DetectionInputError, match=message):
        measure_coverage(mask)
