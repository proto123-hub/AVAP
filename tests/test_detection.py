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


def test_make_mask_runs_hsv_then_morph_then_roi_intersection():
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


def test_make_mask_supports_hue_wrap():
    hsv = np.array([[[178, 255, 255], [2, 255, 255], [90, 255, 255]]], dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detect = {"lower": [0.95, 0.9, 0.9], "upper": [0.05, 1.0, 1.0]}

    result = make_mask(image, np.ones((1, 3), dtype=bool), detect)

    np.testing.assert_array_equal(result.foreground != 0, [[True, True, False]])


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
