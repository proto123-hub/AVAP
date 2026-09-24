"""shape_compare (docs/DESIGN.md 6.3): the current foreground C against the golden footprint G."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from avap.alignment import Aligner, AlignStatus, Pose
from avap.detection import (
    DetectionInputError,
    DetectionMask,
    ShapeCompareMeasurement,
    evaluate_shape_compare,
    golden_footprint,
    make_mask,
    make_roi_mask,
    map_footprint,
    measure_shape_compare,
)
from avap.recipe import PARAM_SPECS, Rule, load_recipe
from avap.synth import (
    BEAD_THICKNESS,
    BEAD_X0,
    BEAD_X1,
    BEAD_Y,
    MATERIAL,
    apply_pose,
    draw_golden,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "recipes" / "sample_synth.json"

# 모든 파라미터를 느슨하게 둔 규칙 - 증인 테스트가 한 파라미터만 조인다.
PERMISSIVE = {"iou_min": 0.0, "excess_max": 1.0, "deficit_max": 1.0}
# 합성 세트 판정용. ok 는 복원 pose 에서도 IoU >= 0.979 · deficit <= 0.021 로 여유가 있다.
STRICT = {"iou_min": 0.95, "excess_max": 0.1, "deficit_max": 0.1}


def _rule(**params) -> Rule:
    return Rule("shape_compare", tuple(sorted(params.items())))


def _masks(roi, foreground, footprint):
    """0/1 lists -> (DetectionMask, G) as 0/255 uint8."""
    as_mask = lambda rows: np.array(rows, dtype=np.uint8) * 255  # noqa: E731
    return DetectionMask(foreground=as_mask(foreground), roi=as_mask(roi)), as_mask(footprint)


FULL = [[1, 1, 1, 1]] * 2
# G 4px, C 6px, 겹침 2px -> IoU 2/8, excess 4/6, deficit 2/4
G_4 = [[1, 1, 1, 1], [0, 0, 0, 0]]
C_6 = [[0, 0, 1, 1], [1, 1, 1, 1]]


def _sample():
    recipe = load_recipe(SAMPLE)
    return recipe, recipe.rois[0]


def _roi_centre(recipe, roi_config):
    x, y, w, h = roi_config.rect_golden
    width, height = recipe.golden_size
    return (x + w / 2.0) * width, (y + h / 2.0) * height


def _compare(recipe, roi_config, aligner, g0, image, rule):
    aligned = aligner.align(image)
    assert aligned.status is AlignStatus.OK and aligned.pose is not None
    roi = make_roi_mask(roi_config.rect_golden, aligned.pose, recipe.golden_size)
    mask = make_mask(image, roi, roi_config.detect)
    return aligned.pose, evaluate_shape_compare(mask, map_footprint(g0, aligned.pose, roi), rule)


def _thick_bead() -> np.ndarray:
    image = draw_golden("ng_missing")
    cv2.line(image, (BEAD_X0, BEAD_Y), (BEAD_X1, BEAD_Y), MATERIAL, 50)
    return image


SCENARIOS = {
    "ok": lambda: draw_golden("ok"),
    "ng_missing": lambda: draw_golden("ng_missing"),
    "ng_broken": lambda: draw_golden("ng_broken"),
    "ng_excess": _thick_bead,
}


# ── 정의 (6.3) ────────────────────────────────────────────────────────────

def test_identical_masks_measure_one_zero_zero():
    mask, footprint = _masks(FULL, G_4, G_4)
    m = measure_shape_compare(mask, footprint)
    assert (m.iou, m.excess, m.deficit) == (1.0, 0.0, 0.0)
    assert (m.footprint_pixels, m.foreground_pixels, m.intersection_pixels) == (4, 4, 4)


def test_partial_overlap_matches_the_three_definitions():
    mask, footprint = _masks(FULL, C_6, G_4)
    m = measure_shape_compare(mask, footprint)
    assert m == ShapeCompareMeasurement(
        iou=2 / 8, excess=4 / 6, deficit=2 / 4,
        footprint_pixels=4, foreground_pixels=6, intersection_pixels=2,
    )


def test_empty_foreground_on_a_valid_footprint_is_zero_zero_one():
    mask, footprint = _masks(FULL, [[0] * 4] * 2, G_4)
    m = measure_shape_compare(mask, footprint)
    assert (m.iou, m.excess, m.deficit) == (0.0, 0.0, 1.0)
    result = evaluate_shape_compare(mask, footprint, _rule(**STRICT))
    assert not result.passed and result.failed_params == ("iou_min", "deficit_max")


def test_foreground_inside_the_footprint_is_pure_deficit():
    mask, footprint = _masks(FULL, [[1, 1, 0, 0], [0, 0, 0, 0]], G_4)
    m = measure_shape_compare(mask, footprint)
    assert (m.iou, m.excess, m.deficit) == (0.5, 0.0, 0.5)


def test_foreground_around_the_footprint_is_pure_excess():
    mask, footprint = _masks(FULL, FULL, G_4)
    m = measure_shape_compare(mask, footprint)
    assert (m.iou, m.excess, m.deficit) == (0.5, 0.5, 0.0)


def test_disjoint_masks_are_all_excess_and_all_deficit():
    mask, footprint = _masks(FULL, [[0, 0, 0, 0], [1, 1, 1, 1]], G_4)
    m = measure_shape_compare(mask, footprint)
    assert (m.iou, m.excess, m.deficit) == (0.0, 1.0, 1.0)
    # 셋 다 떨어지면 PARAM_SPECS 선언 순서로 보고한다 (Evidence 순서가 실행마다 같도록)
    result = evaluate_shape_compare(mask, footprint, _rule(**STRICT))
    assert result.failed_params == ("iou_min", "excess_max", "deficit_max")
    assert result.failed_params == tuple(PARAM_SPECS["shape_compare"])


def test_measurements_are_fractions_that_agree_with_the_pixel_counts():
    # L7(0~1) 과 세 정의의 분모를 무작위 마스크로 교차 확인한다 - 분모를 바꿔 쓰면
    # 대칭 입력(|G| == |C|)에서는 안 보이고 크기가 다른 입력에서만 드러난다.
    rng = np.random.default_rng(6_3)
    roi = np.full((12, 12), 255, dtype=np.uint8)
    for _ in range(200):
        golden = (rng.random((12, 12)) < rng.uniform(0.05, 0.9)).astype(np.uint8) * 255
        golden[rng.integers(12), rng.integers(12)] = 255          # G 는 비지 않는다
        current = (rng.random((12, 12)) < rng.uniform(0.0, 0.9)).astype(np.uint8) * 255
        m = measure_shape_compare(DetectionMask(foreground=current, roi=roi), golden)
        g, c = np.count_nonzero(golden), np.count_nonzero(current)
        both = np.count_nonzero((golden != 0) & (current != 0))
        assert all(0.0 <= v <= 1.0 for v in (m.iou, m.excess, m.deficit))
        assert (m.footprint_pixels, m.foreground_pixels, m.intersection_pixels) == (g, c, both)
        assert math.isclose(m.iou, both / (g + c - both))
        assert math.isclose(m.deficit, (g - both) / g)
        assert math.isclose(m.excess, (c - both) / c if c else 0.0)
        assert m.iou <= 1.0 - m.excess + 1e-12 and m.iou <= 1.0 - m.deficit + 1e-12


def test_empty_footprint_is_a_measurement_error_not_a_pass():
    mask, footprint = _masks(FULL, G_4, [[0] * 4] * 2)
    with pytest.raises(DetectionInputError, match="footprint G: empty"):
        measure_shape_compare(mask, footprint)
    with pytest.raises(DetectionInputError, match="footprint G: empty"):
        evaluate_shape_compare(mask, footprint, _rule(**PERMISSIVE))


def test_footprint_outside_the_roi_is_refused():
    # 자르지 않은 G0 를 넘기면 프레임이 검사하지 않는 골든 픽셀이 deficit 으로 잡힌다.
    roi = [[1, 1, 0, 0], [1, 1, 0, 0]]
    mask, footprint = _masks(roi, [[1, 1, 0, 0], [0, 0, 0, 0]], G_4)
    with pytest.raises(DetectionInputError, match="outside ROI"):
        measure_shape_compare(mask, footprint)


# ── 판정 경계 · L1 증인 ───────────────────────────────────────────────────

def _flip(param, measured, passing, failing):
    mask, footprint = _masks(FULL, C_6, G_4)
    at = evaluate_shape_compare(mask, footprint, _rule(**{**PERMISSIVE, param: passing}))
    past = evaluate_shape_compare(mask, footprint, _rule(**{**PERMISSIVE, param: failing}))
    return at, past


@pytest.mark.parametrize("param, attr, direction", [
    ("iou_min", "iou", math.inf),
    ("excess_max", "excess", -math.inf),
    ("deficit_max", "deficit", -math.inf),
])
def test_each_parameter_flips_the_verdict_at_its_measured_value(param, attr, direction):
    mask, footprint = _masks(FULL, C_6, G_4)
    measured = getattr(measure_shape_compare(mask, footprint), attr)
    assert 0.0 < measured < 1.0, "증인 입력은 경계 안쪽 값을 내야 한다"
    at, past = _flip(param, measured, measured, math.nextafter(measured, direction))
    assert at.passed and at.failed_params == (), "경계값은 포함(PASS)이다"
    assert not past.passed and past.failed_params == (param,)


def test_permissive_endpoints_pass_the_worst_case_and_are_not_dead_values():
    # iou_min=0 · excess_max=1 · deficit_max=1 은 최악(서로소, IoU 0 · excess 1 · deficit 1)도
    # 통과시키는 허용 끝점이다. 죽은 값이 아닌 근거는 위 증인 테스트의 반전이다.
    mask, footprint = _masks(FULL, [[0, 0, 0, 0], [1, 1, 1, 1]], G_4)
    assert evaluate_shape_compare(mask, footprint, _rule(**PERMISSIVE)).passed


def test_optional_parameters_are_not_checked_when_absent():
    mask, footprint = _masks(FULL, [[0, 0, 0, 0], [1, 1, 1, 1]], G_4)
    assert evaluate_shape_compare(mask, footprint, _rule(iou_min=0.0)).passed


def test_witnesses_above_cover_every_shape_compare_parameter():
    assert set(PARAM_SPECS["shape_compare"]) == {"iou_min", "excess_max", "deficit_max"}


# ── 합성 골든 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario, failed", [
    ("ok", ()),
    ("ng_missing", ("iou_min", "deficit_max")),
    ("ng_broken", ("iou_min", "deficit_max")),
    ("ng_excess", ("iou_min", "excess_max")),
])
def test_shape_compare_verdict_is_invariant_across_recovered_poses(scenario, failed):
    # L4: 판정뿐 아니라 어느 파라미터가 떨어뜨렸는지까지 pose 와 무관해야 한다.
    recipe, roi_config = _sample()
    golden = draw_golden("ok")
    aligner = Aligner(recipe.alignment, golden)
    g0 = golden_footprint(golden, roi_config.rect_golden, roi_config.detect)
    source = SCENARIOS[scenario]()
    rng = np.random.default_rng(20260823)

    for tx, ty, theta in zip(
        rng.uniform(-20.0, 20.0, 6),
        rng.uniform(-20.0, 20.0, 6),
        rng.uniform(-1.5, 1.5, 6),
    ):
        image = apply_pose(source, float(tx), float(ty), float(theta))
        _pose, result = _compare(recipe, roi_config, aligner, g0, image, _rule(**STRICT))
        assert result.passed is (not failed) and result.failed_params == failed


# ── 방향 (6.3): 앵커 고정 · 도포만 180° 회전 ─────────────────────────────

def _symmetric_bead(layer):
    cv2.line(layer, (BEAD_X0, BEAD_Y), (BEAD_X1, BEAD_Y), 255, BEAD_THICKNESS)


def _wedge_bead(layer):
    # 왼쪽이 두껍고 오른쪽이 얇은 쐐기 - 180° 회전하면 두꺼운 쪽이 반대편으로 간다.
    points = np.array([[BEAD_X0, BEAD_Y - 15], [BEAD_X1, BEAD_Y - 5],
                       [BEAD_X1, BEAD_Y + 5], [BEAD_X0, BEAD_Y + 15]], dtype=np.int32)
    cv2.fillPoly(layer, [points], 255)


def _board_with(draw, recipe, roi_config, rotate):
    """Board with fixed anchors and the coating alone, optionally turned 180 deg about the ROI."""
    width, height = recipe.golden_size
    layer = np.zeros((height, width), dtype=np.uint8)
    draw(layer)
    if rotate:
        turn = cv2.getRotationMatrix2D(_roi_centre(recipe, roi_config), 180.0, 1.0)
        layer = cv2.warpAffine(layer, turn, (width, height), flags=cv2.INTER_NEAREST)
    image = draw_golden("ng_missing")
    image[layer != 0] = MATERIAL
    return image


def _orientation(draw):
    recipe, roi_config = _sample()
    golden = _board_with(draw, recipe, roi_config, rotate=False)
    turned = _board_with(draw, recipe, roi_config, rotate=True)
    aligner = Aligner(recipe.alignment, golden)
    g0 = golden_footprint(golden, roi_config.rect_golden, roi_config.detect)
    results = []
    for image in (golden, turned):
        pose, result = _compare(recipe, roi_config, aligner, g0, image, _rule(**STRICT))
        # 앵커는 움직이지 않았다 - 정렬이 회전을 흡수했다면 이 시험은 방향을 못 본다.
        assert abs(pose.tx) < 0.5 and abs(pose.ty) < 0.5 and abs(pose.theta_deg) < 0.05
        results.append(result)
    return results


def test_an_asymmetric_coating_turned_180_degrees_fails():
    upright, turned = _orientation(_wedge_bead)
    assert upright.passed and upright.measurement.iou == 1.0
    assert not turned.passed
    assert turned.measurement.iou < STRICT["iou_min"]
    # 같은 양을 반대로 발랐다 - 넘친 만큼 모자란다.
    assert turned.measurement.foreground_pixels == turned.measurement.footprint_pixels
    assert turned.measurement.excess == turned.measurement.deficit > 0.0


def test_a_symmetric_coating_turned_180_degrees_is_the_same_footprint():
    # 180° 회전이 항상 비등가는 아니다 - 대칭 비드는 같은 footprint 이므로 같은 판정이어야 한다.
    upright, turned = _orientation(_symmetric_bead)
    assert upright.passed and turned.passed
    assert turned.measurement == upright.measurement


# ── API 경계 ──────────────────────────────────────────────────────────────

def test_evaluate_refuses_a_rule_from_another_tool():
    mask, footprint = _masks(FULL, G_4, G_4)
    with pytest.raises(DetectionInputError, match="shape_compare Rule required"):
        evaluate_shape_compare(mask, footprint, Rule("coverage", (("min", 0.1),)))
    with pytest.raises(DetectionInputError, match="shape_compare Rule required"):
        evaluate_shape_compare(mask, footprint, {"tool": "shape_compare", "iou_min": 0.5})


@pytest.mark.parametrize("footprint, message", [
    (None, "footprint"),
    (np.full((3, 4), 255, dtype=np.uint8), "footprint"),
    (np.full((2, 4), 7, dtype=np.uint8), "footprint"),
    (np.full((2, 4), 255, dtype=np.float32), "footprint"),
])
def test_bad_footprint_is_a_detection_input_error(footprint, message):
    mask, _footprint = _masks(FULL, G_4, G_4)
    with pytest.raises(DetectionInputError, match=message):
        measure_shape_compare(mask, footprint)


@pytest.mark.parametrize("mask", [
    None,
    "mask",
    DetectionMask(foreground=np.zeros((2, 4), np.uint8), roi=np.zeros((2, 4), np.uint8)),
])
def test_bad_mask_is_a_detection_input_error(mask):
    _mask, footprint = _masks(FULL, G_4, G_4)
    with pytest.raises(DetectionInputError):
        measure_shape_compare(mask, footprint)
