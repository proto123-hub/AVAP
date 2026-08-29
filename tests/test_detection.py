"""Phase 2 single-mask contract and coverage-tool regression tests."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from avap.detection import (
    BLOB_FILTERS,
    BlobMeasurement,
    CoverageMeasurement,
    DetectionInputError,
    DetectionMask,
    evaluate_blob,
    evaluate_coverage,
    make_mask,
    make_roi_mask,
    measure_blobs,
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


def test_roi_cut_does_not_erode_a_coating_that_crosses_the_roi_edge():
    # 계약의 앞쪽 절반. ROI 컷을 OPEN 앞으로 옮기면 경계에서 잘린 단면이 커널보다
    # 좁은 조각이 되어 OPEN 에 통째로 갉힌다. OPEN 을 먼저 돌리면 도포 전체가
    # 하나의 형상으로 평가되므로 살아남고, 그 뒤 컷이 ROI 안쪽만 남긴다.
    roi = np.zeros((120, 120), dtype=np.uint8)
    roi[40:71, 100:120] = 255                 # ROI 는 x >= 100
    bright = np.zeros((120, 120), dtype=np.uint8)
    bright[50:55, 70:103] = 255               # 도포가 경계를 가로지른다 - ROI 안 단면은 3열
    image = np.repeat(bright[:, :, None], 3, axis=2)
    detect = {
        "space": "hsv",
        "lower": [0.0, 0.0, 0.5],
        "upper": [1.0, 0.2, 1.0],
        "morph": {"kernel": "rect", "size": 5, "open_iter": 1, "close_iter": 0},
    }

    foreground = make_mask(image, roi, detect).foreground

    # 단면 3열 x 5행. 컷이 먼저 오면 5x5 커널이 3열을 지워 0px 가 된다.
    assert np.count_nonzero(foreground) == 15, "ROI 컷이 OPEN 앞으로 갔다"


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


def test_closing_cannot_grow_past_a_roi_touching_the_frame_edge():
    # 마지막 ROI 컷이 생산 경로에서 실제로 필요한 이유. pose 가 ROI 를 프레임 가장자리로
    # 밀면, CLOSE 의 침식 단계가 프레임 바깥을 채워진 것으로 보는 OpenCV 경계 처리 때문에
    # 안쪽이라면 깎였을 픽셀이 살아남아 ROI 폴리곤 밖에 놓인다. 볼록 ROI 라도 열리는 경로다.
    # (ROI 가 프레임에 '잘리는' 경우는 make_roi_mask 가 거부하므로 존재할 수 없다.)
    width, height = 960, 720
    roi = make_roi_mask(
        (0.04, 0.36, 0.08, 0.20), Pose(tx=-36.75, ty=0.0, theta_deg=-3.0), (width, height)
    )
    assert np.nonzero(roi)[1].min() == 0, "이 회귀는 ROI 가 프레임 좌변에 닿아야 성립한다"
    # 백색 프레임 = HSV 임계를 프레임 전체가 통과 -> 첫 컷 뒤 전경이 곧 ROI 마스크가 된다.
    # 반례가 입력 내용에 의존하지 않으므로 seed/threshold 없이 재현된다.
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    detect = {
        "space": "hsv",
        "lower": [0.0, 0.0, 0.5],
        "upper": [1.0, 0.2, 1.0],
        "morph": {"kernel": "ellipse", "size": 5, "open_iter": 0, "close_iter": 2},
    }

    foreground = make_mask(image, roi, detect).foreground

    # 마지막 컷이 없으면 이 조건에서 198px 가 ROI 밖에 남는다.
    assert not np.any(cv2.bitwise_and(foreground, cv2.bitwise_not(roi))), (
        "CLOSE 가 프레임 가장자리에서 ROI 밖으로 새어 나갔다 - 마지막 ROI 컷 누락"
    )


def test_closing_cannot_grow_past_a_concave_roi_edge():
    # CLOSE 가 그리는 다리가 ROI 밖을 지나는 경로 중 하나. L자 ROI 의 두 팔에 조각을
    # 하나씩 두면 다리가 오목 코너 바깥을 지난다. 생산 경로에서 실제로 열리는 다른 경로는
    # 아래 test_closing_cannot_grow_past_a_roi_touching_the_frame_edge 가 고정한다.
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


# ── blob (docs/DESIGN.md 6.1) ────────────────────────────────────────────

def _blob_rule(**params: float) -> Rule:
    return Rule("blob", tuple(sorted(params.items())))


def test_blob_measures_the_shapes_of_docs_design_6_1():
    # 6.1 이 문구로만 정한 값을 수치로 고정한다. 3x3 정사각은 circularity 원값이
    # 1.767 이라 클램프가 없으면 1 을 넘고, contourArea 를 hull 로 쓰면
    # solidity 가 9/4 = 2.25 가 된다. 20x5 는 AR 이 정확히 4.0 이어야 한다.
    foreground = np.zeros((40, 40), dtype=np.uint8)
    foreground[2:5, 2:5] = 1          # 3x3 정사각
    foreground[10:30, 10:15] = 1      # 20x5 직사각
    mask = _mask(foreground)

    square, rectangle = sorted(measure_blobs(mask), key=lambda b: b.pixels)

    assert (square.pixels, rectangle.pixels) == (9, 100)
    assert square.circularity == 1.0            # 원값 1.767 -> 클램프
    assert square.solidity == 1.0               # contourArea 였다면 2.25
    assert square.aspect_ratio == 1.0
    assert rectangle.aspect_ratio == pytest.approx(4.0)
    assert rectangle.solidity == 1.0


def test_blob_handles_components_too_thin_for_minarearect():
    # minAreaRect 는 1픽셀에 (0,0), 1x10 선분에 (0,9) 를 준다. 장변/단변을 그대로
    # 쓰면 nan / inf 가 되어 "AR 은 항상 >=1" 이 깨진다. 픽셀 폭(+1)으로 읽는다.
    foreground = np.zeros((30, 30), dtype=np.uint8)
    foreground[2, 2] = 1              # 1픽셀 (외곽 둘레 0 -> circularity 정의 불가)
    foreground[10, 5:15] = 1          # 1x10 수평선
    mask = _mask(foreground)

    dot, line = sorted(measure_blobs(mask), key=lambda b: b.pixels)

    assert dot.aspect_ratio == 1.0
    assert dot.circularity == 1.0
    assert dot.solidity == 1.0
    assert line.aspect_ratio == pytest.approx(10.0)
    assert all(np.isfinite(b.aspect_ratio) for b in (dot, line))


def test_blob_solidity_drops_below_one_for_a_hollow_coating():
    # 6.1: 분자가 구멍을 제외하므로 속이 빈 도포는 1 미만이어야 한다.
    hollow = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(hollow, (30, 30), 20, 1, -1)
    cv2.circle(hollow, (30, 30), 10, 0, -1)
    filled = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(filled, (30, 30), 20, 1, -1)

    (hollow_blob,) = measure_blobs(_mask(hollow))
    (filled_blob,) = measure_blobs(_mask(filled))

    assert hollow_blob.solidity < 0.8 < filled_blob.solidity <= 1.0


def test_blob_area_denominator_is_the_roi_mask_not_its_bounding_box():
    # 6.1 / L4: 분모는 사상된 실제 ROI 마스크 픽셀 수다. 회전 ROI 에서 마스크
    # 픽셀 수와 bbox 넓이가 어긋나므로 둘 중 무엇을 썼는지 값이 갈린다.
    roi = make_roi_mask(
        (0.25, 0.25, 0.5, 0.5), Pose(tx=0.0, ty=0.0, theta_deg=30.0), (200, 200)
    )
    roi_pixels = int(np.count_nonzero(roi))
    ys, xs = np.nonzero(roi)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    assert roi_pixels != bbox_area, "이 회귀는 두 분모가 달라야 성립한다"

    foreground = np.zeros((200, 200), dtype=np.uint8)
    foreground[95:105, 95:105] = 255
    foreground = cv2.bitwise_and(foreground, roi)
    (blob,) = measure_blobs(DetectionMask(foreground, roi))

    assert blob.area == pytest.approx(blob.pixels / roi_pixels)
    assert blob.area != pytest.approx(blob.pixels / bbox_area)


def test_blob_counts_diagonal_contact_as_one_component():
    # 6.1: coverage 와 동일한 8-연결 기준. 4-연결이면 2개로 세어진다.
    foreground = np.zeros((10, 10), dtype=np.uint8)
    foreground[2, 2] = 1
    foreground[3, 3] = 1

    assert len(measure_blobs(_mask(foreground))) == 1


def test_blob_empty_foreground_measures_zero_components():
    # 6.1: 성분 0개는 결측이 아니라 실측값. coverage 의 continuity=None 과 다르다.
    assert measure_blobs(_mask(np.zeros((10, 10), dtype=np.uint8))) == ()


def test_blob_measures_a_component_touching_the_image_border():
    # 측정은 성분 bbox 로 크롭해 수행한다. 프레임 가장자리에 붙은 성분에서
    # 크롭이 외곽선을 잘라먹으면 값이 조용히 틀어진다.
    edge = np.zeros((40, 40), dtype=np.uint8)
    edge[0:3, 0:3] = 1
    interior = np.zeros((40, 40), dtype=np.uint8)
    interior[10:13, 10:13] = 1

    (at_edge,) = measure_blobs(_mask(edge))
    (inside,) = measure_blobs(_mask(interior))

    assert (at_edge.pixels, at_edge.solidity, at_edge.aspect_ratio) == (
        inside.pixels,
        inside.solidity,
        inside.aspect_ratio,
    )


def test_blob_filters_remove_before_counting():
    # 6.1 의 순서 계약: 측정 -> 실제 제거 -> 통과분만 개수 판정.
    # 원본 3개 중 작은 2개가 area_min 에 걸려 사라지고 1개만 세어진다.
    foreground = np.zeros((60, 60), dtype=np.uint8)
    foreground[5:8, 5:8] = 1        # 9px
    foreground[15:18, 15:18] = 1    # 9px
    foreground[30:50, 30:50] = 1    # 400px
    mask = _mask(foreground)
    assert len(measure_blobs(mask)) == 3

    # 9/3600 = 0.0025, 400/3600 = 0.111
    result = evaluate_blob(mask, _blob_rule(count_min=1, count_max=1, area_min=0.01))

    assert len(result.kept) == 1
    assert len(result.rejected) == 2
    assert result.passed
    # 제거가 없었다면 3 > count_max=1 로 실패했어야 한다.
    assert not evaluate_blob(mask, _blob_rule(count_min=1, count_max=1)).passed


def test_blob_rejection_carries_the_threshold_that_removed_it():
    foreground = np.zeros((60, 60), dtype=np.uint8)
    foreground[5:8, 5:8] = 1
    mask = _mask(foreground)

    result = evaluate_blob(mask, _blob_rule(count_min=0, count_max=5, area_min=0.01))

    (rejection,) = result.rejected
    assert rejection.param == "area_min"
    assert rejection.operator == "<"
    assert rejection.threshold == 0.01
    assert rejection.measured == pytest.approx(9 / 3600)
    assert rejection.blob.pixels == 9


def test_blob_count_bounds_are_inclusive_and_report_the_failed_side():
    foreground = np.zeros((40, 40), dtype=np.uint8)
    foreground[5:10, 5:10] = 1
    foreground[20:25, 20:25] = 1
    mask = _mask(foreground)

    assert evaluate_blob(mask, _blob_rule(count_min=2, count_max=2)).passed
    assert evaluate_blob(mask, _blob_rule(count_min=3, count_max=5)).failed_params == (
        "count_min",
    )
    assert evaluate_blob(mask, _blob_rule(count_min=0, count_max=1)).failed_params == (
        "count_max",
    )


def test_blob_every_shape_parameter_is_wired_to_a_filter():
    # VSGP 의 advisor->슬라이더 사고와 같은 계약. 스펙에만 있고 필터에 없는 키는
    # 로드는 되는데 아무도 읽지 않는 죽은 파라미터가 된다(L1).
    spec_keys = set(PARAM_SPECS["blob"]) - {"count_min", "count_max"}
    filter_keys = {name for name, _field, _op in BLOB_FILTERS}
    assert filter_keys == spec_keys

    # 필드와 부등호는 파라미터 *이름*에서 유도한다. 표에서 읽어와 비교하면
    # 표가 틀렸을 때 기대값도 같이 틀어져 아무것도 검사하지 못한다.
    for name, field, operator in BLOB_FILTERS:
        stem, _, bound = name.rpartition("_")
        assert field == stem, f"{name}: {stem} 을 재야 하는데 {field} 를 읽는다"
        assert operator == ("<" if bound == "min" else ">")
        assert hasattr(BlobMeasurement(1, 0.1, 1.0, 1.0, 1.0), field)


def test_blob_rejects_a_rule_from_another_tool():
    with pytest.raises(DetectionInputError):
        evaluate_blob(_mask(np.zeros((5, 5), dtype=np.uint8)), _rule(min=0.1))


def test_every_blob_filter_flips_the_verdict_on_its_own_threshold():
    # L1 감도 프로브: 필터 7종 각각이 실제로 blob 을 제거하고, 반대쪽으로 옮기면
    # 남긴다. 한쪽만 확인하면 부등호가 뒤집혀도 통과한다 - area_min 만 쓰던
    # 초안이 실제로 area_max 부등호 변이를 놓쳤다.
    hollow = np.zeros((80, 80), dtype=np.uint8)
    hollow[20:40, 15:65] = 1          # 50x20 -> AR 2.5
    hollow[25:35, 25:55] = 0          # 속을 비워 solidity/circularity 를 1 미만으로
    mask = _mask(hollow)
    (blob,) = measure_blobs(mask)
    assert blob.circularity < 1.0 and blob.solidity < 1.0 and blob.aspect_ratio > 1.0
    assert 0.0 < blob.area < 1.0

    for name, _field, _operator in BLOB_FILTERS:
        # 방향도 이름에서 유도한다 - BLOB_FILTERS 의 부등호를 그대로 쓰면
        # 부등호가 뒤집혔을 때 기대값이 함께 뒤집혀 통과해 버린다.
        stem, _, bound = name.rpartition("_")
        is_min = bound == "min"
        measured = float(getattr(blob, stem))
        removing = measured * (1.01 if is_min else 0.99)
        keeping = measured * (0.99 if is_min else 1.01)

        removed = evaluate_blob(
            mask, _blob_rule(count_min=0, count_max=9, **{name: removing})
        )
        kept = evaluate_blob(
            mask, _blob_rule(count_min=0, count_max=9, **{name: keeping})
        )

        assert removed.kept == (), f"{name}: 임계를 넘겼는데 제거되지 않았다"
        assert [r.param for r in removed.rejected] == [name]
        assert removed.rejected[0].operator == ("<" if is_min else ">")
        assert kept.rejected == (), f"{name}: 통과해야 할 쪽에서 제거됐다"
        assert kept.kept == (blob,)
