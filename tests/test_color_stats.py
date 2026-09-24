"""color_stats (docs/DESIGN.md 6.2) and the golden footprint G it reads (6.3)."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from avap.alignment import Aligner, AlignStatus, Pose, transform_points
from avap.detection import (
    ColorStatsMeasurement,
    DetectionInputError,
    evaluate_color_stats,
    golden_footprint,
    make_mask,
    make_roi_mask,
    map_footprint,
    measure_color_stats,
)
from avap.recipe import PARAM_SPECS, Rule, load_recipe
from avap.synth import (
    BEAD_THICKNESS, BEAD_X0, BEAD_X1, BEAD_Y, BG, MATERIAL, apply_pose, draw_golden,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "recipes" / "sample_synth.json"

# Pure colours whose OpenCV 8-bit HSV is known exactly.
RED = (0, 0, 255)           # H=0   S=255 V=255
RED_356 = (17, 0, 255)      # H=178 S=255 V=255 (356 degrees)
CYAN = (255, 255, 0)        # H=90  S=255 V=255 (opposite of red)
WRONG_MATERIAL = (40, 40, 200)


def _rule(**params) -> Rule:
    return Rule("color_stats", tuple(sorted(params.items())))


def _solid(bgr, height: int = 4, width: int = 4) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = bgr
    return image


def _everywhere(image: np.ndarray) -> np.ndarray:
    return np.full(image.shape[:2], 255, dtype=np.uint8)


def _distance(bgr, centre) -> float:
    image = _solid(bgr)
    return measure_color_stats(image, _everywhere(image), centre).distance


def _sample():
    recipe = load_recipe(SAMPLE)
    roi_config = recipe.rois[0]
    rule = next(rule for rule in roi_config.rules if rule.tool == "color_stats")
    return recipe, roi_config, rule


def _wrong_material_golden() -> np.ndarray:
    image = draw_golden("ng_missing")
    cv2.line(image, (BEAD_X0, BEAD_Y), (BEAD_X1, BEAD_Y), WRONG_MATERIAL, BEAD_THICKNESS)
    return image


SCENARIOS = {
    "ok": lambda: draw_golden("ok"),
    "ng_missing": lambda: draw_golden("ng_missing"),
    "ng_broken": lambda: draw_golden("ng_broken"),
    "wrong_material": _wrong_material_golden,
}


# ── G: golden footprint (6.3) ────────────────────────────────────────────

def test_golden_footprint_is_the_ordinary_mask_of_the_golden():
    recipe, roi_config, _rule_ = _sample()
    golden = draw_golden("ok")
    roi = make_roi_mask(roi_config.rect_golden, Pose(0.0, 0.0, 0.0), recipe.golden_size)

    g0 = golden_footprint(golden, roi_config.rect_golden, roi_config.detect)

    assert np.array_equal(g0, make_mask(golden, roi, roi_config.detect).foreground)
    assert int(np.count_nonzero(g0)) == 8767      # docs/DESIGN.md 6.2 table


def test_golden_without_material_is_a_configuration_error_not_an_empty_footprint():
    _recipe, roi_config, _rule_ = _sample()
    with pytest.raises(DetectionInputError, match="G0: empty"):
        golden_footprint(draw_golden("ng_missing"), roi_config.rect_golden, roi_config.detect)


def test_identity_pose_maps_the_footprint_onto_itself():
    recipe, roi_config, _rule_ = _sample()
    g0 = golden_footprint(draw_golden("ok"), roi_config.rect_golden, roi_config.detect)
    roi = make_roi_mask(roi_config.rect_golden, Pose(0.0, 0.0, 0.0), recipe.golden_size)

    assert np.array_equal(map_footprint(g0, Pose(0.0, 0.0, 0.0), roi), g0)


@pytest.mark.parametrize("pose", [Pose(12.0, -7.0, 0.0), Pose(0.0, 0.0, 2.5),
                                  Pose(-9.0, 5.0, -1.7)])
def test_footprint_moves_with_transform_points(pose):
    # A 3x3 block must land where transform_points sends its centre - the same
    # convention make_roi_mask uses, so G and the ROI cannot drift apart.
    size = (200, 160)
    g0 = np.zeros((160, 200), dtype=np.uint8)
    g0[69:72, 139:142] = 255                     # centre (140, 70)
    everywhere = np.full((160, 200), 255, dtype=np.uint8)

    mapped = map_footprint(g0, pose, everywhere)

    ys, xs = np.nonzero(mapped)
    target = transform_points(np.array([[140.0, 70.0]]), pose, size)[0]
    assert abs(xs.mean() - target[0]) <= 0.5 and abs(ys.mean() - target[1]) <= 0.5


def test_mapped_footprint_is_binary_inside_the_roi_and_zero_outside_the_frame():
    g0 = np.zeros((40, 60), dtype=np.uint8)
    g0[10:20, 45:60] = 255                        # touches the right frame edge
    roi = np.zeros((40, 60), dtype=np.uint8)
    roi[:, :50] = 255

    mapped = map_footprint(g0, Pose(8.0, 0.0, 0.0), roi)

    assert set(np.unique(mapped).tolist()) <= {0, 255}
    assert not np.any(mapped[:, 50:]), "ROI 밖 픽셀"
    # Shifted right by 8: columns 53..59 would come from 45..51 - but the ROI
    # stops at 49, and nothing wraps in from the left edge.
    assert not np.any(mapped[:, :53])


@pytest.mark.parametrize("pose", [
    Pose(math.nan, 0.0, 0.0), Pose(0.0, math.inf, 0.0), Pose(0.0, 0.0, True),
    (0.0, 0.0, 0.0), None,
])
def test_map_footprint_refuses_a_pose_it_cannot_apply(pose):
    g0 = np.full((4, 4), 255, dtype=np.uint8)
    with pytest.raises(DetectionInputError, match="pose"):
        map_footprint(g0, pose, g0)


def test_map_footprint_refuses_masks_of_different_shapes():
    with pytest.raises(DetectionInputError):
        map_footprint(np.full((4, 4), 255, np.uint8), Pose(0.0, 0.0, 0.0),
                      np.full((4, 5), 255, np.uint8))


@pytest.mark.parametrize("pose", [Pose(-9.0, 6.7, -2.9), Pose(9.0, -6.7, 2.9)])
def test_morphology_order_moves_the_footprint_by_a_few_pixels_inside_the_pose_gate(pose):
    # G0 는 골든 좌표에서 형태학을 거친 뒤 회전되고, C 는 회전된 제품에 형태학을 건다.
    # 축 정렬 커널에서 둘은 교환되지 않는다(Codex #14 리뷰). 샘플 설정에서 그 차이가
    # pose 게이트(3°) 끝에서도 몇 픽셀임을 고정한다 - 비교 대상은 형태학 전 마스크를
    # 먼저 사상하고 프레임에서 같은 make_mask 로 형태학을 거는 순서다.
    recipe, roi_config, _rule_ = _sample()
    width, height = recipe.golden_size
    golden = draw_golden("ok")
    g0 = golden_footprint(golden, roi_config.rect_golden, roi_config.detect)
    roi = make_roi_mask(roi_config.rect_golden, pose, recipe.golden_size)
    spec_order = map_footprint(g0, pose, roi)

    no_morph = tuple((k, v) for k, v in roi_config.detect if k != "morph")
    everywhere = np.full((height, width), 255, dtype=np.uint8)
    raw = map_footprint(make_mask(golden, everywhere, no_morph).foreground, pose, everywhere)
    painted = np.zeros_like(golden)
    painted[:] = BG
    painted[raw != 0] = MATERIAL
    frame_order = make_mask(painted, roi, roi_config.detect).foreground

    differ = int(np.count_nonzero((spec_order != 0) ^ (frame_order != 0)))
    assert differ <= 0.001 * np.count_nonzero(spec_order), differ


# ── measurement properties (6.2) ─────────────────────────────────────────

def test_foreground_cannot_tell_wrong_material_from_missing_but_the_footprint_can():
    # docs/DESIGN.md 6.2 table: the HSV foreground is 8767 / 0 / 0, so a tool
    # that reads C sees wrong material and no material as the same thing.
    recipe, roi_config, rule = _sample()
    roi = make_roi_mask(roi_config.rect_golden, Pose(0.0, 0.0, 0.0), recipe.golden_size)
    g0 = golden_footprint(draw_golden("ok"), roi_config.rect_golden, roi_config.detect)

    measured = {}
    for name in ("ok", "wrong_material", "ng_missing"):
        image = SCENARIOS[name]()
        foreground = make_mask(image, roi, roi_config.detect).foreground
        result = evaluate_color_stats(image, g0, rule)
        measured[name] = (int(np.count_nonzero(foreground)), result)

    assert [measured[n][0] for n in ("ok", "wrong_material", "ng_missing")] == [8767, 0, 0]
    for _fg, result in measured.values():
        # G is read whole: the pixel count is |G|, never |G intersect C|.
        assert result.measurement.footprint_pixels == 8767
    assert measured["ok"][1].passed
    assert not measured["wrong_material"][1].passed
    assert not measured["ng_missing"][1].passed
    # Both fail, but on different colours - the Evidence tells them apart.
    assert measured["wrong_material"][1].measurement.distance > \
        measured["ng_missing"][1].measurement.distance > 0.2


def test_hue_is_circular_on_the_180_period():
    # H=0 and H=178 are both red. The DESIGN.md 6.2 example: D = 0.0348995.
    assert _distance(RED_356, (0.0, 1.0, 1.0)) == pytest.approx(0.0348995, abs=1e-7)
    assert _distance(RED, (178 / 180, 1.0, 1.0)) == pytest.approx(0.0348995, abs=1e-7)
    # Expected hue 0 and 1 are the same colour.
    assert _distance(RED, (0.0, 1.0, 1.0)) == pytest.approx(_distance(RED, (1.0, 1.0, 1.0)))


def test_measured_hue_is_read_on_the_180_period_not_179():
    # Cyan is OpenCV H=90: half a turn only on the 180 period. Read as 90/179 it
    # would sit 0.0088 away from its own centre.
    assert _distance(CYAN, (0.5, 1.0, 1.0)) < 1e-12


def test_a_mixture_of_the_two_reds_does_not_average_to_cyan():
    image = _solid(RED, 4, 4)
    image[:, 2:] = RED_356
    region = _everywhere(image)
    near_red = measure_color_stats(image, region, (0.0, 1.0, 1.0)).distance
    near_cyan = measure_color_stats(image, region, (0.494444, 1.0, 1.0)).distance
    assert near_red < 0.03 < 0.9 < near_cyan


@pytest.mark.parametrize("grey", [(0, 0, 0), (128, 128, 128), (255, 255, 255)])
def test_hue_does_not_matter_without_saturation_or_value(grey):
    # OpenCV gives H=0 to every grey; the expected centre's hue must not matter
    # at s=0, and neither hue nor saturation at v=0.
    base_s0 = _distance(grey, (0.0, 0.0, 0.5))
    for hue in (0.25, 0.5, 0.9):
        assert _distance(grey, (hue, 0.0, 0.5)) == pytest.approx(base_s0, abs=1e-12)
    base_v0 = _distance(grey, (0.0, 0.0, 0.0))
    for hue, sat in ((0.3, 1.0), (0.7, 0.5)):
        assert _distance(grey, (hue, sat, 0.0)) == pytest.approx(base_v0, abs=1e-12)


def test_distance_is_one_exactly_for_opposite_saturated_hues():
    assert _distance(RED, (0.5, 1.0, 1.0)) == pytest.approx(1.0, abs=1e-12)


def test_distance_never_exceeds_one():
    levels = (0, 51, 102, 153, 204, 255)
    colours = [(b, g, r) for b in levels for g in levels for r in levels]
    image = np.array([colours], dtype=np.uint8)            # 1 x 216
    worst = 0.0
    for h in (0.0, 0.25, 0.5, 0.75):
        for s in (0.0, 0.5, 1.0):
            for v in (0.0, 0.5, 1.0):
                for column in range(image.shape[1]):
                    region = np.zeros(image.shape[:2], dtype=np.uint8)
                    region[0, column] = 255
                    worst = max(worst, measure_color_stats(image, region, (h, s, v)).distance)
    assert worst <= 1.0 + 1e-12
    assert worst == pytest.approx(1.0, abs=1e-12)


def test_opposite_colours_do_not_cancel():
    # Mean phi of red and cyan is (0, 0, 1) - a mean-colour distance to a white
    # centre would be 0. Per-pixel RMS keeps each pixel's distance of 1.
    image = _solid(RED, 4, 4)
    image[:, 2:] = CYAN
    result = evaluate_color_stats(image, _everywhere(image),
                                  _rule(expect_hsv_center=(0.0, 0.0, 1.0), max_dist=0.4))
    assert result.measurement.distance == pytest.approx(0.5, abs=1e-12)
    assert not result.passed


def test_distance_is_the_root_mean_square_not_the_mean_distance():
    # Half the pixels sit at distance 0 and half at 2 from red. RMS gives
    # sqrt((0 + 4) / 2) / 2 = 0.7071; a plain mean of distances would give 0.5.
    # Every other test here uses pixels at one common distance, where the two
    # agree - this is the input that tells them apart.
    image = _solid(RED, 4, 4)
    image[:, 2:] = CYAN
    measured = measure_color_stats(image, _everywhere(image), (0.0, 1.0, 1.0)).distance
    assert measured == pytest.approx(math.sqrt(2.0) / 2.0, abs=1e-12)


def test_only_pixels_inside_the_footprint_are_read():
    image = _solid(RED, 4, 4)
    image[:, 2:] = CYAN
    region = np.zeros((4, 4), dtype=np.uint8)
    region[:, :2] = 255
    measurement = measure_color_stats(image, region, (0.0, 1.0, 1.0))
    assert measurement == ColorStatsMeasurement(distance=pytest.approx(0.0, abs=1e-12),
                                                footprint_pixels=8)


# ── L1: every parameter flips the verdict on a fixed input ───────────────

def test_max_dist_flips_the_verdict_at_the_measured_distance():
    image = _solid(RED_356)
    region = _everywhere(image)
    centre = (0.0, 1.0, 1.0)
    measured = measure_color_stats(image, region, centre).distance

    assert evaluate_color_stats(image, region, _rule(expect_hsv_center=centre,
                                                     max_dist=measured)).passed
    assert evaluate_color_stats(image, region, _rule(
        expect_hsv_center=centre, max_dist=math.nextafter(measured, 1.0))).passed
    below = evaluate_color_stats(image, region, _rule(
        expect_hsv_center=centre, max_dist=math.nextafter(measured, 0.0)))
    assert not below.passed and below.failed_params == ("max_dist",)


def test_max_dist_one_is_the_permissive_endpoint_not_a_dead_value():
    # The worst colour (D = 1) still passes at max_dist = 1: every D is <= 1.
    # That does not make 1 dead - the flip above happens at any D below it.
    image = _solid(RED)
    result = evaluate_color_stats(image, _everywhere(image),
                                  _rule(expect_hsv_center=(0.5, 1.0, 1.0), max_dist=1.0))
    assert result.measurement.distance == pytest.approx(1.0, abs=1e-12)
    assert result.passed


@pytest.mark.parametrize(
    "channel, matching, other",
    [
        ("H", (0.0, 1.0, 1.0), (0.5, 1.0, 1.0)),
        ("S", (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
        ("V", (0.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
    ],
)
def test_each_channel_of_the_expected_centre_flips_the_verdict(channel, matching, other):
    # Only one component differs between the two centres; the image and
    # max_dist are fixed. Each channel therefore reaches the verdict on its own.
    differing = [i for i in range(3) if matching[i] != other[i]]
    assert differing == ["HSV".index(channel)]
    image = _solid(RED)
    region = _everywhere(image)
    assert evaluate_color_stats(image, region,
                                _rule(expect_hsv_center=matching, max_dist=0.3)).passed
    assert not evaluate_color_stats(image, region,
                                    _rule(expect_hsv_center=other, max_dist=0.3)).passed


def test_witnesses_above_cover_every_color_stats_parameter():
    assert set(PARAM_SPECS["color_stats"]) == {"expect_hsv_center", "max_dist"}


# ── L4: the verdict does not depend on the recovered pose ───────────────

@pytest.mark.parametrize(
    "scenario, expected",
    [("ok", True), ("ng_missing", False), ("ng_broken", True), ("wrong_material", False)],
)
def test_color_stats_verdict_is_invariant_across_recovered_poses(scenario, expected):
    # ng_broken passes on purpose: its colour is right; the gap is coverage's job.
    recipe, roi_config, rule = _sample()
    golden = draw_golden("ok")
    aligner = Aligner(recipe.alignment, golden)
    g0 = golden_footprint(golden, roi_config.rect_golden, roi_config.detect)
    source = SCENARIOS[scenario]()
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
        roi = make_roi_mask(roi_config.rect_golden, aligned.pose, size)
        footprint = map_footprint(g0, aligned.pose, roi)
        assert evaluate_color_stats(image, footprint, rule).passed is expected


# ── API boundary: DetectionInputError, never a foreign exception ─────────

BAD_CENTRES = [
    pytest.param([True, 0.0, 0.5], id="H-true"),
    pytest.param([0.0, False, 0.5], id="S-false"),
    pytest.param([0.0, 0.0, True], id="V-true"),
    pytest.param(np.array([True, False, True]), id="bool-array"),
    pytest.param([math.nan, 0.0, 0.5], id="nan"),
    pytest.param([0.0, math.inf, 0.5], id="inf"),
    pytest.param([-0.1, 0.0, 0.5], id="negative"),
    pytest.param([0.0, 0.0, 1.1], id="above-one"),
    pytest.param([10 ** 400, 0.0, 0.5], id="huge-int"),
    pytest.param([0.0, 0.5], id="two-values"),
    pytest.param([0.0, 0.5, 0.5, 0.5], id="four-values"),
    pytest.param("0.0", id="string"),
    pytest.param(None, id="none"),
    pytest.param([[0.0], [0.5], [0.5]], id="nested"),
    pytest.param(["0", 0.5, 0.5], id="string-element"),
]


@pytest.mark.parametrize("centre", BAD_CENTRES)
def test_bad_expected_centre_is_a_detection_input_error(centre):
    image = _solid(RED)
    with pytest.raises(DetectionInputError, match="expect_hsv_center"):
        measure_color_stats(image, _everywhere(image), centre)


@pytest.mark.parametrize("centre", [(0, 0, 1), [1, 1, 0], np.array([0.25, 0.5, 0.75])])
def test_integer_and_array_centres_are_accepted(centre):
    image = _solid(RED)
    measure_color_stats(image, _everywhere(image), centre)


@pytest.mark.parametrize(
    "image, footprint, message",
    [
        (np.zeros((4, 4, 3), np.float32), np.full((4, 4), 255, np.uint8), "image_bgr"),
        (np.zeros((4, 4), np.uint8), np.full((4, 4), 255, np.uint8), "image_bgr"),
        (np.zeros((0, 4, 3), np.uint8), np.zeros((0, 4), np.uint8), "image_bgr"),
        (np.zeros((4, 4, 3), np.uint8), np.full((4, 5), 255, np.uint8), "footprint"),
        (np.zeros((4, 4, 3), np.uint8), np.full((4, 4), 7, np.uint8), "footprint"),
        (np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8), "G: empty"),
        (np.zeros((4, 4, 3), np.uint8), None, "footprint"),
    ],
)
def test_bad_image_or_footprint_is_a_detection_input_error(image, footprint, message):
    with pytest.raises(DetectionInputError, match=message):
        measure_color_stats(image, footprint, (0.0, 0.0, 0.0))


def test_evaluate_refuses_a_rule_from_another_tool():
    image = _solid(RED)
    with pytest.raises(DetectionInputError, match="color_stats Rule"):
        evaluate_color_stats(image, _everywhere(image), Rule("coverage", (("min", 0.1),)))


@pytest.mark.parametrize("bound, index", [(b, i) for b in ("lower", "upper") for i in range(3)])
def test_make_mask_refuses_a_bool_inside_hsv_bounds(bound, index):
    # Same defect class as issue #11 on the mask API: np.asarray would have
    # turned [True, 0, 0] into [1.0, 0, 0] without a word.
    detect = {"space": "hsv", "lower": [0.0, 0.0, 0.0], "upper": [1.0, 1.0, 1.0]}
    detect[bound] = list(detect[bound])
    detect[bound][index] = bool(detect[bound][index])
    image = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(DetectionInputError, match="three numeric"):
        make_mask(image, np.full((4, 4), 255, np.uint8), detect)
