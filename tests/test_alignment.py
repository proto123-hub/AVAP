"""Phase 1 two-anchor alignment against deterministic synthetic ground truth."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import avap.alignment as alignment_module
from avap.alignment import (
    Aligner,
    AlignmentConfigError,
    AlignFailCode,
    AlignStatus,
    Pose,
    normalized_shift_frac,
    transform_points,
)
from avap.recipe import load_recipe, parse_recipe
from avap.synth import BG, apply_pose, draw_golden


REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "recipes" / "sample_synth.json"


@pytest.fixture(scope="module")
def golden_bgr() -> np.ndarray:
    return draw_golden("ok")


@pytest.fixture(scope="module")
def sample_alignment():
    return load_recipe(SAMPLE).alignment


def _alignment_with(*, min_score: float = 0.2, **gate_overrides):
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for anchor in data["alignment"]["anchors"]:
        anchor["min_score"] = min_score
    data["alignment"]["pose_gates"].update(gate_overrides)
    return parse_recipe(data).alignment


def _apply_similarity(
    image: np.ndarray,
    *,
    scale: float,
    tx: float = 0.0,
    ty: float = 0.0,
    theta_deg: float = 0.0,
) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), theta_deg, scale
    )
    matrix[:, 2] += (tx, ty)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BG,
    )


def _assert_unknown(result, code: AlignFailCode) -> None:
    assert result.status is AlignStatus.UNKNOWN
    assert result.fail_code is code
    assert result.pose is None  # downstream cannot use a rejected pose as fallback


def test_recovers_30_seeded_poses_within_phase1_budget(golden_bgr, sample_alignment):
    aligner = Aligner(sample_alignment, golden_bgr)
    rng = np.random.default_rng(20260822)

    for index in range(30):
        tx, ty = rng.uniform(-30.0, 30.0, size=2)
        theta_deg = float(rng.uniform(-2.0, 2.0))
        image = apply_pose(golden_bgr, float(tx), float(ty), theta_deg)

        result = aligner.align(image)

        assert result.status is AlignStatus.OK, f"pose {index}: {result}"
        assert result.fail_code is None
        assert result.pose is not None
        assert len(result.anchor_matches) == 2
        translation_error = math.hypot(result.pose.tx - tx, result.pose.ty - ty)
        assert translation_error <= 2.0, f"pose {index}: translation"
        assert abs(result.pose.theta_deg - theta_deg) <= 0.5, f"pose {index}: theta"


def test_image_size_mismatch_has_exact_unknown_code(golden_bgr, sample_alignment):
    aligner = Aligner(sample_alignment, golden_bgr)

    for image in (golden_bgr[:-1, :, :], golden_bgr[:, :-1, :]):
        _assert_unknown(
            aligner.align(image), AlignFailCode.IMAGE_SIZE_MISMATCH
        )


def test_low_anchor_score_has_exact_unknown_code(golden_bgr, sample_alignment):
    no_landmarks = np.full_like(golden_bgr, 17)
    result = Aligner(sample_alignment, golden_bgr).align(no_landmarks)

    _assert_unknown(result, AlignFailCode.ANCHOR_SCORE_LOW)


def test_one_low_anchor_is_enough_to_reject(golden_bgr, sample_alignment):
    image = golden_bgr.copy()
    first = sample_alignment.anchors[0]
    height, width = image.shape[:2]
    sx, sy, sw, sh = (
        int(round(first.search[0] * width)),
        int(round(first.search[1] * height)),
        int(round((first.search[0] + first.search[2]) * width)),
        int(round((first.search[1] + first.search[3]) * height)),
    )
    image[sy:sh, sx:sw] = 17

    result = Aligner(sample_alignment, golden_bgr).align(image)

    _assert_unknown(result, AlignFailCode.ANCHOR_SCORE_LOW)
    assert result.anchor_matches[0].score < first.min_score
    assert result.anchor_matches[1].score >= sample_alignment.anchors[1].min_score


@pytest.mark.parametrize("scale", [1.04, 0.96])
def test_scale_out_of_range_has_exact_unknown_code(golden_bgr, scale):
    alignment = _alignment_with(scale_tol=0.01)
    image = _apply_similarity(golden_bgr, scale=scale)
    result = Aligner(alignment, golden_bgr).align(image)

    _assert_unknown(result, AlignFailCode.SCALE_OUT_OF_RANGE)


def test_shift_out_of_range_has_exact_unknown_code(golden_bgr):
    alignment = _alignment_with(max_shift_frac=0.001)
    image = apply_pose(golden_bgr, tx=12.0, ty=-9.0, theta_deg=0.0)
    result = Aligner(alignment, golden_bgr).align(image)

    _assert_unknown(result, AlignFailCode.SHIFT_OUT_OF_RANGE)


def test_shift_fraction_contract_is_normalized_per_axis():
    size = (960, 720)
    assert normalized_shift_frac(Pose(960.0, 720.0, 0.0), size) == pytest.approx(1.0)
    assert normalized_shift_frac(Pose(960.0, 0.0, 0.0), size) == pytest.approx(
        1 / np.sqrt(2.0)
    )


@pytest.mark.parametrize("theta_deg", [1.25, -1.25])
def test_rotation_out_of_range_has_exact_unknown_code(golden_bgr, theta_deg):
    alignment = _alignment_with(max_rotation_deg=0.25)
    image = apply_pose(golden_bgr, tx=0.0, ty=0.0, theta_deg=theta_deg)
    result = Aligner(alignment, golden_bgr).align(image)

    _assert_unknown(result, AlignFailCode.ROTATION_OUT_OF_RANGE)


def test_gate_boundaries_are_inclusive_and_next_float_outside_rejects(golden_bgr):
    permissive = _alignment_with(
        min_score=0.01, scale_tol=1.0, max_shift_frac=1.0,
        max_rotation_deg=10.0,
    )

    score_image = apply_pose(golden_bgr, tx=7.0, ty=-5.0, theta_deg=0.4)
    score_probe = Aligner(permissive, golden_bgr).align(score_image)
    assert score_probe.pose is not None
    exact_anchors = tuple(
        replace(anchor, min_score=match.score)
        for anchor, match in zip(permissive.anchors, score_probe.anchor_matches)
    )
    exact_score = replace(permissive, anchors=exact_anchors)
    assert Aligner(exact_score, golden_bgr).align(score_image).status is AlignStatus.OK
    outside_anchors = (
        replace(exact_anchors[0], min_score=math.nextafter(exact_anchors[0].min_score, 1.0)),
        exact_anchors[1],
    )
    _assert_unknown(
        Aligner(replace(permissive, anchors=outside_anchors), golden_bgr).align(score_image),
        AlignFailCode.ANCHOR_SCORE_LOW,
    )

    cases = (
        (
            _apply_similarity(golden_bgr, scale=1.02),
            "scale_tol", AlignFailCode.SCALE_OUT_OF_RANGE,
            lambda result: abs(result.scale_ratio - 1.0),
        ),
        (
            apply_pose(golden_bgr, tx=12.0, ty=-9.0, theta_deg=0.0),
            "max_shift_frac", AlignFailCode.SHIFT_OUT_OF_RANGE,
            lambda result: normalized_shift_frac(result.pose, (960, 720)),
        ),
        (
            apply_pose(golden_bgr, tx=0.0, ty=0.0, theta_deg=-1.25),
            "max_rotation_deg", AlignFailCode.ROTATION_OUT_OF_RANGE,
            lambda result: abs(result.pose.theta_deg),
        ),
    )
    for image, field, fail_code, measure in cases:
        probe = Aligner(permissive, golden_bgr).align(image)
        assert probe.status is AlignStatus.OK and probe.pose is not None
        value = float(measure(probe))
        exact = replace(permissive, **{field: value})
        assert Aligner(exact, golden_bgr).align(image).status is AlignStatus.OK
        outside = replace(permissive, **{field: math.nextafter(value, 0.0)})
        _assert_unknown(Aligner(outside, golden_bgr).align(image), fail_code)


def test_nonfinite_golden_is_config_error_and_nonfinite_input_is_unknown(
    golden_bgr, sample_alignment
):
    bad_golden = golden_bgr.astype(np.float32)
    bad_golden[0, 0, 0] = np.nan
    with pytest.raises(AlignmentConfigError, match="NaN"):
        Aligner(sample_alignment, bad_golden)

    bad_image = golden_bgr.astype(np.float32)
    bad_image[0, 0, 0] = np.inf
    result = Aligner(sample_alignment, golden_bgr).align(bad_image)
    _assert_unknown(result, AlignFailCode.NUMERIC_NONFINITE)


def test_nonfinite_match_coordinate_cannot_produce_ok(
    monkeypatch, golden_bgr, sample_alignment
):
    monkeypatch.setattr(
        alignment_module, "_match_patch", lambda _window, _patch: (math.nan, 0.0, 0.99)
    )

    result = Aligner(sample_alignment, golden_bgr).align(golden_bgr)

    _assert_unknown(result, AlignFailCode.NUMERIC_NONFINITE)


def test_parabolic_subpixel_peak_flat_boundary_and_nonfinite():
    assert alignment_module._parabolic_offset(-0.5625, 0.9375, 0.4375) == pytest.approx(0.25)
    assert alignment_module._parabolic_offset(1.0, 1.0, 1.0) == 0.0

    response = np.array(
        [[0.0, 0.2, 0.1], [0.3, 1.0, 0.7], [0.1, 0.5, 0.2]],
        dtype=np.float32,
    )
    x, y, score = alignment_module._peak(response)
    assert 0.0 < x - 1.0 < 0.5
    assert 0.0 < y - 1.0 < 0.5
    assert score == pytest.approx(1.0)

    boundary = np.array([[1.0, 0.5], [0.4, 0.3]], dtype=np.float32)
    assert alignment_module._peak(boundary) == (0.0, 0.0, 1.0)

    response[0, 0] = np.nan
    assert all(math.isnan(value) for value in alignment_module._peak(response))


def test_transform_points_matches_synth_pose_and_round_trips(
    golden_bgr, sample_alignment
):
    image = apply_pose(golden_bgr, tx=17.25, ty=-11.5, theta_deg=1.3)
    result = Aligner(sample_alignment, golden_bgr).align(image)
    assert result.status is AlignStatus.OK and result.pose is not None

    height, width = golden_bgr.shape[:2]
    golden_size = (width, height)
    points = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [220.0, 180.0],
         [width / 2.0, height / 2.0]],
        dtype=np.float64,
    )
    mapped = transform_points(points, result.pose, golden_size)

    expected_matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), result.pose.theta_deg, 1.0
    )
    expected_matrix[:, 2] += (result.pose.tx, result.pose.ty)
    expected = np.column_stack([points, np.ones(len(points))]) @ expected_matrix.T
    np.testing.assert_allclose(mapped, expected, atol=1e-6)

    restored = transform_points(
        mapped, result.pose, golden_size, inverse=True
    )
    assert np.max(np.linalg.norm(restored - points, axis=1)) < 1e-6
