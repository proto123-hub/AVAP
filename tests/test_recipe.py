"""Recipe loader — the schema side of Design Law L1 (dead parameters cannot load)."""
import copy
import json
from pathlib import Path

import pytest

from avap.recipe import RecipeError, compute_fingerprint, load_recipe, parse_recipe

SAMPLE = Path(__file__).resolve().parents[1] / "recipes" / "sample_synth.json"


def _sample_dict() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_sample_recipe_loads():
    r = load_recipe(SAMPLE)
    assert r.recipe_id == "SYNTH_BEAD_V1"
    assert r.golden_size == (960, 720)
    assert len(r.alignment.anchors) == 2
    assert len(r.rois) == 1
    assert {rule.tool for rule in r.rois[0].rules} == {"blob", "coverage", "color_stats"}


def test_fingerprint_is_stable_and_12hex():
    d = _sample_dict()
    fp1, fp2 = compute_fingerprint(d), compute_fingerprint(copy.deepcopy(d))
    assert fp1 == fp2
    assert len(fp1) == 12 and all(c in "0123456789abcdef" for c in fp1)
    assert parse_recipe(d).fingerprint == fp1


def test_declared_fingerprint_mismatch_rejected():
    d = _sample_dict()
    d["meta"]["fingerprint"] = compute_fingerprint(d)
    parse_recipe(d)  # 일치 → 통과
    d["rois"][0]["rules"][1]["min"] = 0.3  # 본문 변조
    with pytest.raises(RecipeError, match="fingerprint 불일치"):
        parse_recipe(d)


def test_unknown_param_rejected_by_name():
    # L1의 스키마 측: 어느 tool도 소비하지 않는 키는 로드 자체가 거부된다.
    d = _sample_dict()
    d["rois"][0]["rules"][0]["banana_threshold"] = 0.5
    with pytest.raises(RecipeError, match="banana_threshold"):
        parse_recipe(d)


def test_unknown_tool_rejected():
    d = _sample_dict()
    d["rois"][0]["rules"].append({"tool": "telepathy", "min": 0.1})
    with pytest.raises(RecipeError, match="telepathy"):
        parse_recipe(d)


def test_out_of_range_fraction_rejected():
    # L7: 전 필드 0~1 분수 — %/분수 혼용(같은 숫자가 20배 다른 의미) 차단.
    d = _sample_dict()
    d["rois"][0]["rules"][1]["min"] = 20.0  # percent가 아니라 분수여야 함
    with pytest.raises(RecipeError, match=r"0~1 분수|범위"):
        parse_recipe(d)


def test_missing_required_param_rejected():
    d = _sample_dict()
    d["rois"][0]["rules"][1].pop("min")
    with pytest.raises(RecipeError, match="필수 파라미터 'min'"):
        parse_recipe(d)


def test_single_anchor_rejected():
    d = _sample_dict()
    d["alignment"]["anchors"] = d["alignment"]["anchors"][:1]
    with pytest.raises(RecipeError, match="2개 이상"):
        parse_recipe(d)


def test_close_anchors_rejected():
    # 회전 정밀도는 앵커 이격에서 나온다 (§4.4 교차 비판 반영).
    d = _sample_dict()
    a0 = d["alignment"]["anchors"][0]
    a1 = d["alignment"]["anchors"][1]
    a1["origin"] = [a0["origin"][0] + 0.02, a0["origin"][1], 0.0625, 0.08333]
    with pytest.raises(RecipeError, match="이격 부족"):
        parse_recipe(d)


def test_empty_rois_rejected():
    d = _sample_dict()
    d["rois"] = []
    with pytest.raises(RecipeError, match="rois가 비어 있음"):
        parse_recipe(d)


def test_roi_outside_frame_rejected():
    d = _sample_dict()
    d["rois"][0]["rect_golden"] = [0.9, 0.9, 0.3, 0.3]
    with pytest.raises(RecipeError, match="벗어남"):
        parse_recipe(d)


# ── Phase 0 hardening: 미지 키 거부(전 레벨) + pose_gates 범위 (외부 검증 발견) ──

def test_unknown_top_level_key_rejected():
    d = _sample_dict()
    d["extra_block"] = {"anything": 1}
    with pytest.raises(RecipeError, match="알 수 없는 키 'extra_block'"):
        parse_recipe(d)


def test_unknown_roi_key_rejected():
    d = _sample_dict()
    d["rois"][0]["custom_threshold"] = 0.5  # 소비자 없는 파라미터의 전형
    with pytest.raises(RecipeError, match="알 수 없는 키 'custom_threshold'"):
        parse_recipe(d)


def test_unknown_pose_gate_key_rejected():
    d = _sample_dict()
    d["alignment"]["pose_gates"]["max_shift_px"] = 40  # frac을 px로 오타 낸 상황
    with pytest.raises(RecipeError, match="알 수 없는 키 'max_shift_px'"):
        parse_recipe(d)


def test_pose_gate_fractions_range_enforced():
    # 외부 검증에서 실제 통과했던 값 3종 — 전부 거부돼야 한다 (L7)
    for key, bad in (("max_shift_frac", 20), ("anchor_dist_tol_frac", -1), ("scale_tol", 25)):
        d = _sample_dict()
        d["alignment"]["pose_gates"][key] = bad
        with pytest.raises(RecipeError, match=key):
            parse_recipe(d)


def test_underscore_annotation_keys_allowed():
    d = _sample_dict()
    d["_comment"] = "주석은 어느 레벨에서든 허용"
    d["rois"][0]["_why"] = "이 ROI는 상단 도포부"
    parse_recipe(d)  # 예외 없이 통과


def test_bad_morph_rejected():
    d = _sample_dict()
    d["rois"][0]["detect"]["morph"]["size"] = 0
    with pytest.raises(RecipeError, match="morph.size"):
        parse_recipe(d)
