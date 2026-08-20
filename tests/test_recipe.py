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


# ── 2차 외부 검증 반영: 타입 가드 + required 제거 + 메타데이터 예외 타입화 ──

def test_min_score_basis_non_object_rejected_not_crash():
    # 이전에는 "garbage" 문자열이 조용히 통과했다 (미지 키 검사가 non-dict를 스킵)
    d = _sample_dict()
    d["alignment"]["min_score_basis"] = "garbage"
    with pytest.raises(RecipeError, match="min_score_basis.*객체여야 함"):
        parse_recipe(d)


def test_morph_string_gives_recipe_error_not_attribute_error():
    # 이전에는 morph: "ellipse"가 RecipeError 대신 AttributeError로 충돌했다
    d = _sample_dict()
    d["rois"][0]["detect"]["morph"] = "ellipse"
    with pytest.raises(RecipeError, match="morph.*객체여야 함"):
        parse_recipe(d)


def test_non_object_blocks_rejected_everywhere():
    for path_desc, mutate in [
        ("meta", lambda d: d.__setitem__("meta", "x")),
        ("golden", lambda d: d.__setitem__("golden", 3)),
        ("alignment", lambda d: d.__setitem__("alignment", [1])),
        ("pose_gates", lambda d: d["alignment"].__setitem__("pose_gates", "wide")),
        ("provenance", lambda d: d.__setitem__("provenance", "me")),
        ("anchors 원소", lambda d: d["alignment"]["anchors"].__setitem__(0, "a")),
        ("rois 원소", lambda d: d["rois"].__setitem__(0, 7)),
    ]:
        d = _sample_dict()
        mutate(d)
        with pytest.raises(RecipeError, match="객체여야 함|배열이어야 함"), \
                _no_crash(path_desc):
            parse_recipe(d)


import contextlib

@contextlib.contextmanager
def _no_crash(desc):
    try:
        yield
    except RecipeError:
        raise
    except Exception as e:  # AttributeError 등으로 새면 검증기 결함
        raise AssertionError(f"{desc}: RecipeError가 아닌 {type(e).__name__} — {e}")


def test_root_must_be_object():
    with pytest.raises(RecipeError, match="루트"):
        parse_recipe([1, 2])


def test_anchor_required_key_now_rejected_as_dead_param():
    # hardening 1차에서 required를 검증 후 모델에서 버렸다 — 그 자체가 새 죽은
    # 파라미터(L1 위반, 외부 검증 발견). 정렬 엔진이 소비하는 Phase 1까지 금지.
    d = _sample_dict()
    d["alignment"]["anchors"][0]["required"] = True
    with pytest.raises(RecipeError, match="알 수 없는 키 'required'"):
        parse_recipe(d)


def test_min_score_basis_field_values_validated():
    for key, bad in (("golden_n", -1), ("p5", 2.0), ("margin", -0.1)):
        d = _sample_dict()
        d["alignment"]["min_score_basis"] = {"golden_n": 30, "p5": 0.8, "margin": 0.1}
        d["alignment"]["min_score_basis"][key] = bad
        with pytest.raises(RecipeError, match=f"min_score_basis.{key}"):
            parse_recipe(d)
