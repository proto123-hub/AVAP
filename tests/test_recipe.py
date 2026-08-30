"""Recipe loader — the schema side of Design Law L1 (dead parameters cannot load)."""
import copy
import json
import math
from pathlib import Path

import pytest

from avap.recipe import (
    PARAM_SPECS,
    RecipeError,
    _min_max_pairs,
    compute_fingerprint,
    load_recipe,
    parse_recipe,
)

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
    with pytest.raises(RecipeError, match="정확히 2개"):
        parse_recipe(d)


def test_three_anchors_rejected():
    d = _sample_dict()
    d["alignment"]["anchors"].append(copy.deepcopy(d["alignment"]["anchors"][0]))
    d["alignment"]["anchors"][2]["id"] = "third"
    with pytest.raises(RecipeError, match="정확히 2개"):
        parse_recipe(d)


def test_anchor_ids_must_be_unique():
    d = _sample_dict()
    d["alignment"]["anchors"][1]["id"] = d["alignment"]["anchors"][0]["id"]
    with pytest.raises(RecipeError, match="id는 서로 달라야"):
        parse_recipe(d)


@pytest.mark.parametrize("bad_id", ["", "   ", 1, None])
def test_anchor_id_must_be_nonempty_string(bad_id):
    d = _sample_dict()
    d["alignment"]["anchors"][0]["id"] = bad_id
    with pytest.raises(RecipeError, match="비어 있지 않은 문자열"):
        parse_recipe(d)


def test_ids_that_would_collide_after_string_coercion_are_rejected():
    d = _sample_dict()
    d["alignment"]["anchors"][0]["id"] = 1
    d["alignment"]["anchors"][1]["id"] = "1"
    with pytest.raises(RecipeError, match="비어 있지 않은 문자열"):
        parse_recipe(d)


def test_anchor_search_must_contain_origin():
    d = _sample_dict()
    d["alignment"]["anchors"][0]["search"] = [0.0, 0.0, 0.1, 0.1]
    with pytest.raises(RecipeError, match="origin 전체를 포함"):
        parse_recipe(d)


def test_close_anchors_rejected():
    # 회전 정밀도는 앵커 이격에서 나온다 (§4.4 교차 비판 반영).
    d = _sample_dict()
    a0 = d["alignment"]["anchors"][0]
    a1 = d["alignment"]["anchors"][1]
    a1["origin"] = [a0["origin"][0] + 0.02, a0["origin"][1], 0.0625, 0.08333]
    a1["search"] = [a1["origin"][0] - 0.01, a1["origin"][1] - 0.01,
                    a1["origin"][2] + 0.02, a1["origin"][3] + 0.02]
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
    # 외부 검증에서 실제 통과했던 범위 밖 값은 전부 거부돼야 한다 (L7)
    for key, bad in (("max_shift_frac", 20), ("scale_tol", 25)):
        d = _sample_dict()
        d["alignment"]["pose_gates"][key] = bad
        with pytest.raises(RecipeError, match=key):
            parse_recipe(d)


@pytest.mark.parametrize(
    ("where", "key"),
    [
        ("anchor", "min_score"),
        ("gate", "max_rotation_deg"),
        ("origin", None),
    ],
)
def test_bool_is_not_accepted_as_phase1_number(where, key):
    d = _sample_dict()
    if where == "anchor":
        d["alignment"]["anchors"][0][key] = True
    elif where == "gate":
        d["alignment"]["pose_gates"][key] = True
    else:
        d["alignment"]["anchors"][0]["origin"][0] = True
    with pytest.raises(RecipeError):
        parse_recipe(d)


def test_removed_alignment_fields_are_rejected_as_dead_parameters():
    for where, key, value in [
        ("anchor", "patch", "anchors/a.png"),
        ("gate", "anchor_dist_tol_frac", 0.01),
    ]:
        d = _sample_dict()
        block = (d["alignment"]["anchors"][0]
                 if where == "anchor" else d["alignment"]["pose_gates"])
        block[key] = value
        with pytest.raises(RecipeError, match=key):
            parse_recipe(d)


def test_v1_0_recipe_is_rejected_with_explicit_upgrade_deltas():
    d = _sample_dict()
    d["avap_recipe"] = "1.0"
    for index, anchor in enumerate(d["alignment"]["anchors"]):
        anchor["patch"] = f"anchors/a{index + 1}.png"
    d["alignment"]["pose_gates"]["anchor_dist_tol_frac"] = 0.01

    with pytest.raises(RecipeError) as caught:
        parse_recipe(d)

    message = str(caught.value)
    assert "스키마 버전 불일치" in message
    assert "patch" in message
    assert "anchor_dist_tol_frac" in message


def test_underscore_annotation_keys_allowed():
    d = _sample_dict()
    d["_comment"] = "주석은 어느 레벨에서든 허용"
    d["rois"][0]["_why"] = "이 ROI는 상단 도포부"
    parse_recipe(d)  # 예외 없이 통과


def test_bad_morph_rejected():
    for size in (0, 2):
        d = _sample_dict()
        d["rois"][0]["detect"]["morph"]["size"] = size
        with pytest.raises(RecipeError, match="morph.size"):
            parse_recipe(d)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([0.0, 0.0, 102 / 255], [1.0, 1.0, 102 / 255]),
        ([89 / 179, 0.0, 0.0], [89 / 179, 1.0, 1.0]),
        ([0.95, 0.0, 0.0], [0.05, 1.0, 1.0]),
    ],
    ids=["v-grid-aligned-single-bin", "h-grid-aligned-single-bin", "hue-wrap"],
)
def test_nonempty_quantized_hsv_bands_load(lower, upper):
    d = _sample_dict()
    d["rois"][0]["detect"]["lower"] = lower
    d["rois"][0]["detect"]["upper"] = upper

    parse_recipe(d)


@pytest.mark.parametrize(
    ("lower", "upper", "empty_channel"),
    [
        ([0.0, 0.0, 0.5], [1.0, 1.0, 0.5], "V"),
        ([0.5, 0.0, 0.0], [0.5, 1.0, 1.0], "H"),
        ([0.95, 0.5, 0.0], [0.05, 0.5, 1.0], "S"),
    ],
    ids=["v-off-grid-zero-width", "h-off-grid-zero-width", "hue-wrap-with-empty-s"],
)
def test_empty_quantized_hsv_bands_are_rejected(lower, upper, empty_channel):
    d = _sample_dict()
    d["rois"][0]["detect"]["lower"] = lower
    d["rois"][0]["detect"]["upper"] = upper

    with pytest.raises(
        RecipeError,
        match=rf"양자화 후 빈 HSV 밴드 \({empty_channel}\)$",
    ):
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


def test_explicit_null_rejected_everywhere():
    # 키 부재(기본값 OK)와 "key": null(작성된 값)은 다르다 — null이 기본값으로
    # 조용히 대체되면 죽은 설정의 탄생 경로가 된다 (외부 검증 발견 + 재현:
    # pose_gates=null이 max_shift_frac 기본값 0.05로 통과했다).
    for desc, mutate in [
        ("pose_gates", lambda d: d["alignment"].__setitem__("pose_gates", None)),
        ("alignment", lambda d: d.__setitem__("alignment", None)),
        ("min_score_basis", lambda d: d["alignment"].__setitem__("min_score_basis", None)),
        ("morph", lambda d: d["rois"][0]["detect"].__setitem__("morph", None)),
        ("provenance", lambda d: d.__setitem__("provenance", None)),
        ("meta", lambda d: d.__setitem__("meta", None)),
        ("anchors 원소", lambda d: d["alignment"]["anchors"].__setitem__(0, None)),
    ]:
        d = _sample_dict()
        mutate(d)
        with pytest.raises(RecipeError, match="객체여야 함|배열이어야 함"), _no_crash(desc):
            parse_recipe(d)


def test_absent_optional_blocks_still_use_defaults():
    # sentinel 도입이 "생략 시 기본값" 동작을 깨지 않아야 한다
    d = _sample_dict()
    d["alignment"].pop("min_score_basis", None)
    d["alignment"]["pose_gates"].pop("scale_tol", None)
    d["rois"][0]["detect"].pop("morph", None)
    r = parse_recipe(d)
    assert r.alignment.scale_tol == 0.02  # 기본값


def test_min_above_max_rejected():
    # 통과 가능한 측정값이 존재하지 않는 구간은 로드 단계에서 막는다.
    d = _sample_dict()
    d["rois"][0]["rules"][0]["area_min"] = 0.9
    d["rois"][0]["rules"][0]["area_max"] = 0.1
    with pytest.raises(RecipeError, match="area_min"):
        parse_recipe(d)


def test_bare_min_max_pair_is_checked_too():
    # coverage 는 접두사 없는 min/max 를 쓴다 — 접미사 규칙이 이쪽도 잡아야 한다.
    d = _sample_dict()
    d["rois"][0]["rules"][1]["min"] = 0.8
    d["rois"][0]["rules"][1]["max"] = 0.3
    with pytest.raises(RecipeError, match="min"):
        parse_recipe(d)


def test_min_equal_to_max_is_allowed():
    d = _sample_dict()
    d["rois"][0]["rules"][0]["area_min"] = 0.5
    d["rois"][0]["rules"][0]["area_max"] = 0.5
    parse_recipe(d)  # 한 점만 통과하는 구간은 모순이 아니다


def test_min_max_sweep_pairs_exactly_the_bounds_that_have_partners():
    # 짝짓기 결과를 고정한다. 스펙에 새 min/max 쌍이 생기면 여기서 먼저 깨지고,
    # 짝 도출이 틀어져 없던 쌍을 만들어내도 깨진다.
    # max_dist(짝 없는 거리 한계)와 solidity_min/continuity_min/iou_min 이
    # 쌍으로 잡히지 않는다는 것이 이 테스트가 지키는 내용이다.
    assert _min_max_pairs(PARAM_SPECS["blob"]) == (
        ("count_min", "count_max"),
        ("area_min", "area_max"),
        ("circularity_min", "circularity_max"),
        ("aspect_ratio_min", "aspect_ratio_max"),
    )
    assert _min_max_pairs(PARAM_SPECS["coverage"]) == (("min", "max"),)
    assert _min_max_pairs(PARAM_SPECS["color_stats"]) == ()
    assert _min_max_pairs(PARAM_SPECS["shape_compare"]) == ()


def test_min_without_a_max_partner_loads():
    # solidity_min / continuity_min 은 스펙에 짝이 없다.
    d = _sample_dict()
    d["rois"][0]["rules"][0]["solidity_min"] = 0.99
    parse_recipe(d)


def test_aspect_ratio_min_below_one_rejected():
    # DESIGN 6.1: 측정값은 항상 >=1 이므로 0~1 구간은 도달 불가능한 죽은 구간이다.
    d = _sample_dict()
    d["rois"][0]["rules"][0]["aspect_ratio_min"] = 0.5
    with pytest.raises(RecipeError, match="aspect_ratio_min"):
        parse_recipe(d)
    d["rois"][0]["rules"][0]["aspect_ratio_min"] = 1.0
    parse_recipe(d)


@pytest.mark.parametrize("bad", ["x", None, [0.1], {"a": 1}])
def test_wrong_typed_bound_reports_recipe_error_not_typeerror(bad):
    # 짝 비교가 개별 검증에 실패한 raw 값을 그대로 '>' 하면 TypeError 로 탈출한다.
    # 로더의 계약은 어떤 입력에도 RecipeError 다.
    d = _sample_dict()
    d["rois"][0]["rules"][1]["min"] = bad
    d["rois"][0]["rules"][1]["max"] = 0.3
    with pytest.raises(RecipeError):
        parse_recipe(d)


def test_out_of_range_bound_reports_range_error_not_a_pair_error():
    # 범위를 벗어난 값은 짝 비교 대상이 아니다. 5.0 > 0.1 이라 게이트가 없으면
    # 진짜 원인(범위) 위에 짝 오류가 하나 더 얹혀 원인을 흐린다.
    d = _sample_dict()
    d["rois"][0]["rules"][0]["area_min"] = 5.0     # 0~1 밖
    d["rois"][0]["rules"][0]["area_max"] = 0.1
    with pytest.raises(RecipeError) as excinfo:
        parse_recipe(d)

    message = str(excinfo.value)
    assert "범위" in message
    assert "통과할 수 없는 구간" not in message, "범위 실패값이 짝 비교까지 흘러갔다"


@pytest.mark.parametrize(
    "key", ["area_min", "count_min", "count_max", "continuity_min", "aspect_ratio_min"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_bound_rejected(key, bad):
    # json.loads 는 NaN/Infinity 를 그대로 읽는다. int 계열(count_*)에서는
    # int() 가 유한성 검사보다 먼저 돌면 ValueError/OverflowError 로 탈출한다 -
    # frac 계열만 확인하면 그 경로를 통째로 놓친다.
    d = _sample_dict()
    rule = 1 if key == "continuity_min" else 0
    d["rois"][0]["rules"][rule][key] = bad
    with pytest.raises(RecipeError):
        parse_recipe(d)


@pytest.mark.parametrize(
    "key", ["area_min", "count_min", "count_max", "continuity_min", "aspect_ratio_min"]
)
@pytest.mark.parametrize("big", [10**309, -(10**309), 10**400])
def test_oversized_integer_bound_rejected(key, big):
    # json 은 임의 정밀도 정수를 그대로 읽는다. 유한성 검사를 float 로 좁히지
    # 않으면 isfinite() 가 float 변환에서 OverflowError 로 탈출한다 -
    # 파이썬 int 는 언제나 유한하므로 검사 대상이 아니다.
    d = _sample_dict()
    rule = 1 if key == "continuity_min" else 0
    d["rois"][0]["rules"][rule][key] = big
    with pytest.raises(RecipeError):
        parse_recipe(d)


def test_oversized_integer_bound_rejected_through_the_file_path(tmp_path):
    d = _sample_dict()
    path = tmp_path / "oversized.json"
    path.write_text(
        json.dumps(d).replace('"count_min": 1', f'"count_min": {10 ** 309}'),
        encoding="utf-8",
    )
    with pytest.raises(RecipeError):
        load_recipe(path)


def test_non_finite_bound_rejected_through_the_file_path(tmp_path):
    # load_recipe() 경로에서도 같아야 한다 - 실제 레시피 파일로 재현되는 결함이었다.
    d = _sample_dict()
    path = tmp_path / "nonfinite.json"
    path.write_text(
        json.dumps(d).replace('"count_min": 1', '"count_min": NaN'), encoding="utf-8"
    )
    assert math.isnan(json.loads(path.read_text(encoding="utf-8"))
                      ["rois"][0]["rules"][0]["count_min"])

    with pytest.raises(RecipeError):
        load_recipe(path)


def test_non_integer_count_bound_reports_only_the_integer_error():
    # count_min/count_max 는 유일한 int 쌍이다. 정수 검증에 실패한 값이 짝
    # 비교까지 흘러가면 진짜 원인 위에 짝 오류가 얹힌다.
    d = _sample_dict()
    d["rois"][0]["rules"][0]["count_min"] = 1.5
    d["rois"][0]["rules"][0]["count_max"] = 0.5
    with pytest.raises(RecipeError) as excinfo:
        parse_recipe(d)

    message = str(excinfo.value)
    assert "정수" in message
    assert "통과할 수 없는 구간" not in message, "정수 실패값이 짝 비교까지 흘러갔다"
