"""Recipe: the single source of truth for one inspection job.

Design Laws enforced here (docs/DESIGN.md §2):
- L1 (schema side): every rule parameter must be declared in PARAM_SPECS for
  its tool - an unknown key fails the load, so a parameter with no consumer
  cannot exist. (Judgment-side sensitivity probes arrive with the tools in
  Phase 2.)
- L5: fingerprints are sha256 prefixes, verified on load when present.
- L7: every ratio field is a 0..1 fraction; ranges are validated here.

All coordinates are in the golden-image frame, normalized 0..1
([x, y, w, h]); at runtime they are mapped onto each inspection image by
the alignment pose. Fixed pixel coordinates are deliberately unsupported.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from avap.constants import (
    FINGERPRINT_LEN,
    MAX_ROTATION_DEG_LIMIT,
    MIN_ANCHOR_SEPARATION_FRAC,
    RECIPE_SCHEMA_VERSION,
)


def anchor_separation_frac(box_a: tuple[float, float, float, float],
                           box_b: tuple[float, float, float, float]) -> float:
    """Distance between two anchor centres, as a fraction of the diagonal.

    Boxes are in normalized [0,1] per-axis coordinates, so the longest
    possible separation is sqrt(2); dividing by it makes the result a plain
    0..1 fraction comparable to MIN_ANCHOR_SEPARATION_FRAC.

    Exported because avap/preflight.py must judge candidate anchors by exactly
    this measure. A tool that green-lights anchors this validator then rejects
    is worse than no tool: on a 3840x2160 frame, two centres 900px apart score
    20.4% by physical diagonal but 16.6% here, so the picker would have said
    OK to anchors the recipe refuses.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ca = (ax + aw / 2, ay + ah / 2)
    cb = (bx + bw / 2, by + bh / 2)
    return math.hypot(cb[0] - ca[0], cb[1] - ca[1]) / math.sqrt(2.0)


class RecipeError(ValueError):
    """Raised when a recipe file fails validation. Message lists every problem."""


# ── Parameter specs ──────────────────────────────────────────────────────
# The single declaration of every parameter a rule may expose:
#   tool -> key -> (kind, lo, hi, required)
# kind: "frac" 0..1 float · "int" · "float" · "hsv" [h,s,v] each 0..1
# UI widgets, value collection and Advisor suggestions must all derive from
# this table (Design Law L2) - never from a second hand-maintained list.
PARAM_SPECS: dict[str, dict[str, tuple[str, float | None, float | None, bool]]] = {
    "blob": {
        "count_min": ("int", 0, 10000, True),
        "count_max": ("int", 0, 10000, True),
        "area_min": ("frac", 0.0, 1.0, False),
        "area_max": ("frac", 0.0, 1.0, False),
        "circularity_min": ("frac", 0.0, 1.0, False),
        "circularity_max": ("frac", 0.0, 1.0, False),
        "solidity_min": ("frac", 0.0, 1.0, False),
        "aspect_ratio_min": ("float", 0.0, 100.0, False),
        "aspect_ratio_max": ("float", 0.0, 100.0, False),
    },
    "coverage": {
        "min": ("frac", 0.0, 1.0, True),
        "max": ("frac", 0.0, 1.0, False),
        "continuity_min": ("frac", 0.0, 1.0, False),
    },
    "color_stats": {
        "expect_hsv_center": ("hsv", None, None, True),
        "max_dist": ("frac", 0.0, 1.0, True),
    },
    "shape_compare": {
        "iou_min": ("frac", 0.0, 1.0, True),
        "excess_max": ("frac", 0.0, 1.0, False),
        "deficit_max": ("frac", 0.0, 1.0, False),
    },
}


# ── Frozen model ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Anchor:
    id: str
    origin: tuple[float, float, float, float]
    search: tuple[float, float, float, float]
    min_score: float


@dataclass(frozen=True)
class Alignment:
    method: str
    anchors: tuple[Anchor, ...]
    max_shift_frac: float
    max_rotation_deg: float
    scale_tol: float


@dataclass(frozen=True)
class Rule:
    tool: str
    params: tuple[tuple[str, Any], ...]  # sorted (key, value) pairs

    def param(self, key: str) -> Any:
        for k, v in self.params:
            if k == key:
                return v
        raise KeyError(key)


@dataclass(frozen=True)
class Roi:
    id: str
    label: str
    rect_golden: tuple[float, float, float, float]
    detect: tuple[tuple[str, Any], ...]  # sorted (key, value) pairs of detect block
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    recipe_version: int
    fingerprint: str
    golden_image_sha: str
    golden_size: tuple[int, int]
    alignment: Alignment
    rois: tuple[Roi, ...]
    raw: str = field(repr=False)  # canonical JSON the fingerprint covers


# ── Validation helpers ───────────────────────────────────────────────────

# Allowed keys per block. Keys starting with "_" are annotations - always
# allowed (documentation convention), never consumed by code. Anything else
# not listed here fails the load: an unknown key is either a typo or a
# parameter nothing consumes, and both must be loud (L1).
#
# L1 metadata exceptions (judgment never reads these; they are provenance,
# not tunable parameters, so sensitivity probes do not apply):
#   - top-level "provenance": free-form JSON object (author, dates, ...)
#   - "alignment.min_score_basis": calibration basis record (golden_n, p5,
#     margin) - documents WHERE min_score came from, does not change it.
# Both must still BE objects; their contents are covered by the fingerprint.
ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "": frozenset({"avap_recipe", "meta", "golden", "alignment", "rois", "provenance"}),
    "meta": frozenset({"recipe_id", "recipe_version", "fingerprint",
                       "created_by", "created_at"}),
    "golden": frozenset({"image_sha", "size"}),
    "alignment": frozenset({"method", "anchors", "pose_gates", "min_score_basis"}),
    "alignment.min_score_basis": frozenset({"golden_n", "p5", "margin"}),
    # 앵커는 정확히 2개가 모두 필수라 별도 "required"는 같은 사실을 중복하는
    # 죽은 파라미터다 (L1).
    "alignment.anchors[]": frozenset({"id", "origin", "search", "min_score"}),
    "alignment.pose_gates": frozenset({"max_shift_frac", "max_rotation_deg", "scale_tol"}),
    "rois[]": frozenset({"id", "label", "rect_golden", "detect", "rules"}),
    "rois[].detect": frozenset({"space", "lower", "upper", "morph"}),
    "rois[].detect.morph": frozenset({"kernel", "size", "open_iter", "close_iter"}),
}


# Distinguishes an absent key from an explicit null: absence falls back to
# defaults, but a written "key": null is an authored value and must be loud -
# silently substituting defaults is how dead configuration is born.
_MISSING = object()


def _dict_of(errors: list[str], where: str, value: Any) -> dict:
    """Coerce a block to a dict; a present non-object (null 포함) is a
    validation error, never an AttributeError crash further down."""
    if value is _MISSING:
        return {}
    if isinstance(value, dict):
        return value
    kind = "null" if value is None else type(value).__name__
    errors.append(f"{where}: JSON 객체여야 함 - {kind} {value!r} "
                  "(생략하려면 키 자체를 빼라)")
    return {}


def _list_of(errors: list[str], where: str, value: Any) -> list:
    if value is _MISSING:
        return []
    if isinstance(value, list):
        return value
    kind = "null" if value is None else type(value).__name__
    errors.append(f"{where}: JSON 배열이어야 함 - {kind} {value!r}")
    return []


def _check_unknown_keys(errors: list[str], where: str, block: Any, allowed_id: str) -> None:
    if not isinstance(block, dict):
        return
    allowed = ALLOWED_KEYS[allowed_id]
    for key in block:
        if key.startswith("_"):
            continue  # annotation
        if key not in allowed:
            errors.append(
                f"{where or 'recipe'}: 알 수 없는 키 '{key}' - 오타이거나 소비자 없는 "
                f"파라미터 (L1). 허용: {', '.join(sorted(allowed))} (주석은 '_' 접두)"
            )


def _check_frac(errors: list[str], where: str, value: Any, hi: float = 1.0) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 < value <= hi):
        errors.append(f"{where}: (0, {hi}] 범위 분수여야 함 (L7) - {value!r}")


def _freeze(d: dict) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((k, _immutable(v)) for k, v in d.items()))


def _immutable(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_immutable(x) for x in v)
    if isinstance(v, dict):
        return _freeze(v)
    return v


def _check_rect(errors: list[str], where: str, rect: Any) -> None:
    if (not isinstance(rect, list)) or len(rect) != 4 or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and math.isfinite(v) for v in rect
    ):
        errors.append(f"{where}: [x, y, w, h] 4개 숫자여야 함 - {rect!r}")
        return
    x, y, w, h = rect
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        errors.append(f"{where}: 좌표는 0~1 분수여야 함 (L7) - {rect!r}")
    elif x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
        errors.append(f"{where}: 사각형이 골든 프레임을 벗어남 - {rect!r}")


def _check_params(errors: list[str], where: str, tool: str, params: dict) -> None:
    spec = PARAM_SPECS.get(tool)
    if spec is None:
        errors.append(
            f"{where}: 알 수 없는 tool '{tool}' (허용: {', '.join(sorted(PARAM_SPECS))})"
        )
        return
    for key, value in params.items():
        if key not in spec:
            errors.append(
                f"{where}: tool '{tool}'이 소비하지 않는 파라미터 '{key}' - "
                f"죽은 파라미터 금지 (L1). 허용 키: {', '.join(sorted(spec))}"
            )
            continue
        kind, lo, hi, _req = spec[key]
        if kind == "hsv":
            if (not isinstance(value, list)) or len(value) != 3 or not all(
                isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in value
            ):
                errors.append(f"{where}.{key}: [h, s, v] 각 0~1 이어야 함 - {value!r}")
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{where}.{key}: 숫자여야 함 - {value!r}")
            continue
        if kind == "int" and int(value) != value:
            errors.append(f"{where}.{key}: 정수여야 함 - {value!r}")
        if lo is not None and hi is not None and not (lo <= value <= hi):
            unit = " (0~1 분수, L7)" if kind == "frac" else ""
            errors.append(f"{where}.{key}: 범위 [{lo}, {hi}] 밖{unit} - {value!r}")
    for key, (_kind, _lo, _hi, required) in spec.items():
        if required and key not in params:
            errors.append(f"{where}: tool '{tool}'의 필수 파라미터 '{key}' 누락")


def compute_fingerprint(data: dict) -> str:
    """sha256 prefix over the canonical recipe body, excluding meta.fingerprint."""
    body = json.loads(json.dumps(data))  # deep copy
    meta = body.get("meta")
    if isinstance(meta, dict):  # 비정상 타입은 검증이 잡는다 - 지문 계산은 크래시 금지
        meta.pop("fingerprint", None)
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:FINGERPRINT_LEN]


# ── Loader ───────────────────────────────────────────────────────────────

def load_recipe(path: str | Path) -> Recipe:
    """Load and validate a recipe file. Raises RecipeError listing all problems."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RecipeError(f"recipe 읽기 실패: {p} - {e}") from e
    return parse_recipe(data)


def parse_recipe(data: dict) -> Recipe:
    if not isinstance(data, dict):
        raise RecipeError(f"recipe 루트는 JSON 객체여야 함 - {type(data).__name__}")
    errors: list[str] = []

    _check_unknown_keys(errors, "", data, "")
    prov = data.get("provenance", _MISSING)
    if prov is not _MISSING and not isinstance(prov, dict):
        kind = "null" if prov is None else type(prov).__name__
        errors.append(f"provenance: JSON 객체여야 함 - {kind} {prov!r}")

    if data.get("avap_recipe") != RECIPE_SCHEMA_VERSION:
        errors.append(
            f"avap_recipe 스키마 버전 불일치: {data.get('avap_recipe')!r} "
            f"(지원: {RECIPE_SCHEMA_VERSION})"
        )

    meta = _dict_of(errors, "meta", data.get("meta", _MISSING))
    _check_unknown_keys(errors, "meta", meta, "meta")
    recipe_id = meta.get("recipe_id")
    if not recipe_id:
        errors.append("meta.recipe_id 누락")
    version = meta.get("recipe_version")
    if not isinstance(version, int) or version < 1:
        errors.append(f"meta.recipe_version은 1 이상의 정수여야 함 - {version!r}")

    golden = _dict_of(errors, "golden", data.get("golden", _MISSING))
    _check_unknown_keys(errors, "golden", golden, "golden")
    golden_sha = golden.get("image_sha", "")
    size = golden.get("size")
    if (not isinstance(size, list)) or len(size) != 2 or not all(
        isinstance(v, int) and v > 0 for v in size
    ):
        errors.append(f"golden.size는 [width, height] 양의 정수여야 함 - {size!r}")
        size = [1, 1]

    # alignment
    al = _dict_of(errors, "alignment", data.get("alignment", _MISSING))
    _check_unknown_keys(errors, "alignment", al, "alignment")
    msb = _dict_of(errors, "alignment.min_score_basis",
                   al.get("min_score_basis", _MISSING))
    _check_unknown_keys(errors, "alignment.min_score_basis", msb, "alignment.min_score_basis")
    gn = msb.get("golden_n")
    if gn is not None and (not isinstance(gn, int) or isinstance(gn, bool) or gn < 0):
        errors.append(f"alignment.min_score_basis.golden_n: 0 이상 정수여야 함 - {gn!r}")
    for k in ("p5", "margin"):
        v = msb.get(k)
        if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool)
                              or not (0.0 <= v <= 1.0)):
            errors.append(f"alignment.min_score_basis.{k}: 0~1 이어야 함 - {v!r}")
    method = al.get("method")
    if method != "template_2anchor":
        errors.append(f"alignment.method 미지원: {method!r} (v1: template_2anchor)")
    anchors_raw = _list_of(errors, "alignment.anchors", al.get("anchors", _MISSING))
    anchors: list[Anchor] = []
    if len(anchors_raw) != 2:
        errors.append(f"alignment.anchors는 정확히 2개여야 함 - {len(anchors_raw)}개")
    anchor_ids: list[str] = []
    for i, a in enumerate(anchors_raw):
        where = f"alignment.anchors[{i}]"
        a = _dict_of(errors, where, a)
        _check_unknown_keys(errors, where, a, "alignment.anchors[]")
        for k in ("id", "origin", "search"):
            if k not in a:
                errors.append(f"{where}: '{k}' 누락")
        _check_rect(errors, f"{where}.origin", a.get("origin", []))
        _check_rect(errors, f"{where}.search", a.get("search", []))
        anchor_id = a.get("id")
        if not isinstance(anchor_id, str) or not anchor_id.strip():
            errors.append(f"{where}.id: 비어 있지 않은 문자열이어야 함 - {anchor_id!r}")
        else:
            anchor_ids.append(anchor_id)
        origin, search = a.get("origin"), a.get("search")
        if (isinstance(origin, list) and len(origin) == 4
                and isinstance(search, list) and len(search) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) for v in origin + search)):
            ox, oy, ow, oh = origin
            sx, sy, sw, sh = search
            if (ox < sx - 1e-9 or oy < sy - 1e-9
                    or ox + ow > sx + sw + 1e-9
                    or oy + oh > sy + sh + 1e-9):
                errors.append(f"{where}.search: origin 전체를 포함해야 함")
        score = a.get("min_score", 0.7)
        if (not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score) or not (0.0 < score <= 1.0)):
            errors.append(f"{where}.min_score: 0~1 이어야 함 - {score!r}")
        if not errors:
            anchors.append(
                Anchor(
                    id=str(a["id"]),
                    origin=tuple(a["origin"]),
                    search=tuple(a["search"]),
                    min_score=float(score),
                )
            )
    if len(anchor_ids) != len(set(anchor_ids)):
        errors.append("alignment.anchors[].id는 서로 달라야 함")
    # anchor separation: rotation precision needs distant anchors
    if len(anchors) >= 2:
        dist = anchor_separation_frac(anchors[0].origin, anchors[1].origin)
        if dist < MIN_ANCHOR_SEPARATION_FRAC:
            errors.append(
                f"앵커 간 이격 부족: 대각선 대비 {dist:.3f} < "
                f"{MIN_ANCHOR_SEPARATION_FRAC} - 회전 정밀도가 무너짐 (DESIGN.md 5)"
            )

    gates = _dict_of(errors, "alignment.pose_gates", al.get("pose_gates", _MISSING))
    _check_unknown_keys(errors, "alignment.pose_gates", gates, "alignment.pose_gates")
    _check_frac(errors, "pose_gates.max_shift_frac", gates.get("max_shift_frac", 0.05))
    _check_frac(errors, "pose_gates.scale_tol", gates.get("scale_tol", 0.02))
    max_rot = gates.get("max_rotation_deg", 3.0)
    if (not isinstance(max_rot, (int, float)) or isinstance(max_rot, bool)
            or not math.isfinite(max_rot)
            or not (0.0 < max_rot <= MAX_ROTATION_DEG_LIMIT)):
        errors.append(
            f"pose_gates.max_rotation_deg: 0~{MAX_ROTATION_DEG_LIMIT} 이어야 함 - {max_rot!r}"
        )

    # rois
    rois_raw = _list_of(errors, "rois", data.get("rois", _MISSING))
    if not rois_raw:
        errors.append("rois가 비어 있음 - 검사 항목 0개 recipe는 저장 불가")
    rois: list[Roi] = []
    for i, r in enumerate(rois_raw):
        where = f"rois[{i}]"
        r = _dict_of(errors, where, r)
        roi_id = r.get("id") or f"roi_{i}"
        _check_unknown_keys(errors, where, r, "rois[]")
        _check_rect(errors, f"{where}.rect_golden", r.get("rect_golden", []))
        detect = _dict_of(errors, f"{where}.detect", r.get("detect", _MISSING))
        _check_unknown_keys(errors, f"{where}.detect", detect, "rois[].detect")
        if detect.get("space", "hsv") != "hsv":
            errors.append(f"{where}.detect.space 미지원: {detect.get('space')!r} (v1: hsv)")
        morph_raw = detect.get("morph", _MISSING)
        morph = _dict_of(errors, f"{where}.detect.morph", morph_raw)
        if morph:
            _check_unknown_keys(errors, f"{where}.detect.morph", morph, "rois[].detect.morph")
            if morph.get("kernel", "ellipse") not in ("ellipse", "rect", "cross"):
                errors.append(f"{where}.detect.morph.kernel 미지원: {morph.get('kernel')!r}")
            ksize = morph.get("size", 5)
            if (not isinstance(ksize, int) or isinstance(ksize, bool)
                    or not (1 <= ksize <= 99) or ksize % 2 == 0):
                errors.append(f"{where}.detect.morph.size: 1~99 홀수 정수여야 함 - {ksize!r}")
            for it in ("open_iter", "close_iter"):
                v = morph.get(it, 1)
                if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= 10):
                    errors.append(f"{where}.detect.morph.{it}: 0~10 정수여야 함 - {v!r}")
        for bound in ("lower", "upper"):
            v = detect.get(bound)
            if (not isinstance(v, list)) or len(v) != 3 or not all(
                isinstance(x, (int, float)) and 0.0 <= x <= 1.0 for x in v
            ):
                errors.append(f"{where}.detect.{bound}: [h, s, v] 각 0~1 이어야 함 (L7) - {v!r}")
        rules_raw = _list_of(errors, f"{where}.rules", r.get("rules", _MISSING))
        if not rules_raw:
            errors.append(f"{where}: rules가 비어 있음")
        rules: list[Rule] = []
        for j, rule in enumerate(rules_raw):
            rwhere = f"{where}.rules[{j}]"
            rule = _dict_of(errors, rwhere, rule)
            tool = rule.get("tool")
            params = {k: v for k, v in rule.items() if k != "tool"}
            if tool is None:
                errors.append(f"{rwhere}: 'tool' 누락")
                continue
            _check_params(errors, rwhere, tool, params)
            rules.append(Rule(tool=str(tool), params=_freeze(params)))
        rois.append(
            Roi(
                id=str(roi_id),
                label=str(r.get("label", roi_id)),
                rect_golden=tuple(r.get("rect_golden", (0, 0, 1, 1))),
                detect=_freeze(detect),
                rules=tuple(rules),
            )
        )

    # fingerprint
    expected = compute_fingerprint(data)
    declared = meta.get("fingerprint")
    if declared is not None and declared != expected:
        errors.append(
            f"fingerprint 불일치: 선언 {declared!r} ≠ 계산 {expected!r} - "
            "recipe 본문이 저장 후 변조됨 (L5)"
        )

    if errors:
        raise RecipeError("recipe 검증 실패:\n- " + "\n- ".join(errors))

    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return Recipe(
        recipe_id=str(recipe_id),
        recipe_version=int(version),
        fingerprint=expected,
        golden_image_sha=str(golden_sha),
        golden_size=(int(size[0]), int(size[1])),
        alignment=Alignment(
            method=str(method),
            anchors=tuple(anchors),
            max_shift_frac=float(gates.get("max_shift_frac", 0.05)),
            max_rotation_deg=float(max_rot),
            scale_tol=float(gates.get("scale_tol", 0.02)),
        ),
        rois=tuple(rois),
        raw=canonical,
    )
