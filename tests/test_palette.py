"""Contrast is enforced here, not documented in a design file.

The suite walks the palette itself rather than a hand-listed set of pairs:
a hand-listed set silently omits whatever colour is added next.
"""
import re
from pathlib import Path

import pytest

from avap.ui import palette as P
from avap.ui.palette import (
    AAA_GOAL, BODY_MAX, BODY_MIN, PALETTE, UI_MIN, VERDICT_GLYPH,
    bgr, composite, contrast, rgb,
)

REPO = Path(__file__).resolve().parents[1]


# ── The maths itself, before it is trusted to judge anything ──────────────

def test_contrast_matches_known_values():
    assert contrast("#000000", "#FFFFFF") == pytest.approx(21.0, abs=0.01)
    assert contrast("#777777", "#777777") == pytest.approx(1.0, abs=1e-9)
    # WCAG's own worked example: the darkest grey passing 4.5:1 on white.
    assert contrast("#767676", "#FFFFFF") == pytest.approx(4.54, abs=0.01)
    # Symmetric regardless of argument order.
    assert contrast("#3DDC84", "#333333") == pytest.approx(contrast("#333333", "#3DDC84"))


def test_linearisation_uses_the_piecewise_threshold():
    # Below 0.04045 the curve is linear; a plain 2.2 gamma approximation is
    # wrong here and drifts once composite() feeds it near-black values.
    assert P._linear(10) == pytest.approx(10 / 255 / 12.92, rel=1e-12)
    assert P._linear(200) == pytest.approx(((200 / 255 + 0.055) / 1.055) ** 2.4, rel=1e-12)


# ── The palette, walked exhaustively ─────────────────────────────────────

def _declared_pairs():
    for name, colour in PALETTE.items():
        for bg in colour.on:
            yield name, bg, colour.target


def test_every_declared_pair_meets_its_target():
    failures = []
    for fg, bg, target in _declared_pairs():
        r = contrast(fg, bg)
        if r < target:
            failures.append(f"{fg} on {bg}: {r:.2f} < {target}")
    assert not failures, "명암비 미달:\n  " + "\n  ".join(failures)


def test_body_text_stays_under_the_upper_bound():
    # Near-white on near-black haloes on emissive panels over a long shift.
    for fg in ("TEXT_PRIMARY", "TEXT_SECONDARY"):
        for bg in ("BG_BASE", "BG_PANEL", "BG_SURFACE", "BG_HOVER"):
            assert contrast(fg, bg) <= BODY_MAX, f"{fg} on {bg} 과대비"


def test_foreground_without_declared_backgrounds_is_rejected():
    # An undeclared pair is an unchecked pair, so every non-background colour
    # must say where it is allowed to sit. Overlay casings are the exception:
    # they sit on arbitrary image pixels and are covered by their own test.
    casings = {"OUTLINE_DARK", "OUTLINE_LIGHT"}
    backgrounds = {n for n in PALETTE if n.startswith("BG_")}
    for name, colour in PALETTE.items():
        if name in backgrounds or name in casings:
            continue
        assert colour.on, f"{name}: on 미선언 — 검증되지 않는 색은 존재할 수 없다"


def test_worst_case_body_text_clears_aa_with_margin():
    # Shop-floor ambient light and projector gamma both eat contrast, so the
    # colours that carry meaning are checked against the brightest surface
    # they may sit on rather than the average one.
    for fg in ("TEXT_PRIMARY", "TEXT_SECONDARY", "ACCENT", "PASS", "FAIL", "UNKNOWN"):
        assert contrast(fg, "BG_HOVER") >= BODY_MIN, f"{fg}: 최악 배경에서 AA 미달"


def test_aaa_shortfalls_are_bounded_and_only_where_expected():
    # AAA is reported, not gated (see AAA_GOAL). What is asserted is that the
    # shortfall stays small and confined to the reds — if a neutral or the
    # accent starts missing 7:1, the ramp has drifted and that is a defect.
    for fg, bg, target in _declared_pairs():
        if target < BODY_MIN:
            continue
        r = contrast(fg, bg)
        if r >= AAA_GOAL:
            continue
        assert r >= 4.5, f"{fg} on {bg}: {r:.2f} — AAA 미달은 허용해도 AA는 아니다"
        assert fg in {"FAIL", "TEXT_SECONDARY", "ACCENT", "INK_ON_BRIGHT"}, (
            f"{fg} on {bg}: {r:.2f} — 예상 밖 토큰이 AAA에 미달, 램프가 흘렀는지 확인"
        )


# ── The user's stated requirement, as an assertion ───────────────────────

def test_backgrounds_are_pure_neutral():
    # "검은색 회색 배경" — a hue on screen must mean something.
    for name, colour in PALETTE.items():
        if name.startswith("BG_"):
            r, g, b = rgb(name)
            assert r == g == b, f"{name} {colour.hex}: 배경은 무채색이어야 함 (R=G=B)"


def test_backgrounds_are_dark_but_not_pure_black():
    for name in (n for n in PALETTE if n.startswith("BG_")):
        r, _, _ = rgb(name)
        assert 0x10 <= r <= 0x40, f"{name}: 순검정 회피 + 다크 유지"


# ── Verdict legibility without colour ────────────────────────────────────

def test_verdict_carries_three_channels():
    assert set(VERDICT_GLYPH) == {"PASS", "FAIL", "UNKNOWN"}
    assert len(set(VERDICT_GLYPH.values())) == 3, "글리프가 겹치면 색이 유일 채널이 된다"


def test_verdict_cards_separate_by_luminance_not_hue():
    # Hue between PASS and FAIL text collapses under red-green CVD, so the
    # cards are separated by luminance polarity instead: PASS/UNKNOWN are
    # neutral surfaces, FAIL is a bright plate. That difference survives
    # dichromacy, greyscale printing and projector desaturation alike.
    assert contrast("FAIL_PLATE", "BG_SURFACE") >= UI_MIN


def test_unknown_is_neutral_so_it_never_reads_as_a_defect():
    r, g, b = rgb("UNKNOWN")
    assert r == g == b, "UNKNOWN에 색조를 주면 FAIL과 같은 경보로 읽힌다"


def test_red_is_reserved_for_product_fail():
    # Tool states (failed alignment, disabled controls) must not borrow red:
    # the verdict would read as a defect when it is 'no information'.
    reds = {"FAIL", "FAIL_PLATE"}
    for name, colour in PALETTE.items():
        if name in reds:
            continue
        r, g, b = rgb(name)
        assert not (r > 160 and r - max(g, b) > 60), (
            f"{name} {colour.hex}: 적색 계열은 제품 FAIL 전용"
        )


# ── Image overlays: legible on any pixel, not just on our own surfaces ────

def test_overlay_casing_is_legible_over_every_possible_grey():
    # The guarantee is per-pixel: for any background, at least one of the two
    # casings must stand out. Measuring the stroke against its own halo would
    # prove nothing about the image underneath.
    worst = min(
        max(contrast("OUTLINE_DARK", f"#{v:02X}{v:02X}{v:02X}"),
            contrast("OUTLINE_LIGHT", f"#{v:02X}{v:02X}{v:02X}"))
        for v in range(256)
    )
    assert worst >= UI_MIN, f"임의 픽셀 위 오버레이 보장 실패: 최악 {worst:.2f}"


def test_overlay_widths_keep_the_casing_visible():
    assert P.OVERLAY_LIGHT_W == P.OVERLAY_STROKE_W + 2
    assert P.OVERLAY_DARK_W == P.OVERLAY_STROKE_W + 4
    assert P.OVERLAY_STROKE_W >= 2, "1px 획은 축소 표시에서 사라진다"


# ── Derivation is single-source ──────────────────────────────────────────

def test_bgr_is_the_hex_with_channels_swapped():
    for name in PALETTE:
        r, g, b = rgb(name)
        assert bgr(name) == (b, g, r)


def test_composited_tint_lands_between_its_endpoints():
    tint = composite("ACCENT", "BG_SURFACE", P.MASK_TINT_ALPHA)
    assert re.fullmatch(r"#[0-9A-F]{6}", tint)
    low, high = sorted((P.luminance("BG_SURFACE"), P.luminance("ACCENT")))
    assert low <= P.luminance(tint) <= high


def test_no_colour_literals_outside_the_palette():
    # Includes OpenCV's 3-tuple form, which a hex-only grep would miss.
    hex_lit = re.compile(r"#[0-9a-fA-F]{6}\b")
    bgr_lit = re.compile(r"\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)")
    # synth.py is exempt: it paints the *content* of a fake inspection image
    # (board, material, background), whose colours are chosen against the HSV
    # detection thresholds, not against the UI. The palette governs what the
    # app draws over an image, never what an image contains.
    exempt = {"palette.py", "synth.py"}
    offenders = []
    for path in (REPO / "avap").rglob("*.py"):
        if path.name in exempt:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, label in ((hex_lit, "hex"), (bgr_lit, "bgr")):
            for m in pattern.finditer(text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(REPO)}:{line} {label} {m.group(0)}")
    assert not offenders, "팔레트 밖 색 리터럴:\n  " + "\n  ".join(offenders)
