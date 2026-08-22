"""Contrast is enforced here, not documented in a design file.

The suite walks the palette itself rather than a hand-listed set of pairs:
a hand-listed set silently omits whatever colour is added next.
"""
import ast
import colorsys
import re
from pathlib import Path

import pytest

from avap.ui import palette as P
from avap.ui.palette import (
    AAA_GOAL, BODY_MAX, BODY_MIN, CVD_MATRICES, CVD_MIN, PALETTE, UI_MIN,
    VERDICT_GLYPH, bgr, composite, contrast, cvd_contrast, rgb, simulate_cvd,
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


def test_no_colour_can_declare_a_target_below_the_ui_minimum():
    # Every other check trusts `target`. A future colour declaring target=0.0
    # would satisfy the exhaustive walk, the `on` check and the AAA check all
    # at once, opting out of the gate without omitting a single declaration.
    backgrounds = {n for n in PALETTE if n.startswith("BG_")}
    for name, colour in PALETTE.items():
        if name in backgrounds:
            continue
        assert colour.target >= UI_MIN, (
            f"{name}: target {colour.target} < {UI_MIN} - 검사를 스스로 무력화하는 선언"
        )


def test_disabled_text_is_held_to_the_body_target_it_claims():
    # The module declines the WCAG disabled exemption in words; this is the
    # number that makes the words true.
    from avap.ui.palette import DISABLED_MIN
    assert DISABLED_MIN == BODY_MIN
    assert contrast("TEXT_DISABLED", "BG_HOVER") >= BODY_MIN


def test_disabled_text_still_reads_as_dimmer_than_secondary():
    # Raising TEXT_DISABLED to clear the body target must not make it a second
    # TEXT_SECONDARY - disabled has to look disabled.
    assert P.luminance("TEXT_DISABLED") < P.luminance("TEXT_SECONDARY")
    assert contrast("TEXT_SECONDARY", "BG_HOVER") > contrast("TEXT_DISABLED", "BG_HOVER")


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
        # TEXT_DISABLED is here by construction: it is held to the body target
        # (see DISABLED_MIN) yet must stay visibly dimmer than TEXT_PRIMARY, so
        # it lands between AA and AAA on the brighter surfaces.
        assert fg in {"FAIL", "TEXT_SECONDARY", "TEXT_DISABLED", "ACCENT",
                      "INK_ON_BRIGHT"}, (
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


def test_cvd_simulation_behaves_like_a_dichromat_projection():
    # Same rule as the contrast maths above: the transform is checked before it
    # is trusted to judge the palette. A wrong matrix would silently "prove"
    # whatever the palette happened to be.
    for kind in CVD_MATRICES:
        for c in ("#FF0000", "#00FF00", "#3DDC84", "FAIL_PLATE"):
            once = simulate_cvd(c, kind)
            assert simulate_cvd(once, kind) == once, f"{kind}: 투영이 아니다"
        # Greys carry no chroma to lose.
        for grey in ("BG_SURFACE", "BG_HOVER", "UNKNOWN"):
            assert simulate_cvd(grey, kind) == PALETTE[grey].hex.upper()
        # The blue-yellow axis is what dichromats keep.
        assert simulate_cvd("#0000FF", kind) == "#0000FF"
        # Red and green must land on one confusion line: hue is gone.
        r, g = simulate_cvd("#FF0000", kind), simulate_cvd("#00FF00", kind)
        assert rgb(r)[0] == rgb(r)[1] and rgb(g)[0] == rgb(g)[1]


def test_verdict_cards_separate_by_luminance_under_dichromacy():
    # The claim this file exists to enforce. PASS/UNKNOWN are neutral surfaces,
    # FAIL is a bright plate; that polarity has to survive red-green CVD, where
    # hue does not. Plain sRGB contrast cannot show this - see the next test.
    for kind in CVD_MATRICES:
        r = cvd_contrast("FAIL_PLATE", "BG_SURFACE", kind)
        assert r >= CVD_MIN, f"{kind}: 판정 카드 분리 {r:.2f} < {CVD_MIN}"


def test_the_documented_failing_red_is_actually_rejected():
    # palette.py states that darkening FAIL_PLATE toward #FF4438 breaks protan
    # separation. That sentence was previously unenforced: #FF4438 passes plain
    # sRGB contrast (4.42) just as the shipped colour does, so the old
    # assertion could not tell them apart. Now it can.
    assert contrast("#FF4438", "BG_SURFACE") >= UI_MIN          # indistinguishable here
    assert cvd_contrast("#FF4438", "BG_SURFACE", "protanopia") < CVD_MIN   # and here it is not
    assert cvd_contrast("FAIL_PLATE", "BG_SURFACE", "protanopia") >= CVD_MIN


def test_ink_on_the_bright_plate_survives_dichromacy_too():
    for kind in CVD_MATRICES:
        assert cvd_contrast("INK_ON_BRIGHT", "FAIL_PLATE", kind) >= BODY_MIN


def test_hue_alone_would_not_have_carried_the_verdict():
    # Why the card architecture exists at all: PASS text against FAIL text is
    # never a reliable separation, in any vision.
    assert contrast("PASS", "FAIL") < CVD_MIN
    for kind in CVD_MATRICES:
        assert cvd_contrast("PASS", "FAIL", kind) < CVD_MIN


def test_unknown_is_neutral_so_it_never_reads_as_a_defect():
    r, g, b = rgb("UNKNOWN")
    assert r == g == b, "UNKNOWN에 색조를 주면 FAIL과 같은 경보로 읽힌다"


def _alarm_hue(token_or_hex: str) -> bool:
    """Does this colour read as the red-to-magenta alarm family?

    Channel arithmetic is not enough: `r - max(g, b)` is zero for a saturated
    pink like #FF9FFF (b is also 255), so a magenta tool state would have
    slipped past while still reading as an alarm. Hue and chroma decide.
    """
    r, g, b = (v / 255 for v in rgb(token_or_hex))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    deg = h * 360
    # 290..360..30 spans magenta through red to orange-red. The lower bound is
    # measured, not guessed: saturated pink sits at hue 300, so a 320 cut-off
    # would have let exactly the case this predicate was widened for through.
    # Below 290 reads as violet, which is not an alarm.
    in_band = deg >= 290 or deg <= 30
    return in_band and s >= 0.35 and v >= 0.45


def test_red_is_reserved_for_product_fail():
    # Tool states (failed alignment, disabled controls) must not borrow red:
    # the verdict would read as a defect when it is 'no information'.
    reds = {"FAIL", "FAIL_PLATE"}
    for name, colour in PALETTE.items():
        if name in reds:
            continue
        assert not _alarm_hue(name), f"{name} {colour.hex}: 경보 색상은 제품 FAIL 전용"


def test_the_alarm_predicate_covers_magenta_not_just_red():
    # The predicate this reservation rests on, checked on its own before it is
    # trusted. The first two are the cases the previous channel test missed.
    for pink in ("#FF9FFF", "#FF44FF", "#E060C0"):
        assert _alarm_hue(pink), f"{pink}: 자홍 계열이 경보로 분류되지 않는다"
    for red in ("#FF5C2E", "#FF7A6B", "#FF4438"):
        assert _alarm_hue(red)
    # Not alarms: neutrals, the blue accent, the green PASS.
    for safe in ("#C8C8C8", "#7ABAFF", "#3DDC84", "#333333", "#9C9C9C"):
        assert not _alarm_hue(safe), f"{safe}: 경보가 아닌 색이 경보로 분류됨"


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


# OpenCV drawing calls whose `color` argument is a raw BGR triple.
_CV2_DRAWING = {
    "line", "rectangle", "circle", "putText", "polylines", "drawContours",
    "fillPoly", "fillConvexPoly", "arrowedLine", "ellipse", "drawMarker",
}
_COLOUR_NAME = re.compile(r"colou?r|bgr|rgb", re.IGNORECASE)
HEX_LITERAL = re.compile(r"#[0-9a-fA-F]{6}\b")


def _is_int_triple(node) -> bool:
    return (isinstance(node, ast.Tuple) and len(node.elts) == 3
            and all(isinstance(e, ast.Constant) and isinstance(e.value, int)
                    for e in node.elts))


def find_colour_literals(source: str) -> list[tuple[int, str]]:
    """Colour literals in one module: hex strings, and BGR triples in colour use.

    A bare three-integer tuple is NOT a colour. An image shape, a version, a
    3-D point are all `(int, int, int)`, and a scan that rejects every one of
    them would fail CI on code that never draws anything. So a triple counts
    only where it is used as a colour: the `color=` keyword, an argument to a
    cv2 drawing call, or a binding whose name says colour.
    """
    found = [(source[: m.start()].count("\n") + 1, f"hex {m.group(0)}")
             for m in HEX_LITERAL.finditer(source)]
    tree = ast.parse(source)
    for node in ast.walk(tree):
        suspects = []
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in _CV2_DRAWING:
                suspects += node.args
            suspects += [k.value for k in node.keywords
                         if k.arg and _COLOUR_NAME.search(k.arg)]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(_COLOUR_NAME.search(n) for n in names):
                suspects.append(node.value)
        for s in suspects:
            if _is_int_triple(s):
                found.append((s.lineno, "bgr "
                              + repr(tuple(e.value for e in s.elts))))
    return sorted(found)


def test_the_literal_scanner_tells_colours_from_ordinary_tuples():
    # The scanner decides what CI rejects, so it is checked on its own first.
    assert find_colour_literals("cv2.line(im, a, b, (0, 0, 255), 2)")
    assert find_colour_literals("draw(im, color=(0, 0, 255))")
    assert find_colour_literals("draw(im, line_colour=(0, 0, 255))")
    assert find_colour_literals("BOX_COLOR = (0, 0, 255)")
    assert find_colour_literals("s = '#FF0000'")
    # These are not colours and must not fail a build.
    assert not find_colour_literals("shape = (1200, 1600, 3)")
    assert not find_colour_literals("VERSION = (1, 5, 0)")
    assert not find_colour_literals("return (x0, y0, 3)")
    assert not find_colour_literals("cv2.resize(im, (640, 480, 3))")


def test_no_colour_literals_outside_the_palette():
    # synth.py is exempt: it paints the *content* of a fake inspection image
    # (board, material, background), whose colours are chosen against the HSV
    # detection thresholds, not against the UI. The palette governs what the
    # app draws over an image, never what an image contains.
    exempt = {"palette.py", "synth.py"}
    offenders = []
    for path in (REPO / "avap").rglob("*.py"):
        if path.name in exempt:
            continue
        for line, what in find_colour_literals(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(REPO)}:{line} {what}")
    assert not offenders, "팔레트 밖 색 리터럴:\n  " + "\n  ".join(offenders)
