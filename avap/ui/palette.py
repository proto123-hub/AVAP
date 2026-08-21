"""Single source for every colour and stroke width the app draws.

Widget styling (Qt hex) and image overlays (OpenCV BGR) are both derived from
this table, so a colour can never mean one thing in the panel and another on
the image. Nothing else in the codebase may hold a colour literal.

Neutral by construction: every background is R=G=B, so a hue on screen always
carries meaning (verdict or interaction) and never decoration.

Contrast is not documented here, it is enforced. tests/test_palette.py walks
the product of every foreground and the backgrounds it declares in `on`, and
fails the build below target. A new colour without an `on` list is rejected
too, so the check cannot be dodged by leaving a pair undeclared.

Scope of that proof: it shows the specified sRGB codes meet the ratios under
standard observer conditions. Font weight, antialiasing, shop-floor ambient
light, panel viewing angle and projector gamma are outside the maths and stay
a real-image check.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Targets ──────────────────────────────────────────────────────────────
# WCAG 2.1 AA. Body text 1.4.3 (4.5), UI components and graphics 1.4.11 (3.0).
BODY_MIN = 4.5
UI_MIN = 3.0
# Disabled controls are formally exempt from 1.4.3. We decline the exemption:
# a greyed-out Inspect button has to explain why it cannot be pressed.
DISABLED_MIN = 3.0
# Upper bound. Near-white on near-black haloes on emissive panels and tires
# the eye over a shift; this stops a later "improvement" to #FFFFFF on #000000.
BODY_MAX = 17.0


@dataclass(frozen=True)
class Colour:
    hex: str
    role: str
    on: tuple[str, ...] = ()      # background tokens this must stay legible on
    target: float = BODY_MIN      # ratio every pair in `on` must clear


# ── Backgrounds — pure neutral, R=G=B ────────────────────────────────────
# Steps are 1.10 / 1.13 / 1.20 apart: surfaces alone cannot separate panels,
# which is why BORDER_STRONG is a functional element, not decoration.
_BACKGROUNDS = {
    "BG_BASE":    Colour("#121212", "App canvas, image viewer letterbox"),
    "BG_PANEL":   Colour("#1C1C1C", "The three panel faces"),
    "BG_SURFACE": Colour("#262626", "Inputs, table rows, cards, toolbar, status bar, tooltip"),
    "BG_HOVER":   Colour("#333333", "Hover / selected / pressed. Worst case for every foreground"),
}

_ALL_BG = tuple(_BACKGROUNDS)

# ── Foregrounds ──────────────────────────────────────────────────────────
_FOREGROUNDS = {
    "TEXT_PRIMARY": Colour(
        "#F0F0F0", "Body text, verdict wording, measured values. Not pure white", _ALL_BG),
    "TEXT_SECONDARY": Colour(
        "#B4B4B4", "Units, threshold labels, column headers, hints", _ALL_BG),
    "TEXT_DISABLED": Colour(
        "#8A8A8A", "Disabled control labels", _ALL_BG, DISABLED_MIN),
    "BORDER_STRONG": Colour(
        "#808080", "Input outline, panel edge, scrollbar handle, ROI edit handle", _ALL_BG, UI_MIN),
    "ACCENT": Colour(
        "#7ABAFF",
        "Focus ring, slider fill, selection, text-selection background, default "
        "button, ROI box, mask tint. Blue survives red-green CVD, so interaction "
        "keeps the one axis dichromats retain",
        _ALL_BG),
    "INK_ON_BRIGHT": Colour(
        "#0A0A0A", "The only ink for bright fills", ("ACCENT", "FAIL_PLATE")),
}

# ── Verdict ──────────────────────────────────────────────────────────────
# Three channels carry a verdict, in this order: the word (PASS/FAIL/UNKNOWN),
# the glyph, then colour. Colour alone never decides — see VERDICT_GLYPH.
#
# Card architecture is what makes the verdict survive colour blindness:
# PASS and UNKNOWN are neutral surfaces with coloured text (on screen most of
# the shift, so kept dim), FAIL is a bright plate with dark ink (brief, loud).
# That luminance polarity separates the cards by 3.54:1 even under protanopia,
# where hue alone manages 2.03:1. Darkening FAIL_PLATE back toward #FF4438
# drops protan to 2.97 and breaks it.
_VERDICT = {
    "PASS": Colour(
        "#3DDC84", "PASS wording, left bar, evidence row, aligned anchor", _ALL_BG),
    "FAIL": Colour(
        "#FF7A6B", "FAIL wording on dark surfaces, left bar, evidence row", _ALL_BG),
    "FAIL_PLATE": Colour(
        "#FF5C2E", "FAIL verdict card fill only. Its text is INK_ON_BRIGHT",
        ("BG_PANEL",), UI_MIN),
    "UNKNOWN": Colour(
        "#C8C8C8",
        "UNKNOWN wording, dashed border, failed anchor. Neutral on purpose: "
        "'no information', not 'warning'. Red stays reserved for product FAIL",
        _ALL_BG),
}

# ── Image overlay casing ─────────────────────────────────────────────────
# An overlay colour cannot be legible on an arbitrary image on its own: over
# mid grey every verdict colour falls under 3:1. So every stroke is drawn as
# a black/white casing pair, whose worst case over all 256 greys is 4.61 —
# independent of what the image contains.
_OVERLAY = {
    "OUTLINE_DARK":  Colour("#000000", "Overlay outermost casing (w+4)"),
    "OUTLINE_LIGHT": Colour("#FFFFFF", "Overlay middle casing (w+2)"),
}

PALETTE: dict[str, Colour] = {**_BACKGROUNDS, **_FOREGROUNDS, **_VERDICT, **_OVERLAY}

# ── Non-colour constants that carry the same meaning ──────────────────────
# Shape is what separates verdicts when colour is gone, so the widths live
# beside the colours and are asserted by the same tests. Colour guarded by CI
# while thickness drifts free would void the accessibility argument silently.
OVERLAY_STROKE_W = 2        # the coloured stroke itself
OVERLAY_LIGHT_W = 4         # OUTLINE_LIGHT, = stroke + 2
OVERLAY_DARK_W = 6          # OUTLINE_DARK,  = stroke + 4
FOCUS_RING_W = 2
VERDICT_BAR_W = 8           # card left bar
DASH_PATTERN = (6, 4)       # UNKNOWN / failed anchor, so it reads without colour

MASK_TINT_ALPHA = 0.35      # detection mask fill. Never the only cue —
                            # an ACCENT outline must accompany it
SEPARATOR_ALPHA = 0.45      # decorative rules, derived from BORDER_STRONG

VERDICT_GLYPH = {"PASS": "✓", "FAIL": "✕", "UNKNOWN": "?"}

# AA is the gate; AAA (7:1) is reported, not enforced. Forcing every pair to
# 7:1 would push FAIL toward pink (a red light enough to clear 7:1 on #333333
# stops reading as a red) and cost the domain convention that red means NG.
# An exemption list was tried instead and reached six entries, at which point
# it documents the palette rather than constraining it.
AAA_GOAL = 7.0


# ── Colour maths ─────────────────────────────────────────────────────────

def rgb(token_or_hex: str) -> tuple[int, int, int]:
    """(R, G, B) 0-255 for a token name or a #RRGGBB literal."""
    h = PALETTE[token_or_hex].hex if token_or_hex in PALETTE else token_or_hex
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def bgr(token_or_hex: str) -> tuple[int, int, int]:
    """(B, G, R) for OpenCV. The only channel swap in the codebase."""
    r, g, b = rgb(token_or_hex)
    return b, g, r


def qss(token: str) -> str:
    """#RRGGBB for Qt stylesheets."""
    return PALETTE[token].hex


def _linear(channel: int) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(token_or_hex: str) -> float:
    """WCAG 2.1 relative luminance."""
    r, g, b = rgb(token_or_hex)
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def contrast(a: str, b: str) -> float:
    """WCAG 2.1 contrast ratio. Raw float — round only when reporting."""
    la, lb = luminance(a), luminance(b)
    if la < lb:
        la, lb = lb, la
    return (la + 0.05) / (lb + 0.05)


def composite(fg: str, bg: str, alpha: float) -> str:
    """Flatten fg over bg at alpha, as #RRGGBB.

    Tints are derived, never hand-picked: a hand-picked tint is a colour no
    test recalculates, which is how the predecessor shipped a 4.45:1 fill.
    """
    fr, fg_, fb = rgb(fg)
    br, bg_, bb = rgb(bg)
    mix = lambda f, b: round(f * alpha + b * (1 - alpha))
    return f"#{mix(fr, br):02X}{mix(fg_, bg_):02X}{mix(fb, bb):02X}"
