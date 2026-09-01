"""Re-check the two equivalence claims behind blob measurement.

The bbox crop in measure_blobs() and the hull (rather than whole-contour) input
to the aspect-ratio rect are both chosen for their structure, not for a speed
number: neither changes any measurement, so no test can fail if one is reverted.
This script is that missing evidence - it runs the current code against a source
variant with each choice undone and reports whether the results still agree.

Deterministic: no timings, so two runs print identical bytes.  Output is
ASCII-punctuation only, so a Windows CP949 console can print it.

    python tools/bench_blob.py

Needs only what the package already needs (numpy, opencv-python).  Run it from
anywhere - the repository root is put on sys.path below.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2, numpy as np
from avap.detection import DetectionMask, measure_blobs

SRC = (_ROOT / "avap" / "detection.py").read_text(encoding="utf-8")

def variant(old, new):
    ns = {}
    assert SRC.count(old) == 1
    exec(compile(SRC.replace(old, new), "<variant>", "exec"), ns)
    return ns

def key(bs):
    return [(b.pixels, b.area, b.circularity, b.solidity, b.aspect_ratio) for b in bs]

CORN = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]], np.float32)

def ar_of(src):
    pts = np.concatenate([src.reshape(-1, 2) + o for o in CORN]).astype(np.float32)
    (_, (w, h), _) = cv2.minAreaRect(pts)
    return max(w, h) / min(w, h) if min(w, h) > 0 else float("inf")

# [1] bbox crop does not change any measurement (equivalent mutant)
rng = np.random.default_rng(0)
W, H = 960, 720
fg = np.zeros((H, W), np.uint8)
for _ in range(60):
    x, y = rng.integers(0, W - 40), rng.integers(0, H - 40)
    cv2.rectangle(fg, (int(x), int(y)), (int(x) + 30, int(y) + 20), 255, -1)
roi = np.full((H, W), 255, np.uint8)
mask = DetectionMask(fg, roi)
nocrop = variant("        window = labels[top : top + height, left : left + width]",
                 "        window = labels")
m2 = nocrop["DetectionMask"](fg, roi)
print("[1] bbox crop - %dx%d, %d components" % (W, H, len(measure_blobs(mask))))
print("    measurements bit-exact equal: %s"
      % (key(measure_blobs(mask)) == key(nocrop["measure_blobs"](m2))))

# [2] AR: expanding the hull == expanding the whole contour
by_contour = variant("            [hull.reshape(-1, 2) + offset for offset in _PIXEL_CORNERS]",
                     "            [contour.reshape(-1, 2) + offset for offset in _PIXEL_CORNERS]")
rng = np.random.default_rng(0)
checked = worst = 0
for trial in range(400):
    m = np.zeros((90, 90), np.uint8)
    kind = trial % 4
    if kind == 0:
        box = cv2.boxPoints(((45.0, 45.0), (float(rng.integers(4, 60)), float(rng.integers(2, 40))),
                             float(rng.integers(0, 180))))
        cv2.fillPoly(m, [box.astype(np.int32)], 255)
    elif kind == 1:
        a = np.radians(float(rng.integers(0, 180))); L = int(rng.integers(4, 35))
        cv2.line(m, (45 - int(L * np.cos(a)), 45 - int(L * np.sin(a))),
                    (45 + int(L * np.cos(a)), 45 + int(L * np.sin(a))), 255, int(rng.integers(1, 4)))
    elif kind == 2:
        cv2.ellipse(m, (45, 45), (int(rng.integers(3, 35)), int(rng.integers(2, 25))),
                    float(rng.integers(0, 180)), 0, 360, 255, -1)
    else:
        cv2.circle(m, (45, 45), int(rng.integers(5, 25)), 255, -1)
        m &= (rng.random((90, 90)) > 0.15).astype(np.uint8) * 255
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cs) != 1 or len(cs[0]) < 2:
        continue
    c = cs[0]; checked += 1
    worst = max(worst, abs(ar_of(c) - ar_of(cv2.convexHull(c))))
print("\n[2] AR hull == contour - %d shapes (rot-rect / line / ellipse / speckle)" % checked)
print("    max absolute difference: %.3e" % worst)

saw = np.zeros((H, W), np.uint8)
saw[100:620, 100:860] = 255
for x in range(100, 860, 2):
    saw[100:110, x:x + 1] = 0; saw[610:620, x + 1:x + 2] = 0
for y in range(100, 620, 2):
    saw[y:y + 1, 100:110] = 0; saw[y + 1:y + 2, 850:860] = 0
saw_mask = DetectionMask(saw, np.full((H, W), 255, np.uint8))
sm2 = by_contour["DetectionMask"](saw, np.full((H, W), 255, np.uint8))
contour = max(cv2.findContours(saw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
hull = cv2.convexHull(contour)
print("    sawtooth component (production path CHAIN_APPROX_SIMPLE):")
print("      contour %d pts -> %d corners | hull %d pts -> %d corners"
      % (len(contour), len(contour) * 4, len(hull), len(hull) * 4))
print("    AR equal: %s" % (measure_blobs(saw_mask)[0].aspect_ratio
                            == by_contour["measure_blobs"](sm2)[0].aspect_ratio))

# [3] AR spread across rotations
corner_ar, plus1_ar = [], []
for ang in range(0, 90, 10):
    m = np.zeros((240, 240), np.uint8)
    cv2.fillPoly(m, [cv2.boxPoints(((120.0, 120.0), (100.0, 25.0), float(ang))).astype(np.int32)], 255)
    c = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    (_, (w, h), _) = cv2.minAreaRect(c); plus1_ar.append((max(w, h) + 1) / (min(w, h) + 1))
    corner_ar.append(ar_of(cv2.convexHull(c)))
print("\n[3] AR spread - 100x25 rect @ 240x240, 0..80 deg step 10 (true value 4.0)")
print("    pixel-centre (+1): %.4f    pixel-corner: %.4f"
      % (max(plus1_ar) - min(plus1_ar), max(corner_ar) - min(corner_ar)))

# [4] orientation invariance
h = np.zeros((40, 40), np.uint8); h[10, 5:15] = 255
dg = np.zeros((40, 40), np.uint8)
for i in range(10):
    dg[5 + i, 5 + i] = 255
print("\n[4] orientation invariance - 10px horizontal / 10px diagonal")
for name, img in [("horizontal", h), ("diagonal  ", dg)]:
    mk = DetectionMask(img, np.full((40, 40), 255, np.uint8))
    print("    %s AR = %.4f" % (name, measure_blobs(mk)[0].aspect_ratio))
