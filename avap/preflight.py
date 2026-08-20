"""Phase 0.5 preflight surveys — run on the PC that holds the real line images.

Two questions must be answered with data BEFORE the alignment engine is
tuned (docs/AVAP_DESIGN_REVIEW.md §7 Phase 0.5 — the cross-review found the
original plans assumed these numbers instead of measuring them):

1. offset — how far do products actually move between shots?
   → sets the anchor search-window size and the pose gates.
2. anchor — is a candidate landmark stable across the whole set?
   → NCC score distribution decides min_score (percentile basis, same
     philosophy as the VSGP advisor: p5 minus a margin).

Both are surveys, not the alignment engine: rough numbers with an honest
method note beat precise-looking numbers from an unvalidated pipeline.

Usage (run inside the folder that holds one Position's images):
    python -m avap.preflight offset --ref golden.png --images ./pos6_ok --out offset.csv
    python -m avap.preflight anchor --ref golden.png --box 700,495,80,90 --images ./pos6_ok
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from avap.io_utils import SUPPORTED_EXTS, imread_u

# Survey resolution: full frames are downscaled so FFTs stay fast; survey
# precision (~1px at survey scale) is plenty for sizing search windows.
SURVEY_MAX_W = 640
ANGLE_SWEEP_DEG = 4.0
ANGLE_STEP_DEG = 0.25
# Below this phase-correlation response the measurement is content-driven
# noise (e.g. the applied material differs wildly between shots) — it is
# excluded from the summary statistics but still reported, never silently.
MIN_SURVEY_RESPONSE = 0.35


def _gray_small(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Grayscale float32, downscaled to <= SURVEY_MAX_W. Returns (image, scale)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, SURVEY_MAX_W / g.shape[1])
    if scale < 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return g.astype(np.float32), scale


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def estimate_pose(ref_bgr: np.ndarray, img_bgr: np.ndarray) -> dict:
    """Estimate (|shift|, theta) of img relative to ref.

    Method: sweep small rotations, de-rotate, cv2.phaseCorrelate with a
    Hanning window; keep the angle with the highest correlation response.
    Note: the reported shift is measured after de-rotation, so its
    components are in the de-rotated frame — the magnitude is what sizes
    the search window, and that is preserved.
    """
    ref, scale = _gray_small(ref_bgr)
    img, _ = _gray_small(img_bgr)
    if ref.shape != img.shape:
        raise ValueError(f"해상도 불일치: ref {ref.shape} vs image {img.shape}")
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)

    samples = []  # (angle, response, dx, dy)
    steps = int(round(ANGLE_SWEEP_DEG / ANGLE_STEP_DEG))
    for i in range(-steps, steps + 1):
        angle = i * ANGLE_STEP_DEG
        candidate = _rotate(img, -angle) if angle else img
        (dx, dy), response = cv2.phaseCorrelate(ref, candidate, win)
        samples.append((angle, response, dx, dy))

    k = max(range(len(samples)), key=lambda j: samples[j][1])
    angle, response, dx, dy = samples[k]
    # Sub-step angle: parabola through the response peak and its neighbors —
    # without this the estimate snaps to the ANGLE_STEP_DEG grid.
    if 0 < k < len(samples) - 1:
        r0, r1, r2 = samples[k - 1][1], response, samples[k + 1][1]
        denom = r0 - 2 * r1 + r2
        if abs(denom) > 1e-12:
            offset = 0.5 * (r0 - r2) / denom
            if abs(offset) <= 1.0:
                angle += offset * ANGLE_STEP_DEG

    return {"theta_deg": angle, "dx": dx / scale, "dy": dy / scale,
            "response": response, "shift_px": math.hypot(dx, dy) / scale}


def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in SUPPORTED_EXTS and p.is_file())


def survey_offset(ref_path: Path, images_dir: Path, out_csv: Path | None) -> dict:
    """Measure pose spread of every image in a folder against a reference."""
    ref = imread_u(ref_path)
    rows = []
    files = _image_files(images_dir)
    for i, f in enumerate(files):
        if f.resolve() == ref_path.resolve():
            continue
        try:
            pose = estimate_pose(ref, imread_u(f))
        except ValueError as e:
            rows.append({"file": f.name, "error": str(e)})
            continue
        low = pose["response"] < MIN_SURVEY_RESPONSE
        rows.append({"file": f.name, "low_confidence": int(low),
                     **{k: round(v, 3) for k, v in pose.items()}})
        flag = "  (저신뢰 — 통계 제외)" if low else ""
        print(f"  [{i + 1}/{len(files)}] {f.name}: |shift|={pose['shift_px']:.1f}px "
              f"theta={pose['theta_deg']:+.2f}deg resp={pose['response']:.3f}{flag}")

    ok = [r for r in rows if "error" not in r and not r["low_confidence"]]
    shifts = sorted(r["shift_px"] for r in ok)
    thetas = sorted(abs(r["theta_deg"]) for r in ok)
    summary = {
        "n": len(ok),
        "n_low_confidence": sum(1 for r in rows if r.get("low_confidence")),
        "n_error": sum(1 for r in rows if "error" in r),
        "shift_px_p50": shifts[len(shifts) // 2] if shifts else None,
        "shift_px_max": shifts[-1] if shifts else None,
        "theta_deg_p50": thetas[len(thetas) // 2] if thetas else None,
        "theta_deg_max": thetas[-1] if thetas else None,
    }
    if out_csv is not None:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "shift_px", "dx", "dy",
                                               "theta_deg", "response",
                                               "low_confidence", "error"])
            w.writeheader()
            w.writerows(rows)
    return summary


def score_anchor(ref_bgr: np.ndarray, box: tuple[int, int, int, int],
                 img_bgr: np.ndarray, margin: int) -> float:
    """Best NCC score of the reference patch inside a margin-expanded window."""
    x, y, w, h = box
    patch = cv2.cvtColor(ref_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    ih, iw = img_bgr.shape[:2]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(iw, x + w + margin), min(ih, y + h + margin)
    window = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if window.shape[0] < h or window.shape[1] < w:
        return -1.0
    res = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def survey_anchor(ref_path: Path, box: tuple[int, int, int, int],
                  images_dir: Path, margin: int, out_csv: Path | None) -> dict:
    """NCC score distribution of one anchor candidate across a folder."""
    ref = imread_u(ref_path)
    rows = []
    for f in _image_files(images_dir):
        if f.resolve() == ref_path.resolve():
            continue
        score = score_anchor(ref, box, imread_u(f), margin)
        rows.append({"file": f.name, "ncc": round(score, 4)})
        print(f"  {f.name}: NCC={score:.3f}")

    scores = sorted(r["ncc"] for r in rows)
    n = len(scores)
    summary = {
        "n": n,
        "ncc_min": scores[0] if n else None,
        "ncc_p5": scores[max(0, int(n * 0.05) - 1)] if n else None,
        "ncc_p50": scores[n // 2] if n else None,
        # min_score 권고: p5 − 0.10 마진 (advisor 백분위 철학, §4.4)
        "min_score_suggestion": round(scores[max(0, int(n * 0.05) - 1)] - 0.10, 3)
        if n else None,
    }
    if out_csv is not None:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "ncc"])
            w.writeheader()
            w.writerows(rows)
    return summary


def _print_summary(title: str, summary: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="AVAP Phase 0.5 사전 조사 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("offset", help="로딩 오차(이동·회전) 분포 실측")
    p1.add_argument("--ref", required=True, help="기준 이미지 (골든 후보)")
    p1.add_argument("--images", required=True, help="같은 Position 이미지 폴더")
    p1.add_argument("--out", default=None, help="결과 CSV 경로 (utf-8-sig)")

    p2 = sub.add_parser("anchor", help="앵커 후보 NCC 점수 분포 스크리닝")
    p2.add_argument("--ref", required=True, help="기준 이미지 (골든 후보)")
    p2.add_argument("--box", required=True, help="앵커 박스 x,y,w,h (기준 이미지 픽셀)")
    p2.add_argument("--images", required=True, help="같은 Position 이미지 폴더")
    p2.add_argument("--margin", type=int, default=80,
                    help="탐색창 확장(px) — offset 조사에서 나온 최대 이동보다 크게")
    p2.add_argument("--out", default=None, help="결과 CSV 경로 (utf-8-sig)")

    args = ap.parse_args()
    if args.cmd == "offset":
        s = survey_offset(Path(args.ref), Path(args.images),
                          Path(args.out) if args.out else None)
        _print_summary("로딩 오차 분포 (탐색창·pose gate 근거)", s)
        print("  -> recipe 권고: search 창은 max_shift보다 크게, "
              "max_rotation_deg는 theta_max + 여유")
    else:
        box = tuple(int(v) for v in args.box.split(","))
        if len(box) != 4:
            ap.error("--box는 x,y,w,h 4개 정수")
        s = survey_anchor(Path(args.ref), box, Path(args.images),
                          args.margin, Path(args.out) if args.out else None)
        _print_summary("앵커 후보 점수 분포", s)
        print("  -> min_score_suggestion은 p5-0.10 초기 권고치 — 실운영 전 재확인")


if __name__ == "__main__":
    main()
