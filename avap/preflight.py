"""Phase 0.5 preflight surveys - run on the PC that holds the real line images.

Two questions must be answered with data BEFORE the alignment engine is
tuned (docs/DESIGN.md §11 Phase 0.5 - the design review found the
original plans assumed these numbers instead of measuring them):

1. offset - how far do products actually move between shots?
   → sets the anchor search-window size and the pose gates.
2. anchor - is a candidate landmark stable across the whole set?
   → NCC score distribution decides min_score (percentile basis, same
     philosophy as the percentile-based threshold advisor: p5 minus a margin).

Both are surveys, not the alignment engine: rough numbers with an honest
method note beat precise-looking numbers from an unvalidated pipeline.

Usage (run inside the folder that holds one Position's images):
    python -m avap.preflight offset --ref golden.png --images ./ok_set --out offset.csv
    python -m avap.preflight pick   --ref golden.png
    python -m avap.preflight anchor --ref golden.png --box 700,495,80,90 --images ./ok_set

`pick` exists because `anchor` needs --box in source pixels, and reading
those off a 4K frame by hand is where this survey stalls.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from avap.constants import MIN_ANCHOR_SEPARATION_FRAC
from avap.io_utils import SUPPORTED_EXTS, imread_u
from avap.recipe import anchor_separation_frac

# Survey resolution: full frames are downscaled so FFTs stay fast; survey
# precision (~1px at survey scale) is plenty for sizing search windows.
SURVEY_MAX_W = 640
ANGLE_SWEEP_DEG = 4.0
ANGLE_STEP_DEG = 0.25
# Below this phase-correlation response the measurement is content-driven
# noise (e.g. the applied material differs wildly between shots) - it is
# excluded from the summary statistics but still reported, never silently.
MIN_SURVEY_RESPONSE = 0.35

# Anchor picking view. highgui has no zoom, so a full frame is scaled down
# to fit an ordinary screen before boxes can be drawn on it.
PICK_MAX_W = 1400
PICK_MAX_H = 800
# NCC peak localisation error assumed when reporting the angular precision
# two picked anchors can support (DESIGN.md section 5 targets +-0.5 deg).
NCC_PEAK_ERROR_PX = 1.0
TARGET_THETA_DEG = 0.5


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
    components are in the de-rotated frame - the magnitude is what sizes
    the search window, and that is preserved.
    """
    if ref_bgr.shape[:2] != img_bgr.shape[:2]:
        rh, rw = ref_bgr.shape[:2]; ih, iw = img_bgr.shape[:2]
        raise ValueError(
            f"해상도 불일치: ref {rw}x{rh} vs image {iw}x{ih} - 원본 크기가 달라도 "
            f"다운스케일 후엔 같아져 가짜 측정이 나온다. 같은 카메라 설정의 세트인지 확인"
        )
    ref, scale = _gray_small(ref_bgr)
    img, _ = _gray_small(img_bgr)
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
    # Sub-step angle: parabola through the response peak and its neighbors -
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
        flag = "  (저신뢰 - 통계 제외)" if low else ""
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
    if ref_bgr.shape[:2] != img_bgr.shape[:2]:
        rh, rw = ref_bgr.shape[:2]; ih, iw = img_bgr.shape[:2]
        raise ValueError(f"해상도 불일치: ref {rw}x{rh} vs image {iw}x{ih}")
    x, y, w, h = box
    rh, rw = ref_bgr.shape[:2]
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > rw or y + h > rh:
        # numpy 슬라이싱은 범위 밖을 조용히 잘라 패치가 요청보다 작아진다 -
        # 그 상태의 매칭은 측정이 아니다. sentinel(-1.0)로 돌려주면 유효 측정으로
        # 집계돼 n=1·exit 0이 되므로 반드시 예외로 (외부 검증 발견).
        raise ValueError(f"앵커 박스가 기준 이미지를 벗어남: box={box}, ref {rw}x{rh}")
    patch = cv2.cvtColor(ref_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    ih, iw = img_bgr.shape[:2]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(iw, x + w + margin), min(ih, y + h + margin)
    window = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if window.shape[0] < h or window.shape[1] < w:
        raise ValueError(f"탐색창({window.shape[1]}x{window.shape[0]})이 "
                         f"패치({w}x{h})보다 작음: box={box}, margin={margin}")
    res = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def anchor_summary(scores: list[float]) -> dict:
    """NCC score distribution -> min_score recommendation (p5 - 0.10 margin).

    Uses real interpolated percentiles (np.percentile): the earlier
    nearest-rank shortcut `scores[int(n*0.05)-1]` returned the MINIMUM for
    n<40 samples, dragging the recommendation down to the worst outlier at
    exactly the 30~50-image sample sizes this survey targets.
    """
    if not scores:
        return {"n": 0, "ncc_min": None, "ncc_p5": None, "ncc_p50": None,
                "min_score_suggestion": None}
    arr = np.asarray(scores, dtype=float)
    p5 = float(np.percentile(arr, 5))
    return {
        "n": int(arr.size),
        "ncc_min": round(float(arr.min()), 4),
        "ncc_p5": round(p5, 4),
        "ncc_p50": round(float(np.percentile(arr, 50)), 4),
        "min_score_suggestion": round(p5 - 0.10, 3),
    }


def survey_anchor(ref_path: Path, box: tuple[int, int, int, int],
                  images_dir: Path, margin: int, out_csv: Path | None) -> dict:
    """NCC score distribution of one anchor candidate across a folder."""
    ref = imread_u(ref_path)
    rows = []
    for f in _image_files(images_dir):
        if f.resolve() == ref_path.resolve():
            continue
        try:
            score = score_anchor(ref, box, imread_u(f), margin)
        except ValueError as e:
            rows.append({"file": f.name, "error": str(e)})
            continue
        rows.append({"file": f.name, "ncc": round(score, 4)})
        print(f"  {f.name}: NCC={score:.3f}")

    summary = anchor_summary([r["ncc"] for r in rows if "error" not in r])
    summary["n_error"] = sum(1 for r in rows if "error" in r)
    if out_csv is not None:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "ncc", "error"])
            w.writeheader()
            w.writerows(rows)
    return summary


def view_scale(shape: tuple, max_w: int = PICK_MAX_W, max_h: int = PICK_MAX_H) -> float:
    """Factor that fits an image inside the picking window. Never upscales."""
    h, w = shape[:2]
    return min(1.0, max_w / w, max_h / h)


def box_to_source(box_view: tuple[int, int, int, int], scale: float,
                  shape: tuple) -> tuple[int, int, int, int]:
    """Map a box drawn on the scaled view back to source pixels.

    Both edges are mapped and then differenced, rather than scaling w/h
    independently, so the right/bottom edge lands where the operator drew
    it. The result is clamped inside the image: a picker that handed the
    next command a box score_anchor() rejects would be worse than no
    picker at all.
    """
    ih, iw = shape[:2]
    x, y, w, h = box_view
    x0 = max(0, min(int(round(x / scale)), iw - 1))
    y0 = max(0, min(int(round(y / scale)), ih - 1))
    x1 = max(x0 + 1, min(int(round((x + w) / scale)), iw))
    y1 = max(y0 + 1, min(int(round((y + h) / scale)), ih))
    return x0, y0, x1 - x0, y1 - y0


def anchor_separation_note(boxes: list[tuple[int, int, int, int]],
                           shape: tuple) -> tuple[bool, str]:
    """Judge whether two anchors sit far enough apart to carry the angle.

    Two-point rigid estimation turns an NCC peak error into an angle error
    of roughly (error / separation) radians, so anchors picked side by side
    cannot reach the design target no matter how good each match is. The
    separation floor is the same constant the recipe validator uses.
    """
    if len(boxes) != 2:
        return False, "앵커는 정확히 2개가 필요하다 (2점 강체 추정)."
    h, w = shape[:2]
    # Normalize per axis first, exactly as the recipe validator does. Measuring
    # against the physical diagonal instead would green-light anchors the
    # validator then rejects on any non-square frame.
    norm = [(x / w, y / h, bw / w, bh / h) for x, y, bw, bh in boxes]
    frac = anchor_separation_frac(norm[0], norm[1])

    (x0, y0, w0, h0), (x1, y1, w1, h1) = boxes
    sep_px = math.hypot((x1 + w1 / 2) - (x0 + w0 / 2), (y1 + h1 / 2) - (y0 + h0 / 2))
    theta_err = math.degrees(NCC_PEAK_ERROR_PX / sep_px) if sep_px > 0 else float("inf")
    detail = (f"앵커 간격 {sep_px:.0f}px (recipe 기준 {frac * 100:.1f}%), "
              f"NCC 1px 오차 기준 각도 정밀도 약 {theta_err:.2f}deg "
              f"(목표 {TARGET_THETA_DEG}deg)")
    if frac < MIN_ANCHOR_SEPARATION_FRAC:
        return False, (f"[경고] {detail}\n"
                       f"  recipe 하한 {MIN_ANCHOR_SEPARATION_FRAC * 100:.0f}% 미만이라 "
                       f"검증에서 거부된다. 더 멀리 떨어진 두 곳을 다시 고를 것.")
    if theta_err > TARGET_THETA_DEG:
        # Passing the recipe floor while missing the design target is still a
        # fail: reporting "cannot meet the precision" and exiting 0 would be
        # the documented-but-unenforced pattern this project keeps removing.
        return False, (f"[경고] {detail}\n"
                       f"  recipe 하한은 통과했으나 각도 정밀도가 목표에 못 미친다. "
                       f"더 멀리 떨어뜨리거나, 감수하고 진행하려면 anchor 명령에 "
                       f"--box 를 직접 지정할 것.")
    return True, f"[OK] {detail}"


def pick_anchors(ref_path: Path) -> list[tuple[int, int, int, int]]:
    """Open the reference image and let the operator drag two anchor boxes."""
    ref = imread_u(ref_path)
    scale = view_scale(ref.shape)
    view = ref if scale == 1.0 else cv2.resize(
        ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    print(f"기준 이미지 {ref.shape[1]}x{ref.shape[0]}"
          + (f" (화면 표시 {scale * 100:.0f}%)" if scale < 1.0 else ""))
    print("도포 영역 '밖'의 안정적인 랜드마크 2곳에 박스를 드래그하고 각각 ENTER, "
          "둘 다 고른 뒤 ESC.")
    # ASCII window title: highgui does not render Korean reliably on Windows.
    try:
        picked = cv2.selectROIs("AVAP - pick 2 anchors (ENTER each, ESC done)",
                                view, showCrosshair=True)
    except cv2.error as e:
        raise RuntimeError(
            "창을 열 수 없다. requirements.txt는 opencv-python-headless(GUI 없음)를 "
            "설치한다. 기존 venv에서는 두 OpenCV 배포판을 제거한 뒤 데스크톱용만 "
            "설치할 것:\n"
            "    python -m pip uninstall -y opencv-python opencv-python-headless\n"
            "    python -m pip install -r requirements-desktop.txt\n"
            "  새 venv에서는 requirements.txt 대신 requirements-desktop.txt만 설치한다.\n"
            "  SSH/서버처럼 화면 자체가 없으면 anchor 명령에 --box x,y,w,h 를 직접 "
            f"지정할 것.\n  원인: {e}"
        ) from e
    finally:
        # A headless build raises here too; letting that escape would bury the
        # message above under an OpenCV traceback.
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return [box_to_source(tuple(int(v) for v in b), scale, ref.shape)
            for b in picked if b[2] > 0 and b[3] > 0]


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
                    help="탐색창 확장(px) - offset 조사에서 나온 최대 이동보다 크게")
    p2.add_argument("--out", default=None, help="결과 CSV 경로 (utf-8-sig)")

    p3 = sub.add_parser("pick", help="기준 이미지에서 앵커 박스 2개를 마우스로 지정")
    p3.add_argument("--ref", required=True, help="기준 이미지 (골든 후보)")

    args = ap.parse_args()
    if args.cmd == "pick":
        try:
            boxes = pick_anchors(Path(args.ref))
        except RuntimeError as e:
            print(f"[FAIL] {e}")
            sys.exit(1)
        if len(boxes) != 2:
            print(f"[FAIL] 박스 {len(boxes)}개 지정됨 - 정확히 2개가 필요하다.")
            sys.exit(1)
        ok, note = anchor_separation_note(boxes, imread_u(Path(args.ref)).shape)
        print("\n=== 앵커 박스 (원본 픽셀) ===")
        for i, b in enumerate(boxes, 1):
            print(f"  앵커{i}: {','.join(str(v) for v in b)}")
        print(f"\n{note}\n")
        print("이어서 실행할 명령:")
        for i, b in enumerate(boxes, 1):
            print(f"  python -m avap.preflight anchor --ref \"{args.ref}\" "
                  f"--box {','.join(str(v) for v in b)} "
                  f"--images <이미지 폴더> --out anchor{i}.csv")
        sys.exit(0 if ok else 1)
    if args.cmd == "offset":
        s = survey_offset(Path(args.ref), Path(args.images),
                          Path(args.out) if args.out else None)
        _print_summary("로딩 오차 분포 (탐색창·pose gate 근거)", s)
        if s["n"] == 0:
            # 이미지 0장·전량 저신뢰·전량 해상도 불일치 - 어느 쪽이든 측정은 없다.
            print("[FAIL] 유효 측정 0건 - 이 요약으로는 아무것도 결정할 수 없다. "
                  f"(저신뢰 {s['n_low_confidence']}건 / 오류 {s['n_error']}건)")
            sys.exit(1)
        print("  -> recipe 권고: search 창은 max_shift보다 크게, "
              "max_rotation_deg는 theta_max + 여유")
    else:
        box = tuple(int(v) for v in args.box.split(","))
        if len(box) != 4:
            ap.error("--box는 x,y,w,h 4개 정수")
        s = survey_anchor(Path(args.ref), box, Path(args.images),
                          args.margin, Path(args.out) if args.out else None)
        _print_summary("앵커 후보 점수 분포", s)
        if s["n"] == 0:
            print(f"[FAIL] 유효 측정 0건 - min_score 권고 불가. (오류 {s.get('n_error', 0)}건)")
            sys.exit(1)
        print("  -> min_score_suggestion은 p5-0.10 초기 권고치 - 실운영 전 재확인")


if __name__ == "__main__":
    main()
