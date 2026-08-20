"""Synthetic benchmark generator: random pose + ground-truth sidecar.

This is the only way CI can measure the align→detect→judge path without the
real line images (which never enter the repo). Two rules from the
predecessor project's post-mortem apply:
- results carry benchmark_kind="synthetic" — synthetic scores are for
  regression detection only, never quoted as real-image performance;
- generation is deterministic: same seed → byte-identical files, so the
  benchmark itself cannot drift silently.

Usage:
    python -m avap.synth --out output/synth --n 30 --seed 1234
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from avap.io_utils import imwrite_u

CANVAS_W, CANVAS_H = 960, 720

# Colors (BGR) are designed relative to the sample recipe's HSV thresholds:
# MATERIAL sits inside detect lower/upper (V≈0.65, S≈0.02), BOARD outside
# (V≈0.24) — 합성 색은 판정 임계와의 관계로 설계한다 (선행 프로젝트 규율).
BG = (24, 26, 28)
BOARD = (60, 62, 64)
BOSS = (205, 205, 200)
CONNECTOR = (38, 32, 30)
MATERIAL = (168, 166, 164)

# Golden-frame geometry (pixels). The sample recipe's anchors/ROI mirror these.
BOSS_A = (220, 180)   # screw boss centers — alignment anchors
BOSS_B = (740, 540)
BOSS_R = 18
BEAD_Y = 300
BEAD_X0, BEAD_X1 = 350, 610
BEAD_THICKNESS = 30

SCENARIOS = ("ok", "ng_missing", "ng_broken")


def draw_golden(scenario: str = "ok") -> np.ndarray:
    """Board at the golden pose. `scenario` controls the applied material."""
    img = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    img[:] = BG
    cv2.rectangle(img, (160, 120), (800, 600), BOARD, -1)
    cv2.rectangle(img, (300, 480), (660, 560), CONNECTOR, -1)
    for center in (BOSS_A, BOSS_B):
        cv2.circle(img, center, BOSS_R, BOSS, -1)
        cv2.circle(img, center, BOSS_R // 3, (90, 92, 94), -1)

    if scenario == "ok":
        cv2.line(img, (BEAD_X0, BEAD_Y), (BEAD_X1, BEAD_Y), MATERIAL, BEAD_THICKNESS)
    elif scenario == "ng_broken":
        gap0, gap1 = 460, 520  # missing middle segment → continuity failure
        cv2.line(img, (BEAD_X0, BEAD_Y), (gap0, BEAD_Y), MATERIAL, BEAD_THICKNESS)
        cv2.line(img, (gap1, BEAD_Y), (BEAD_X1, BEAD_Y), MATERIAL, BEAD_THICKNESS)
    elif scenario == "ng_missing":
        pass  # no material at all
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return img


def apply_pose(img: np.ndarray, tx: float, ty: float, theta_deg: float) -> np.ndarray:
    """Rotate about the canvas center by theta, then translate by (tx, ty)."""
    m = cv2.getRotationMatrix2D((CANVAS_W / 2, CANVAS_H / 2), theta_deg, 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(
        img, m, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=BG,
    )


def generate_set(
    out_dir: str | Path,
    n: int = 30,
    seed: int = 1234,
    max_shift: float = 40.0,
    max_rot_deg: float = 3.0,
) -> list[Path]:
    """Write n images + one JSON sidecar each. Deterministic for a given seed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written: list[Path] = []
    for i in range(n):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        tx = round(float(rng.uniform(-max_shift, max_shift)), 2)
        ty = round(float(rng.uniform(-max_shift, max_shift)), 2)
        theta = round(float(rng.uniform(-max_rot_deg, max_rot_deg)), 3)
        img = apply_pose(draw_golden(scenario), tx, ty, theta)

        stem = f"synth_{i:03d}_{scenario}"
        img_path = out / f"{stem}.png"
        imwrite_u(img_path, img)
        sidecar = {
            "benchmark_kind": "synthetic",
            "expected_verdict": "PASS" if scenario == "ok" else "FAIL",
            "file": img_path.name,
            "pose": {"theta_deg": theta, "tx": tx, "ty": ty},
            "scenario": scenario,
            "seed": seed,
        }
        sidecar_path = out / f"{stem}.json"
        sidecar_path.write_text(
            json.dumps(sidecar, sort_keys=True, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        written.extend([img_path, sidecar_path])
    return written


def write_golden(out_path: str | Path) -> Path:
    """Write the zero-pose golden image the sample recipe references."""
    p = Path(out_path)
    imwrite_u(p, draw_golden("ok"))
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="AVAP 합성 벤치마크 생성기")
    ap.add_argument("--out", default="output/synth", help="출력 폴더")
    ap.add_argument("--n", type=int, default=30, help="이미지 수")
    ap.add_argument("--seed", type=int, default=1234, help="난수 seed (결정성)")
    ap.add_argument("--golden", action="store_true", help="골든 이미지도 함께 출력")
    args = ap.parse_args()
    files = generate_set(args.out, n=args.n, seed=args.seed)
    if args.golden:
        files.append(write_golden(Path(args.out) / "golden.png"))
    print(f"[synth] {len(files)}개 파일 생성 → {args.out} (seed={args.seed})")


if __name__ == "__main__":
    main()
