"""Preflight surveys validated against synth images with known ground truth.

The synth sidecars carry the injected pose, so the offset survey's own
accuracy is itself measured — the survey tool never becomes an unverified
number source (Design Law L5 in spirit).
"""
import json
from pathlib import Path

import pytest

from avap.io_utils import imread_u
from avap.preflight import estimate_pose, score_anchor, survey_anchor, survey_offset
from avap.synth import BOSS_B, BOSS_R, generate_set, write_golden


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synth")
    generate_set(d, n=9, seed=11, max_shift=30.0, max_rot_deg=2.0)
    write_golden(d / "golden.png")
    return d


def test_estimate_pose_recovers_known_offsets(synth_dir):
    ref = imread_u(synth_dir / "golden.png")
    checked = 0
    for sidecar in sorted(synth_dir.glob("synth_*.json")):
        gt = json.loads(sidecar.read_text(encoding="utf-8"))
        if gt["scenario"] != "ok":
            continue  # 내용이 다른 이미지(비드 유무)는 자세 추정 검증 대상이 아님
        img = imread_u(synth_dir / gt["file"])
        pose = estimate_pose(ref, img)
        gt_mag = (gt["pose"]["tx"] ** 2 + gt["pose"]["ty"] ** 2) ** 0.5
        # 조사 도구 목표 정밀도: 탐색창·게이트 값 결정용이므로 ±3px / ±0.75° 면 충분
        # (정밀 정렬은 Phase 1의 2-앵커 NCC 엔진 몫 — 이 도구는 분포 조사기다)
        assert abs(pose["shift_px"] - gt_mag) < 3.0, (
            f"{gt['file']}: |shift| 추정 {pose['shift_px']:.1f} vs GT {gt_mag:.1f}"
        )
        assert abs(abs(pose["theta_deg"]) - abs(gt["pose"]["theta_deg"])) < 0.75, (
            f"{gt['file']}: theta 추정 {pose['theta_deg']} vs GT {gt['pose']['theta_deg']}"
        )
        checked += 1
    assert checked >= 3  # 측정 0건이면 이 테스트 자체가 거짓 녹색이다


def test_survey_offset_summary_and_csv(synth_dir, tmp_path):
    out = tmp_path / "offset.csv"
    s = survey_offset(synth_dir / "golden.png", synth_dir, out)
    # 내용이 크게 다른 이미지(ng_missing)는 저신뢰로 분리 집계돼야 하고,
    # 신뢰 가능한 측정만으로 낸 최대 이동은 주입 범위(±30px) 근처여야 한다.
    assert s["n"] >= 5 and s["n_error"] == 0
    assert s["n"] + s["n_low_confidence"] >= 9
    assert 0 < s["shift_px_max"] < 60
    assert 0 <= s["theta_deg_max"] <= 2.75
    text = out.read_bytes()
    assert text.startswith(b"\xef\xbb\xbf")  # utf-8-sig — 한국어 Excel 호환


def test_anchor_screening_on_stable_landmark(synth_dir, tmp_path):
    # 골든의 screw boss를 앵커 후보로 — 모든 변형본에서 고점수가 나와야 정상
    x, y = BOSS_B
    box = (x - BOSS_R - 12, y - BOSS_R - 12, (BOSS_R + 12) * 2, (BOSS_R + 12) * 2)
    out = tmp_path / "anchor.csv"
    s = survey_anchor(synth_dir / "golden.png", box, synth_dir, margin=80, out_csv=out)
    assert s["n"] >= 9
    assert s["ncc_min"] > 0.6, f"안정 랜드마크인데 NCC 최저 {s['ncc_min']}"
    assert s["min_score_suggestion"] is not None
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")


def test_anchor_screening_flags_bad_anchor(synth_dir):
    # 도포부(비드) 위 박스는 시나리오에 따라 내용이 사라지므로 점수가 무너져야 한다
    ref = imread_u(synth_dir / "golden.png")
    bead_box = (430, 270, 100, 60)
    scores = []
    for sc in sorted(synth_dir.glob("synth_*_ng_missing.png")):
        scores.append(score_anchor(ref, bead_box, imread_u(sc), margin=80))
    assert scores and min(scores) < 0.5, (
        f"도포부 박스가 미도포 이미지에서도 고점수({scores}) — 스크리닝이 나쁜 앵커를 못 거름"
    )
