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


def test_anchor_summary_p5_is_not_the_minimum():
    # 이전 구현 scores[int(n*0.05)-1]은 n<40에서 최솟값을 반환해, 이상치 1장이
    # min_score 권고를 통째로 끌어내렸다 (외부 검증 발견). 보간 백분위여야 한다.
    from avap.preflight import anchor_summary
    scores = [0.30] + [0.80 + 0.003 * i for i in range(39)]  # n=40, 이상치 1개
    s = anchor_summary(scores)
    assert s["n"] == 40
    assert s["ncc_min"] == 0.30
    assert s["ncc_p5"] > 0.5, f"p5={s['ncc_p5']} — 여전히 이상치에 끌려감"
    assert s["min_score_suggestion"] == round(s["ncc_p5"] - 0.10, 3)
    assert anchor_summary([])["min_score_suggestion"] is None


def test_resolution_mismatch_rejected_before_downscale(synth_dir):
    # 1280x960 vs 640x480은 다운스케일 후 크기가 같아져 그럴듯한 가짜 측정
    # (실측 재현: response 0.517)을 냈다 (외부 검증 발견). 원본 크기에서 거부해야 한다.
    import cv2
    from avap.preflight import estimate_pose, score_anchor
    ref = imread_u(synth_dir / "golden.png")
    doubled = cv2.resize(ref, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    with pytest.raises(ValueError, match="해상도 불일치"):
        estimate_pose(ref, doubled)
    with pytest.raises(ValueError, match="해상도 불일치"):
        score_anchor(ref, (10, 10, 40, 40), doubled, margin=80)


def test_survey_records_resolution_mismatch_as_error(synth_dir, tmp_path):
    import cv2
    from avap.io_utils import imwrite_u
    d = tmp_path / "mixed"
    d.mkdir()
    ref = imread_u(synth_dir / "golden.png")
    imwrite_u(d / "big.png", cv2.resize(ref, None, fx=2, fy=2))
    s = survey_offset(synth_dir / "golden.png", d, None)
    assert s["n"] == 0 and s["n_error"] == 1  # 가짜 측정이 아니라 오류로 집계


def test_cli_exits_nonzero_when_no_valid_measurement(synth_dir, tmp_path):
    # 이미지 0장 폴더에서 성공 종료하면 "측정 0건인데 녹색" — Phase 0 원칙 위반.
    import subprocess, sys
    from pathlib import Path
    empty = tmp_path / "empty"
    empty.mkdir()
    repo = Path(__file__).resolve().parents[1]
    for cmd in (
        ["offset", "--ref", str(synth_dir / "golden.png"), "--images", str(empty)],
        ["anchor", "--ref", str(synth_dir / "golden.png"), "--box", "700,495,80,90",
         "--images", str(empty)],
    ):
        r = subprocess.run([sys.executable, "-m", "avap.preflight", *cmd],
                           cwd=repo, capture_output=True, text=True, timeout=120)
        assert r.returncode != 0, f"{cmd[0]}: 빈 폴더인데 성공 종료\n{r.stdout}"
        assert "[FAIL]" in r.stdout


def test_out_of_bounds_anchor_box_is_error_not_valid_measurement(synth_dir, tmp_path):
    # 범위 밖 박스가 -1.0 sentinel로 유효 측정 집계돼 n=1·suggestion=-1.1·exit 0이
    # 되던 결함 (외부 검증 발견 + 재현 확인). ValueError → error 행이어야 한다.
    from avap.preflight import score_anchor, survey_anchor
    ref = imread_u(synth_dir / "golden.png")
    h, w = ref.shape[:2]
    bad_box = (w - 40, h - 40, 80, 90)
    with pytest.raises(ValueError, match="벗어남"):
        score_anchor(ref, bad_box, imread_u(sorted(synth_dir.glob("synth_*.png"))[0]),
                     margin=10)
    s = survey_anchor(synth_dir / "golden.png", bad_box, synth_dir, margin=10,
                      out_csv=None)
    assert s["n"] == 0 and s["n_error"] >= 9
    assert s["min_score_suggestion"] is None  # 쓰레기 권고(-1.1) 재발 금지


def test_cli_stdout_survives_cp949(synth_dir, tmp_path):
    # Windows 기본 콘솔(CP949)에서 em dash 등 인코딩 불가 문자가 stdout에 있으면
    # 성공한 조사도 마지막 출력에서 UnicodeEncodeError로 죽는다 (외부 검증 발견 +
    # 재현: '—' position 82). 성공·실패 경로 전부 CP949 stdout으로 실행한다.
    import os, subprocess, sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONIOENCODING": "cp949"}
    empty = tmp_path / "empty"
    empty.mkdir()
    golden = str(synth_dir / "golden.png")
    cases = [  # (인자, 기대 returncode==0)
        (["offset", "--ref", golden, "--images", str(synth_dir)], True),
        (["offset", "--ref", golden, "--images", str(empty)], False),
        (["anchor", "--ref", golden, "--box", "700,495,80,90",
          "--images", str(synth_dir)], True),
        (["anchor", "--ref", golden, "--box", "700,495,80,90",
          "--images", str(empty)], False),
    ]
    for args, expect_ok in cases:
        # 자식은 cp949로 쓰므로 부모도 cp949로 읽는다 (UTF-8로 읽으면 테스트가 죽음)
        r = subprocess.run([sys.executable, "-m", "avap.preflight", *args],
                           cwd=repo, env=env, capture_output=True,
                           encoding="cp949", errors="replace", timeout=180)
        assert "UnicodeEncodeError" not in r.stderr, f"{args[0]}: CP949 크래시\n{r.stderr}"
        assert (r.returncode == 0) == expect_ok, \
            f"{args}: rc={r.returncode}\n{r.stdout}\n{r.stderr}"
