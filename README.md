# AVAP — AI Vision Analysis Program

이미지 기반 **독립 비전 검사 프로그램**. 이미지를 읽어(JPEG/BMP 등 포맷 무관) 제품 위치를 정렬(fixture)하고, ROI 안에서 도포 형태·blob·coverage·색상을 검출해 스스로 PASS/FAIL을 판정한다.

- 파이프라인: `정렬(2-앵커 NCC) → golden 좌표 ROI 사상 → 단일 마스크 검출 → 측정·필터·규칙 → Evidence 내장 판정`
- 정렬 실패 = **UNKNOWN** (조용한 full-frame 폴백 없음)
- 설계 문서: [docs/DESIGN.md](docs/DESIGN.md) — 설계 법칙 L1~L7을 테스트로 강제한다

## 실행

```bash
pip install -r requirements.txt
python -m pytest -q                          # 테스트 전체
python -m avap.synth --out output/synth --golden   # 합성 벤치마크 생성 (결정적)
```

## Phase 0.5 사전 조사 (실사 이미지가 있는 PC에서)

정렬 엔진의 탐색창 크기·게이트 값은 추정이 아니라 실측에서 나온다. Position 폴더별로:

```bash
# 1. 로딩 오차 분포 - 제품이 실제로 얼마나 움직이는가
python -m avap.preflight offset --ref <골든> --images <폴더> --out offset.csv

# 2. 앵커 박스 2개 지정 (도포 영역 '밖'의 안정적 랜드마크)
#    창을 띄우므로 새 venv에는 requirements.txt 대신 아래 파일을 설치한다:
#      python -m pip install -r requirements-desktop.txt
python -m avap.preflight pick --ref <골든>

# 3. 앵커별 NCC 점수 분포 - min_score의 근거
python -m avap.preflight anchor --ref <골든> --box x,y,w,h --images <폴더> --out anchor1.csv
```

`requirements.txt`는 headless OpenCV라 창을 열 수 없다. `pick`만 데스크톱 빌드를
요구하고, 나머지 조사와 테스트는 headless로 그대로 돌아간다.
이미 `requirements.txt`를 설치한 venv를 재사용하려면 두 OpenCV 배포판을 먼저
제거한 뒤 데스크톱 파일을 설치한다. 둘은 같은 `cv2` import를 공유해 공존할 수 없다.

```bash
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install -r requirements-desktop.txt
```

`pick`은 원본 픽셀 좌표와 이어서 붙여넣을 `anchor` 명령을 그대로 출력한다.
앵커가 recipe 이격 하한에 못 미치거나 각도 정밀도 목표(±0.5°)를 못 맞추면
exit 1 - 감수하고 진행하려면 `anchor`에 `--box`를 직접 지정한다.

## Recipe 1.0 → 1.1

Phase 1 정렬 엔진은 확정 골든 이미지에서 앵커 패치를 직접 crop/cache한다. 따라서
1.0 recipe는 자동 변환하지 않고 명시적으로 거부한다. 기존 recipe를 올릴 때는:

1. `avap_recipe`를 `"1.1"`로 변경한다.
2. 각 `alignment.anchors[]`의 `patch`를 삭제한다.
3. `alignment.pose_gates.anchor_dist_tol_frac`를 삭제한다. 두 앵커 거리비는
   `scale_tol` 하나가 검사한다.
4. `meta.recipe_version`을 올리고 기존 `meta.fingerprint`를 제거한 뒤 새 지문으로
   저장한다.
5. 골든 이미지의 SHA·크기와 `origin`/`search` 박스를 다시 확인한 뒤 테스트한다.

## 현재 상태

Phase 0·0.5·1 완료: 유니코드 안전 이미지 IO · recipe 1.1 로더 · Home-PC
실사 사전 조사 · 2-앵커 NCC 정렬/서브픽셀 pose 복원 · UNKNOWN 6원인 코드 ·
합성 pose 회귀 벤치마크 · CI 3 job.

Phase 1 실사 검증은 `benchmark_kind=real`인 별도 비공개 run artifact로 보존했다.
확정 전 provisional golden과 OK 이미지로만 구성한 holdout에서 same-position은
OK 40/40(UNKNOWN 0/40), cross-position은 OK 0/40(UNKNOWN 40/40)이었다. 이는
**정렬의 제한된 검증**이며 OK/NG 판별이나 생산 대체 성능 주장이 아니다. 공개
문서에는 집계와 artifact SHA-256만 남기고 원본·경로·unit ID는 저장소에 넣지
않는다.

다음: Phase 2(검출·판정 + CLI).
