# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
**AVAP (AI Vision Analysis Program)** — 이미지 기반 독립 비전 검사 프로그램 (General 용도)
- 파이프라인: 정렬(2-앵커 NCC fixture) → golden 좌표 ROI 사상 → 단일 마스크 검출(HSV) → 측정→필터→규칙 → Evidence 내장 PASS/FAIL/UNKNOWN
- 설계 문서: `docs/DESIGN.md` — **모든 설계 판단은 이 문서를 먼저 본다**
- 선행 내부 프로젝트의 사후 분석에서 도출된 설계 법칙을 처음부터 테스트로 강제하는 재설계다. 선행 코드는 "파일 복사"가 아니라 검증된 커널만 발췌 이식한다.
- 제약: 예산 0원(오픈소스만) · 비개발자 운영 · Windows 배포(EXE 예정) · UI 한국어 · docstring 영어

## Design Laws (위반 = CI 실패. 전문은 docs/DESIGN.md §2)
1. **L1** 노출된 모든 recipe 파라미터는 판정에 실제 영향 — 스키마 순회 감도 프로브(verdict 플립 필수). 스키마 측은 `recipe.py`가 소비자 없는 키를 거부
2. **L2** UI·엔진·Advisor는 `recipe.PARAM_SPECS` 하나에서 파생 — 두 번째 손관리 목록 금지
3. **L3** 임계 상수는 `avap/constants.py` 한 곳 — 리터럴 재정의는 `test_source_discipline`이 차단
4. **L4** 정렬 불변성 — 이동+회전 변형본이 동일 verdict
5. **L5** 수치는 run 지문(benchmark_kind 포함) 없이 존재 불가 — synthetic 성적을 실사 성능으로 인용 금지
6. **L6** 마스크 생성기는 시스템에 1개 (`detect.make_mask` 예정) — 병렬 검출 경로 금지
7. **L7** 단위는 전 필드 0~1 분수 — % 혼용 금지, 스키마가 검증
- 이미지 IO는 `avap/io_utils.py`의 `imread_u/imwrite_u`만 사용 — bare `cv2.imread`는 한글 경로에서 조용히 실패하며 테스트가 차단한다
- 이미지·모델 파일 커밋 금지 — 테스트 이미지는 `avap/synth.py`로 런타임 생성

## Commands
```bash
pip install -r requirements.txt              # CI와 동일한 한 줄
python -m pytest -q                          # 테스트 (SKIP은 CI에서 실패 처리)
python -m avap.synth --out output/synth --n 30 --seed 1234 --golden   # 합성 벤치마크

# Phase 0.5 사전 조사 — 라인 이미지가 있는 PC에서 실행 (파일럿 공정 2종)
python -m avap.preflight offset --ref <골든후보> --images <같은 공정 폴더> --out offset.csv
python -m avap.preflight anchor --ref <골든후보> --box x,y,w,h --images <폴더> --out anchor.csv
#  offset: 로딩 오차 분포(|shift|·θ 백분위) → 탐색창·pose gate 값의 근거
#  anchor: 앵커 후보 NCC 분포 → min_score 권고(p5−0.10). 저신뢰 측정은 분리 집계
```

## Architecture (Phase 0 시점)
```
avap/
  constants.py   공유 상수 (L3의 단일 정의처)
  io_utils.py    유니코드 안전 이미지 IO (유일한 이미지 파일 접점)
  recipe.py      recipe 로드·스키마 검증·frozen 스냅샷·지문. PARAM_SPECS 내장 (L2)
  synth.py       합성 벤치마크: 랜덤 pose + GT 사이드카, seed 결정성 (CI가 매 커밋 측정)
recipes/         sample_synth.json — synth 골든 이미지 대응 참조 recipe (테스트 픽스처)
tests/           pytest. source_discipline 테스트가 IO·상수 규율을 소스 레벨에서 강제
```

## Roadmap (docs/DESIGN.md §11)
- Phase 0.5: 실사 사전 조사 — 로딩 오차 분포 실측·앵커 스크리닝 (파일럿 공정 2종, 사용자 확정)
- Phase 1: 정렬 엔진 — 2-앵커 NCC + 2점 강체(스케일 1 고정, `estimateAffinePartial2D` 사용 금지 — similarity라 스케일 오차가 샘) + 서브픽셀 파라볼라 보간 + 4중 게이트. 목표 ≤2px/±0.5°
- Phase 2: detect/measure/tools(blob·coverage·color_stats·shape_compare)/judge/report/batch
- Phase 3: PyQt5 GUI (QGraphicsView, 골든 위 ROI/앵커 편집, HSV 스포이드)
- 보류 목록(착수 전 문답 필수): M패턴, YOLO tool, ECC/ORB 폴백, Advisor 이식, EXE
