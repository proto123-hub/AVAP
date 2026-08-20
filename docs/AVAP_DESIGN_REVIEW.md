# AVAP 설계 재검토 — VSGP 전면 재검토 및 신규 프로그램 설계안

- 작성일: 2026-08-20
- 방법: 병렬 감사 4건(core/ui/utils/tests·CI) + ROI 정렬 기술조사 + 독립 설계안 3종(MVP-first / Tool-pipeline / Validation-first) 생성 + 적대적 교차 비판 → 본 문서로 통합
- 상태: **설계 제안** — 사용자 승인 전. 미결 결정사항은 §9

---

## 0. 결론 요약

1. **Two-Track 확정.** VSGP는 회사 제출용으로 동결 보존한다. 신규 프로그램 **AVAP(AI Vision Analysis Program)** 를 General 용도로 새로 설계한다.
2. **개선이 아니라 재설계가 맞다.** VSGP의 문제는 기능 부족이 아니라 구조다 — "남의 검사기(FA+/Cognex) setting을 통신도 정보도 없이 추측 제안"하는 포지션 자체와, 그 위에 쌓인 3대 병리(죽은 파라미터·이원 검출 경로·검증 없는 수치)는 부분 수리가 불가능하다.
3. **AVAP = 스스로 판정하는 독립 검사 파이프라인.** `정렬(fixture) → ROI 사상 → 단일 마스크 검출 → 측정→필터→규칙 → Evidence 내장 판정` 한 방향 흐름 하나만 만든다.
4. **핵심 신기능은 정렬이다.** 제품 로딩 오차(평행이동+소회전)를 2-앵커 NCC 템플릿 매칭으로 흡수하고, ROI는 골든 이미지 좌표계에 저장해 매 이미지마다 변환으로 사상한다. 정렬 실패 = UNKNOWN(조용한 full-frame 폴백 금지).
5. **VSGP에서 검증된 커널은 발췌 이식한다.** 파일 복사가 아니라 커널 단위(§3) — blob 기하 측정, HSV+모폴로지, 좌표 매핑, 워커 패턴, run 지문(provenance), QSS 테마, 백분위 Advisor 산식.
6. **일정은 4~6주(1인 기준).** 첫 설계안의 2~3주는 교차 비판에서 비현실 판정. Phase 0.5(실사 사전 조사)를 신설했다.

---

## 1. 배경 — 확정된 방향 (사용자 결정사항)

| 항목 | 내용 |
|---|---|
| 이름 | **AVAP** (AI Vision Analysis Program). VSGP라는 이름은 회사 용도로 보존 |
| 용도 | General — 회사 전용이 아닌 범용 비전 검사 보조/판정 프로그램 |
| 목표 | 이미지(JPEG/BMP 등 포맷 무관)를 읽어 ROI를 설정하고, 도포 형태·blob 크기·coverage(면적/용량)·색상을 검출해 어디에 발렸는지 확인하고 **스스로 PASS/FAIL 판정** |
| 탈피할 구조 | 검사 로직·변수를 모르는 외부 검사기에 통신 없이 setting을 추측 제안하는 형태 |
| 핵심 요구 | 제품 로딩 오차(이동+약간의 회전)가 있어도 ROI가 제품을 따라감 — **정렬 필수** |
| 샘플 | IDC 실사 이미지 우선 (OneDrive 88GB, 저장소 커밋 금지 유지) |
| 제약 | 예산 0원 / Python+OpenCV+PyQt5 / 비개발자 운영 / Windows·EXE 배포 이력 / UI 한국어 |

---

## 2. VSGP 진단 — 왜 개선이 아니라 재설계인가

### 2.1 포지션의 구조적 한계
VSGP는 "FA+ 검사기가 어떤 로직·변수로 검사하는지 모르는 채, 통신도 없이 setting 참고치를 추측 제안"하는 도구였다. 제안값이 실제 검사기에서 어떤 효과를 내는지 **검증 루프가 원천적으로 존재하지 않는다.** 이 포지션에서는 아무리 내부 품질을 올려도 신뢰할 수 있는 도구가 되지 못한다. AVAP는 판정 주체를 자기 자신으로 옮겨 이 문제를 소멸시킨다.

### 2.2 3대 병리 (사고 이력 — CLAUDE.md 계약 1~10, 감사로 재확인)

**① 죽은 파라미터.** 슬라이더·Advisor가 만지는 `blob_spec` 전체(area·circularity·AR·solidity·coverage)가 판정에 무력하다(계약 9 — `BlobAnalyzer`는 전 contour를 담고 verdict만 라벨링, `judge()`는 `len(blobs)`만 셈). ROI Position(2·3·5·6)은 `and not roi_boxes` 게이트 한 항 때문에 품질 검사가 아예 실행되지 않아 `quality_check: true`가 죽은 플래그다(계약 8). `advisor_weights` 2종도 읽히기만 하고 효과 0(감사에서 신규 확인 — `advisor.py:105, 211`). **설정 파일이 자기 플래그의 사망을 주석으로 경고해야 하는 상태**(`positions.json:3`)까지 갔다.

**② 이원 검출 경로.** 같은 TG를 `Detector._detect_by_color`(settings.json HSV)와 `OKNGJudge._feature_mask`(positions.json의 **다른** HSV)가 두 번 검출한다. 이것이 "슬라이더를 움직여도 Pos2·3·5·6 판정이 안 움직이는" 현상의 직접 원인. 여기에 blob_spec/blob_spec_color 이중 스펙 + 런타임 병합·역병합 기계(~70줄)가 얹혀 Ctrl+S 한 번에 설정이 영구 손상되는 취약성(계약 1)을 낳았다.

**③ 검증 없는 수치.** 같은 val 120장이 모드에 따라 97.5%/71.7%로 갈리는데 결과 JSON에 모드 필드가 없어 추적 불가였고, 항상 skip되는 CI accuracy job이 "측정 근거"로 수개월 인용됐다(계약 6). run 지문 블록(`batch_validate.py:73-101`)이 사후 해법이었다.

부수 병리: 파일명 정규식(`Pos N`) 기반 판정 라우팅(깨지면 전량 UNKNOWN, 계약 4), 고정 픽셀 ROI(로딩 오차 추적 불가), %/분수 단위 혼재(0.1이 파일에 따라 20배 다른 의미), 유니코드 경로 미대응(bare `cv2.imread` — 한글 경로에서 조용히 None).

### 2.3 그럼에도 건진 것
VSGP는 실패 기록이 축적된 저장소다. 검증된 알고리즘 커널, frozen EXE·CP949·OneDrive 실전 교훈, 계약 고정 테스트 패턴, 실패 누적 문서(BUILD.md) — 이것이 AVAP의 출발 자본이다. §3에 상세.

---

## 3. 계승 자산 맵

**원칙: 파일 복사가 아니라 커널 발췌 이식.** 코드를 통째로 가져오면 VSGP의 구조 가정(파일명 결합, 이원 경로)이 따라 들어온다.

### 3.1 가져갈 것

| 자산 | 출처 | AVAP 용도 | 난이도 |
|---|---|---|---|
| blob 기하 측정 커널 `_measure_blob` | `core/blob_analyzer.py:144-189` | 측정 엔진 (순수 함수 그대로) | as-is |
| HSV inRange + morph open→close | `core/detector.py:403-433` | 단일 마스크 생성기 (재질명 하드코딩 제거) | minor |
| ROI 마스크 교집합 검출 패턴 | `detector.py:317-326` | 정렬-후-검사 뒷단 | as-is |
| 백분위 임계 추천 `[p5−0.5σ, p95+0.5σ]` + clamp + 15/30% severity | `core/advisor.py:209-274` | "OK 샘플 N장 자동 임계" (입력을 OK 라벨 세트로 교체 — 순환 참조 절단) | minor |
| GF bead 연속성(최대 연결성분 비율)·TG M패턴(3분할+비대칭) | `ok_ng_judge.py:357-493` | 형상 검사 알고리즘 (자체 HSV 재검출 제거, 마스크 입력형으로) | minor |
| run 지문 블록 (모드·sha·표본 해시·seed) | `batch_validate.py:73-101` | **모든** 결과의 필수 provenance | as-is |
| 검증 하네스 골격 (혼동행렬·그룹별 FP/FN·NG recall) | `batch_validate.py` | golden-set 회귀 측정기 (GT를 사이드카 매니페스트로) | minor |
| 판정근거 3단 출력 + 임계 대비 위치 바 | `explain_verdict.py:33-46, 98-162` | Evidence 포매터로 격하 (재계산 경로 소멸) | minor |
| 설정 도구 5중 안전장치 (중앙값→미리보기→표본 게이트→경고→근거 기록) | `roi_tuner.py` | 캘리브레이션 도구 표준 절차 | minor |
| 변경 이력 CSV + 급변 플래그 + utf-8-sig | `setting_log.py` | recipe 변경 감사 추적 (임계는 공유 상수 모듈로 이전) | as-is |
| 좌표 매핑 수렴 함수 + 커서 줌 + 드래그 이벤트 골격 + 줌 불변 핸들 | `ui/image_panel.py:302-361, 468-508` | ROI/앵커 편집기 기반 | minor |
| 워커 패턴 (QThread + 디바운스 + 배치 취소) | `ui/main_window.py:52-154` | 스냅샷 전달 + dirty coalescing으로 수정 후 이식 | minor |
| 다크 테마 QSS + 3분할 QSplitter + LabeledSlider | `ui/styles.py` 외 | 파일째/골격 이식 | as-is |
| 엔진 상태 원인별 라벨 UX (7종 status) | `main_window.py:359-385` | 정렬/검출 상태 표시로 일반화 | minor |
| 계약 고정 테스트 패턴 (TEST 6/10/13/14/15) | `tests/test_core.py` | "사후 고정"→"설계 시점 스펙"으로 승격 | minor |
| CI 구조 (quick-check·core-tests·no-binary-leak) + 최소 설치 = CI = 문서 단일 진실 | `.github/workflows/ci.yml` | 계승 (accuracy job은 §4.8 방식으로 교체) | as-is |
| frozen EXE 교훈 일습 (부팅 순서·diag_log·resource_path·build_env_check·CP949·BOM CSV) | `main.py`, `utils/`, `docs/BUILD.md` | 1일차부터 내장 | as-is |
| 합성 생성기 기법 (주입 RNG·시나리오 seed·임계 기준 색 설계) | `demo_generator.py` | pose 랜덤화 + GT 사이드카로 재작성 | major |
| OneDrive mtime 불신 + 파일명 촬영시각 파싱 | `roi_tuner.py:51-71` | 표본 선정 유틸 | as-is |

### 3.2 버릴 것 (구조적 폐기)

| 폐기 대상 | 이유 |
|---|---|
| blob_spec/blob_spec_color 이중 스펙 + 병합·역병합 기계 | 상처 조직. AVAP는 런타임 병합·역병합 자체를 금지 |
| YOLO bbox→채운 사각형→기하 측정 경로 | 측정을 거짓으로 만든 원인. 모델 도입 시 seg 마스크만 기하에 연결 |
| 파일명 정규식 Position 라우팅 (판정·튜닝·UI 전반) | recipe 명시 선택 + 정렬로 대체. 파일명은 참고 메타데이터 |
| ROI Position 우회 판정 경로 (`_feature_mask` 병렬 HSV) | 마스크 생성기는 시스템에 정확히 1개 |
| "ROI 있으면 품질검사 스킵" 조건부 게이트 | 규칙 스킵은 금지 — 스킵이 필요하면 Evidence(skipped+사유)로 기록 |
| 측정=라벨링(거르지 않는 필터) 의미론 | 측정→**실제 필터**→판정 한 방향으로 수리 |
| QLabel 풀프레임 재렌더 파이프라인 | 실사 수 MP에서 마우스 이벤트마다 full-frame resize. QGraphicsView로 교체 |
| 환경변수(VSGP_COLOR_ONLY) 숨은 전역 모드 | 명시적 생성자 인자로. 환경변수는 CLI 경계에서 번역만 |
| SKIP=PASS 테스트 러너 + 수동 등록 리스트 | pytest 전환. SKIP은 CI에서 실패 |
| 항상 skip인데 녹색인 accuracy job | 데이터 없는 측정 job의 성공 표시 금지 (§4.8) |
| 사고 동작을 무기한 고정하는 characterization 테스트 | AVAP에서는 같은 상태가 **실패하는** 테스트여야 함 |

---

## 4. AVAP 설계 (3안 통합 + 교차 비판 반영)

통합 기준(교차 비판 판정): **뼈대는 안1(MVP-first)**, 안2에서 공유 상수 모듈·앵커 편집 UX·캘리브레이션 철학, 안3에서 run 블록의 타입 수준 강제·required/forbidden 존·benchmark_kind·실측 게이트 규율을 채택. 플러그인 인프라·v1 폴백 체인·파라미터 3중 선언·도구 6종은 기각.

### 4.1 설계 법칙 (Design Laws — 테스트가 강제, 위반 시 CI 실패)

| # | 법칙 | 강제 수단 |
|---|---|---|
| L1 | **노출된 모든 recipe 파라미터는 판정에 실제 영향을 준다** | recipe 스키마를 pytest가 parametrize 순회. 파라미터마다 감도 프로브(합성 golden + 극단값 → **verdict 플립을 반드시 1회 포함**) 필수. 프로브 없거나 불변이면 수집 단계 실패. "중간값 변화"는 인정 안 함(게이밍 방지 — 기계 검증) |
| L2 | **UI·엔진·Advisor는 단일 파라미터 명세를 공유한다** | 파라미터 명세는 recipe 모듈 내장 스키마 상수(PARAM_SPECS) 한 곳. 슬라이더 생성·값 수집·추천 적용 전부 여기서 파생. 미등록 키 적용은 예외로 즉사 |
| L3 | **임계 상수는 한 곳에만 정의된다** | 공유 상수 모듈(`constants.py`) + 리터럴 중복 grep 테스트 |
| L4 | **정렬 불변성** | 같은 이미지의 이동+회전 변형본(±20px·±2°)이 원본과 동일 verdict — 합성 전 장 CI 상시 |
| L5 | **수치는 지문 없이 존재할 수 없다** | run 블록(recipe sha·이미지 sha·pose·benchmark_kind)을 `InspectionRecord` 생성자의 **필수 필드**로 — 타입 수준 강제 |
| L6 | **단일 마스크 생성기** | 판정·오버레이·캘리브레이션 도구 전부 `detect.make_mask` 하나만 import — 병렬 검출 경로 grep 테스트 |
| L7 | **단위는 전 필드 0~1 분수** | 스키마 검증에 포함 (VSGP의 %/분수 20배 혼선 차단) |
| 메타룰 | 문서화된 "파일 간 계약"이 3개를 넘으면 설계 리뷰 트리거 — 계약 수는 설계 부채의 계기판 | CLAUDE.md 운영 규칙 |

### 4.2 파이프라인과 패키지 구조

```
이미지 로드(imread_u) → 정렬(align) ─실패→ UNKNOWN + 원인 코드
                          │성공(pose: tx,ty,θ)
                          ▼
        golden 좌표 ROI를 pose로 이미지에 역투영(fillPoly)
                          ▼
        make_mask(단일 생성기: HSV inRange + morph + ROI ∩)
                          ▼
        measure(기하) → filter(실제 거름) → tools(규칙 평가)
                          ▼
        InspectionRecord{verdict, align, evidences[], run(필수)}
                          ▼
        GUI 패널 / CLI explain / JSON / CSV = 전부 이 레코드의 포매터
```

```
avap/
  constants.py     공유 상수 (급변 임계 15/30% 등 — L3)
  io_utils.py      imread_u/imwrite_u (np.fromfile+imdecode), SUPPORTED_EXTS 단일 정의
  recipe.py        로드·스키마 검증·frozen 스냅샷·sha12 지문·PARAM_SPECS 내장
  align.py         2-앵커 NCC 정렬 (§4.4)
  detect.py        단일 마스크 생성기 (L6)
  measure.py       blob 기하 측정 (VSGP 커널 as-is)
  tools.py         도구 4종 = 이름 붙은 순수 함수 dict (§4.5)
  judge.py         오케스트레이터 → InspectionRecord (§4.6)
  report.py        run 블록·JSON·utf-8-sig CSV·한국어 텍스트 포매터
  batch.py         배치 러너 + golden-set 회귀 (GT: 사이드카 매니페스트)
  synth.py         합성 벤치마크 (랜덤 pose + GT 사이드카)
  ui/              PyQt5 3분할, QGraphicsView 뷰어, ROI/앵커 편집기, palette.py
  main.py/boot.py  부팅: diag 로거 → 엔진 선로드 슬롯(v1 no-op) → check_deps → Qt
tests/             pytest (L1~L7 + 좌표 왕복 + 한글 경로 + CP949)
docs/              BUILD.md·INCIDENTS.md 빈 틀부터 (실패 누적 형식 계승)
```

### 4.3 Recipe 모델 — 단일 파일 = 단일 소스

원칙: ① 모든 좌표는 **골든 이미지 좌표계 0~1 정규화** — 런타임에 정렬 pose로만 사상, 고정 픽셀 금지. ② 저장은 항상 **새 버전 파일 + 지문**(런타임 수정·병합 금지 — VSGP 계약 1·2의 원천 차단). ③ 소비자 없는 키는 로더가 거부.

```jsonc
{
  "avap_recipe": "1.0",
  "meta": {"recipe_id": "IDC_POS6_TG", "recipe_version": 3, "fingerprint": "a1b2c3d4e5f6"},
  "golden": {"image_sha": "9f8e7d6c5b4a", "size": [2448, 2048]},
  "alignment": {
    "method": "template_2anchor",
    "anchors": [
      {"id": "screw_boss_L", "patch": "anchors/a1.png",
       "origin": [0.08, 0.12, 0.05, 0.05], "search": [0.02, 0.06, 0.17, 0.17], "min_score": 0.7},
      {"id": "conn_frame_R", "patch": "anchors/a2.png",
       "origin": [0.85, 0.80, 0.06, 0.05], "search": [0.79, 0.74, 0.18, 0.17], "min_score": 0.7}
    ],
    "min_score_basis": {"golden_n": 0, "note": "미보정 초기값 — Phase 2에서 OK 세트 p5−마진으로 갱신"},
    "pose_gates": {"max_shift_frac": 0.05, "max_rotation_deg": 3.0,
                   "anchor_dist_tol_frac": 0.01, "scale_tol": 0.02}
  },
  "rois": [
    {"id": "tg_bead_upper", "label": "TG 상단 도포",
     "rect_golden": [0.30, 0.20, 0.35, 0.10],
     "detect": {"space": "hsv", "lower": [0, 0, 0.58], "upper": [1.0, 0.10, 0.70],
                "morph": {"kernel": "ellipse", "size": 5, "open_iter": 1, "close_iter": 2}},
     "rules": [
       {"tool": "blob",     "area_min": 0.001, "area_max": 0.5, "count_min": 1, "count_max": 2},
       {"tool": "coverage", "min": 0.05, "continuity_min": 0.6},
       {"tool": "color_stats", "expect_hsv_center": [0, 0.04, 0.64], "max_dist": 0.15}
     ]}
  ],
  "provenance": {"created_by": "...", "created_at": "2026-08-20"}
}
```

주: 앵커 패치·골든 이미지는 sha 참조로만 연결(실사 커밋 금지 유지). 지문 불일치 시 로드 거부 + 재연결 안내.

### 4.4 정렬 엔진 — 핵심 신기능

**기술 조사 결론(6개 후보 비교 — 부록 B): 2-앵커 NCC 템플릿 매칭.**
골든 이미지에서 운영자가 도포 영역 밖 안정 랜드마크 2곳(멀수록 좋음)에 박스를 치면, 각 앵커를 `cv2.matchTemplate(TM_CCOEFF_NORMED)` + 피라미드 + 제한 탐색창으로 찾고, 두 좌표쌍에서 강체 변환(tx, ty, θ)을 산출한다.

선정 이유: 구현·이해 난이도 최저(전부 OpenCV 표준 API), NCC 점수가 [0,1] 그대로 신뢰도, 전역 밝기 변화에 강건, 운영자가 만지는 것은 **앵커 박스 위치와 점수 임계 둘뿐** — 상용 검사기의 model region/acceptance threshold와 같은 멘탈 모델이라 학습 비용 ~0. ORB/AKAZE는 다이캐스트 저텍스처·반복 구조(핀 어레이) 리스크로 1차 부적합(VSGP `roi_tuner`의 "은색-위-은색 전역 탐색 실패" 실측 교훈과 동일 계열).

**교차 비판 반영 정정 2건:**
- `cv2.estimateAffinePartial2D`는 rigid가 아니라 **similarity**(스케일 포함)를 추정한다 — 2점이면 해가 정확 결정되어 NCC 위치 오차 1px이 그대로 스케일·회전 오차로 샌다. **스케일을 1로 고정한 2점 강체 닫힌형 추정**(θ = 매칭 벡터 각도차, t = 회전 후 중심 이동)으로 구현하고, 측정된 앵커 간 거리비는 `scale_tol` 게이트로만 사용한다.
- 정수 격자 NCC의 이론 바닥은 ±0.5px — NCC 응답 3×3 **파라볼라 보간**으로 서브픽셀화하고, 정밀도 목표는 ≤1px이 아닌 **≤2px / ±0.5°** 로 설정. 회전 정밀도를 위해 **앵커 최소 이격** 제약을 recipe 검증에 포함.

**신뢰도 = 4중 게이트의 논리곱** (하나라도 실패 → `alignment_status=FAILED` → 판정 **UNKNOWN**):
① 앵커별 NCC ≥ min_score ② 앵커 간 거리 일관성(±tol, 스케일 게이트 겸용) ③ 포즈 한계(|θ|≤3°, |shift|≤한계) ④ 이미지 크기 = 골든 크기.
실패는 시끄럽게: UI에 앵커 적색 + "어느 앵커가 몇 점으로 임계 얼마에 미달" 원인 코드, 이력 CSV 기록. **조용한 full-frame 폴백은 코드 경로 자체가 없다.**

**v1 감량(교차 비판 채택):** ECC 정밀화·ORB 폴백·wide-search 재시도·전처리 옵션(gradient-NCC, CLAHE)은 v1 **제외**. 정렬 노브 10개+를 비개발자에게 노출하면 "누구도 튜닝 못 할 옵션 = 죽은 파라미터의 새 이름"이다. recipe 스키마 버전업으로 추후 추가하되, 실사에서 raw-gray NCC가 조명에 흔들리면 gradient-NCC 전처리가 1순위 업그레이드 경로.

### 4.5 검출 도구 — 4종 (dict가 곧 레지스트리, 플러그인 인프라 없음)

각 도구는 `(mask, blobs, params) → Evidence[]` 순수 함수. 사용자 요구와의 대응을 명시한다:

| 도구 | 측정·판정 | 사용자 요구 대응 |
|---|---|---|
| **blob** | 측정→**실제 필터**(통과분만 셈 — VSGP 계약 9 수리)→개수 min~max. 제거 blob마다 사유(실측 vs 임계) Evidence | blob size, 발렸는지 유무 |
| **coverage** | ROI 내 마스크 픽셀 비율(0~1) + `continuity_min`(최대 연결성분 비율 — GF bead 끊김) + **required/forbidden 존**(기대 위치 미도포 / 금지 영역 도포를 각각 검출) | Capa(면적), 어디에 발렸는지 |
| **color_stats** *(교차 비판 지적 신설)* | 도포부(마스크 내부) 평균 HSV vs 기대 중심·허용 거리. **오재질 도포(TG 자리에 GF)·변색을 '미도포'와 구분** — HSV 마스크만으로는 둘 다 "빈 마스크"로 보여 같은 Evidence가 나오는 맹점의 해소. 구현은 `cv2.mean` 수준 | 색상 감지 |
| **shape_compare** *(교차 비판 지적 신설)* | 골든 도포 footprint 마스크와 현재 마스크 대조: 초과 도포 영역·부족 영역·IoU. **정렬이 있어야 가능해지는 가장 직접적인 '발림 형태' 검사** — 정렬 기능의 최대 배당 | 발리는 형태, 모양 잡아내기 |

보류 목록(명시적): M패턴 도구(VSGP Pos4 전용 요구 확정 시 — dict 구조상 추가 비용 = 함수 1개), YOLO tool(데모 이후), OCR(요구에 없음 — 기각).

### 4.6 판정 엔진 — Evidence가 1급

```
InspectionRecord {
  verdict: PASS | FAIL | UNKNOWN      # 단일 어휘 (VSGP의 OK/NG vs PASS/FAIL 이중 어휘 폐기)
  align:   {status, pose(tx,ty,θ), anchor_scores[], fail_code?}
  rois:    [{roi_id, evidences: [{rule, param, measured, threshold, op, passed, note}]}]
  run:     {recipe_sha12, recipe_version, app_version, image_sha12, pose,
            benchmark_kind: synthetic|real, timestamp}   # 필수 필드 — 없으면 생성 불가 (L5)
}
```

- UNKNOWN은 정렬 실패·검사 항목 0개(`all([])` 함정의 1급 상태화)·로드 실패에만. **불량이 아니며 색상도 구분.**
- FAIL은 어느 rule이든 `passed=false`. 규칙을 끄는 조건부 게이트 금지 — 스킵은 Evidence(skipped+사유)로.
- explain·GUI·JSON·CSV는 전부 이 레코드의 포매터 — VSGP explain_verdict가 judge 바깥에서 근거를 재구성하다 내부 조건을 문자열로 하드코딩하게 된 병렬 경로가 구조적으로 불가능해진다.

### 4.7 UI 계획 (PyQt5, 한국어)

- 3분할(QSplitter: 이미지 / recipe / 결과) + 시그널 결합, 다크 QSS 파일째 이식.
- 뷰어는 **QGraphicsView** 신규(이미지 아이템 + 뷰 공간 오버레이) — VSGP QLabel 풀프레임 재렌더는 실사 수 MP에서 불가. 좌표 매핑·드래그 골격·줌 불변 핸들만 발췌 이식.
- **편집은 항상 골든 이미지 위에서** — 검사 이미지 위 ROI는 표시 전용(pose 역투영 폴리곤). 따라서 v1에 회전 shape 편집기가 필요 없다(회전은 정렬이 흡수 — UI 복잡도를 실제로 줄이는 결정). ROI **생성·삭제**를 1급 기능으로(VSGP는 수정만 가능했음). 같은 편집기가 앵커 박스도 편집하고, **현재 이미지에서의 매칭 점수를 즉시 표시**(안2 채택 — 앵커 품질을 보며 위치를 잡는다).
- **HSV 스포이드**(교차 비판 지적 신설): 이미지에서 도포부를 드래그하면 그 영역 HSV 분포(백분위)로 lower/upper를 제안 — 6축 슬라이더 수동 조정은 VSGP에서 이미 고통이었다. 구현 반나절, 비개발자 운영 제약의 핵심.
- UNKNOWN 운영(교차 비판 지적): v1은 최소한 UNKNOWN 사유 표시 + 배치 결과에서 UNKNOWN 목록 export. 수동 정렬 오버라이드는 Phase 4.
- 워커: recipe 스냅샷(deepcopy) 전달 + busy-drop 대신 dirty 플래그 coalescing. 렌더 함수 하나를 화면·export 공용. verdict 색상은 palette.py에서 Qt hex·BGR 동시 파생.

### 4.8 테스트·CI

- **pytest 전환.** L1 감도 테스트의 본질이 스키마 순회 parametrize라 순수 스크립트로는 안 된다. SKIP은 CI에서 실패. F5용 얇은 러너만 유지.
- CI 4 job: quick-check / core-tests(최소 설치 = 문서와 단일 진실) / **synthetic-benchmark** / no-binary-leak.
- **synthetic-benchmark job**: `synth.py`가 랜덤 pose(tx,ty,θ) + GT 사이드카(정답 pose·기대 verdict)를 가진 합성 세트를 CI에서 생성 → 정렬→검출→판정 전 경로를 **매 커밋 실측**. "측정값 0건이면 실패" — VSGP의 항상-skip 녹색 job의 정반대.
- **합성 성적의 순환성 라벨링**(교차 비판 반영): golden에서 오려낸 패치를 golden 변형본에서 찾는 것은 거의 자기 매칭이라 조명·글레어라는 진짜 난제를 재지 않는다. 합성 성적은 **회귀 감지 전용**(benchmark_kind=synthetic)으로만 라벨하고, **정렬·정확도 성능 주장은 실사 실측(benchmark_kind=real)에만 근거**한다는 규칙을 문서화.
- 실사 실측은 로컬 프로토콜: run 블록 JSON을 `results/`에 커밋, 문서 인용 시 지문 12자리 병기.
- 한글 경로 픽스처(`한글폴더/테스트.jpg`) 로드, 사용자 대면 출력 CP949 일괄 인코딩 검사, 좌표 3단 왕복(widget↔image↔golden, 회전 포함) <2px — 1일차부터.

---

## 5. 교차 비판에서 걸러낸 것 (안 만들기로 한 목록)

| 기각 항목 | 출처 | 이유 |
|---|---|---|
| Tool 추상 베이스 + registry 데코레이터 + 체인 컨텍스트 | 안2 | 도구 4종에 플러그인 인프라는 추측성 추상화. dict로 충분 |
| YoloTool/OcrTool 확장 슬롯 | 안2 | OCR은 요구에 없음. YOLO는 필요 시점에 함수 1개 추가 |
| v1 정렬 폴백 체인(ECC+ORB+wide-search) + 노브 10개 | 안2·3 | 비개발자가 못 만지는 옵션 = 죽은 파라미터 재생산 |
| 파라미터 3중 선언(코드 레지스트리+recipe+프로브 id 상호참조) | 안3 | "파일 간 이중 소스"를 "파일 내 3중 소스"로 옮겼을 뿐. recipe 내장 스키마 단층으로 |
| 죽은 파라미터 3중 강제 중 "읽힘 추적 프록시" | 안1·3 | frozen dataclass와 구현 충돌·최고 비용. consumer 선언 + 감도 프로브 2중으로 충분 |
| 도구 6종 (m_pattern·bead_continuity·presence 독립) | 안3 | continuity는 coverage 옵션, presence는 coverage 존 모드로 흡수. 6→4종 |
| docs-lint CI·INCIDENTS 빈 틀을 Phase 0부터 | 안2·3 | 문서가 생기는 시점으로 미뤄도 소급 비용 0 |
| 앵커 캘리브레이션 도구화·2단 프로브 규율 v1 필수 | 안3 | 철학은 채택(min_score_basis 필드), 도구화는 Phase 4 |
| FP ≤ 5 목표 | 안3 | 근거 없는 상향. ≤10 + 미달 시 명시적 결정점(안1) |
| 2~3주 일정 | 안1 | 정렬(재사용 0)+QGraphicsView 신규를 1인 3주는 비현실. 4~6주 |

---

## 6. 정직한 한계 (세미나에서 먼저 말할 것)

1. **"용량(Capa)"은 단일 2D 이미지로 측정 불가.** AVAP가 재는 것은 면적 coverage와 프록시(면적×평균 비드폭, 도포 길이)다. 체적 측정에는 3D(스테레오/구조광) 장비가 필요하며 이는 예산 0원 범위 밖 — recipe와 문서에 선제 명시한다. 세미나 예상 질문 1순위.
2. **"기존 검사 프로그램을 대체할 만한 수준"은 실측 전까지 주장하지 않는다.** VSGP 계약 6의 교훈 그대로 — 표현 사용 조건: 실사 val 세트 실측 JSON(run 블록, benchmark_kind=real) 확보 후. 그전까지는 "대체 가능성을 검증 중인 독립 판정 엔진".
3. **UNKNOWN율은 운영 지표다.** 정렬 성공률 90% 목표의 뒷면은 UNKNOWN 10% — 데모에는 충분하나 라인 운영 기준으로는 판정 불가율이다. 실측 보고에 UNKNOWN율을 반드시 병기한다.
4. **정렬 성능의 전제(앵커 가용성·오차 범위·글레어 안정성)는 실사 검증 전까지 가정이다.** 그래서 Phase 0.5를 만들었다(§7).
5. 카메라·지그 물리 이동 시 골든 프레임 전체가 무효화된다 — v1은 정렬 실패율 상승으로만 나타나므로, 데모 기간은 운영 절차(카메라 변경 시 recipe 재생성)로 커버.

---

## 7. 단계 계획 (1인 기준 4~6주)

| Phase | 목표 | 검증 가능한 성공 기준 |
|---|---|---|
| **0** 골격·안전망 (3일) | 패키지 골격, io_utils, recipe 로더+스키마, synth.py(랜덤 pose+GT 사이드카), pytest CI | 한글 경로 로드 통과 · 합성 30장 seed 고정 바이트 동일 · 미소비 키 거부 테스트 통과 · CI 녹색+SKIP 0 |
| **0.5** 실사 사전 조사 (2~3일, *교차 비판 반영 신설*) | ① 로딩 오차 분포 실측(같은 Position 수십 장 반자동 정합 — 이동·회전 범위) ② Position별 앵커 후보 스크리닝(도포 밖·프레임 내·글레어 안정) ③ recipe 부트스트랩 CLI(GUI 없이 골든 위 클릭으로 앵커·ROI 좌표 추출) | 오차 분포 히스토그램 확보(탐색창·게이트 값의 근거) · 대상 Position에 앵커 2개 확보 또는 대안 결정 · 부트스트랩으로 recipe 1벌 생성 |
| **1** 정렬 엔진 (1~1.5주) | 2-앵커 NCC + 2점 강체 추정(스케일 고정) + 서브픽셀 보간 + 4중 게이트 + 원인 코드 | 합성(±40px·±3°): 복원 오차 ≤2px·≤0.5°, 게이트 오탐 0 · 실사 Position당 20장 정렬 성공률 ≥90%, 실패 전량 원인 코드(조용한 통과 0건) · L4 통과 |
| **2** 검출·판정 + CLI (1.5주) | detect/measure/tools/judge/report/batch. Evidence + run 블록 | L1: 전 파라미터 감도 프로브 보유·verdict 플립 포함(불변 0개) · **실사 Position 2종** 배치 실측: VSGP Color 대비 FP 개선(해당 영역 FP ≤10 — 미달 시 가설 재검토 결정점) · run 블록 부재 시 생성 불가 테스트 · explain 재계산 경로 grep 0건 |
| **3** GUI + 데모 (1.5~2주) | QGraphicsView 뷰어, 골든 위 ROI/앵커 생성·수정·삭제, HSV 스포이드, Evidence 패널, 배치+CSV | 시연: 실사 OK/NG 각 5장 파일명 무관 로드→정렬→판정→근거 한국어 표시 · 좌표 3단 왕복 <2px · 저장 시 새 버전+지문+이력 CSV(원본 불변) · 정렬 실패 시 UNKNOWN+적색 앵커+원인(FAIL 오표시 0건) · CP949 통과 |
| **4** 데모 이후 결정점 (보류 목록) | val 120 전량 실측, OK 라벨 자동 임계(Advisor 이식), ECC/ORB 폴백, M패턴, 수동 정렬 오버라이드, EXE(BUILD.md 규칙 재적용), 세미나 덱 | 각 항목 착수 전 "데모/현장 요구 중 무엇에 기여하는가" 문답(VSGP 가드레일 계승). EXE 착수 시 첫 주 스모크 빌드 + 기능별 상태 라벨 |

지연 시 감량 순서(사전 확정): ROI 생성/삭제 UI → 배치 GUI(CLI 대체) → 편집기 자체(부트스트랩 CLI + 시각 확인만).

---

## 8. Two-Track 저장소 전략 (결정 필요)

| 옵션 | 장점 | 단점 |
|---|---|---|
| **A. 신규 저장소 `AVAP`** (권장) | 회사용(VSGP)/General용(AVAP) 정체성 완전 분리 · VSGP 동결 보존이 자연스러움 · 향후 공개(General) 가능성 열림 | 저장소 생성·CI 세팅 1회 비용 · VSGP 코드 발췌 시 참조 불편(잠깐) |
| B. 현 저장소 `avap/` 서브패키지 | 발췌 이식 마찰 최소 · CI·세션 권한 그대로 | 회사 제출물에 General 코드가 섞임 · "갈아엎되 남겨둔다"는 의도와 어긋남 · 2MB 코드 상한(Critical Rule 3) 압박 |
| C. 현 저장소 별도 브랜치 | 없음에 가까움 | 브랜치는 Two-Track이 아니라 분기 — 병합 압력만 생김 |

권장: **A**. 초기 1~2주 발췌 기간만 로컬에서 VSGP를 참조 클론으로 두면 B의 장점도 흡수된다.

## 9. 미결 결정사항 (사용자 확인 필요)

1. **저장소 위치** — §8의 A/B/C. A 권장.
2. **v1 검사 대상 Position** — Phase 2 실측을 2종으로 감량했다. 후보: Pos6(TG 2개, ROI 검증 이력 풍부) + Pos1(GF, full-frame FP 12건의 원흉 — 정렬+협 ROI 효과가 가장 극적으로 보일 곳). 이 선택이 Phase 0.5 앵커 스크리닝 대상을 정한다.
3. **v1 도구 4종 확정** — blob / coverage(+continuity+존) / color_stats / shape_compare (§4.5). M패턴은 보류.
4. **"AVAP" 표기** — 프로그램 정식 명칭·파일명 규약(avap_recipe 등)을 이 이름으로 고정할지.

---

## 부록 A. 설계안 3종 요지와 채택 내역

| 안 | 렌즈 | 최종 채택 | 기각 |
|---|---|---|---|
| 1 MVP-first | 2~3주 데모 가능한 최소 일관 설계 | **통합안의 뼈대**: 도구=순수 함수 dict, 골든 위 축정렬 rect만 편집(회전은 정렬이 흡수), PARAM_SPECS 내장, 명시적 보류 목록, 감량 순서 사전 확정 | 2~3주 일정, val 120 전량 실측 기준, 읽힘 추적, ≤1px 목표 |
| 2 Tool-pipeline | 상용 검사기 job/tool 워크플로의 zero-budget 재해석 | constants.py 단일 상수 모듈, 앵커 편집기=ROI 편집기 모드+실시간 매칭 점수, min_score 캘리브레이션 철학, 계약 3개 초과 시 설계 리뷰 메타룰 | 플러그인 인프라, v1 폴백 체인 전부, 3중 상호참조 선언, 9주 로드맵 |
| 3 Validation-first | VSGP 사고 3건의 구조적 차단이 아키텍처를 주도 | run 블록의 타입 수준 강제, required/forbidden 존, benchmark_kind, 실측을 Phase 게이트로, "대체" 표현 자제 규율 | 도구 6종, 코드측 ParamDesc 레지스트리, 읽힘 추적 프록시, FP≤5, 기간 없는 계획 |

## 부록 B. 정렬 기술 후보 비교 (조사 요지)

| 후보 | IDC 적합성 | 판정 |
|---|---|---|
| **NCC 템플릿(2-앵커) + 피라미드** | 고정 카메라·평면·소오차 전제와 정확히 일치. 점수=신뢰도, 튜닝 노브 2개 | **1차 채택** |
| ORB/AKAZE + RANSAC | 다이캐스트 저텍스처·핀 어레이 반복 구조에서 위험. 파라미터 불투명 | 폴백 2차 (v1 제외) |
| findTransformECC | 서브픽셀 최고지만 수렴 반경 좁음·로컬 최적 조용한 실패 | 정밀화 옵션 (v1 제외) |
| phaseCorrelate (+log-polar) | 소각도에서 log-polar 피크 불안정, 대면적 내용 변화에 취약 | 부적합 |
| 에지/chamfer 매칭 | 턴키 API 부재·자작 비용 최대·Canny 임계가 비개발자에 최악 | 과대 비용 (gradient-NCC 절충안만 업그레이드 경로로) |
| 윤곽/Hu moments | 변환을 주지 않음·근대칭 형상에서 방향각 불안정 | 부적합 |

실사 검증 필요 가정(정렬 성능의 전제): 앵커 후보의 존재·글레어 안정성, 로딩 오차 범위(±px·±°), 조명 변화가 전역 게인 수준, 해상도 불변. → Phase 0.5에서 실측.
