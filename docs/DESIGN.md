# AVAP 설계 문서

- 상태: 승인된 설계 (Phase 1 구현·제한된 실사 검증 완료)
- 배경: 선행 내부 프로젝트의 운영 경험(사후 분석)에서 도출된 설계 법칙을 처음부터 테스트로 강제하는 재설계

## 1. 목적과 포지션

**AVAP(AI Vision Analysis Program)** 은 이미지 기반 독립 비전 검사 프로그램이다. 이미지를 읽어(JPEG/BMP 등 포맷 무관) 제품 위치를 정렬하고, ROI 안에서 도포 형태·blob·coverage·색상을 검출해 스스로 PASS/FAIL을 판정한다.

선행 도구는 "외부 검사 장비의 setting 값을 통신 없이 추측 제안"하는 포지션이어서 제안값의 효과를 검증할 루프가 원천적으로 없었다. AVAP는 판정 주체를 자기 자신으로 옮겨 이 문제를 소멸시킨다.

제약: 예산 0원(오픈소스만) · Python + OpenCV + PyQt5 · 비개발자 운영(설치·실행 단순) · Windows/EXE 배포 · UI 한국어.

## 2. 설계 법칙 (Design Laws — 테스트가 강제, 위반 시 CI 실패)

선행 프로젝트에서 실제로 벌어진 세 가지 사고 — **죽은 파라미터**(노출됐지만 판정에 영향 없는 설정), **이원 검출 경로**(같은 대상을 서로 다른 기준으로 두 번 검출), **검증 없는 수치**(실행 조건 없이 인용되는 성능 숫자) — 를 구조적으로 차단한다.

| # | 법칙 | 강제 수단 |
|---|---|---|
| L1 | 노출된 모든 recipe 파라미터는 판정에 실제 영향을 준다 | 스키마 순회 감도 프로브(verdict 플립 1회 필수). 소비자 없는 키는 로더가 거부 |
| L2 | UI·엔진·자동 임계 추천은 단일 파라미터 명세(PARAM_SPECS)를 공유 | 슬라이더 생성·값 수집·추천 적용 전부 한 곳에서 파생 |
| L3 | 임계 상수는 `constants.py` 한 곳에만 정의 | 리터럴 중복 grep 테스트 |
| L4 | 정렬 불변성 — 이동+회전 변형본의 pose/좌표 사상, Phase 2부터 verdict까지 동일 | 합성 세트 CI 상시 |
| L5 | 수치는 run 지문(recipe sha·이미지 sha·pose·benchmark_kind) 없이 존재 불가 | InspectionRecord 생성자의 필수 필드 (타입 수준 강제) |
| L6 | 마스크 생성기는 시스템에 정확히 1개 | 병렬 검출 경로 grep 테스트 |
| L7 | 단위는 전 필드 0~1 분수 | 스키마 검증 포함 |

메타룰: 문서화된 "파일 간 계약"이 3개를 넘으면 설계 리뷰 트리거 — 계약 수는 설계 부채의 계기판.

## 3. 파이프라인

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

## 4. Recipe 모델 — 단일 파일 = 단일 소스

- 모든 좌표는 **골든 이미지 좌표계 0~1 정규화** — 런타임에 정렬 pose로만 사상, 고정 픽셀 금지.
- 저장은 항상 **새 버전 파일 + 지문** — 런타임 수정·병합 금지.
- 소비자 없는 키는 로더가 거부. 참조 골격: `recipes/sample_synth.json`.

## 5. 정렬 엔진 (핵심 기능)

**2-앵커 NCC 템플릿 매칭.** 골든 이미지에서 운영자가 도포 영역 밖 안정 랜드마크 2곳에 박스를 치면, 확정 골든에서 두 패치를 한 번 crop/cache하고 각 앵커를 `matchTemplate(TM_CCOEFF_NORMED)` + 실측 제한 탐색창으로 찾는다. 별도 patch 파일 생명주기는 두지 않는다. 두 좌표쌍에서 **스케일 1 고정 2점 강체 닫힌형 추정**으로 (tx, ty, θ)를 산출한다 (`estimateAffinePartial2D`는 similarity 추정이라 사용 금지 — NCC 위치 오차가 스케일·회전 오차로 샌다). NCC 응답 3×3 파라볼라 보간으로 서브픽셀화. Pose는 골든 프레임 중심 회전 후의 `tx, ty`(px)와 OpenCV 이미지 좌표계의 `θ`로 정의한다. 목표 정밀도 ≤2px / ±0.5°.

선정 이유: 전부 OpenCV 표준 API, NCC 점수 [0,1]이 그대로 신뢰도, 전역 밝기 변화에 강건, 운영자가 만지는 것은 앵커 박스 위치와 점수 임계 둘뿐(상용 검사기의 model region / acceptance threshold와 같은 멘탈 모델). 특징점 방식(ORB/AKAZE)은 저텍스처 금속 표면·반복 구조에서 위험해 1차 후보에서 제외.

**신뢰도 = 4중 게이트의 논리곱** (하나라도 실패 → UNKNOWN): ① 이미지 크기 = 골든 크기 ② 앵커별 NCC ≥ min_score ③ 두 앵커 거리비가 1±scale_tol ④ 포즈 한계. `max_shift_frac`은 `hypot(tx/W, ty/H) / sqrt(2)`로 정의한다. NaN/inf는 별도 `NUMERIC_NONFINITE` UNKNOWN이다. 실패는 시끄럽게 — 앵커를 UNKNOWN 무채색 + 파선 + `?` 글리프로 표시 + 원인 코드 + 이력 기록 (적색은 제품 FAIL 전용으로 예약한다. 정렬 실패는 불량이 아니라 '정보 없음'이므로 빨간 앵커는 §7의 UNKNOWN 정의와 모순된다 — `tests/test_palette.py::test_red_is_reserved_for_product_fail`가 강제). **조용한 full-frame 폴백은 코드 경로 자체가 없다.** min_score는 OK calibration 세트 점수 분포의 p5 − 마진으로 정하고 별도 validation 세트로 확인한다(백분위 철학).

v1 감량: 피라미드는 실측된 좁은 탐색창에서 이득 없이 계약만 늘리므로 제외한다. 실제 latency가 목표를 넘을 때 회귀 벤치마크와 함께 재검토한다. ECC 정밀화·특징점 폴백·wide-search·전처리 옵션도 제외 — 운영자가 튜닝할 수 없는 노브는 죽은 파라미터의 재생산이다. 조명 강건성이 부족하면 gradient-NCC 전처리가 1순위 업그레이드 경로.

## 6. 검출 도구 4종 (dict가 곧 레지스트리 — 플러그인 인프라 없음)

| 도구 | 측정·판정 |
|---|---|
| blob | 측정 → **실제 필터**(통과분만 셈) → 개수 min~max. 제거 blob마다 사유(실측 vs 임계) Evidence |
| coverage | ROI 내 마스크 비율(0~1) + continuity(최대 연결성분 — 끊김 검출) + required/forbidden 존(기대 위치 미도포 / 금지 영역 도포) |
| color_stats | 도포부 평균 HSV vs 기대 중심·허용 거리 — 오재질·변색을 '미도포'와 구분 (HSV 마스크만으로는 둘 다 빈 마스크로 보임) |
| shape_compare | 골든 도포 footprint와 현재 마스크 대조(초과·부족·IoU) — 정렬이 있어야 가능한 가장 직접적인 도포 형태 검사 |

### 6.1 blob 측정 정의 (구현 전 확정 — 재해석 금지)

blob은 파라미터가 8개로 가장 많아 정의가 흔들리면 곧바로 "죽은 파라미터"가 된다. 아래는 구현이 따라야 할 확정 규약이다.

| 항목 | 확정 |
|---|---|
| **blob의 정체** | ROI 마스크의 **8-연결 성분**(connected component). contour는 blob을 정의하지 않고, 이미 확정된 성분의 형상 기술자(둘레·볼록껍질)를 뽑는 데만 쓴다 |
| **면적 분모** | `성분 픽셀 수 / ROI 마스크 픽셀 수`. `rect_golden`의 기하 넓이가 아니라 **pose로 사상된 실제 ROI 마스크 픽셀 수** — 회전 시 잘림·보간 때문에 둘이 어긋나므로, 기하 넓이를 쓰면 L4(정렬 불변)가 깨진다 |
| **circularity** | `4π · 성분 픽셀 수 / 외곽 둘레²`, 1.0으로 클램프(픽셀화로 1 초과 가능). 구멍은 둘레에 넣지 않는다(외곽 contour만) |
| **solidity** | `성분 픽셀 수 / 볼록껍질 면적`. 분자가 구멍을 제외하므로 속이 빈 도포는 1 미만으로 떨어진다 |
| **aspect ratio** | `minAreaRect의 장변 / 단변` — 항상 ≥1, 회전 불변 |
| **ROI 경계 성분** | ROI 마스크와 AND한 뒤 남은 부분으로 측정. 경계 접촉에 예외를 두지 않는다(예외 = 조건부 게이트 = 금지) |
| **빈 마스크** | 성분 0개는 결측이 아니라 정상 측정값 `count=0`. `count_min>0`이면 FAIL(미도포). coverage의 `continuity=None` 규약과 다르다 — continuity는 정의 자체가 불가능하지만 개수는 0이 실측값이다 |
| **순서** | 측정 → 면적·형상 파라미터로 **실제 제거** → 통과분만 `count_min~count_max` 판정. 제거된 성분마다 사유 Evidence(파라미터·실측·임계·연산자)를 남긴다 |

coverage 도구와 맞물리는 두 지점 — **동일한 ROI 마스크**, **동일한 8-연결 기준** — 은 같은 값을 써야 한다. 다르면 같은 픽셀 뭉치가 도구마다 다르게 세어지는 이원 경로(L6 위반)가 된다.

**방향(orientation)은 blob이 잡지 않는다.** 장/단 비는 "가늘고 김"만 재므로, 가로로 가야 할 비드가 세로로 간 경우는 통과한다. 그 검사는 골든 footprint와 대조하는 `shape_compare`의 몫이다 — 도구별 역할을 겹치지 않게 둔다.

> `aspect_ratio_min`의 스키마 하한이 현재 `0.0`이지만 위 정의상 측정값은 항상 ≥1이므로 0~1 구간은 도달 불가능한 죽은 구간이다. blob 구현과 함께 `PARAM_SPECS`의 하한을 `1.0`으로 좁힌다.

보류 목록(착수 전 "무엇에 기여하는가" 문답 필수): 특정 패턴 전용 검사, 딥러닝 검출 tool, ECC/특징점 폴백, 자동 임계 추천 이식, EXE 패키징.

## 7. 판정 엔진 — Evidence가 1급

```
InspectionRecord {
  verdict: PASS | FAIL | UNKNOWN      # 단일 어휘
  align:   {status, pose(tx,ty,θ), anchor_scores[], fail_code?}
  rois:    [{roi_id, evidences: [{rule, param, measured, threshold, op, passed, note}]}]
  run:     {recipe_sha12, recipe_version, app_version, image_sha12, pose,
            benchmark_kind: synthetic|real, timestamp}   # 필수 — 없으면 생성 불가 (L5)
}
```

- UNKNOWN은 정렬 실패·검사 항목 0개(`all([])` 함정의 1급 상태화)·로드 실패에만. 불량이 아니며 색상도 구분.
- 규칙을 끄는 조건부 게이트 금지 — 스킵은 Evidence(skipped+사유)로 기록.
- 설명(explain)·GUI·JSON·CSV는 전부 이 레코드의 포매터 — 판정 근거를 바깥에서 재구성하는 병렬 경로가 구조적으로 불가능.

## 8. UI 계획 (PyQt5 · 한국어)

- 3분할(이미지/recipe/결과), QGraphicsView 뷰어(대형 이미지 성능).
- **편집은 항상 골든 이미지 위에서** — 검사 이미지 위 ROI는 표시 전용. 회전은 정렬이 흡수하므로 v1에 회전 shape 편집기가 필요 없다. ROI 생성·삭제 1급. 같은 편집기가 앵커도 편집하고 현재 이미지 매칭 점수를 즉시 표시.
- HSV 스포이드: 도포부를 드래그하면 그 영역 HSV 분포(백분위)로 범위 제안 — 6축 슬라이더 수동 조정 대체.
- UNKNOWN 운영: 사유 표시 + 배치 결과 UNKNOWN 목록 export.
- 워커: recipe 스냅샷(deepcopy) 전달 + dirty coalescing. 렌더 함수 하나를 화면·export 공용.

## 9. 테스트·CI

- pytest (SKIP은 CI에서 실패). CI: tests / synthetic-benchmark / no-binary-leak.
- synthetic-benchmark: 랜덤 pose + GT 사이드카 합성 세트로 Phase 1은 정렬 pose를 매 커밋 실측하고, Phase 2부터 검출·판정까지 확장한다. **측정값 0건·비유한 오차·pose 다양성 붕괴는 실패.**
- 합성 성적은 회귀 감지 전용(benchmark_kind=synthetic) — 성능 주장은 실사 실측(real)에만 근거. 실사 run JSON은 원본 경로·unit ID와 함께 비공개 보존하고, 공개 저장소에는 집계·artifact SHA-256·evidence boundary만 남긴다.
- 한글 경로 픽스처 로드, 사용자 대면 출력 CP949 인코딩 검사, 좌표 왕복(<2px) — 1일차부터.

## 10. 정직한 한계

1. **부피(volume)는 단일 2D 이미지로 측정 불가** — 면적 coverage와 프록시(면적×평균 폭)만 제공. 체적 측정은 3D 장비 영역.
2. **성능 주장은 실사 실측 후에만** — benchmark_kind=real 결과 확보 전까지 "기존 검사기 대체 수준" 표현 금지.
3. **UNKNOWN율은 운영 지표** — 실측 보고에 반드시 병기.
4. 카메라·지그 물리 이동 시 골든 프레임 전체 무효화 — 운영 절차(변경 시 recipe 재생성)로 커버.

## 11. 단계 계획

| Phase | 목표 | 검증 가능한 성공 기준 |
|---|---|---|
| 0 ✅ | 골격·안전망 — io/recipe/synth + 설계 법칙 테스트 | 한글 경로 로드 · seed 결정성 · 미소비 키 거부 · CI 녹색+SKIP 0 |
| 0.5 ✅ | 실사 사전 조사 (파일럿 공정 2종) — 로딩 오차 분포 실측 · 앵커 후보 스크리닝 (`avap/preflight.py`) | 오차 분포 확보(탐색창·게이트 값의 근거) · 공정별 앵커 2개 확보 |
| 1 ✅ | 정렬 엔진 | 합성: 복원 오차 ≤2px·≤0.5°, 게이트 오탐 0 · provisional golden 기반 실사 20장/공정 정렬 40/40, 반대 공정 승인 0/40 · L4 pose/좌표 계약 통과(verdict는 Phase 2) |
| 2 | 검출·판정 + CLI | L1 전 파라미터 프로브(불변 0개) · 파일럿 2종 실사 실측(FP 목표 미달 시 명시적 결정점) · run 블록 강제 |
| 3 | GUI + 데모 | 실사 OK/NG 시연 · 좌표 왕복 <2px · 새 버전+지문 저장 · UNKNOWN 오표시 0건 · CP949 통과 |
| 4 | 결정점 (보류 목록) | 각 항목 착수 전 기여 문답 |

지연 시 감량 순서(사전 확정): ROI 생성/삭제 UI → 배치 GUI(CLI 대체) → 편집기 자체(부트스트랩 CLI + 시각 확인만).
