# AVAP — AI Vision Analysis Program

이미지 기반 **독립 비전 검사 프로그램**. 이미지를 읽어(JPEG/BMP 등 포맷 무관) 제품 위치를 정렬(fixture)하고, ROI 안에서 도포 형태·blob·coverage·색상을 검출해 스스로 PASS/FAIL을 판정한다.

- 파이프라인: `정렬(2-앵커 NCC) → golden 좌표 ROI 사상 → 단일 마스크 검출 → 측정·필터·규칙 → Evidence 내장 판정`
- 정렬 실패 = **UNKNOWN** (조용한 full-frame 폴백 없음)
- 설계 문서: [docs/DESIGN.md](docs/DESIGN.md) — 설계 법칙 L1~L7을 테스트로 강제한다

## 실행 (Phase 0)

```bash
pip install -r requirements.txt
python -m pytest -q                          # 테스트 전체
python -m avap.synth --out output/synth --golden   # 합성 벤치마크 생성 (결정적)
```

## 현재 상태

Phase 0 (골격·안전망): 유니코드 안전 이미지 IO · recipe 로더+스키마 검증(죽은 파라미터 거부) · 합성 벤치마크 생성기(랜덤 pose + GT 사이드카) · CI 3 job.

다음: Phase 0.5(실사 사전 조사 — 로딩 오차 분포·앵커 스크리닝) → Phase 1(정렬 엔진).
