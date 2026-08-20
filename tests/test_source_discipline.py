"""Source-level rules that keep the predecessor project's pathologies out.

- io discipline: cv2.imread/imwrite only inside avap/io_utils.py (L6/io).
- constants discipline: the shared change thresholds exist once (L3).
"""
import re
from pathlib import Path

AVAP_DIR = Path(__file__).resolve().parents[1] / "avap"


def _sources(exclude: set[str] = frozenset()) -> list[Path]:
    return [p for p in AVAP_DIR.rglob("*.py") if p.name not in exclude]


def test_no_direct_cv2_image_io():
    offenders = []
    for src in _sources(exclude={"io_utils.py"}):
        text = src.read_text(encoding="utf-8")
        if re.search(r"cv2\.(imread|imwrite)\b", text):
            offenders.append(src.name)
    assert not offenders, (
        f"cv2.imread/imwrite 직접 호출 금지 (한글 경로에서 조용히 실패): {offenders} — "
        "avap.io_utils.imread_u/imwrite_u를 쓸 것"
    )


def test_change_thresholds_defined_once():
    # 0.15 / 0.30 급변 임계가 constants.py 밖에 리터럴로 재정의되면 이중 정의 사고의 재판이다.
    offenders = []
    pattern = re.compile(r"(NOTICE_CHANGE_FRAC|LARGE_CHANGE_FRAC)\s*=")
    for src in _sources(exclude={"constants.py"}):
        if pattern.search(src.read_text(encoding="utf-8")):
            offenders.append(src.name)
    assert not offenders, f"급변 임계 재정의 금지 (L3): {offenders}"
