"""Unicode-safe image IO — the only module allowed to touch image files.

cv2.imread/imwrite silently fail on non-ASCII (Korean) paths on Windows,
so every read goes through np.fromfile + cv2.imdecode and every write
through cv2.imencode + tofile. Direct cv2.imread/imwrite calls elsewhere
in avap/ are rejected by tests/test_source_discipline.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageIOError(RuntimeError):
    """Raised when an image cannot be read or written."""


def imread_u(path: str | Path) -> np.ndarray:
    """Read an image (BGR) from a path that may contain non-ASCII characters."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise ImageIOError(f"지원하지 않는 확장자: {p.suffix} ({p.name})")
    if not p.is_file():
        raise ImageIOError(f"파일 없음: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageIOError(f"이미지 디코드 실패: {p}")
    return img


def imwrite_u(path: str | Path, image: np.ndarray) -> None:
    """Write an image to a path that may contain non-ASCII characters."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ImageIOError(f"지원하지 않는 확장자: {ext} ({p.name})")
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ImageIOError(f"이미지 인코드 실패: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(p))
