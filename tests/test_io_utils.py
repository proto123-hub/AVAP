"""Unicode-path image IO — the VSGP gap (bare cv2.imread) must never return."""
import numpy as np
import pytest

from avap.io_utils import ImageIOError, imread_u, imwrite_u


def test_korean_path_roundtrip(tmp_path):
    # 한국어 Windows 현장의 실제 경로 형태 — OneDrive 한글 폴더/파일명
    p = tmp_path / "한글 폴더" / "테스트 이미지.png"
    img = np.zeros((32, 48, 3), np.uint8)
    img[:, :, 1] = 200
    imwrite_u(p, img)
    back = imread_u(p)
    assert back.shape == (32, 48, 3)
    assert (back == img).all()


def test_missing_file_raises(tmp_path):
    with pytest.raises(ImageIOError, match="파일 없음"):
        imread_u(tmp_path / "없는파일.png")


def test_unsupported_ext_raises(tmp_path):
    with pytest.raises(ImageIOError, match="확장자"):
        imread_u(tmp_path / "문서.txt")
    with pytest.raises(ImageIOError, match="확장자"):
        imwrite_u(tmp_path / "문서.txt", np.zeros((4, 4, 3), np.uint8))
