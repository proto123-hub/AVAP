"""Synthetic benchmark determinism — same seed must give byte-identical files.

If this drifts, CI regression numbers become incomparable between commits
(the measured-nothing-but-green failure mode in a new costume).
"""
import json
from pathlib import Path

from avap.synth import SCENARIOS, generate_set, write_golden


def _read_all(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_same_seed_is_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_set(a, n=6, seed=42)
    generate_set(b, n=6, seed=42)
    fa, fb = _read_all(a), _read_all(b)
    assert list(fa) == list(fb)
    for name in fa:
        assert fa[name] == fb[name], f"{name}: 같은 seed인데 바이트가 다름"


def test_different_seed_differs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_set(a, n=3, seed=1)
    generate_set(b, n=3, seed=2)
    fa, fb = _read_all(a), _read_all(b)
    assert any(fa[n] != fb[n] for n in fa if n.endswith(".png"))


def test_sidecar_carries_ground_truth(tmp_path):
    generate_set(tmp_path, n=6, seed=7)
    sidecars = sorted(tmp_path.glob("*.json"))
    assert len(sidecars) == 6
    for sc in sidecars:
        d = json.loads(sc.read_text(encoding="utf-8"))
        assert d["benchmark_kind"] == "synthetic"  # 합성 성적의 실사 오인 인용 차단
        assert d["scenario"] in SCENARIOS
        assert set(d["pose"]) == {"tx", "ty", "theta_deg"}
        expected = "PASS" if d["scenario"] == "ok" else "FAIL"
        assert d["expected_verdict"] == expected
        assert (tmp_path / d["file"]).is_file()


def test_golden_write(tmp_path):
    p = write_golden(tmp_path / "golden.png")
    assert p.is_file() and p.stat().st_size > 0
