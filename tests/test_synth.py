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


def test_zero_n_rejected():
    # n=0 + --golden 은 golden.png 하나로 "측정 0건" CI 검사를 통과시켰다 (외부 검증 발견).
    # 벤치마크 생성기 자체가 측정 0건 세트를 만들 수 없어야 한다.
    from avap.synth import generate_set
    import pytest
    with pytest.raises(ValueError, match="1 이상"):
        generate_set("/tmp/should_not_exist_avap", n=0)


def test_sidecar_pose_values_are_finite_numbers(tmp_path):
    # CI의 "측정 존재" 검사와 같은 기준 — pose가 null/NaN이면 측정이 아니다.
    import json, math
    from avap.synth import generate_set
    generate_set(tmp_path, n=4, seed=3)
    sidecars = sorted(tmp_path.glob("synth_*.json"))
    assert sidecars
    for sc in sidecars:
        gt = json.loads(sc.read_text(encoding="utf-8"))
        for k in ("tx", "ty", "theta_deg"):
            v = gt["pose"][k]
            assert isinstance(v, (int, float)) and not isinstance(v, bool) \
                and math.isfinite(v), f"{sc.name}.pose.{k} = {v!r}"
        assert gt["expected_verdict"] in ("PASS", "FAIL")
