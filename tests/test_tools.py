from pathlib import Path

import pytest

from tools.benchmark_method import run_benchmark
from tools.validate_evidence import validate_candidate, validate_legacy, validate_method_receipt


def test_method_benchmark_is_deterministic() -> None:
    first = run_benchmark(
        tokens=4097,
        query_heads=4,
        kv_heads=2,
        head_dim=8,
        layers=3,
        iterations=2,
        warmup=0,
        seed=11,
        device="cpu",
        dtype="float32",
    )
    second = run_benchmark(
        tokens=4097,
        query_heads=4,
        kv_heads=2,
        head_dim=8,
        layers=3,
        iterations=2,
        warmup=0,
        seed=11,
        device="cpu",
        dtype="float32",
    )

    assert first["selection"] == second["selection"]
    assert first["selection"]["retained_tokens_layer_0"] < 4097
    assert len(first["timing_ms"]["pyramidkv_selection"]["samples"]) == 2


def test_method_benchmark_rejects_non_compressing_input() -> None:
    with pytest.raises(ValueError, match="exceed the default compression threshold"):
        run_benchmark(
            tokens=4096,
            query_heads=4,
            kv_heads=2,
            head_dim=8,
            layers=3,
            iterations=1,
            warmup=0,
            seed=0,
            device="cpu",
            dtype="float32",
        )


def test_checked_in_evidence_passes_validator() -> None:
    root = Path(__file__).parents[1]
    validate_candidate(root / "evidence/current/2026-09-30-candidate-qualification.json")
    validate_legacy(root / "evidence/legacy/2026-08-13-ascend910b2")
    validate_method_receipt(root / "evidence/current/2026-10-08-method-npu/result.json")
