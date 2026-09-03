import hashlib
import json
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "legacy" / "2026-08-13-ascend910b2"
EXPECTED_SHA256 = {
    "provenance.json": "84b4a0fb8ce2d92f6b8fb49f22df650f7127f56c55f43e69ff987e1ffb034385",
    "performance-eager-prefix-aggregate.json": ("65a17d6bde2d350d2d4410e7fbe813d25f2c74d0b879372518efc50ee8556424"),
    "quality-summary.json": ("1a7c05d896149c08e350ad9489e72d91bff6b2d27ef31f27cb658dba794a3392"),
}


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_public_legacy_evidence_is_byte_identical_to_recorded_sources() -> None:
    for name, expected in EXPECTED_SHA256.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected


def test_performance_ratios_match_published_metrics() -> None:
    evidence = _load("performance-eager-prefix-aggregate.json")

    assert evidence["core_sha"] == "4861aab3af39e721c1b5a8b27b72c4f6bebda888"
    assert evidence["ascend_sha"] == "a0a2ffffce472220061c3e9696defa6b8ddbe63a"
    assert evidence["hardware"] == "Ascend 910B2"
    assert evidence["execution_mode"] == "EAGER"
    assert evidence["prefix_caching"] is True

    for workload in ("short", "long"):
        for metric in (
            "request_throughput",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "mean_e2e_ms",
        ):
            disabled = evidence["metrics"]["disabled"][workload][metric]
            enabled = evidence["metrics"]["enabled"][workload][metric]
            assert enabled / disabled == pytest.approx(evidence["enabled_to_disabled_ratios"][workload][metric])

    reduction = evidence["active_kv_reduction_per_7168_request"]
    assert reduction["source_blocks"] - reduction["destination_blocks"] == reduction["released_blocks"]
    assert reduction["released_blocks"] * reduction["bytes_per_qwen_bf16_block"] == reduction["released_bytes"]
    assert reduction["released_bytes"] / 2**30 == reduction["released_gib"]


def test_quality_and_provenance_share_the_same_exact_pair() -> None:
    provenance = _load("provenance.json")
    performance = _load("performance-eager-prefix-aggregate.json")
    quality = _load("quality-summary.json")

    identity = (provenance["core_sha"], provenance["ascend_sha"])
    assert (performance["core_sha"], performance["ascend_sha"]) == identity
    assert (quality["core_sha"], quality["ascend_sha"]) == identity
    assert quality["comparisons"]["llama"]["mean_drop"] == 0.8086
    assert quality["comparisons"]["qwen"]["mean_drop"] == 2.415
    assert quality["comparisons"]["llama"]["passed"] is True
    assert quality["comparisons"]["qwen"]["passed"] is True
