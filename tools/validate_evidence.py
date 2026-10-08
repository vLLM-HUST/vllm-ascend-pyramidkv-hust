#!/usr/bin/env python3
"""Validate the repository's structured qualification evidence.

The validator is deliberately conservative: a record can be internally
consistent and still remain unqualified until it has been produced on the
declared Ascend host.  This catches stale or self-contradictory receipts before
they are linked from release documentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "evidence/current/2026-09-30-candidate-qualification.json"
DEFAULT_LEGACY = ROOT / "evidence/legacy/2026-08-13-ascend910b2"
DEFAULT_METHOD = ROOT / "evidence/current/2026-10-08-method-npu/result.json"


def _require(mapping: dict[str, Any], key: str, expected: type) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required field: {key}")
    value = mapping[key]
    if not isinstance(value, expected):
        raise ValueError(f"{key} must be {expected.__name__}, got {type(value).__name__}")
    return value


def validate_candidate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if _require(data, "passed", bool) is not True:
        raise ValueError("candidate qualification is not marked passed")
    _require(data, "date", str)
    _require(data, "scope", str)
    model = _require(data, "model", dict)
    if model.get("id") != "Qwen/Qwen3.5-35B-A3B":
        raise ValueError("candidate model is not the qualified Qwen3.5-35B model")
    host = _require(data, "host", dict)
    if not str(host.get("cann", "")).startswith("9.1"):
        raise ValueError("candidate host must declare CANN 9.1")
    configuration = _require(data, "configuration", dict)
    expected = {
        "tp": 2,
        "prefix_caching": True,
        "mtp_draft_tokens": 2,
        "async_scheduling": True,
        "chunked_prefill": True,
        "graph_mode": "FULL_AND_PIECEWISE",
        "mamba_cache_mode": "align",
        "query_window": 8,
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"configuration.{key} must be {value!r}")
    transactions = _require(data, "transactions", dict)
    commits = _require(transactions, "scheduler_commits", int)
    acknowledgements = _require(transactions, "worker_acknowledgements", dict)
    if commits <= 0 or any(value != commits for value in acknowledgements.values()):
        raise ValueError("scheduler commits and per-worker acknowledgements disagree")
    if not data.get("limitations"):
        raise ValueError("qualification must document its limitations")


def validate_legacy(directory: Path) -> None:
    performance = json.loads((directory / "performance-eager-prefix-aggregate.json").read_text(encoding="utf-8"))
    quality = json.loads((directory / "quality-summary.json").read_text(encoding="utf-8"))
    if performance.get("evidence_complete") is not True:
        raise ValueError("legacy performance evidence is incomplete")
    for workload, metrics in performance["metrics"]["disabled"].items():
        if not isinstance(metrics, dict) or workload not in performance["metrics"]["enabled"]:
            continue
        for metric in ("request_throughput", "mean_ttft_ms", "mean_tpot_ms", "mean_e2e_ms"):
            disabled = metrics[metric]
            enabled = performance["metrics"]["enabled"][workload][metric]
            ratio = performance["enabled_to_disabled_ratios"][workload][metric]
            if abs(enabled / disabled - ratio) > 1e-9:
                raise ValueError(f"legacy ratio mismatch for {workload}.{metric}")
    if quality.get("passed") is not True:
        raise ValueError("legacy quality evidence is not marked passed")


def validate_method_receipt(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "method-level" not in _require(data, "scope", str):
        raise ValueError("method receipt scope must identify it as method-level")
    shape = _require(data, "shape", dict)
    selection = _require(data, "selection", dict)
    tokens = _require(shape, "tokens", int)
    retained = _require(selection, "retained_tokens_layer_0", int)
    if not 0 < retained < tokens:
        raise ValueError("method receipt must show a positive compressed selection")
    timing = _require(data, "timing_ms", dict)
    for name in ("baseline_full_kv_copy", "pyramidkv_selection"):
        samples = _require(_require(timing, name, dict), "samples", list)
        if not samples or any(not isinstance(value, (int, float)) or value < 0 for value in samples):
            raise ValueError(f"method receipt has invalid {name} samples")
    if not _require(data, "limitations", list):
        raise ValueError("method receipt must document its limitations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--method", type=Path, default=DEFAULT_METHOD)
    args = parser.parse_args()
    validate_candidate(args.candidate)
    validate_legacy(args.legacy)
    validate_method_receipt(args.method)
    print("evidence validation passed")


if __name__ == "__main__":
    main()
