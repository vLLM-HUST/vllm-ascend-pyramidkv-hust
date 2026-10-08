#!/usr/bin/env python3
"""Run a reproducible, method-level PyramidKV micro-benchmark.

This intentionally does not start vLLM or claim serving performance.  It is a
small smoke benchmark for comparing the cost of PyramidKV selection with a
full-K/V copy under the same synthetic tensor shape.  The JSON output is useful
when qualifying a host because it records the exact shape, dtype, device, and
selection checksum alongside the timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from vllm_ascend_pyramidkv.provider import PyramidKVAscendConfig, select_pyramid_indices

DEFAULT_CONFIG = {
    "max_capacity_prompt": 512,
    "min_compression_prompt_tokens": 4096,
    "window_size": 8,
    "kernel_size": 7,
    "pooling": "maxpool",
    "beta": 20,
    "kv_cache_granularity": "kv_head",
    "gqa_score_aggregation": "mean",
    "merge": None,
}


def _synchronize(device: torch.device) -> None:
    """Synchronize accelerator work without importing accelerator packages."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    npu = getattr(torch, "npu", None)
    if device.type == "npu" and npu is not None:
        npu.synchronize()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _timed(fn: Any, *, device: torch.device) -> float:
    _synchronize(device)
    start = time.perf_counter()
    fn()
    _synchronize(device)
    return (time.perf_counter() - start) * 1000


def run_benchmark(
    *,
    tokens: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    layers: int,
    iterations: int,
    warmup: int,
    seed: int,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    if tokens <= 4096:
        raise ValueError("tokens must exceed the default compression threshold (4096)")
    if layers <= 1 or iterations <= 0 or warmup < 0:
        raise ValueError("layers must exceed one; iterations must be positive; warmup cannot be negative")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query_heads must be a positive multiple of kv_heads")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")

    selected_device = torch.device(device)
    torch.manual_seed(seed)
    tensor_dtype = getattr(torch, dtype.removeprefix("torch."), None)
    if not isinstance(tensor_dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    config = PyramidKVAscendConfig.from_dict(DEFAULT_CONFIG)
    query = torch.randn(1, query_heads, config.window_size, head_dim, device=selected_device, dtype=tensor_dtype)
    key = torch.randn(1, kv_heads, tokens, head_dim, device=selected_device, dtype=tensor_dtype)
    value = torch.randn_like(key)

    def select() -> tuple[torch.Tensor | None, int]:
        return select_pyramid_indices(query, key, config, layer_index=0, num_hidden_layers=layers)

    def baseline_copy() -> tuple[torch.Tensor, torch.Tensor]:
        return key.clone(), value.clone()

    for _ in range(warmup):
        select()
        baseline_copy()
    _synchronize(selected_device)

    selection_ms = [_timed(select, device=selected_device) for _ in range(iterations)]
    baseline_ms = [_timed(baseline_copy, device=selected_device) for _ in range(iterations)]
    selected, retained = select()
    if selected is None:
        raise RuntimeError("benchmark input did not cross the compression boundary")

    checksum = hashlib.sha256(selected.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {
        "schema_version": 1,
        "scope": "PyramidKV method-level micro-benchmark; not end-to-end serving evidence",
        "seed": seed,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(selected_device),
        "dtype": str(tensor_dtype).removeprefix("torch."),
        "shape": {
            "tokens": tokens,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "layers": layers,
            "window_size": config.window_size,
        },
        "selection": {
            "retained_tokens_layer_0": retained,
            "retained_ratio_layer_0": retained / tokens,
            "selected_indices_sha256": checksum,
        },
        "timing_ms": {
            "baseline_full_kv_copy": {
                "median": statistics.median(baseline_ms),
                "p95": _percentile(baseline_ms, 95),
                "samples": baseline_ms,
            },
            "pyramidkv_selection": {
                "median": statistics.median(selection_ms),
                "p95": _percentile(selection_ms, 95),
                "samples": selection_ms,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4097)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(
        tokens=args.tokens,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        layers=args.layers,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
