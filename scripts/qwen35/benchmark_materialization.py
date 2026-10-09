"""Compare complete compression methods on NPU; this is NOT a serving benchmark.

Pass an immutable checkout of the previous method.py as --baseline-source.
The baseline and candidate consume identical queries/pages in one process;
whole-cache equality includes source pages, destination padding, and MTP.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from vllm_ascend_kvcompress.methods.base import (
    CompressionRequest,
    LayerCache,
    ModelShape,
    QueryBatchSpan,
    QueryObservation,
)
from vllm_ascend_kvcompress.transaction import CompressionPlan

import vllm_ascend_pyramidkv.method as candidate


def run(args):
    spec = importlib.util.spec_from_file_location("pyramidkv_baseline", args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    if args.output.exists():
        raise ValueError("Refusing to overwrite a diagnostic")
    torch.npu.set_device(0)
    torch.manual_seed(17)
    torch.npu.manual_seed(17)
    config = json.loads(args.config.read_text())["method_config"]
    shape = ModelShape("qwen3_5_moe_text", 40, 16, 2, 256, 1000000.0, False, 64, tuple(range(3, 40, 4)))
    vconfig = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        speculative_config=SimpleNamespace(
            method="mtp", num_speculative_tokens=2, num_speculative_tokens_per_batch_size=None
        ),
    )
    records = []
    with torch.inference_mode():
        for n in args.lengths:
            source_count = (n + 127) // 128
            destination_count = 16
            caches = tuple(
                LayerCache(
                    f"model.layers.{i}.self_attn.attn" if i < 40 else "mtp.layers.0.self_attn.attn",
                    i,
                    torch.randn(source_count + destination_count, 128, 1, 256, device="npu", dtype=torch.bfloat16),
                    torch.randn(source_count + destination_count, 128, 1, 256, device="npu", dtype=torch.bfloat16),
                )
                for i in (*range(3, 40, 4), 40)
            )
            methods = [module.PyramidKVMethod(config, vconfig, shape) for module in (baseline, candidate)]
            src = tuple(torch.randperm(source_count).tolist())
            dst = tuple(range(source_count, source_count + destination_count))
            plan = CompressionPlan(n, 2048, src, dst)
            request = CompressionRequest(
                "diagnostic",
                n,
                n,
                (src,),
                (dst,),
                torch.tensor(src, device="npu", dtype=torch.int32),
                torch.tensor(dst, device="npu", dtype=torch.int32),
                plan,
            )
            queries = [torch.randn(8, 8, 256, device="npu", dtype=torch.bfloat16) for _ in caches[:-1]]
            for method in methods:
                method.bind_model_runner(SimpleNamespace(max_num_reqs=1, device=torch.device("npu:0")), caches)
                for layer, query in zip(caches[:-1], queries, strict=True):
                    method.capture_query(layer, query, (QueryBatchSpan("diagnostic", 0, 8),))
                method.complete_query_observation(
                    QueryObservation("diagnostic", plan, n, 8, shape.full_attention_layer_indices)
                )
            original = [(c.k_cache.clone(), c.v_cache.clone()) for c in caches]
            baseline_result = methods[0].compress(request)
            expected = [(c.k_cache.clone(), c.v_cache.clone()) for c in caches]
            for layer, (key, value) in zip(caches, original, strict=True):
                layer.k_cache.copy_(key)
                layer.v_cache.copy_(value)
            candidate_result = methods[1].compress(request)
            assert candidate_result == baseline_result
            assert all(
                torch.equal(layer.k_cache, key) and torch.equal(layer.v_cache, value)
                for layer, (key, value) in zip(caches, expected, strict=True)
            ), "Whole-cache mismatch"
            assert all(
                torch.equal(layer.k_cache[:source_count], key[:source_count])
                and torch.equal(layer.v_cache[:source_count], value[:source_count])
                for layer, (key, value) in zip(caches, original, strict=True)
            ), "Source cache changed"
            del original, expected
            for _ in range(5):
                for method in methods:
                    method.compress(request)
            torch.npu.synchronize()
            timings = [[], []]
            for repeat in range(args.iterations):
                for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    methods[index].compress(request)
                    torch.npu.synchronize()
                    timings[index].append(1000 * (time.perf_counter() - start))
            medians = [statistics.median(t) for t in timings]
            record = {
                "input_tokens": n,
                "whole_cache_bitwise_equal": True,
                "source_unchanged": True,
                "baseline_ms": timings[0],
                "candidate_ms": timings[1],
                "median_ms": medians,
                "median_reduction_percent": 100 * (1 - medians[1] / medians[0]),
            }
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k not in {"baseline_ms", "candidate_ms"}}), flush=True)
            del methods, caches, queries, request
            torch.npu.empty_cache()
    result = {
        "scope": "isolated operator diagnostic; no serving throughput, HBM, or task-quality claim",
        "hardware": args.hardware,
        "device_name": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "seed": 17,
        "warmups": 5,
        "iterations": args.iterations,
        "order": "alternating baseline/candidate",
        "geometry": "BF16, local 8 Q / 1 KV head, dim 256, 10 target + 1 MTP layers",
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.baseline_source,
                Path(candidate.__file__),
                Path(candidate.__file__).with_name("provider.py"),
                args.config,
                Path(__file__),
            )
        },
        "results": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hardware", required=True, help="Verified hardware inventory label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 24576, 32704])
    run(parser.parse_args())
