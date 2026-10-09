# Selected-page materialization diagnostic

This candidate for [issue #1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1)
reduces the work needed to materialize PyramidKV selections. Scoring still sees the same
full K tensor, with the same arithmetic and top-k order. The method resolves each request's
page addresses once, reads only selected V rows, and reuses the first layer's selected
addresses for MTP K/V. Destination addressing is shared across layers and K/V copies.
Per-head selections, chronological tail, private destinations, and nonuniform layer lengths
are preserved. No model/hardware compatibility gate or compression budget changes.

The initial test is an **isolated operator diagnostic**, not a service throughput result.
It ran on a verified 910B2 device with BF16, local 8 query heads / 1 KV head, head dimension
256, ten full-attention layers and one MTP cache. Both implementations receive identical
queries and scrambled source pages, alternating order across 30 repetitions after five
warmups. The baseline is `47580977a2dca108054470b3132811bd690020a6`.

| Input tokens | Original median (ms) | Candidate median (ms) | Reduction |
| --- | ---: | ---: | ---: |
| 8,192 | 17.8141 | 11.4263 | 35.86% |
| 24,576 | 17.9188 | 11.4653 | 36.02% |
| 32,704 | 18.1324 | 11.6188 | 35.92% |

Whole K/V caches match bit for bit at all three lengths, including MTP and untouched
padding; source caches remain unchanged. CPU regression tests additionally cover distinct
per-head selections, scattered destinations and partial pages. The full local suite passed
78 tests with one opt-in NPU test skipped; the separate diagnostic above actually ran on NPU.

[Raw timing samples, logs, source hashes and checksums](../evidence/current/2026-10-09-selected-page-diagnostic)
are retained. Temporary allocations and individual operator events do not establish a
service HBM reduction. A fresh-process, same-input serving comparison is in progress;
this diagnostic does **not** supersede the earlier measured throughput regressions.

Reproduce in the qualified isolated runtime, while it is not processing requests:

```bash
PYTHONPATH=/path/to/candidate/src python scripts/qwen35/benchmark_materialization.py \
  --baseline-source /path/to/4758097/src/vllm_ascend_pyramidkv/method.py \
  --config scripts/qwen35/pyramidkv-aligned.json \
  --hardware Ascend910B2 \
  --output /path/to/new-diagnostic.json
```

Use the baseline's immutable checkout with the unchanged provider. The benchmark script
records hashes of both methods, the provider, configuration and itself. Revert the method
commit or point the runtime back to the baseline checkout to restore old materialization;
no cache-state migration is needed between fresh processes.
