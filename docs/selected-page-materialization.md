# Selected-page materialization diagnostic

This operator-level optimization for [issue #1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1)
reduces the work needed to materialize PyramidKV selections. Scoring still sees the same full K
tensor, with the same arithmetic and top-k order. The method resolves each request's page addresses
once, reads only selected V rows, and reuses the first layer's selected addresses for MTP K/V.
Destination addressing is shared across layers and K/V copies. Per-head selections, chronological
tail, private destinations, and nonuniform layer lengths are preserved. No model/hardware
compatibility gate or compression budget changes.

## Historical observations and original data

The [complete historical narrative](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4139441f928744af14f89c0cb95cd999cbd6915d/archive/mod-targeted-20261023/pyramid-selected/snapshot/docs/selected-page-materialization.md)
preserves the measurements, failures, original comparison identities and repeatability limits.
An isolated operator improvement is not a service throughput, HBM or capacity improvement.
The serving series did not establish acceleration; negative results and text differences remain
in the original evidence. Compression-off/on and old/new materialization are different comparisons.

- [Interleaved operator timing samples](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-selected-page-diagnostic/interleaved-method.json)
- [Operator diagnostic log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-selected-page-diagnostic/interleaved-method.log)
- [Initial serving comparison](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-selected-page-serving/serving-comparison.json)
- [Initial serving limitations](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-selected-page-serving/serving-limitations.json)
- [Live-cache shadow diagnostic](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-selected-page-serving/shadow-summary.json)

The separate [fresh-process ABBA report and raw archive](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/a2684c3701237c0b5a8cb6e73d25ce305d6f0e71/reports/pyramidkv-selected-page-abba-20261009)
remain canonical for that series; the initial serving comparison is not its substitute.
Archive contents are historical inputs, not executable installation instructions.

## Reproduction method

Reproduce in the qualified isolated runtime, while it is not processing requests:

```bash
PYTHONPATH=/path/to/candidate/src python scripts/qwen35/benchmark_materialization.py \
  --baseline-source /path/to/4758097/src/vllm_ascend_pyramidkv/method.py \
  --config scripts/qwen35/pyramidkv-aligned.json \
  --hardware Ascend910B2 \
  --output /path/to/new-diagnostic.json
```

Use the baseline's immutable checkout with the unchanged provider. The benchmark script records
hashes of both methods, the provider, configuration and itself. Revert the method commit or point
the runtime back to the baseline checkout to restore old materialization; no cache-state migration
is needed between fresh processes.
