# Runtime metadata and fixed-budget reproduction

The [independent default-budget review](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/8c8ac7b0d7288b7549c14cdbc55a323bc12c7ad4/reports/pyramidkv-metadata-net-cost-20261028/batched/README.md)
preserves both the original negative evidence and the batched metadata follow-up.
Reduced metadata host-call time and lower logical KV ownership do not establish
service acceleration or lower device HBM. The distinct fixed-budget protocol is
[pinned before measurement](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/30dfdc59b6bfc03f96ce286846096f80e590d4c9/docs/benchmark/pyramidkv-memory-pressure.md).
It must be evaluated independently from the default-budget service.

The [completed fixed-budget pressure report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/c4093ae07d49ced6ac534befb62c10498ccbd08f/reports/pyramidkv-metadata-net-cost-20261028/memory-pressure/README.md)
records the full matched matrix, recomputation counters, latency tradeoffs and
bounded conclusion. Read both reports when deciding whether to enable compression.

## Prepared-host configuration

Use the exact model, host, native operators, package identities and source set
recorded in the current benchmark capsule. Historical smoke-installation pins
identify earlier experiments. This guide is not a fresh-server installer or a
public-wheel acceptance receipt.

The batched adapter source is
[`f33785844be5bff7716c9464dd8444ef0b4b05e9`](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/commit/f33785844be5bff7716c9464dd8444ef0b4b05e9),
reviewed in [shared PR #33](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/33).
Load that source with the recorded distribution/Manager environment before using
its optional flag; setting the flag does not upgrade an installed adapter.

From this repository root, on the prepared, allocated host:

```bash
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B
export PYRAMIDKV_CONFIG="$PWD/scripts/qwen35/pyramidkv-aligned.json"
export VLLM_ASCEND_KVCOMPRESS_BATCHED_METADATA=1
export PYRAMIDKV_KV_CACHE_MEMORY_BYTES=2147483648
export PYRAMIDKV_ENABLED=1
bash scripts/qwen35/serve.sh
```

`PYRAMIDKV_KV_CACHE_MEMORY_BYTES` sets the vLLM `--kv-cache-memory-bytes`
argument per worker. Omit it for the original reservation, or set it explicitly
for a frozen budget comparison. Keep the same value and source in both arms;
use `PYRAMIDKV_ENABLED=0` in a separate fresh process for native compression OFF.
All other serving flags and the explicit aligned method profile must match.
Use the device/port allocation and shutdown procedure in the
[prepared-host runbook](qwen35-serving-smoke.md).

The measured pressure launch used the same byte-valued CLI argument directly.
The new environment switch exposes that argument for reproduction; argv checks
are not a new NPU qualification. Do not silently replace a failing budget with a
larger one or a shorter context and call it the same experiment.

## Interpreting the comparison

Record output-token throughput, latency, failures, preemptions, MTP counters,
observed HBM and KV-pool occupancy together. A preallocated pool stays reserved
when request-owned blocks are released. Explicitly reserving less KV memory is a
configuration change, so compare both arms at that same reservation before
attributing a service benefit to compression.

Unique request cache salts prevent cross-request prefix reuse. A request resumed
after preemption can still report cached tokens from its own earlier work;
preserve those counters and explain them rather than claiming an entirely
uncached pressure run. Retain all cells and both process pairs, not only a
favorable cell. Fixed-budget results do not establish maximum concurrency or
universal acceleration. The small output-consistency smoke does not replace the
original development/holdout quality evaluation or the separate issue #8 contract.
