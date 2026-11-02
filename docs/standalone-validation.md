# Independent adapter validation protocol

Run against the exact source or installed wheel under review. Preserve commands,
source identities, package metadata, device identity and failures. Do not reuse
historical shared-adapter receipts as evidence for this implementation.

## Independence and package gates

1. Build the wheel and sdist; inspect wheel METADATA and entry points. Neither
   runtime requirements nor bundle dependencies may require KVCompress.
2. Install the wheel in a fresh environment without KVCompress, vLLM or torch.
   Run `python scripts/check_standalone_install.py`; disabled entry-point loading
   must succeed while imports of all host/runtime modules are blocked.
3. Install optional Manager; enable, inspect, plan, render, disable and forget the
   PyramidKV bundle alone. Verify exclusive `vllm-ascend.kv-cache.mutation` claims.
4. In a matched host environment without KVCompress, run the entire pytest suite,
   `ruff check .`, `ruff format --check .`, and build. Preserve skipped host/device
   cases explicitly. Test conflicting direct and Manager activation separately.

## Semantic and lifecycle gates

The tests cover provider selection/budget math, GQA, threshold boundaries,
chronological tail retention, independent-head top-k materialization, transaction
identity and auxiliary MTP data. Adapter regressions cover private destination
allocation, APC suffix recomputation, deferred async frees, commit acknowledgement,
preemption/abort, mixed and reordered rows, synthetic FULL capture, layer-specific
write slots, rejected drafts, immutable correction storage and native fallback.

Run `PYRAMIDKV_RUN_NPU_TESTS=1 pytest -q tests/npu` on an available Ascend device.
The integer metadata oracle compares the local batch implementation with the
actual Ascend BlockTable kernel, including CPU and NPU sequence-length metadata.
The selection oracle independently checks the device's GQA/pooling/materialization.

## Serving gate

Use a dedicated free port/devices and the complete profile from
[installation](install-and-rollback.md), with no old extension distribution or
old activation variables. Launch from the installed distribution or the recorded
editable source. Verify actual import paths and the general plugin registration.

```bash
MODEL_PATH=/path/to/Qwen3.5-35B-A3B \
PYRAMIDKV_CONFIG="$PWD/scripts/qwen35/pyramidkv-aligned.json" \
SERVE_PORT=8011 bash scripts/qwen35/serve.sh > /tmp/pyramidkv-server.log 2>&1
# After health is ready, in a second shell:
python scripts/qwen35/smoke.py --base-url http://127.0.0.1:8011 \
  --model-path /path/to/Qwen3.5-35B-A3B --server-log /tmp/pyramidkv-server.log \
  --output /tmp/pyramidkv-smoke
```

Require expected short/long output, request-matched scheduler and both TP worker
commits, prefix-cache reuse and new MTP draft/acceptance counters. Exercise both
native and batched metadata, repeated long prompts and overlapping requests;
check for engine errors after requests finish. Stop owned processes and confirm
device release. This is functional validation, not a performance benchmark.

Keep measured request/metric/log originals in the organization's benchmark
repository when publishing them, following [original custody](benchmark-originals.md).
A PR may summarize test/functional checks and their limitations. Promotion of
quality, throughput, latency or HBM claims requires a separate fresh protocol and
canonical evidence; this architecture change makes no such claim.
