# vLLM Ascend PyramidKV HUST

Owner-maintained extraction of the PyramidKV provider work preserved in the
archived vLLM-HUST and vLLM-Ascend-HUST repositories.

**Status: capability preview, not a runnable alpha.**

The package is discoverable as `org.vllm-hust.ascend-pyramidkv`, but its
Manifest 0.2 carrier is deliberately marked `import_only`. Extension Manager
inspection works and enablement fails closed. Importing the top-level package
never patches or activates vLLM.

The package retains the schema-v1 configuration, fail-closed capability
matrix, request state, CPU-testable selection semantics, and the current
Qwen2.5 grouped-GQA device oracle without claiming serving compatibility.

## Host ownership and current interface gap

The former host-side Drafts in `vllm-hust` and `vllm-ascend-hust` were
withdrawn. The confirmed shared lifecycle owner is
[`vllm-ascend-kvcompress-hust`](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust),
whose documented external method namespace is
`vllm_ascend_kvcompress.methods`. Its merged
[PR #9](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/9)
provides the accepted final-prefill query-window observation contract.

One method-neutral gap remains: `CompressionResult` can describe per-layer
physical lengths, but the shared adapter does not yet apply those lengths to
per-layer decode slots and attention metadata. PyramidKV requires different
retained lengths at different layers, so replacing them with one uniform
length would change the algorithm or expose invalid cache slots. Follow-up is
tracked in
[`vllm-ascend-kvcompress-hust#3`](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/3).
Until that public consumption path is merged, this repository publishes no
runtime provider or method entry point.

## Inspect the capability preview

```bash
python -m pip install \
  "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@9fb467447e95d753f7002b28575d6802f4347181"
python -m pip install --no-deps .
vllm-hust-ext extension inspect org.vllm-hust.ascend-pyramidkv
```

Inspection must report that the implementation is `import_only`. Do not enable
or launch it through Extension Manager; no current host version is advertised
as serving-compatible.

For standalone provider tests, install the test extra in an environment with a
host-compatible PyTorch build:

```bash
python -m pip install -e ".[test]"
pytest -q
```

On an Ascend test host, run the opt-in grouped-GQA device oracle with:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 PYRAMIDKV_RUN_NPU_TESTS=1 \
  pytest -q tests/npu/test_selection.py
```

Do not install a generic PyTorch wheel over an existing torch-npu environment.
Use the matched PyTorch supplied by the Ascend host and install this package
with `--no-deps` when appropriate.

## Current boundary

- No `vllm.general_plugins` entry point is registered.
- No `vllm_ascend.kv_cache_compression_providers` or
  `vllm_ascend_kvcompress.methods` entry point is registered.
- No import-time monkey patching is performed.
- Runtime activation is blocked while the shared host lacks per-layer physical
  state consumption.
- Historical NPU, LongBench, and performance results are supporting evidence,
  not measurements of this repository's current head.
- The current provider head has a real-device Qwen 40-to-8 GQA selection
  oracle on Ascend 910B2/CANN 9.0; this does not replace serving validation and
  will not be rerun as a substitute.
- The requested target is CANN 9.1, Qwen3.5-35B-A3B BF16, TP=2, APC, MTP=2,
  async scheduling, `FULL_AND_PIECEWISE`, and `mamba_cache_mode=align`. None of
  that combination is currently claimed as supported.
- Release promotion requires the remaining shared interface followed by
  exact-head installation, activation, serving correctness, rollback, quality,
  capacity, latency, throughput, and HBM evidence.

See:

- [provenance](PROVENANCE.md)
- [maintainers](MAINTAINERS.md)
- [support matrix](docs/support-matrix.md)
- [host contract](docs/host-contract.md)
- [install and rollback](docs/install-and-rollback.md)
- [legacy evidence inventory](evidence/legacy/README.md)
- [public historical Ascend 910B2 result](evidence/legacy/2026-08-13-ascend910b2/README.md)
- [current provider-only Ascend 910B2 oracle](evidence/current/2026-09-03-provider-npu/README.md)
