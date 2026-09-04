# vLLM Ascend PyramidKV HUST

Owner-maintained extraction of the PyramidKV provider work preserved in the
archived vLLM-HUST and vLLM-Ascend-HUST repositories.

**Status: active development alpha; exact-head NPU validation is pending.**

The package is discoverable as `org.vllm-hust.ascend-pyramidkv` and registers
`pyramidkv_ascend` in the provider namespace owned by vLLM-Ascend-HUST. It is
loaded lazily only when vLLM-HUST configuration selects that provider.
Importing the top-level package never patches or activates vLLM.

The active alpha retains the schema-v1 configuration, fail-closed capability
matrix, request state, and CPU-testable selection semantics while using the
provider-neutral transactional interfaces in the paired Core and Ascend hosts.

## Install the active alpha

```bash
python -m pip install \
  "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@9fb467447e95d753f7002b28575d6802f4347181"
python -m pip install --no-deps .
vllm-hust-ext extension inspect org.vllm-hust.ascend-pyramidkv
```

Install it into the same environment as the exact paired vLLM-HUST and
vLLM-Ascend-HUST heads. Enable compression explicitly at launch:

```bash
vllm serve MODEL \
  --kv-cache-compression-config \
  '{"provider":"pyramidkv_ascend","provider_config":{"max_capacity_prompt":512,"min_compression_prompt_tokens":4096,"window_size":8,"kernel_size":7,"pooling":"maxpool","beta":20,"kv_cache_granularity":"kv_head","gqa_score_aggregation":"mean","merge":null}}'
```

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
- No import-time monkey patching is performed.
- Runtime activation requires the paired schema-v1 Core and Ascend host branches.
- Historical NPU, LongBench, and performance results are supporting evidence,
  not measurements of this repository's current head.
- The current provider head has a real-device Qwen 40-to-8 GQA selection
  oracle; this does not replace paired-host end-to-end validation.
- Release promotion still requires exact-head real-device correctness, rollback,
  quality, capacity, and performance evidence.

See:

- [provenance](PROVENANCE.md)
- [maintainers](MAINTAINERS.md)
- [support matrix](docs/support-matrix.md)
- [host contract](docs/host-contract.md)
- [install and rollback](docs/install-and-rollback.md)
- [legacy evidence inventory](evidence/legacy/README.md)
- [public historical Ascend 910B2 result](evidence/legacy/2026-08-13-ascend910b2/README.md)
- [current provider-only Ascend 910B2 oracle](evidence/current/2026-09-03-provider-npu/README.md)
