# vLLM Ascend PyramidKV HUST

**PyramidKV is a standalone, default-off vLLM/Ascend plugin.** Version
`0.4.0.dev0` owns its scheduler, worker, transaction and metadata adapter. It
neither installs nor imports `vllm-ascend-kvcompress-hust`, and does not need that
plugin at runtime. The general entry point is
`vllm.general_plugins:ascend_pyramidkv`; importing this package or discovering
its disabled entry point does not import PyTorch or vLLM.

The serving compatibility target remains **Qwen/Qwen3.5-35B-A3B**, BF16, TP=2,
CANN 9.1, APC, MTP2, async scheduling, chunked prefill,
`FULL_AND_PIECEWISE`, and `mamba_cache_mode=align`. Unsupported model/cache/host
seams fail closed. A manifest version range is a discovery filter, not a promise
that every vLLM/Ascend combination works. See the [support matrix](docs/support-matrix.md).

## Install and activate

Provision a matched vLLM, vLLM-Ascend, torch/torch-npu and CANN stack first.
The wheel intentionally does not ask pip to replace hardware-specific host
packages. No KVCompress package or Extension Manager is needed for direct use:

```bash
python -m pip install .
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_pyramidkv
export VLLM_ASCEND_PYRAMIDKV_ENABLED=1
export VLLM_ASCEND_PYRAMIDKV_CONFIG="$PWD/scripts/qwen35/pyramidkv-aligned.json"
export VLLM_ASCEND_PYRAMIDKV_EXPERIMENTAL_MTP2=1
# Inside the prepared host environment:
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B
PYRAMIDKV_CONFIG="$VLLM_ASCEND_PYRAMIDKV_CONFIG" bash scripts/qwen35/serve.sh
```

The launcher explicitly enables the plugin; library discovery alone does not.
The launcher retains the historical `pyramidkv.json` smoke defaults. Select
`pyramidkv-aligned.json` explicitly to use the 1368/beta-2 configuration. It uses
an included worker for the pinned host's Mamba ABI bridge; it no longer requires
the separate dev-hub worker helpers. A matched native host can override
`PYRAMIDKV_WORKER_CLS`. See [installation and rollback](docs/install-and-rollback.md)
for host preparation, direct configuration and the optional Manager path.

**Migration from 0.2:** restart the process; disable the shared KVCompress
bundle/environment; install this wheel; change the config provider to
`ascend_pyramidkv`; select the new general plugin and `VLLM_ASCEND_PYRAMIDKV_*`
variables. Keep `method: pyramidkv` and the existing `method_config` values.
Old provider names are rejected, not silently translated. Two active KV-cache
mutation adapters are rejected. Other applications may keep KVCompress installed;
PyramidKV does not use it.

**Migration from 0.3:** plugin activation and `method_config` are unchanged.
Version 0.4 removes the retired schema-v1 provider runtime: `get_provider`,
`PyramidKVContractProposal`, `provider`, `registry`, root `contracts` and
`legacy_contracts`. Algorithm tooling now imports `PyramidKVConfig`,
`PyramidKVSelection`, `select_pyramid_indices`, `materialize_pyramid_kv` and
`select_pyramid_kv` from `vllm_ascend_pyramidkv.algorithm`. There is one runtime:
the independent adapter. Historical evidence remains linked to its original source.

## Architecture and semantics

The local [adapter contract](docs/host-contract.md) covers private destination
allocation, immutable APC source blocks, query observation across prefill chunks,
acknowledged worker commits, deferred release, abort/preemption, separate semantic
RoPE/Mamba and physical attention lengths, layer-specific FULL graph slots and
MTP draft rollback. The existing GQA score aggregation, pooling, PyramidKV layer
budgets, strict prompt threshold, top-k order, trailing window and selected-page
K/V materialization are retained.

`VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA=1` opts into the migrated PR33 metadata
optimization. It preserves captured slot addresses and uses a native fallback
when the layout is unsupported. It remains off by default; it is not a speedup
claim. The [dependency audit and design](docs/superpowers/specs/2026-11-02-standalone-adapter-design.md)
records the source snapshot and excluded dependencies. [NOTICE](NOTICE) retains
Apache-2.0/MIT attribution for the owned runtime.

## Verification

```bash
# Use the host's matched torch; do not replace it with a generic wheel.
python -m pip install -e '.[test]' build
python scripts/fetch_benchmark_originals.py /tmp/pyramidkv-originals
VLLM_HUST_BENCHMARK_ORIGINALS=/tmp/pyramidkv-originals OMP_NUM_THREADS=2 pytest -q
ruff check .
ruff format --check .
python -m build
# In a clean environment with the built wheel installed:
python scripts/check_standalone_install.py
# On the prepared Ascend host:
ASCEND_RT_VISIBLE_DEVICES=0 PYRAMIDKV_RUN_NPU_TESTS=1 pytest -q tests/npu
```

The ordinary suite runs metadata/algorithm tests on CPU; tests needing the actual
host skip when it is absent. Device tests are opt-in. The NPU metadata oracle
compares integer lengths and slots against the native Ascend BlockTable kernel,
including CPU/NPU metadata, reordered batches, mixed compressed rows and stable
storage. The [validation protocol](docs/standalone-validation.md) also requires
fresh installed-wheel and serving checks. Historical receipts do not qualify
this revision, and no new quality, throughput or HBM benefit is claimed here.

## Historical evidence and ongoing evaluation

The old shared-adapter quality and performance experiments retain their exact
versions and limitations in the [canonical benchmark originals](docs/benchmark-originals.md).
The [five-primary-dataset program](docs/five-dataset-program.md),
[evaluation protocol](docs/issue-1-evaluation-protocol.md),
[experimental protocols](docs/benchmark/README.md),
[fixed-budget reproduction](docs/runtime-performance-boundary.md),
[selected-page materialization](docs/selected-page-materialization.md),
[historical serving smoke](docs/qwen35-serving-smoke.md) and
[historical joint Manager launch](docs/qwen35-managed-launch.md) remain available.
They must not be read as current installation instructions or as measurements
of the independent adapter.

See also [provenance](PROVENANCE.md), [maintainers](MAINTAINERS.md), and
[MOD metadata](MOD_METADATA.json). Performance promotion requires new exact-revision qualification.
