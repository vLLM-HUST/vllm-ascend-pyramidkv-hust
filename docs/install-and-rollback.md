# Standalone installation, migration and rollback

## Host preparation

Install this plugin into a prepared vLLM/Ascend environment. Keep the host's
matched torch/torch-npu, Triton Ascend, CANN libraries and native operators.
Pip dependencies are empty intentionally: replacing those components during
plugin installation is unsafe. Direct activation needs neither KVCompress nor
Extension Manager. `python -m pip install .` builds the standalone distribution.

The reproduced source host is vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804`
and Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`, with the prepared native
operators and CANN 9.1 environment. Package metadata is `0.23.0+empty` and
`0.23.0.post1`; the launcher's `VLLM_VERSION=0.25.1` selects that host's ABI branch,
not its installed version. See [support boundaries](support-matrix.md).

## Public candidate artifacts

For a released package, download the wheel, sdist, provenance and checksums from
the [0.4.0rc2.dev0 GitHub prerelease](releases/0.4.0rc2.dev0.md). Verify the checksums and
install the wheel with `--no-deps` into the prepared host environment. Extract the
sdist for the launcher and configuration below. No PyPI availability is implied.

The rc2 notes distinguish its accepted C128 eager replay check from the earlier
rc1 graph/MTP2 and quality evidence. The shared model-comparison launcher defaults
to C32 and an automatic pool; reproducing the C128/558-block acceptance requires
the frozen launch and request bank linked in the rc2 evidence, not just that
launcher's defaults. Use [rc1's pinned artifacts](releases/0.4.0rc1.md) when
reproducing its separately qualified results.

## Direct launch

```bash
export VLLM_ASCEND_PYRAMIDKV_ENABLED=1
export VLLM_ASCEND_PYRAMIDKV_CONFIG="$PWD/scripts/qwen35/pyramidkv-aligned.json"
export VLLM_ASCEND_PYRAMIDKV_EXPERIMENTAL_MTP2=1
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_pyramidkv
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B
export PYRAMIDKV_CONFIG="$VLLM_ASCEND_PYRAMIDKV_CONFIG"
bash scripts/qwen35/serve.sh
```

The configuration may be a JSON file path or inline JSON:

```json
{"schema_version":1,"provider":"ascend_pyramidkv","method":"pyramidkv","method_config":{"max_capacity_prompt":1368,"min_compression_prompt_tokens":4096,"window_size":8,"kernel_size":7,"pooling":"maxpool","beta":2,"kv_cache_granularity":"kv_head","gqa_score_aggregation":"mean","merge":null}}
```

The included reproduction launcher explicitly enables the plugin, chooses a
compatible local Mamba-ABI worker, and sets the complete supported profile. It
accepts `SERVE_HOST`, `SERVE_PORT`, `ASCEND_RT_VISIBLE_DEVICES`,
`PYRAMIDKV_KV_CACHE_MEMORY_BYTES`, `PYRAMIDKV_CONFIG` and `PYRAMIDKV_WORKER_CLS`. The latter
can select a native worker on a matched host. Dev-hub helper modules are not
required. The default `pyramidkv.json` is the historical smoke profile; choose
the aligned file explicitly when reproducing that configuration.

The optional `VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA=1` enables batched metadata
with guarded native fallback. The optional
`VLLM_ASCEND_PYRAMIDKV_QWEN_GDN_LIST_COMPAT=1` is only for the documented older
list-based GDN operator ABI; keep it unset on the tensor ABI. These choices do
not broaden the supported model or execution profile.

## Optional Extension Manager

Install a compatible `vllm-hust-ext` (0.2.0.dev0 or 0.3.x). Configure and enable
only the PyramidKV bundle; there is no `requires_extensions` dependency:

```bash
vllm-hust-ext extension inspect org.vllm-hust.ascend-pyramidkv
vllm-hust-ext extension configure org.vllm-hust.ascend-pyramidkv \
  --file scripts/qwen35/pyramidkv-aligned.json
vllm-hust-ext extension enable org.vllm-hust.ascend-pyramidkv
vllm-hust-ext extension plan org.vllm-hust.ascend-pyramidkv
```

Manager launch must still supply the complete host/model profile and
`VLLM_ASCEND_PYRAMIDKV_EXPERIMENTAL_MTP2=1`. A directly supplied config environment
variable overrides Manager state. The manifest claims
`vllm-ascend.kv-cache.mutation` exclusively, matching the shared adapter's claim.

## Migration from the shared adapter

1. Stop the old service and allow workers to release their device allocations.
2. Disable `org.vllm-hust.ascend-kvcompress` in Manager if enabled; remove
   `VLLM_ASCEND_KVCOMPRESS_ENABLED`, its config/experimental variables and the
   `ascend_kvcompress` selection from the new process environment.
3. Install this distribution. KVCompress may be uninstalled if no other workload
   uses it; it is not used here even if it remains installed.
4. Change the JSON provider from `ascend_kvcompress` to `ascend_pyramidkv`; retain
   `method: pyramidkv` and all algorithm options. Replace the environment prefix
   and general plugin name as shown above.
5. Enable PyramidKV directly or through Manager, restart, and run the validation
   protocol. Old provider names fail with an explicit unsupported-provider error.

Conflict checks inspect old activation state and already-loaded old patch
markers without importing the shared package. Do not enable both plugins in the
same process. Hooks and captured graphs are process-owned; live hot replacement
is unsupported.

## Rollback

Stop the service, disable `org.vllm-hust.ascend-pyramidkv`, unset its activation
variables, and start a fresh process. For the supplied launcher use
`PYRAMIDKV_ENABLED=0` and ensure Manager does not also enable the bundle. Verify
absence of compression commits and use an uncompressed KV cache. Uninstalling
a wheel or changing an environment variable cannot unpatch an existing process.
To reproduce the old shared implementation, use its separately archived exact
versions/configuration in a fresh environment; do not mix old and new adapters.
