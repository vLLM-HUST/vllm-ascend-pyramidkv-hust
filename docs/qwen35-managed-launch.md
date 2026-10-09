# Qwen3.5 joint Manager launch

This follows the [prepared-host runbook](qwen35-serving-smoke.md) and the
[paired evaluation](issue-1-evaluation-protocol.md). The original Manager pin
`98903e4` rejects joint adapter/method plans on `vllm.environment` even though
one bundle only declares an empty environment. Manager
[PR #40](https://github.com/vLLM-HUST/extension-manager/pull/40), pinned at
`d22088cf6a45aeb7c47e39101607a00d87bf2006`, compares shared settings at their
actual merge boundaries and preserves same-key and exclusive-resource conflicts.

## Prepare the isolated Manager state

Install the pinned Manager from [installation instructions](install-and-rollback.md)
in the prepared runtime environment. The shared adapter remains pinned to the
host-compatible graph-slot fix `19f322130e1b3841da953b1b421bcf3259e9b8dc`.
The model, CANN, Torch and core/Ascend source requirements are unchanged.
Use a separate state directory so historical configuration remains reproducible:

```bash
export XDG_CONFIG_HOME="$PWD/.manager-qwen/config"
export XDG_STATE_HOME="$PWD/.manager-qwen/state"
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file scripts/qwen35/pyramidkv-aligned.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension enable org.vllm-hust.ascend-pyramidkv
unset VLLM_ASCEND_KVCOMPRESS_ENABLED VLLM_ASCEND_KVCOMPRESS_CONFIG
unset VLLMHUST_EXT_ENABLED_BUNDLES VLLM_EXTENSION_MANIFESTS VLLM_EXTENSION_BUNDLES
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
export VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2=1
export VLLM_VERSION=0.25.1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0,1
vllm-hust-ext run --dry-run -- python -c pass
```

The dry-run must include both bundle IDs and the `ascend_kvcompress` plugin.
The method configuration comes from Manager state, with budget 1368, beta 2 and
threshold 4096. A direct configuration environment variable would take precedence,
so it is explicitly removed. `VLLM_VERSION` is the host ABI selector; the actual
installed core remains 0.23.0, not 0.25.1.

## Launch and check

Keep the prepared-host CANN paths, source overlays, worker imports and runtime
environment from the earlier runbook. Launch the actual server through Manager:

```bash
vllm-hust-ext run -- python -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --host 127.0.0.1 --port 8000 --served-model-name qwen35-pyramidkv \
  --tensor-parallel-size 2 --distributed-executor-backend mp --worker-cls pipeline_worker.Worker \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 --kv-cache-memory-bytes 8589934592 \
  --seed 17 --enable-prefix-caching --mamba-cache-mode align \
  --enable-chunked-prefill --async-scheduling --enable-prompt-tokens-details \
  --additional-config '{"enable_cpu_binding":false}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[3,6,12],"max_cudagraph_capture_size":12}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --generation-config vllm > manager-server.log 2>&1
```

After `/health` succeeds, run the existing smoke on an otherwise idle server:

```bash
python scripts/qwen35/smoke.py --model-path "$MODEL_PATH" \
  --server-log manager-server.log --output /path/to/new-smoke-output
```

The smoke requires expected model text, request-matched scheduler commits and both
TP acknowledgements, prefix-cache hits and new MTP draft/acceptance counters.
HTTP 200 and saved enable intent are insufficient. Use SIGTERM on the Manager
parent to stop the process tree, then verify the port closes and the device
processes exit before another launch. This is a lifecycle/functional check, not a
new paired quality or performance measurement.

The 64 GiB pod may require the same startup-only model-file cache advice as the
paired runbook. Stop that monitor before measured requests. The archived launch
wrapper removes HTTP proxies for local readiness checks so readiness detection
cannot accidentally route through an external proxy.

## Recorded device evidence

The managed-launch receipt is stored under
[`evidence/current/2026-10-09-manager-joint-launch`](../evidence/current/2026-10-09-manager-joint-launch).
It records the exact wheel, selected process environment, Manager configuration,
server command, raw smoke requests/responses, metrics and matched commits.
Historical direct-launch and paired-evaluation archives remain unchanged.
