#!/usr/bin/env bash
# Run inside the pinned, prepared Ascend environment described in the runbook.
set -eo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3.5-35B-A3B directory}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# ABI branch selector used by the pinned host, not its installed package version.
export VLLM_VERSION=0.25.1
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_kvcompress
export VLLM_ASCEND_KVCOMPRESS_ENABLED=${PYRAMIDKV_ENABLED:-1}
export VLLM_ASCEND_KVCOMPRESS_CONFIG="${PYRAMIDKV_CONFIG:-$script_dir/pyramidkv.json}"
export VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2=1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=4 TORCHINDUCTOR_AUTOGRAD_CACHE=0
exec python -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --host "${SERVE_HOST:-127.0.0.1}" --port "${SERVE_PORT:-8000}" \
  --served-model-name qwen35-pyramidkv \
  --tensor-parallel-size 2 --distributed-executor-backend mp --worker-cls pipeline_worker.Worker \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 --kv-cache-memory-bytes 8589934592 \
  --seed 17 --enable-prefix-caching --mamba-cache-mode align \
  --enable-chunked-prefill --async-scheduling --enable-prompt-tokens-details \
  --additional-config '{"enable_cpu_binding":false}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[3,6,12],"max_cudagraph_capture_size":12}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --generation-config vllm
