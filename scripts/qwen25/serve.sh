#!/usr/bin/env bash
# Run in the prepared pinned CANN 9.1 / torch-npu environment.
set -eo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-14B-Instruct}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-1}
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_VERSION=0.25.1
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_pyramidkv
export VLLM_ASCEND_PYRAMIDKV_ENABLED=${PYRAMIDKV_ENABLED:-1}
export VLLM_ASCEND_PYRAMIDKV_CONFIG="${PYRAMIDKV_CONFIG:-$script_dir/pyramidkv.json}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=4 TORCHINDUCTOR_AUTOGRAD_CACHE=0
exec python -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --host "${SERVE_HOST:-127.0.0.1}" --port "${SERVE_PORT:-8015}" \
  --served-model-name qwen25-pyramidkv \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --worker-cls "${PYRAMIDKV_WORKER_CLS:-vllm_ascend_pyramidkv.adapter.host_worker.PyramidKVHostWorker}" \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.8 --block-size 128 --seed 0 \
  --enable-prefix-caching --enable-chunked-prefill --async-scheduling --enforce-eager \
  --enable-prompt-tokens-details --additional-config '{"enable_cpu_binding":false}' \
  --generation-config vllm
