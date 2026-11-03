#!/usr/bin/env bash
# Run in the prepared pinned CANN 9.1 / torch-npu environment.
set -eo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-14B-Instruct or Qwen3.5-35B-A3B}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_VERSION=0.25.1
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_pyramidkv
export VLLM_ASCEND_PYRAMIDKV_ENABLED=${PYRAMIDKV_ENABLED:-1}
export VLLM_ASCEND_PYRAMIDKV_CONFIG="${PYRAMIDKV_CONFIG:-$script_dir/pyramidkv.json}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=4 TORCHINDUCTOR_AUTOGRAD_CACHE=0
# Dense Qwen2 does not implement Mamba state copying. This is a model-specific
# cache mode, applied identically in the enabled and disabled hybrid arms.
model_type=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_type"])' "$MODEL_PATH/config.json")
model_args=()
if [[ "$model_type" == qwen3_5_moe ]]; then
  model_args+=(--mamba-cache-mode align)
fi
exec python -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --host "${SERVE_HOST:-127.0.0.1}" --port "${SERVE_PORT:-8016}" \
  --served-model-name pyramidkv-model-comparison \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --distributed-executor-backend mp \
  --worker-cls "${PYRAMIDKV_WORKER_CLS:-vllm_ascend_pyramidkv.adapter.host_worker.PyramidKVHostWorker}" \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.8 --block-size 128 --seed 0 \
  --enable-prefix-caching --enable-chunked-prefill --async-scheduling --enforce-eager \
  --enable-prompt-tokens-details --additional-config '{"enable_cpu_binding":false}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --generation-config vllm "${model_args[@]}"
