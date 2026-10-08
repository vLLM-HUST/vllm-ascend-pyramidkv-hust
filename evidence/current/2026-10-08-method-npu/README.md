# Current-head NPU method benchmark — 2026-10-08

This receipt records the checked-in method-level benchmark on one exposed NPU
using the current repository head. It compares PyramidKV selection with a full
K/V tensor copy over the same synthetic Qwen3.5-like grouped-GQA shape
(4097 tokens, 16 query heads, 2 KV heads, head dimension 256, BF16).

It is a reproducibility and algorithm-cost check. It is **not** an end-to-end
vLLM serving result and must not be used to claim quality, capacity, latency,
throughput, HBM savings, or release support. The exact command, source commit,
file hashes, and structured measurements are in [`result.json`](result.json).

Run it on a matched host with:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python tools/benchmark_method.py \
  --device npu:0 --dtype bfloat16 --tokens 4097 \
  --query-heads 16 --kv-heads 2 --head-dim 256 --layers 10 \
  --iterations 3 --warmup 1 --seed 0 --output result.json
```
