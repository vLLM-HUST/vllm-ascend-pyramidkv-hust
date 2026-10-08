# Reproducible method benchmark

The repository now includes a small benchmark for the provider algorithm:

```bash
python tools/benchmark_method.py \
  --device cpu \
  --tokens 4097 \
  --query-heads 16 \
  --kv-heads 2 \
  --head-dim 256 \
  --layers 10 \
  --output /tmp/pyramidkv-method.json
```

The same command can run on a matched Ascend environment by selecting the
device exposed by that environment (for example `--device npu:0`) and the
installed torch dtype. The output records the synthetic tensor shape, PyTorch
and platform versions, selected-index checksum, retained-token ratio, and
median/P95 timings for a full K/V copy and PyramidKV selection.

This is a **method-level micro-benchmark**. It does not start vLLM, measure
request throughput, or establish quality, latency, HBM, or release support.
Those claims require paired baseline/enabled serving runs on the exact target
host and model. The candidate qualification receipt is checked separately:

```bash
python tools/validate_evidence.py
```

The validator checks that the checked-in receipts are internally consistent,
that the candidate names the exact Qwen3.5 model and qualified runtime, and
that known limitations are recorded. It cannot turn a CPU run or a historical
receipt into current-head release evidence.
