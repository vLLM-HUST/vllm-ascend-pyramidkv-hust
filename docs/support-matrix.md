# Support matrix

This matrix separates the recovered historical profile from current active
alpha claims. CPU contract coverage is available; exact-head NPU validation is
still required before release promotion.

## Historical baseline

| Dimension | Legacy validated values |
| --- | --- |
| Device | Ascend 910B2 |
| CANN | 8.5.1 eager; 9.0 eager and validated graph modes |
| Model runner | vLLM V1 |
| Models | Llama-3-8B profile and Qwen2.5-14B profile |
| Dtype | BF16 model and KV cache |
| Attention | Dense full attention, `AscendAttentionBackend` |
| Block size | 128 |
| Parallelism | TP=1, PP=1, PCP=1, DCP=1 |
| Scheduling | Async scheduling on/off; balance scheduling off; DBO off |
| Prefill | Unchunked; chunked on CANN 9.0 |
| Prefix caching | On/off with 128-token hash blocks |
| Graph | `NONE`, `PIECEWISE`, `FULL_DECODE_ONLY` within the recorded restrictions |

The legacy default configuration used a compressed capacity of 512 tokens, an
admission threshold of 4096 tokens, an eight-token query window, max pooling,
kernel size seven, and beta 20. Prompts at or below 4096 remained on the
ordinary path; admission began above that boundary.

## Explicitly unsupported in the legacy baseline

- devices other than Ascend 910B2;
- unlisted model geometries or architectures;
- quantized models or quantized KV cache;
- sliding-window attention, speculative decoding, KV transfer, or KV offload;
- TP, PP, PCP, or DCP greater than one;
- balance scheduling, dual-batch overlap, or KNorm;
- paged-attention graph shapes, `FULL`, or `FULL_AND_PIECEWISE` graph modes;
- chunked prefill on CANN 8.5.1.

## Current repository status

| Capability | Current status |
| --- | --- |
| Package installation and metadata inspection | Available |
| Offline CPU algorithm and compatibility tests | Available |
| Current-head Qwen grouped-GQA NPU oracle | Passed on Ascend 910B2, CANN 9.0, torch-npu 2.9 |
| Extension descriptor activation | Available |
| Current vLLM-HUST host integration | Implemented on paired development branch |
| Current vLLM-Ascend-HUST integration | Implemented on paired development branch |
| CANN 9.1 used by current Ascend main | Fail-closed until exact-head NPU validation |
| Exact-head NPU correctness | Pending |
| Exact-head quality/capacity/performance | Pending |
| Alpha release | Blocked |
