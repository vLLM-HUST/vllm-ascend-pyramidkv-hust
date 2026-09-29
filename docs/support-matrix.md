# Support matrix

This matrix separates the recovered historical profile from the current
capability preview. CPU coverage and a provider-only NPU oracle are available;
there is no current serving support claim.

## Release-validation target

| Dimension | Required value |
| --- | --- |
| Display name | Qwen3.5-35B |
| Exact model ID | `Qwen/Qwen3.5-35B-A3B` |
| Model family | Qwen3.5 hybrid MoE (`qwen3_5_moe_text`) |
| Dtype | BF16 |
| Runtime | CANN 9.1, Ascend, TP=2 |
| Required features | APC, MTP=2, async scheduling, chunked prefill, `FULL_AND_PIECEWISE`, `mamba_cache_mode=align` |

The Qwen2.5-14B result below is retained only as historical/provider-only
evidence. It must not be reported as the release-validation model or used to
claim that the current plugin is runnable.

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
| Historical Qwen2.5-14B grouped-GQA NPU oracle | Passed on Ascend 910B2, CANN 9.0, torch-npu 2.9; not release evidence |
| Extension descriptor activation | Blocked (`import_only`) |
| Former direct Core/Ascend host integration | Withdrawn; not a contribution path |
| Shared lifecycle owner | Confirmed: `vllm-ascend-kvcompress-hust` |
| Query observation interface | Accepted and merged in shared-host PR #9 |
| Per-layer physical-state interface | Eager stage merged in PR #10; MTP2/FULL_AND_PIECEWISE follow-up submitted as shared-host Draft PR #13 |
| External method implementation | CPU prototype covers target layers plus auxiliary MTP cache; runtime entry point intentionally unregistered |
| Prefix-cache recompute admission | Accepted and merged in shared-host PR #12; Query/APC mismatches fail closed |
| CANN 9.1 + `Qwen/Qwen3.5-35B-A3B` target | Unsupported; fail closed pending interface and serving validation |
| TP2/APC/MTP2/async/FULL_AND_PIECEWISE/align target | Code contract implemented; unsupported until exact-stack serving validation |
| Exact-head NPU correctness | Pending |
| Exact-head quality/capacity/performance | Pending |
| Alpha release | Blocked |
