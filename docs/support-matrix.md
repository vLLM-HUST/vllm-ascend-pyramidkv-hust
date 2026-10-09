# Support matrix

This matrix separates the recovered historical profile from the narrowly
qualified current runtime profile. CPU coverage, a provider-only NPU oracle,
and exact-stack functional serving evidence are available. The current paired
evaluation adds bounded quality, capacity, performance, and HBM measurements;
it does not establish a general performance benefit.

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
| Extension descriptor activation | Active Manifest 0.3 external method entry point; explicit shared-owner dependency and exclusive method-name claim |
| Former direct Core/Ascend host integration | Withdrawn; not a contribution path |
| Shared lifecycle owner | Confirmed: `vllm-ascend-kvcompress-hust` |
| Query observation interface | Accepted and merged in shared-host PR #9 |
| Per-layer physical-state interface | Eager stage merged in PR #10; MTP2/FULL_AND_PIECEWISE stage merged in shared-host PR #13 |
| External method implementation | Registered; covers target layers plus auxiliary MTP cache |
| Prefix-cache recompute admission | Accepted and merged in shared-host PR #12; Query/APC mismatches fail closed |
| CANN 9.1 + `Qwen/Qwen3.5-35B-A3B` target | Functional serving smoke passed on Ascend 910B2 |
| TP2/APC/MTP2/async/FULL_AND_PIECEWISE/align target | Functional 5007-token request passed; 5007-to-2048 compression committed on both TP workers |
| Candidate-wheel Manager lifecycle and SWE C4/60s | Passed: 21 successful requests, 10 matching compression transactions on both TP ranks, prefix hits, real MTP acceptance, rollback and device release; see [receipt](../evidence/current/2026-09-30-candidate-qualification.json) |
| October source-install Qwen3.5 smoke | Three successful requests; two 5312-to-2048 commits on both TP ranks, APC hit, and MTP acceptance. Direct activation passed; pinned Manager joint launch conflicts on `vllm.environment`. See [runbook](qwen35-serving-smoke.md) |
| Published 0.9.0 installation acceptance | Pending approved package-source availability; local candidate-wheel qualification does not replace it |
| FULL-graph per-layer cache-write fix | Shared PR #19; evaluated host-compatible source pinned to `19f322130e1b3841da953b1b421bcf3259e9b8dc` |
| Current-source quality regression | Explicit 1368/beta-2 profile passed separate development and holdout gates; mean F1 losses 0.59/0.26 points. Original 512/beta-20 profile failed development gate |
| Current-source capacity | Both arms completed 32704 input + 64 output tokens at concurrency 1 and 4; not a maximum-capacity or capacity-improvement claim |
| Current-source paired performance/HBM | Measurements complete: 8K/24K throughput 2.6%–10.3% lower, peak HBM about 0.4 GiB higher, peak KV-pool usage 29.12% → 21.70%; see [full receipt](../evidence/current/2026-10-09-paired-evaluation/README.md) |
| Alpha release | Pending quality, capacity, and performance evidence review; measurements available, long-input throughput regression remains, functional rollback passed |
