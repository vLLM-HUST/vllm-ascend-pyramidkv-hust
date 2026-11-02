# Standalone support boundary

The independent adapter preserves the existing narrow compatibility gate.
Distribution independence does not broaden model or host support.

| Dimension | Accepted target |
| --- | --- |
| Model | Official `Qwen/Qwen3.5-35B-A3B`, text path |
| Geometry | 40 layers; full attention at 3,7,…,39; Q/KV heads 16/2; head dim 256; rotary dim 64 |
| Precision and parallelism | BF16, TP=2, no quantization, no PP/DP/context parallelism |
| Hardware/runtime | Ascend 910B2 with prepared CANN 9.1 and matched torch-npu |
| Scheduling | APC, chunked prefill, async scheduling |
| Speculation | Qwen3.5 MTP2, explicit experimental gate, one auxiliary MTP cache |
| Graph | `FULL_AND_PIECEWISE`, checked layer-aware FIA replay |
| Cache | Standard Ascend attention, 128-token kernel pages, supported logical pages, Mamba `align` |
| Activation | Independent general plugin; direct environment or optional Manager |

The reproduced source pair is vLLM
`d0f22d2bda562156e4dbf433ce645e1769b4f804` and Ascend
`03766ac696fde5ab1980d80ca0b8543d3580c989`, with native operators prepared as in
the [historical host preparation](qwen35-serving-smoke.md). Other releases must
pass host seam and native serving checks; version metadata alone is insufficient.
The offline selection/provider APIs and small synthetic fixtures cover more
geometries but do not imply serving support for them.

The migrated batched metadata path is opt-in. Its native fallback supports
layouts excluded by that optimization within the accepted host contract; it does
not make an unsupported model/cache configuration acceptable.

Historical shared-adapter quality, capacity, performance and HBM receipts remain
in [benchmark originals](benchmark-originals.md). They are not measurements of
the independent adapter. Fresh functional validation is also not a replacement
for a full quality/performance qualification or authorization for broad promotion.
