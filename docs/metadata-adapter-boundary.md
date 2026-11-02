# Independent adapter ownership

English | [简体中文](metadata-adapter-boundary.zh.md)

The earlier PR26 proposal waited for a shared generic metadata interface.
Version 0.3 instead owns the complete adapter locally, following the requested
removal of installation, import and runtime dependence. See the
[dependency audit/design](superpowers/specs/2026-11-02-standalone-adapter-design.md)
and [implemented host contract](host-contract.md).

The scheduler transaction, binding/query observation, per-layer metadata,
`_metadata_tensor_cache`, `_batched_slot_storage`, immutable correction views,
graph-stable slots, MTP correction, reset and native fallback now live under
`vllm_ascend_pyramidkv.adapter`. The method uses local data classes and a direct
factory; no external-method registry or shared provider is constructed.

[Shared PR33](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/33)
remains a frozen implementation/provenance reference. This change neither
modifies that PR nor invents a shared public API. Checked hooks target vLLM/Ascend
directly, and conflicting active cache mutation adapters are rejected.

The historical protocols remain in [the protocol index](benchmark/README.md).
Their original observations do not qualify the independent adapter. Follow the
[fresh validation protocol](standalone-validation.md) before promotion.
