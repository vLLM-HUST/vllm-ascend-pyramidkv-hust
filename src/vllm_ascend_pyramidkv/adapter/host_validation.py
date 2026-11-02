# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-HUST/vllm-ascend-kvcompress-hust, f337858 (see NOTICE).
"""Shared, fail-closed host checks independent of scheduler and worker state."""

from __future__ import annotations

import os
from typing import Any

import torch

from .config import ASCEND_BLOCK_SIZE
from .contracts import MethodRuntimeSpec, ModelShape


def supports_mtp2(vllm_config: Any) -> bool:
    """Gate the unqualified MTP2 path behind explicit experimental opt-in.

    Compression is armed only at the exact prompt boundary; subsequent
    speculative decode steps cannot trigger another transaction because their
    optimistic KV tail can roll back after acceptance. The attention manager
    sees physical lengths, while Mamba state remains in semantic length space.
    """
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is None:
        return False
    if os.getenv("VLLM_ASCEND_PYRAMIDKV_EXPERIMENTAL_MTP2") != "1":
        return False
    model = vllm_config.model_config
    text_config = getattr(model, "hf_text_config", None)
    return (
        getattr(speculative, "method", None) == "mtp"
        and getattr(speculative, "num_speculative_tokens", None) == 2
        and not getattr(speculative, "num_speculative_tokens_per_batch_size", None)
        and bool(getattr(model, "is_hybrid", False))
        and getattr(text_config, "model_type", None) in {"qwen3_5_text", "qwen3_5_moe_text"}
        and getattr(vllm_config.cache_config, "mamba_cache_mode", None) == "align"
    )


def common_compatibility_reasons(vllm_config: Any, runner: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    cache_config = vllm_config.cache_config
    # Prefix hits may share immutable source blocks with other requests. The
    # scheduler reserves disjoint destination blocks and swaps the requesting
    # row only after the worker has copied into them; its cache_blocks hook
    # also prevents private compacted KV from receiving semantic prefix hashes.
    block_size = int(cache_config.block_size)
    allowed_block_sizes = allowed_scheduler_block_sizes(vllm_config)
    if block_size not in allowed_block_sizes:
        reasons.append(f"logical KV block size must be one of {sorted(allowed_block_sizes)}, got {block_size}")
    if getattr(cache_config, "cache_dtype", "auto") not in {
        "auto",
        "bfloat16",
        "float16",
    }:
        reasons.append("quantized KV cache is unsupported")
    if vllm_config.speculative_config is not None and not supports_mtp2(vllm_config):
        reasons.append(
            "speculative decoding is unqualified; experimental Qwen3.5 MTP2 "
            "requires mamba_cache_mode='align' and explicit opt-in"
        )
    if vllm_config.kv_transfer_config is not None:
        reasons.append("KV transfer is unsupported")
    # The scheduler adapter freezes a request after arming its private-block
    # copy. It makes the block-table switch only after the accepted worker
    # output has acknowledged the transaction, so another in-flight batch
    # cannot decode from a half-written destination.
    model_config = vllm_config.model_config
    if bool(getattr(model_config, "is_hybrid", False)):
        text_config = getattr(model_config, "hf_text_config", None)
        model_type = str(getattr(text_config, "model_type", ""))
        if model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
            reasons.append("only Qwen3.5-style full-attention/Gated-DeltaNet hybrids are supported")
        if getattr(cache_config, "mamba_cache_mode", "none") not in {
            "none",
            "align",
        }:
            reasons.append("hybrid compression requires mamba_cache_mode='none' or 'align'")
    if bool(getattr(model_config, "use_mla", False)):
        reasons.append("MLA cache layouts are unsupported")
    if bool(getattr(model_config, "is_encoder_decoder", False)):
        reasons.append("encoder-decoder models are unsupported")
    aux_output_config = getattr(vllm_config, "aux_output_config", None)
    if bool(getattr(aux_output_config, "enable_return_routed_experts", False)):
        reasons.append("routed-expert output is unsupported on this host ABI")
    parallel = vllm_config.parallel_config
    tensor_parallel_size = int(getattr(parallel, "tensor_parallel_size", 1))
    text_config = getattr(model_config, "hf_text_config", None)
    num_kv_heads = int(getattr(text_config, "num_key_value_heads", 0))
    if tensor_parallel_size <= 0 or num_kv_heads % tensor_parallel_size:
        reasons.append("tensor parallel size must be positive and divide the model KV heads")
    for name, attr in (
        ("pipeline parallel", "pipeline_parallel_size"),
        ("data parallel", "data_parallel_size"),
        ("prefill context parallel", "prefill_context_parallel_size"),
        ("decode context parallel", "decode_context_parallel_size"),
    ):
        size = int(getattr(parallel, attr, 1))
        if size != 1:
            reasons.append(f"{name} size must be one, got {size}")
    runner_type = type(runner)
    if runner_type.__module__ != "vllm_ascend.worker.model_runner_v1" or runner_type.__name__ != "NPUModelRunner":
        reasons.append("the standard v1 Ascend NPUModelRunner is required")
    if bool(getattr(runner, "use_sparse", False)):
        reasons.append("Ascend sparse attention is unsupported")
    if bool(getattr(runner, "use_compress", False)):
        reasons.append("model-native Ascend compression is unsupported")
    return tuple(reasons)


def allowed_scheduler_block_sizes(vllm_config: Any) -> frozenset[int]:
    """Return fail-closed logical page sizes for a supported model family.

    Ascend keeps 128-token kernel cache blocks. Its hybrid cache adapter may
    promote the full-attention manager page to 1152 or 2048 tokens,
    representing one logical attention block as 9 or 16 consecutive kernel
    blocks. The scheduler's cross-group LCM alignment is validated separately.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if not bool(getattr(model_config, "is_hybrid", False)):
        return frozenset({ASCEND_BLOCK_SIZE})
    text_config = getattr(model_config, "hf_text_config", None)
    model_type = str(getattr(text_config, "model_type", ""))
    if model_type in {"qwen3_5_text", "qwen3_5_moe_text"}:
        return frozenset({ASCEND_BLOCK_SIZE, 1152, 2048})
    return frozenset({ASCEND_BLOCK_SIZE})


def find_full_attention_group(groups: Any) -> int:
    """Return the sole full-attention group, allowing Mamba companion groups."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        UniformTypeKVCacheSpecs,
    )

    # A few downstream host-contract tests intentionally use an opaque object
    # for the legacy single-group layout. Real vLLM groups always expose a
    # ``kv_cache_spec``; keep that narrow test-double compatibility without
    # weakening validation for hybrid layouts.
    if len(groups) == 1 and not hasattr(groups[0], "kv_cache_spec"):
        return 0

    full_groups: list[int] = []
    unsupported: list[str] = []
    for index, group in enumerate(groups):
        spec = group.kv_cache_spec
        specs = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if all(isinstance(item, FullAttentionSpec) for item in specs):
            full_groups.append(index)
        elif not all(isinstance(item, MambaSpec) for item in specs):
            unsupported.append(type(spec).__name__)
    if unsupported:
        raise RuntimeError(
            "Ascend KV compression supports only full-attention plus optional "
            "Mamba groups; found " + ", ".join(unsupported)
        )
    if len(full_groups) != 1:
        raise RuntimeError("Ascend KV compression requires exactly one full-attention KV group")
    return full_groups[0]


def unpack_layer_cache(layer_name: str, layer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    cache = getattr(layer, "kv_cache", None)
    if isinstance(cache, list) and len(cache) == 1:
        cache = cache[0]
    impl = getattr(layer, "impl", None)
    unpack = getattr(impl, "_unpack_kv_cache", None)
    if callable(unpack):
        k_cache, v_cache = unpack(cache)
    elif isinstance(cache, (tuple, list)) and len(cache) == 2:
        k_cache, v_cache = cache
    else:
        raise RuntimeError(f"attention layer {layer_name!r} does not expose dense Ascend K/V views")
    if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
        raise RuntimeError(f"attention layer {layer_name!r} K/V bindings are invalid")
    return k_cache, v_cache


def validate_cache_tensor(
    layer_name: str,
    kind: str,
    cache: torch.Tensor,
    model: ModelShape,
    expected_kv_heads: int | None = None,
    expected_block_size: int = ASCEND_BLOCK_SIZE,
) -> None:
    expected_tail = (
        expected_block_size,
        model.num_kv_heads if expected_kv_heads is None else expected_kv_heads,
        model.head_dim,
    )
    if cache.ndim != 4 or tuple(cache.shape[1:]) != expected_tail:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache shape {tuple(cache.shape)} does not end in {expected_tail}"
        )
    if cache.dtype not in {torch.bfloat16, torch.float16}:
        raise RuntimeError(f"attention layer {layer_name!r} {kind} cache dtype {cache.dtype} is unsupported")
    inner_stride = (expected_tail[1] * expected_tail[2], expected_tail[2], 1)
    page_elements = expected_tail[0] * inner_stride[0]
    if cache.stride()[1:] != inner_stride or cache.stride(0) < page_elements:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache must have contiguous "
            "inner blocks and a nonoverlapping page stride"
        )


def validate_method_runtime_spec(name: str, spec: MethodRuntimeSpec) -> None:
    numeric = (
        spec.compression_threshold_tokens,
        spec.required_recompute_tokens,
        spec.max_physical_num_tokens,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in numeric):
        raise ValueError(f"compression method {name!r} exposes invalid runtime limits")
    if spec.max_physical_num_tokens >= spec.compression_threshold_tokens:
        raise ValueError(f"compression method {name!r} maximum must be below its threshold")
    minimum = spec.min_output_tokens_for_compression
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise ValueError(f"compression method {name!r} exposes an invalid output threshold")


def validate_per_layer_host(runner: Any, vllm_config: Any, requires_per_layer_physical_state: bool) -> None:
    if runner is None:
        raise RuntimeError("per-layer physical state requires a bound runner")
    mode = getattr(getattr(runner, "compilation_config", None), "cudagraph_mode", None)
    mode_name = getattr(mode, "name", None)
    if mode_name == "FULL_AND_PIECEWISE":
        if not requires_per_layer_physical_state:
            raise RuntimeError("graph replay requires the method to declare per-layer physical state before capture")
        try:
            from vllm_ascend.attention.attention_v1 import (
                AscendAttentionBackendImpl,
            )
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError("per-layer graph replay backend is unavailable") from error
        if not getattr(
            AscendAttentionBackendImpl,
            "_ascend_pyramidkv_layer_aware_graph_v1",
            False,
        ):
            raise RuntimeError("per-layer graph replay requires the layer-aware FIA update seam")
    elif mode_name != "NONE":
        raise RuntimeError("per-layer physical state supports eager or FULL_AND_PIECEWISE execution only")
    if any(getattr(runner, name, False) for name in ("use_dcp", "pcp_enabled")):
        raise RuntimeError("per-layer physical state is unvalidated with context-parallel attention")
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is not None and not supports_mtp2(vllm_config):
        raise RuntimeError("per-layer physical state supports only the qualified Qwen3.5 MTP2 path")
