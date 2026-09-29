# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""External PyramidKV method for the shared Ascend compression adapter.

The shared host now provides prefix-cache recompute admission for
query-observing methods. This module is registered through the shared adapter's
public method entry-point group, without patching host classes. Extension
Manager activation is limited by exact host checks to the qualified
CANN 9.1/Qwen3.5 serving profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm_ascend_kvcompress.methods.base import (
    CompressionRequest,
    CompressionResult,
    KVCompressionMethod,
    LayerCache,
    MethodRuntimeSpec,
    ModelShape,
    QueryBatchSpan,
    QueryObservation,
)
from vllm_ascend_kvcompress.transaction import CompressionPlan

from vllm_ascend_pyramidkv.provider import (
    PyramidKVAscendConfig,
    materialize_pyramid_kv,
    select_pyramid_indices,
)

METHOD_NAME = "pyramidkv"
ASCEND_CACHE_BLOCK_SIZE = 128
QWEN35_ATTENTION_BLOCK_SIZE = 2048
TARGET_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
TARGET_MODEL_DISPLAY_NAME = "Qwen3.5-35B"
QWEN35_MODEL_TYPES = frozenset({"qwen3_5_moe_text"})
QWEN35_FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))


@dataclass(frozen=True)
class _CompletedObservation:
    slot: int
    plan: CompressionPlan
    semantic_num_tokens: int


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _maximum_retained_tokens(config: PyramidKVAscendConfig) -> int:
    past_budget = config.max_capacity_prompt - config.window_size
    minimum_past = past_budget // config.beta
    return 2 * past_budget - minimum_past + config.window_size


def _gather_paged_prefix(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    if cache.ndim != 4 or cache.shape[1] != ASCEND_CACHE_BLOCK_SIZE:
        raise RuntimeError("PyramidKV requires standard 128-token paged K/V cache")
    required = (num_tokens + ASCEND_CACHE_BLOCK_SIZE - 1) // ASCEND_CACHE_BLOCK_SIZE
    if required <= 0 or block_ids.numel() < required:
        raise RuntimeError("PyramidKV source block table is too short")
    selected = block_ids[:required].to(device=cache.device, dtype=torch.long)
    return cache.index_select(0, selected).reshape(-1, cache.shape[2], cache.shape[3])[:num_tokens]


def _write_paged_prefix(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    values: torch.Tensor,
) -> None:
    num_tokens = int(values.shape[0])
    required = (num_tokens + ASCEND_CACHE_BLOCK_SIZE - 1) // ASCEND_CACHE_BLOCK_SIZE
    if required <= 0 or block_ids.numel() < required:
        raise RuntimeError("PyramidKV destination block table is too short")
    positions = torch.arange(num_tokens, device=cache.device, dtype=torch.long)
    ids = block_ids[:required].to(device=cache.device, dtype=torch.long)
    slots = ids[positions // ASCEND_CACHE_BLOCK_SIZE] * ASCEND_CACHE_BLOCK_SIZE
    slots.add_(positions.remainder(ASCEND_CACHE_BLOCK_SIZE))
    cache.reshape(-1, cache.shape[2], cache.shape[3]).index_copy_(0, slots, values)


class PyramidKVMethod(KVCompressionMethod):
    """PyramidKV behind the public ``vllm_ascend_kvcompress`` method API."""

    def __init__(
        self,
        options: Mapping[str, Any],
        vllm_config: Any,
        model_shape: ModelShape,
    ) -> None:
        self.config = PyramidKVAscendConfig.from_dict(dict(options))
        self.vllm_config = vllm_config
        self.model_shape = model_shape
        alignment = QWEN35_ATTENTION_BLOCK_SIZE if model_shape.is_hybrid else ASCEND_CACHE_BLOCK_SIZE
        self.group_maximum = _round_up(_maximum_retained_tokens(self.config), alignment)
        self.layer_caches: tuple[LayerCache, ...] = ()
        self.materialization_caches: tuple[LayerCache, ...] = ()
        self.auxiliary_mtp_caches: tuple[LayerCache, ...] = ()
        self._layer_slots: dict[int, int] = {}
        self._request_slots: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._query_lengths: dict[tuple[str, int], int] = {}
        self._completed: dict[str, _CompletedObservation] = {}
        self._query_buffers: torch.Tensor | None = None
        self._query_scratch: torch.Tensor | None = None
        self._local_query_heads = 0

    @property
    def name(self) -> str:
        return METHOD_NAME

    @property
    def runtime_spec(self) -> MethodRuntimeSpec:
        return MethodRuntimeSpec(
            requires_private_destination=True,
            # The shared adapter compresses at ``>= threshold`` while the
            # recovered contract admits only prompts strictly above the limit.
            compression_threshold_tokens=(self.config.min_compression_prompt_tokens + 1),
            required_recompute_tokens=self.config.window_size,
            max_physical_num_tokens=self.group_maximum,
        )

    @property
    def query_window_tokens(self) -> int:
        return self.config.window_size

    @property
    def query_layer_indices(self) -> tuple[int, ...]:
        return self.model_shape.full_attention_layer_indices

    @property
    def requires_per_layer_physical_state(self) -> bool:
        return True

    def compatibility_reasons(self, runner: Any) -> tuple[str, ...]:
        reasons: list[str] = []
        shape = self.model_shape
        if shape.model_type not in QWEN35_MODEL_TYPES:
            reasons.append(f"only {TARGET_MODEL_ID} ({TARGET_MODEL_DISPLAY_NAME}) is staged for serving")
        if (
            shape.num_layers,
            shape.num_attention_heads,
            shape.num_kv_heads,
            shape.head_dim,
            shape.effective_rotary_dim,
            shape.full_attention_layer_indices,
        ) != (40, 16, 2, 256, 64, QWEN35_FULL_ATTENTION_LAYERS):
            reasons.append(f"{TARGET_MODEL_ID} attention geometry is required")

        model = self.vllm_config.model_config
        if str(getattr(model, "dtype", "")) not in {"bfloat16", "torch.bfloat16"}:
            reasons.append("BF16 model weights are required")
        if getattr(model, "quantization", None) is not None:
            reasons.append("model quantization is unsupported")
        parallel = self.vllm_config.parallel_config
        if int(getattr(parallel, "tensor_parallel_size", 1)) != 2:
            reasons.append("the staged Qwen3.5 profile requires TP=2")
        cache = self.vllm_config.cache_config
        if getattr(cache, "mamba_cache_mode", None) != "align":
            reasons.append("Qwen3.5 requires mamba_cache_mode='align'")
        if not bool(getattr(cache, "enable_prefix_caching", False)):
            reasons.append("the staged target requires prefix caching")
        speculative = getattr(self.vllm_config, "speculative_config", None)
        if speculative is None or (
            getattr(speculative, "method", None) != "mtp"
            or getattr(speculative, "num_speculative_tokens", None) != 2
            or getattr(speculative, "num_speculative_tokens_per_batch_size", None)
        ):
            reasons.append("the staged target requires Qwen3.5 MTP2")

        scheduler = self.vllm_config.scheduler_config
        if not bool(getattr(scheduler, "async_scheduling", False)):
            reasons.append("the staged target requires async scheduling")
        if not bool(getattr(scheduler, "enable_chunked_prefill", False)):
            reasons.append("the staged target requires chunked prefill")

        mode = getattr(getattr(runner, "compilation_config", None), "cudagraph_mode", None)
        if getattr(mode, "name", str(mode)) != "FULL_AND_PIECEWISE":
            reasons.append("the staged target requires FULL_AND_PIECEWISE execution")
        backend = getattr(runner, "attn_backend", None)
        backend_name = backend.__name__ if isinstance(backend, type) else type(backend).__name__
        if backend_name != "AscendAttentionBackend":
            reasons.append("standard AscendAttentionBackend is required")

        cann = str(getattr(torch.version, "cann", "") or "")
        if not cann.startswith("9.1"):
            reasons.append(f"CANN 9.1 is required, got {cann or '<unknown>'}")
        return tuple(reasons)

    def bind_model_runner(
        self,
        runner: Any,
        layer_caches: tuple[LayerCache, ...],
    ) -> None:
        expected = self.model_shape.full_attention_layer_indices
        auxiliary = tuple(layer for layer in layer_caches if _is_mtp_cache_layer(layer.name))
        target = tuple(layer for layer in layer_caches if not _is_mtp_cache_layer(layer.name))
        actual = tuple(layer.layer_index for layer in target)
        if actual != expected:
            raise RuntimeError(f"PyramidKV full-attention layer order changed: expected {expected}, got {actual}")
        if auxiliary and (
            len(auxiliary) != 1
            or auxiliary[0].name != "mtp.layers.0.self_attn.attn"
            or getattr(self.vllm_config.speculative_config, "method", None) != "mtp"
            or getattr(
                self.vllm_config.speculative_config,
                "num_speculative_tokens",
                None,
            )
            != 2
        ):
            raise RuntimeError("PyramidKV auxiliary cache requires exact Qwen3.5 MTP2")
        tensor_parallel_size = int(getattr(self.vllm_config.parallel_config, "tensor_parallel_size", 1))
        if self.model_shape.num_attention_heads % tensor_parallel_size:
            raise RuntimeError("query heads are not divisible by tensor parallel size")
        self._local_query_heads = self.model_shape.num_attention_heads // tensor_parallel_size
        expected_local_kv_heads = self.model_shape.num_kv_heads // tensor_parallel_size
        for layer in layer_caches:
            if (
                layer.k_cache.shape != layer.v_cache.shape
                or layer.k_cache.ndim != 4
                or tuple(layer.k_cache.shape[1:])
                != (
                    ASCEND_CACHE_BLOCK_SIZE,
                    expected_local_kv_heads,
                    self.model_shape.head_dim,
                )
            ):
                raise RuntimeError(f"PyramidKV layer {layer.name!r} has an incompatible K/V cache")
        max_num_reqs = int(getattr(runner, "max_num_reqs", 0))
        if max_num_reqs <= 0:
            raise RuntimeError("PyramidKV requires a positive runner request capacity")
        device = runner.device
        cache_dtype = layer_caches[0].k_cache.dtype
        shape = (
            max_num_reqs,
            len(target),
            self.config.window_size,
            self._local_query_heads,
            self.model_shape.head_dim,
        )
        self._query_buffers = torch.empty(shape, dtype=cache_dtype, device=device)
        self._query_scratch = torch.empty(shape[2:], dtype=cache_dtype, device=device)
        self.layer_caches = target
        self.materialization_caches = layer_caches
        self.auxiliary_mtp_caches = auxiliary
        self._layer_slots = {layer.layer_index: index for index, layer in enumerate(target)}
        self._free_slots = list(reversed(range(max_num_reqs)))

    def capture_query(
        self,
        layer: LayerCache,
        query: torch.Tensor,
        spans: tuple[QueryBatchSpan, ...],
    ) -> None:
        if self._query_buffers is None or self._query_scratch is None:
            raise RuntimeError("PyramidKV query buffers are not initialized")
        layer_slot = self._layer_slots.get(layer.layer_index)
        if layer_slot is None:
            raise RuntimeError(f"unbound PyramidKV layer index {layer.layer_index}")
        normalized = self._normalize_query(query)
        for span in spans:
            if not 0 <= span.start < span.end <= normalized.shape[0]:
                raise RuntimeError("PyramidKV query span is outside the batch")
            request_slot = self._request_slot(span.request_id)
            buffer = self._query_buffers[request_slot, layer_slot]
            rows = normalized[span.start : span.end]
            key = (span.request_id, layer.layer_index)
            previous = self._query_lengths.get(key, 0)
            self._query_lengths[key] = self._append_query_tail(buffer, previous, rows)

    def complete_query_observation(self, observation: QueryObservation) -> None:
        if observation.window_tokens != self.config.window_size:
            raise RuntimeError("PyramidKV observation window changed")
        expected = tuple(layer.layer_index for layer in self.layer_caches)
        if observation.layer_indices != expected:
            raise RuntimeError("PyramidKV observation layers changed")
        slot = self._request_slots.get(observation.request_id)
        if slot is None:
            raise RuntimeError("PyramidKV has no captured query for the request")
        incomplete = [
            layer.layer_index
            for layer in self.layer_caches
            if self._query_lengths.get((observation.request_id, layer.layer_index), 0) != self.config.window_size
        ]
        if incomplete:
            raise RuntimeError(f"PyramidKV did not capture a complete query window for layers {incomplete}")
        if observation.request_id in self._completed:
            raise RuntimeError("PyramidKV query observation completed twice")
        self._completed[observation.request_id] = _CompletedObservation(
            slot=slot,
            plan=observation.plan,
            semantic_num_tokens=observation.semantic_num_tokens,
        )

    def discard_query_observation(self, request_id: str) -> None:
        self._completed.pop(request_id, None)
        slot = self._request_slots.pop(request_id, None)
        for layer in self.layer_caches:
            self._query_lengths.pop((request_id, layer.layer_index), None)
        if slot is not None:
            self._free_slots.append(slot)

    def compress(self, request: CompressionRequest) -> CompressionResult:
        if self._query_buffers is None or not self.layer_caches:
            raise RuntimeError("PyramidKV method is not bound to K/V cache")
        completed = self._completed.get(request.request_id)
        if completed is None:
            raise RuntimeError("PyramidKV compression has no completed query observation")
        if request.plan is not completed.plan:
            raise RuntimeError("PyramidKV compression plan identity changed")
        if request.semantic_num_tokens != completed.semantic_num_tokens:
            raise RuntimeError("PyramidKV semantic length changed after observation")
        if request.per_layer_physical_num_tokens is not None:
            raise RuntimeError("repeat PyramidKV compression is not supported")

        per_layer: list[tuple[str, int]] = []
        auxiliary_selection: tuple[torch.Tensor, int] | None = None
        participating_layers = len(self.layer_caches)
        for order, layer in enumerate(self.layer_caches):
            key = _gather_paged_prefix(
                layer.k_cache,
                request.source_block_ids_device,
                request.physical_num_tokens,
            )
            value = _gather_paged_prefix(
                layer.v_cache,
                request.source_block_ids_device,
                request.physical_num_tokens,
            )
            query = self._query_buffers[completed.slot, self._layer_slots[layer.layer_index]]
            query = query.permute(1, 0, 2).unsqueeze(0)
            paged_key = key.permute(1, 0, 2).unsqueeze(0)
            paged_value = value.permute(1, 0, 2).unsqueeze(0)
            selected, retained = select_pyramid_indices(
                query,
                paged_key,
                self.config,
                layer_index=order,
                num_hidden_layers=participating_layers,
            )
            if selected is None:
                raise RuntimeError("PyramidKV transaction did not cross admission")
            if auxiliary_selection is None:
                auxiliary_selection = selected, retained
            compact_key, compact_value = materialize_pyramid_kv(
                paged_key,
                paged_value,
                selected,
                retained,
                self.config.window_size,
            )
            _write_paged_prefix(
                layer.k_cache,
                request.destination_block_ids_device,
                compact_key.squeeze(0).permute(1, 0, 2).contiguous(),
            )
            _write_paged_prefix(
                layer.v_cache,
                request.destination_block_ids_device,
                compact_value.squeeze(0).permute(1, 0, 2).contiguous(),
            )
            per_layer.append((layer.name, retained))
        if self.auxiliary_mtp_caches:
            if auxiliary_selection is None:
                raise RuntimeError("PyramidKV has no target selection for MTP cache")
            selected, retained = auxiliary_selection
            for layer in self.auxiliary_mtp_caches:
                key = _gather_paged_prefix(
                    layer.k_cache,
                    request.source_block_ids_device,
                    request.physical_num_tokens,
                )
                value = _gather_paged_prefix(
                    layer.v_cache,
                    request.source_block_ids_device,
                    request.physical_num_tokens,
                )
                compact_key, compact_value = materialize_pyramid_kv(
                    key.permute(1, 0, 2).unsqueeze(0),
                    value.permute(1, 0, 2).unsqueeze(0),
                    selected,
                    retained,
                    self.config.window_size,
                )
                _write_paged_prefix(
                    layer.k_cache,
                    request.destination_block_ids_device,
                    compact_key.squeeze(0).permute(1, 0, 2).contiguous(),
                )
                _write_paged_prefix(
                    layer.v_cache,
                    request.destination_block_ids_device,
                    compact_value.squeeze(0).permute(1, 0, 2).contiguous(),
                )
                per_layer.append((layer.name, retained))
        return CompressionResult(
            physical_num_tokens=self.group_maximum,
            per_layer_physical_num_tokens=tuple(per_layer),
        )

    def _normalize_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim == 2 and query.shape[1] == (self._local_query_heads * self.model_shape.head_dim):
            query = query.reshape(query.shape[0], self._local_query_heads, self.model_shape.head_dim)
        if query.ndim != 3 or tuple(query.shape[1:]) != (
            self._local_query_heads,
            self.model_shape.head_dim,
        ):
            raise RuntimeError("PyramidKV query must have [tokens, local_query_heads, head_dim]")
        if self._query_buffers is not None and query.dtype != self._query_buffers.dtype:
            raise RuntimeError("PyramidKV query dtype does not match the K/V cache")
        return query

    def _request_slot(self, request_id: str) -> int:
        existing = self._request_slots.get(request_id)
        if existing is not None:
            return existing
        if not self._free_slots:
            raise RuntimeError("PyramidKV query buffer capacity is exhausted")
        slot = self._free_slots.pop()
        self._request_slots[request_id] = slot
        return slot

    def _append_query_tail(
        self,
        buffer: torch.Tensor,
        previous: int,
        rows: torch.Tensor,
    ) -> int:
        assert self._query_scratch is not None
        window = self.config.window_size
        count = int(rows.shape[0])
        if count >= window:
            buffer.copy_(rows[-window:])
            return window
        keep = min(previous, window - count)
        if keep:
            self._query_scratch[:keep].copy_(buffer[previous - keep : previous])
        self._query_scratch[keep : keep + count].copy_(rows)
        buffer[: keep + count].copy_(self._query_scratch[: keep + count])
        return keep + count


def create_pyramidkv_method(
    options: Mapping[str, Any],
    vllm_config: Any,
    model_shape: ModelShape,
) -> KVCompressionMethod:
    """Create the external method without patching host classes."""
    return PyramidKVMethod(options, vllm_config, model_shape)


def _is_mtp_cache_layer(layer_name: str) -> bool:
    return ".mtp.layers." in f".{layer_name}."
