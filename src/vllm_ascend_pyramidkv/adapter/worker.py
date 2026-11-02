# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-HUST/vllm-ascend-kvcompress-hust, f337858 (see NOTICE).
"""Worker-side adapter for the upstream-aligned vLLM/Ascend runtime.

The current hosts intentionally expose only the standard ``vllm.general_plugins``
entry point. This module owns its state instead of importing fork-only
KV-compression types. Host integration is validated at activation and cache binding; hooks live in
:mod:`vllm_ascend_pyramidkv.adapter.plugin`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from ..method import create_pyramidkv_method
from .config import ASCEND_BLOCK_SIZE, ProviderSelection
from .contracts import (
    CompressionRequest,
    KVCompressionMethod,
    LayerCache,
    MethodRuntimeSpec,
    ModelShape,
    QueryBatchSpan,
    QueryObservation,
)
from .host_validation import (
    allowed_scheduler_block_sizes,
    common_compatibility_reasons,
    find_full_attention_group,
    unpack_layer_cache,
    validate_cache_tensor,
    validate_method_runtime_spec,
    validate_per_layer_host,
)
from .metadata import AttentionMetadata
from .model import model_shape_from_config
from .transaction import PLAN_ATTRIBUTE, ActiveCompression, blocks_for_tokens

logger = logging.getLogger("vllm." + __name__)

RUNNER_PROVIDER_ATTRIBUTE = "_ascend_pyramidkv_provider_v3"
ATTENTION_QUERY_PROVIDER_ATTRIBUTE = "_ascend_pyramidkv_query_provider_v1"


@dataclass(frozen=True)
class PendingCompression:
    semantic_anchor: int
    physical_anchor: int
    block_ids: tuple[int, ...]
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None


class PyramidKVWorker:
    """Own one model runner's PyramidKV state and compacted length mapping."""

    def __init__(self, vllm_config: Any, selection: ProviderSelection) -> None:
        self.vllm_config = vllm_config
        self.selection = selection
        self.model_shape: ModelShape = model_shape_from_config(vllm_config.model_config)
        self.tensor_parallel_size = int(getattr(vllm_config.parallel_config, "tensor_parallel_size", 1))
        self.attention_group_index = 0
        self.scheduler_block_size = int(vllm_config.cache_config.block_size)
        self.cache_block_size = ASCEND_BLOCK_SIZE
        self.cache_blocks_per_scheduler_block = 1
        self.method = create_pyramidkv_method(
            selection.method_config,
            vllm_config,
            self.model_shape,
        )
        self.runtime_spec: MethodRuntimeSpec = self.method.runtime_spec
        validate_method_runtime_spec(self.method.name, self.runtime_spec)
        self.requires_per_layer_physical_state = self.method.requires_per_layer_physical_state
        if not isinstance(self.requires_per_layer_physical_state, bool):
            raise TypeError("method requires_per_layer_physical_state must be a boolean")
        self.query_window_tokens = self.method.query_window_tokens
        if (
            isinstance(self.query_window_tokens, bool)
            or not isinstance(self.query_window_tokens, int)
            or self.query_window_tokens < 0
        ):
            raise ValueError("method query_window_tokens must be a non-negative integer")
        if self.query_window_tokens and any(
            getattr(type(self.method), name) is getattr(KVCompressionMethod, name)
            for name in (
                "capture_query",
                "complete_query_observation",
                "discard_query_observation",
            )
        ):
            raise TypeError("a query-observing method must implement capture, completion, and discard hooks")
        self.runner: Any | None = None
        self.layer_caches: tuple[LayerCache, ...] = ()
        self.query_layer_caches: tuple[LayerCache, ...] = ()
        self.speculative_cache_layer: LayerCache | None = None
        self.pending: dict[str, PendingCompression] = {}
        self.active: dict[str, ActiveCompression] = {}
        self.metadata = AttentionMetadata(vllm_config, self.active, self.requires_per_layer_physical_state)
        self._query_step_output: Any | None = None
        self._query_seen_layers: set[int] = set()
        self._query_forward_complete = False
        self._query_tracked_ids: set[str] = set()

    def validate_host(self, runner: Any) -> None:
        """Fail closed before binding any cache memory."""
        reasons = list(common_compatibility_reasons(self.vllm_config, runner))
        reasons.extend(f"method {self.method.name}: {reason}" for reason in self.method.compatibility_reasons(runner))
        if reasons:
            raise RuntimeError("Ascend KV compression is incompatible with this launch:\n- " + "\n- ".join(reasons))

    def bind_model_runner(self, runner: Any, kv_cache_config: Any) -> None:
        """Validate and bind dense K/V views after the host allocates its cache."""
        self.validate_host(runner)
        from vllm.model_executor.models.utils import extract_layer_index

        groups = kv_cache_config.kv_cache_groups
        self.attention_group_index = find_full_attention_group(groups)
        group = groups[self.attention_group_index]
        self.scheduler_block_size = int(group.kv_cache_spec.block_size)
        allowed_block_sizes = allowed_scheduler_block_sizes(self.vllm_config)
        if self.scheduler_block_size not in allowed_block_sizes:
            raise RuntimeError(
                "Ascend KV compression does not support logical KV block size "
                f"{self.scheduler_block_size}; allowed sizes are "
                f"{sorted(allowed_block_sizes)}"
            )
        if self.runtime_spec.max_physical_num_tokens % self.scheduler_block_size:
            raise RuntimeError(
                f"compression target must be divisible by the logical KV block size {self.scheduler_block_size}"
            )

        block_tables = runner.input_batch.block_table.block_tables
        if len(block_tables) != len(groups):
            raise RuntimeError("worker block-table groups do not match KV groups")
        attention_block_table = block_tables[self.attention_group_index]
        physical_block_size = int(
            getattr(
                attention_block_table,
                "physical_block_size",
                self.scheduler_block_size,
            )
        )
        if physical_block_size != self.scheduler_block_size:
            raise RuntimeError("worker full-attention block table disagrees with its KV cache group block size")
        self.cache_block_size = int(getattr(attention_block_table, "block_size", self.scheduler_block_size))
        if self.cache_block_size != ASCEND_BLOCK_SIZE or self.scheduler_block_size % self.cache_block_size:
            raise RuntimeError(
                "Ascend KV compression requires 128-token kernel cache blocks "
                "that evenly divide the logical KV block size"
            )
        self.cache_blocks_per_scheduler_block = self.scheduler_block_size // self.cache_block_size

        layer_caches: list[LayerCache] = []
        context = runner.compilation_config.static_forward_context
        for layer_name in group.layer_names:
            layer = context.get(layer_name)
            if layer is None:
                raise RuntimeError(f"attention layer {layer_name!r} is not bound")
            k_cache, v_cache = unpack_layer_cache(layer_name, layer)
            local_kv_heads = self.model_shape.num_kv_heads // self.tensor_parallel_size
            validate_cache_tensor(
                layer_name,
                "K",
                k_cache,
                self.model_shape,
                local_kv_heads,
                self.cache_block_size,
            )
            validate_cache_tensor(
                layer_name,
                "V",
                v_cache,
                self.model_shape,
                local_kv_heads,
                self.cache_block_size,
            )
            layer_caches.append(
                LayerCache(
                    name=layer_name,
                    layer_index=extract_layer_index(layer_name),
                    k_cache=k_cache,
                    v_cache=v_cache,
                )
            )

        self.runner = runner
        self.layer_caches = tuple(layer_caches)
        self.method.bind_model_runner(runner, self.layer_caches)
        self.query_layer_caches = self._resolve_query_layer_caches()
        mtp_caches = tuple(cache for cache in self.layer_caches if ".mtp.layers." in f".{cache.name}.")
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            if len(mtp_caches) != 1:
                raise RuntimeError("qualified MTP2 requires exactly one auxiliary attention cache")
            self.speculative_cache_layer = mtp_caches[0]
        self.metadata.bind(runner, self.layer_caches, self.attention_group_index, self.speculative_cache_layer)
        if self.query_window_tokens:
            for cache in self.query_layer_caches:
                layer = context[cache.name]
                setattr(layer, ATTENTION_QUERY_PROVIDER_ATTRIBUTE, self)
        setattr(attention_block_table, RUNNER_PROVIDER_ATTRIBUTE, self)
        logger.info(
            "Ascend KV compression cache bound method=%s "
            "materialization_layers=%d calibrated_scoring_layers=%d "
            "threshold_tokens=%d target_tokens=%d min_output_tokens=%d "
            "logical_block_size=%d cache_block_size=%d",
            self.method.name,
            len(self.layer_caches),
            len(getattr(self.method, "layer_caches", self.layer_caches)),
            self.runtime_spec.compression_threshold_tokens,
            self.runtime_spec.max_physical_num_tokens,
            self.runtime_spec.min_output_tokens_for_compression,
            self.scheduler_block_size,
            self.cache_block_size,
        )

    def _resolve_query_layer_caches(self) -> tuple[LayerCache, ...]:
        if not self.query_window_tokens:
            return ()
        requested = self.method.query_layer_indices
        if requested is None:
            return self.layer_caches
        if (
            not isinstance(requested, tuple)
            or not requested
            or any(isinstance(index, bool) or not isinstance(index, int) for index in requested)
            or len(set(requested)) != len(requested)
        ):
            raise RuntimeError("query_layer_indices must be a non-empty tuple of unique integers")
        by_index: dict[int, LayerCache] = {}
        duplicate_indices: set[int] = set()
        for cache in self.layer_caches:
            if cache.layer_index in by_index:
                duplicate_indices.add(cache.layer_index)
            else:
                by_index[cache.layer_index] = cache
        if duplicate_indices & set(requested):
            raise RuntimeError("query_layer_indices are ambiguous across bound cache layers")
        if any(index not in by_index for index in requested):
            raise RuntimeError("query_layer_indices contain an unbound model layer")
        return tuple(by_index[index] for index in requested)

    def before_update_states(self, scheduler_output: Any) -> None:
        """Commit the prior step after its synchronous model execution barrier."""
        if self.runner is None:
            return
        reset_ids = set(scheduler_output.finished_req_ids)
        reset_ids.update(getattr(scheduler_output, "preempted_req_ids", ()))
        reset_ids.update(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
        for request_id in reset_ids:
            self.pending.pop(request_id, None)
            self.active.pop(request_id, None)
            if self.query_window_tokens:
                self.method.discard_query_observation(request_id)
                self._query_tracked_ids.discard(request_id)

        for request_id, pending in tuple(self.pending.items()):
            # An async scheduler can execute other requests while this
            # compression output is still in flight. The scheduler freezes
            # this request until acknowledgement, so its next scheduled step
            # is the commit boundary on the worker too.
            if request_id not in scheduler_output.num_scheduled_tokens:
                continue
            request = self.runner.requests.get(request_id)
            if request is None:
                self.pending.pop(request_id, None)
                if self.query_window_tokens:
                    self.method.discard_query_observation(request_id)
                    self._query_tracked_ids.discard(request_id)
                continue
            mutable = list(pending.block_ids)
            group_block_ids = list(request.block_ids)
            group_block_ids[self.attention_group_index] = mutable
            request.block_ids = tuple(group_block_ids)
            request_index = self.runner.input_batch.req_id_to_index.get(request_id)
            if request_index is not None:
                self.runner.input_batch.block_table.add_row(request.block_ids, request_index)
            self.active[request_id] = ActiveCompression(
                pending.semantic_anchor,
                pending.physical_anchor,
                pending.per_layer_physical_num_tokens,
            )
            logger.info(
                "KV compression worker commit acknowledged request_id=%s "
                "semantic_tokens=%d physical_tokens=%d destination_blocks=%d",
                request_id,
                pending.semantic_anchor,
                pending.physical_anchor,
                len(pending.block_ids),
            )
            self.pending.pop(request_id, None)

    def capture_attention_query(self, layer_name: str, query: torch.Tensor) -> None:
        """Observe query rows from the attention custom-op execution path.

        Module forward hooks run while Dynamo traces a piecewise graph but are
        not invoked when the compiled graph is replayed.  Ascend's split
        ``unified_attention_with_output`` custom op does execute its backend
        forward for every prefill replay, so the plugin wrapper calls this
        method there.  Keeping the request spans outside the graph also avoids
        baking one scheduler batch into the compiled program.
        """
        if self._query_step_output is None:
            return
        cache = next(
            (cache for cache in self.query_layer_caches if cache.name == layer_name),
            None,
        )
        if cache is None:
            raise RuntimeError(f"attention layer {layer_name!r} is not a query-observation layer")
        spans = self._query_spans()
        if not spans:
            return
        if query.ndim < 2 or max(span.end for span in spans) > query.shape[0]:
            raise RuntimeError(f"attention layer {cache.name!r} query rows changed")
        self.method.capture_query(cache, query, spans)
        self._query_tracked_ids.update(span.request_id for span in spans)
        self._query_seen_layers.add(cache.layer_index)

    def _query_spans(self) -> tuple[QueryBatchSpan, ...]:
        output = self._query_step_output
        if output is None or self.runner is None:
            return ()
        batch = self.runner.input_batch
        offsets = self.runner.query_start_loc.cpu
        spans: list[QueryBatchSpan] = []
        for request_id, scheduled in output.num_scheduled_tokens.items():
            request = self.runner.requests.get(request_id)
            index = batch.req_id_to_index.get(request_id)
            if request is None or index is None or scheduled <= 0:
                continue
            remaining = int(request.num_prompt_tokens) - int(request.num_computed_tokens)
            if remaining <= 0:
                continue
            start = int(offsets[index])
            end = start + min(int(scheduled), remaining)
            if end > int(offsets[index + 1]):
                raise RuntimeError("prefill query rows exceed the runner batch span")
            spans.append(QueryBatchSpan(request_id, start, end))
        return tuple(spans)

    def begin_query_step(self, scheduler_output: Any) -> None:
        if not self.query_window_tokens:
            return
        for request_id in self._query_tracked_ids - self.runner.requests.keys():
            self.method.discard_query_observation(request_id)
            self._query_tracked_ids.discard(request_id)
        self._query_step_output = scheduler_output
        self._query_seen_layers.clear()
        self._query_forward_complete = False

    def finish_query_step(self) -> None:
        if not self.query_window_tokens:
            return
        try:
            if self._query_spans() and self._query_seen_layers != {
                layer.layer_index for layer in self.query_layer_caches
            }:
                raise RuntimeError(
                    "query observation missed a full-attention layer; the current graph replay path is unvalidated"
                )
            self._query_forward_complete = True
        finally:
            self._query_step_output = None

    def abort_query_step(self, scheduler_output: Any) -> None:
        if not self.query_window_tokens:
            return
        self._query_step_output = None
        self._query_forward_complete = False
        self._query_seen_layers.clear()
        for request_id in scheduler_output.num_scheduled_tokens:
            self.method.discard_query_observation(request_id)
            self._query_tracked_ids.discard(request_id)

    def after_update_states(self, scheduler_output: Any) -> None:
        del scheduler_output
        if self.runner is not None:
            live = self.runner.requests.keys()
            for request_id in (self.pending.keys() | self.active.keys()) - live:
                self.pending.pop(request_id, None)
                self.active.pop(request_id, None)
        self.metadata.update_request_offsets()

    def compress_scheduled_requests(self, scheduler_output: Any) -> None:
        """Run scheduler-authorized transactions after the attention step."""
        if self.runner is None or not self.layer_caches:
            raise RuntimeError("Ascend KV compression provider is not cache-bound")
        plans = getattr(scheduler_output, PLAN_ATTRIBUTE, {})
        for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            plan = plans.get(request_id)
            if plan is None:
                continue
            if request_id in self.pending:
                continue
            request = self.runner.requests.get(request_id)
            if request is None:
                continue
            if _request_max_tokens(request) < self.runtime_spec.min_output_tokens_for_compression:
                continue
            semantic = int(request.num_computed_tokens) + int(scheduled)
            active = self.active.get(request_id)
            if active is None:
                physical = semantic
                if semantic != int(request.num_prompt_tokens):
                    continue
            else:
                physical = active.physical_anchor + semantic - active.semantic_anchor
            if semantic != plan.semantic_anchor:
                raise RuntimeError(
                    f"request {request_id!r} semantic length disagrees with the scheduler compression plan"
                )
            if physical < self.runtime_spec.compression_threshold_tokens:
                continue
            if self.attention_group_index >= len(request.block_ids):
                raise RuntimeError("request has no full-attention block table")
            source_ids = tuple(int(value) for value in request.block_ids[self.attention_group_index])
            required_blocks = blocks_for_tokens(
                self.runtime_spec.max_physical_num_tokens,
                self.scheduler_block_size,
            )
            if len(source_ids) < required_blocks:
                raise RuntimeError(f"request {request_id!r} has too few blocks for compression")
            if source_ids != plan.source_block_ids:
                raise RuntimeError(
                    f"request {request_id!r} source block table disagrees with the scheduler compression plan"
                )
            destination_ids = plan.destination_block_ids
            if len(destination_ids) != required_blocks or (set(destination_ids) & set(source_ids)):
                raise RuntimeError(f"request {request_id!r} requires private destination blocks")
            source_device = _expand_scheduler_block_ids(
                source_ids,
                self.cache_blocks_per_scheduler_block,
                self.runner.device,
            )
            destination_device = _expand_scheduler_block_ids(
                destination_ids,
                self.cache_blocks_per_scheduler_block,
                self.runner.device,
            )
            if self.query_window_tokens:
                if not self._query_forward_complete:
                    raise RuntimeError("query observation requires a completed model forward")
                self.method.complete_query_observation(
                    QueryObservation(
                        request_id=request_id,
                        plan=plan,
                        semantic_num_tokens=semantic,
                        window_tokens=self.query_window_tokens,
                        layer_indices=tuple(layer.layer_index for layer in self.query_layer_caches),
                    )
                )
            try:
                result = self.method.compress(
                    CompressionRequest(
                        request_id=request_id,
                        semantic_num_tokens=semantic,
                        physical_num_tokens=physical,
                        source_block_ids=(source_ids,),
                        destination_block_ids=(destination_ids,),
                        source_block_ids_device=source_device,
                        destination_block_ids_device=destination_device,
                        plan=plan,
                        per_layer_physical_num_tokens=(
                            tuple(
                                (
                                    name,
                                    length + semantic - active.semantic_anchor,
                                )
                                for name, length in active.per_layer_physical_num_tokens
                            )
                            if active is not None and active.per_layer_physical_num_tokens is not None
                            else None
                        ),
                    )
                )
            finally:
                if self.query_window_tokens:
                    self.method.discard_query_observation(request_id)
                    self._query_tracked_ids.discard(request_id)
            compacted = result.physical_num_tokens
            if isinstance(compacted, bool) or not isinstance(compacted, int):
                raise RuntimeError("method returned a non-integer physical length")
            if compacted != self.runtime_spec.max_physical_num_tokens:
                raise RuntimeError(f"method {self.method.name!r} returned unexpected length {compacted}")
            per_layer = self._validate_per_layer_lengths(result.per_layer_physical_num_tokens, compacted)
            self.pending[request_id] = PendingCompression(
                semantic_anchor=semantic,
                physical_anchor=compacted,
                block_ids=destination_ids,
                per_layer_physical_num_tokens=per_layer,
            )

    def _validate_per_layer_lengths(
        self, lengths: tuple[tuple[str, int], ...] | None, maximum: int
    ) -> tuple[tuple[str, int], ...] | None:
        if lengths is None:
            return None
        names = tuple(layer.name for layer in self.layer_caches)
        if not isinstance(lengths, tuple) or len(lengths) != len(names):
            raise RuntimeError("per-layer physical lengths must cover every full-attention layer")
        values: dict[str, int] = {}
        for entry in lengths:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise RuntimeError("invalid per-layer physical length entry")
            name, length = entry
            if not isinstance(name, str) or name in values or name not in names:
                raise RuntimeError("per-layer physical lengths contain an unknown or duplicate layer")
            if isinstance(length, bool) or not isinstance(length, int) or not 0 < length <= maximum:
                raise RuntimeError("per-layer physical lengths must be positive and <= group maximum")
            values[name] = length
        if set(values) != set(names):
            raise RuntimeError("per-layer physical lengths omit a full-attention layer")
        ordered = tuple((name, values[name]) for name in names)
        if any(length != maximum for _, length in ordered):
            self._validate_per_layer_host()
            return ordered
        # Uniform physical lengths retain the original common metadata path.
        return None

    def _validate_per_layer_host(self) -> None:
        validate_per_layer_host(self.runner, self.vllm_config, self.requires_per_layer_physical_state)

    def physical_positions_for_slot_mapping(self, positions: torch.Tensor) -> torch.Tensor:
        return self.metadata.physical_positions_for_slot_mapping(positions)

    def apply_physical_attention_lengths(self) -> None:
        self.metadata.apply_physical_attention_lengths()

    def semantic_gdn_metadata(self, metadata: Any) -> Any:
        return self.metadata.semantic_gdn_metadata(metadata)

    def per_layer_attention_metadata(self, result: Any, num_tokens: int, num_reqs: int) -> Any:
        return self.metadata.per_layer_attention_metadata(result, num_tokens, num_reqs)

    def begin_per_layer_step(self, scheduler_output: Any) -> None:
        self.metadata.begin_per_layer_step(scheduler_output)

    def finish_per_layer_step(self) -> None:
        self.metadata.finish_per_layer_step()


def _request_max_tokens(request: Any) -> int:
    value = getattr(request, "max_tokens", None)
    if value is None:
        sampling_params = getattr(request, "sampling_params", None)
        value = getattr(sampling_params, "max_tokens", None)
    if value is None and getattr(request, "pooling_params", None) is not None:
        value = 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("current vLLM worker request must expose a non-negative output limit")
    return value


def _expand_scheduler_block_ids(
    block_ids: tuple[int, ...],
    cache_blocks_per_scheduler_block: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Map scheduler block IDs to consecutive 128-token cache block IDs."""
    ids = torch.as_tensor(block_ids, device=device, dtype=torch.long)
    if cache_blocks_per_scheduler_block == 1:
        return ids
    offsets = torch.arange(
        cache_blocks_per_scheduler_block,
        device=device,
        dtype=torch.long,
    )
    return (ids.unsqueeze(1) * cache_blocks_per_scheduler_block + offsets).reshape(-1)
