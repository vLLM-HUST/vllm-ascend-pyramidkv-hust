# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-HUST/vllm-ascend-kvcompress-hust, f337858 (see NOTICE).
"""Physical attention metadata with stable graph buffers and semantic GDN views."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from .contracts import LayerCache
from .host_validation import validate_per_layer_host
from .transaction import ActiveCompression


class AttentionMetadata:
    """Own the runner's physical offsets and graph-visible per-layer storage."""

    def __init__(
        self, vllm_config: Any, active: Mapping[str, ActiveCompression], requires_per_layer_physical_state: bool
    ) -> None:
        self.vllm_config = vllm_config
        self.active = active
        self.requires_per_layer_physical_state = requires_per_layer_physical_state
        self.runner: Any | None = None
        self.layer_caches: tuple[LayerCache, ...] = ()
        self.attention_group_index = 0
        self.speculative_cache_layer: LayerCache | None = None
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._request_offsets_cpu: torch.Tensor | None = None
        self._request_offsets_device: torch.Tensor | None = None
        self._physical_positions: torch.Tensor | None = None
        self._semantic_seq_lens_device: torch.Tensor | None = None
        self._semantic_seq_lens_cpu: torch.Tensor | None = None
        self._offset_row_snapshot: tuple[int, ...] = ()
        self._has_active_rows = False
        self._has_per_layer_rows = False
        self._per_layer_metadata_seen = False
        self._per_layer_step_requires_metadata = False
        self._per_layer_slot_buffers: dict[str, torch.Tensor] = {}
        self._metadata_tensor_cache: dict[str, tuple[tuple[int, ...], torch.Tensor]] = {}
        self._batched_slot_storage: torch.Tensor | None = None
        self._physical_lengths_applied = False

    def bind(
        self,
        runner: Any,
        layer_caches: tuple[LayerCache, ...],
        attention_group_index: int,
        speculative_cache_layer: LayerCache | None,
    ) -> None:
        self.runner = runner
        self.layer_caches = layer_caches
        self.attention_group_index = attention_group_index
        self.speculative_cache_layer = speculative_cache_layer
        self._reset_buffers()
        if self.requires_per_layer_physical_state:
            self.validate_host()
        pin_memory = bool(getattr(runner, "pin_memory", False))
        self._request_offsets_cpu = torch.zeros(
            runner.max_num_reqs,
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._request_offsets_device = torch.zeros(runner.max_num_reqs, dtype=torch.int64, device=runner.device)
        self._physical_positions = torch.empty(runner.max_num_tokens, dtype=torch.int64, device=runner.device)
        if (
            self.requires_per_layer_physical_state
            and os.environ.get("VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA", "0") == "1"
        ):
            self._batched_slot_storage = torch.empty(
                (len(self.layer_caches), runner.max_num_tokens),
                dtype=torch.int64,
                device=runner.device,
            )
            self._per_layer_slot_buffers = {
                cache.name: self._batched_slot_storage[index] for index, cache in enumerate(self.layer_caches)
            }
        elif self.requires_per_layer_physical_state:
            self._per_layer_slot_buffers = {
                cache.name: torch.empty(runner.max_num_tokens, dtype=torch.int64, device=runner.device)
                for cache in self.layer_caches
            }
        self._semantic_seq_lens_device = torch.empty_like(runner.seq_lens)
        self._semantic_seq_lens_cpu = torch.empty_like(runner.optimistic_seq_lens_cpu)

    def validate_host(self) -> None:
        validate_per_layer_host(self.runner, self.vllm_config, self.requires_per_layer_physical_state)

    def physical_positions_for_slot_mapping(self, positions: torch.Tensor) -> torch.Tensor:
        """Map writes to compacted slots while preserving semantic RoPE positions."""
        if not self._has_active_rows:
            return positions
        if self.runner is None or self._request_offsets_device is None or self._physical_positions is None:
            raise RuntimeError("compression position buffers are not initialized")
        num_tokens = positions.numel()
        return _shift_positions(
            positions,
            self.runner.req_indices.gpu[:num_tokens],
            self._request_offsets_device,
            self._physical_positions[:num_tokens],
        )

    def apply_physical_attention_lengths(self) -> None:
        """Replace semantic lengths only after host sampling decisions are built."""
        if self._physical_lengths_applied or not self._has_active_rows:
            return
        if (
            self.runner is None
            or self._request_offsets_cpu is None
            or self._request_offsets_device is None
            or self._semantic_seq_lens_device is None
            or self._semantic_seq_lens_cpu is None
        ):
            raise RuntimeError("compression length buffers are not initialized")
        num_reqs = self.runner.input_batch.num_reqs
        self._semantic_seq_lens_device[:num_reqs].copy_(self.runner.seq_lens[:num_reqs])
        self._semantic_seq_lens_cpu[:num_reqs].copy_(self.runner.optimistic_seq_lens_cpu[:num_reqs])
        self.runner.seq_lens[:num_reqs].sub_(self._request_offsets_device[:num_reqs])
        self.runner.optimistic_seq_lens_cpu[:num_reqs].sub_(self._request_offsets_cpu[:num_reqs])
        self._physical_lengths_applied = True

    def semantic_gdn_metadata(self, metadata: Any) -> Any:
        """Give GDN its original recurrent positions, not attention offsets."""
        if not self._physical_lengths_applied:
            return metadata
        if self._semantic_seq_lens_device is None or self._semantic_seq_lens_cpu is None:
            raise RuntimeError("semantic GDN length buffers are not initialized")
        num_reqs = metadata.num_reqs
        return replace(
            metadata,
            seq_lens=self._semantic_seq_lens_device[:num_reqs],
            _seq_lens_cpu=self._semantic_seq_lens_cpu[:num_reqs],
            seq_lens_cpu_upper_bound=self._semantic_seq_lens_cpu[:num_reqs],
        )

    def _metadata_correction_tensor(self, key: str, values: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
        """Reuse immutable row corrections until the batch's values change.

        Constructing a device tensor from Python values on every decode step
        repeats host-to-device transfers for each layer. Entries are bounded by
        layer/field, not request IDs. Replace rather than mutate an old tensor:
        an earlier asynchronous operation may still be consuming it.
        """
        cached = self._metadata_tensor_cache.get(key)
        if cached is not None:
            previous, tensor = cached
            if previous == values and tensor.dtype == reference.dtype and tensor.device == reference.device:
                return tensor
        tensor = torch.tensor(values, dtype=reference.dtype, device=reference.device)
        self._metadata_tensor_cache[key] = values, tensor
        return tensor

    def _batched_per_layer_metadata(
        self,
        metadata: dict,
        common: Any,
        table: Any,
        request_ids: list[str],
        global_offsets: tuple[int, ...],
        num_tokens: int,
        num_reqs: int,
    ) -> Any:
        """Batch integer metadata for the validated single-CP expanded table.

        The persistent row views are the same slot addresses captured by FULL
        graphs. Preserve the native path for other layouts or metadata shapes.
        No K/V, query, attention score, or method-selection arithmetic changes.
        """
        slots = self._batched_slot_storage
        blocks = getattr(getattr(table, "block_table", None), "gpu", None)
        block_size = getattr(table, "block_size", 0)
        if (
            slots is None
            or getattr(table, "dcp_world_size", None) != 1
            or getattr(table, "cp_kv_cache_interleave_size", 1) != 1
            or not isinstance(blocks, torch.Tensor)
            or blocks.ndim != 2
            or not blocks.is_contiguous()
            or not isinstance(block_size, int)
            or block_size <= 0
            or blocks.dtype not in (torch.int32, torch.int64)
            or getattr(table, "logical_block_size", None) != block_size
            or getattr(table, "physical_block_size", 0) != block_size * getattr(table, "blocks_per_phys_block", 0)
        ):
            return None
        bases = [metadata.get(layer.name) for layer in self.layer_caches]
        first = bases[0]
        for base in bases:
            if (
                type(base).__name__ != "AscendMetadata"
                or type(base).__module__ != "vllm_ascend.attention.attention_v1"
                or not isinstance(base.seq_lens, torch.Tensor)
                or base.seq_lens.ndim != 1
                or base.seq_lens.shape != first.seq_lens.shape
                or base.seq_lens.data_ptr() != first.seq_lens.data_ptr()
                or base.seq_lens_list != first.seq_lens_list
                or base.slot_mapping is None
            ):
                return None
        nseq = first.seq_lens.numel()
        if (
            nseq < num_reqs
            or num_tokens > slots.shape[1]
            or first.seq_lens.dtype not in (torch.int32, torch.int64)
            or blocks.device != slots.device
            or blocks.shape[0] < num_reqs
        ):
            return None
        runner = self.runner
        assert runner is not None
        offsets = self._layer_offsets(request_ids)
        deltas = [
            [global_value - value for global_value, value in zip(global_offsets, row, strict=True)] for row in offsets
        ]
        layers = len(bases)
        correction = self._metadata_correction_tensor(
            "batch:deltas",
            tuple(value for row in deltas for value in row + [0] * (nseq - num_reqs)),
            first.seq_lens,
        ).reshape(layers, nseq)
        # The qualified Ascend builder keeps these lengths on CPU, whereas
        # positions/slots live on NPU. Follow the actual metadata device. Keep
        # each call's length storage independent from in-flight older views.
        lengths = first.seq_lens.unsqueeze(0) + correction
        slots[:, num_tokens:].fill_(-1)
        if not any(any(row) for row in offsets):
            slots[:, :num_tokens].copy_(table.slot_mapping.gpu[:num_tokens].unsqueeze(0))
        else:
            positions = runner.positions[:num_tokens]
            req_indices = runner.req_indices.gpu[:num_tokens].to(torch.long)
            offset_tensor = self._metadata_correction_tensor(
                "batch:offsets",
                tuple(value for row in offsets for value in row),
                positions,
            ).reshape(layers, num_reqs)
            physical = positions.unsqueeze(0) - offset_tensor.index_select(1, req_indices)
            indices = physical.div(block_size, rounding_mode="floor") + req_indices.unsqueeze(0) * blocks.shape[1]
            ids = blocks.view(-1).index_select(0, indices.reshape(-1).to(torch.long))
            values = ids.reshape(layers, num_tokens) * block_size + physical.remainder(block_size)
            slots[:, :num_tokens].copy_(values)
        speculative = None
        for index, (layer, base) in enumerate(zip(self.layer_caches, bases, strict=True)):
            seq_list = list(base.seq_lens_list)
            for row, delta in enumerate(deltas[index]):
                seq_list[row] += delta
            metadata[layer.name] = replace(
                base,
                seq_lens=lengths[index, :nseq],
                seq_lens_cpu=lengths[index, :nseq],
                seq_lens_list=seq_list,
                slot_mapping=slots[index, :num_tokens],
            )
            if layer is self.speculative_cache_layer:
                speculative = index
        if self.speculative_cache_layer is not None:
            if speculative is None or common is None:
                raise RuntimeError("MTP2 batched attention metadata is incomplete")
            common = self._speculative_common_metadata(
                common,
                deltas[speculative],
                offsets[speculative],
                slots[speculative, :num_tokens],
                num_reqs,
            )
        return metadata, common

    def per_layer_attention_metadata(self, result: Any, num_tokens: int, num_reqs: int) -> Any:
        """Give each full-attention layer distinct KV lengths and write slots."""
        if not (self.requires_per_layer_physical_state or self._has_per_layer_rows):
            return result
        self.validate_host()
        if self.runner is None or not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("per-layer attention metadata hook was bypassed")
        metadata, common = result
        if not isinstance(metadata, dict):
            raise RuntimeError("per-layer attention requires a single eager metadata batch")
        runner = self.runner
        batch = runner.input_batch
        request_ids = batch.req_ids[:num_reqs]
        if num_tokens < 0:
            raise RuntimeError("per-layer attention batch dimensions changed")
        if len(request_ids) != num_reqs:
            # FULL graph capture builds synthetic decode batches before the
            # runner has admitted any requests.  There are no compressed rows
            # to specialize in that phase, but graph cache-write kernels must
            # capture the same per-layer slot buffers used during replay. Keep
            # failing closed for a real or partially populated input batch.
            batch_num_reqs = int(getattr(batch, "num_reqs", len(batch.req_ids)))
            if self._has_active_rows or batch_num_reqs != 0 or batch.req_ids:
                raise RuntimeError("per-layer attention batch dimensions changed")
            for layer in self.layer_caches:
                base = metadata.get(layer.name)
                slots = self._per_layer_slot_buffers.get(layer.name)
                if (
                    base is None
                    or not isinstance(getattr(base, "slot_mapping", None), torch.Tensor)
                    or slots is None
                    or slots.numel() < num_tokens
                    or base.slot_mapping.numel() < num_tokens
                ):
                    raise RuntimeError("per-layer capture slot buffers are unavailable")
                slots[num_tokens:].fill_(-1)
                slots[:num_tokens].copy_(base.slot_mapping[:num_tokens])
                metadata[layer.name] = replace(base, slot_mapping=slots[:num_tokens])
                if layer is self.speculative_cache_layer:
                    if common is None:
                        raise RuntimeError("MTP2 capture metadata is unavailable")
                    common = replace(common, slot_mapping=slots[:num_tokens])
            self._per_layer_metadata_seen = True
            return metadata, common
        table = batch.block_table.block_tables[self.attention_group_index]
        native = getattr(type(table), "_ascend_pyramidkv_patch_v3_slot_mapping_original", None)
        if not callable(native):
            raise RuntimeError("per-layer attention requires the native slot mapping hook")
        if self._request_offsets_cpu is None:
            raise RuntimeError("per-layer attention offset buffers are unavailable")
        if self._has_active_rows and not self._physical_lengths_applied:
            raise RuntimeError("per-layer attention requires saved semantic lengths")
        global_offsets = tuple(int(value) for value in self._request_offsets_cpu[:num_reqs])
        batched = self._batched_per_layer_metadata(
            metadata, common, table, request_ids, global_offsets, num_tokens, num_reqs
        )
        if batched is not None:
            self._per_layer_metadata_seen = True
            return batched
        original_slots = table.slot_mapping.gpu[:num_tokens].clone()
        speculative_view: tuple[list[int], tuple[int, ...], torch.Tensor, torch.Tensor] | None = None
        try:
            for layer, offsets in zip(self.layer_caches, self._layer_offsets(request_ids), strict=True):
                base = metadata.get(layer.name)
                if (
                    type(base).__name__ != "AscendMetadata"
                    or type(base).__module__ != "vllm_ascend.attention.attention_v1"
                    or base.seq_lens is None
                    or base.seq_lens_list is None
                    or base.slot_mapping is None
                ):
                    raise RuntimeError("per-layer attention backend metadata is unvalidated")
                delta = [global_offset - offset for global_offset, offset in zip(global_offsets, offsets, strict=True)]
                seq_lens = base.seq_lens.clone()
                if any(delta):
                    seq_lens[:num_reqs] += self._metadata_correction_tensor(
                        layer.name + ":delta", tuple(delta), seq_lens
                    )
                seq_lens_list = list(base.seq_lens_list)
                for index, correction in enumerate(delta):
                    seq_lens_list[index] += correction
                slots = self._per_layer_slot_buffers.get(layer.name)
                if slots is None and not self.requires_per_layer_physical_state:
                    slots = torch.empty(
                        max(num_tokens, int(getattr(runner, "max_num_tokens", 0))),
                        dtype=torch.int64,
                        device=original_slots.device,
                    )
                    self._per_layer_slot_buffers[layer.name] = slots
                if slots is None or slots.numel() < num_tokens:
                    raise RuntimeError("per-layer attention slot buffers were not prepared before capture")
                # Replay may use a larger captured bucket than the live batch.
                # Never let padded cache writes reuse a previous request's slots.
                slots[num_tokens:].fill_(-1)
                if offsets == global_offsets:
                    slots[:num_tokens].copy_(original_slots)
                else:
                    positions = runner.positions[:num_tokens]
                    req_indices = runner.req_indices.gpu[:num_tokens]
                    offset_tensor = self._metadata_correction_tensor(layer.name + ":offsets", offsets, positions)
                    native(
                        table,
                        num_reqs,
                        runner.query_start_loc.gpu[: num_reqs + 1],
                        positions - offset_tensor[req_indices.long()],
                    )
                    slots[:num_tokens].copy_(table.slot_mapping.gpu[:num_tokens])
                metadata[layer.name] = replace(
                    base,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens,
                    seq_lens_list=seq_lens_list,
                    slot_mapping=slots[:num_tokens],
                )
                if layer is self.speculative_cache_layer:
                    speculative_view = delta, offsets, seq_lens, slots[:num_tokens]
        finally:
            table.slot_mapping.gpu[:num_tokens].copy_(original_slots)
        if self.speculative_cache_layer is not None:
            if speculative_view is None or common is None:
                raise RuntimeError("MTP2 per-layer attention metadata is incomplete")
            delta, offsets, _, slots = speculative_view
            common = self._speculative_common_metadata(
                common,
                delta,
                offsets,
                slots,
                num_reqs,
            )
        self._per_layer_metadata_seen = True
        return metadata, common

    def _speculative_common_metadata(
        self,
        common: Any,
        delta: list[int],
        offsets: tuple[int, ...],
        slots: torch.Tensor,
        num_reqs: int,
    ) -> Any:
        required = (
            "seq_lens",
            "_seq_lens_cpu",
            "seq_lens_cpu_upper_bound",
            "slot_mapping",
        )
        if any(not hasattr(common, name) for name in required):
            raise RuntimeError("MTP2 common attention metadata ABI changed")

        def adjusted(field: str, value: Any, corrections: list[int] | tuple[int, ...]) -> Any:
            if value is None:
                return None
            if not isinstance(value, torch.Tensor) or value.numel() < num_reqs:
                raise RuntimeError("MTP2 sequence-length metadata ABI changed")
            result = value.clone()
            if any(corrections):
                result[:num_reqs] += self._metadata_correction_tensor("mtp-common:" + field, tuple(corrections), result)
            return result

        changes = {
            "seq_lens": adjusted("seq_lens", common.seq_lens, delta),
            "_seq_lens_cpu": adjusted("_seq_lens_cpu", common._seq_lens_cpu, delta),
            "seq_lens_cpu_upper_bound": adjusted(
                "seq_lens_cpu_upper_bound",
                common.seq_lens_cpu_upper_bound,
                delta,
            ),
            "slot_mapping": slots,
        }
        if hasattr(common, "seq_lens_cpu"):
            changes["seq_lens_cpu"] = adjusted("seq_lens_cpu", common.seq_lens_cpu, delta)
        if hasattr(common, "num_computed_tokens_cpu"):
            changes["num_computed_tokens_cpu"] = adjusted(
                "num_computed_tokens_cpu",
                common.num_computed_tokens_cpu,
                tuple(-offset for offset in offsets),
            )
        try:
            return replace(common, **changes)
        except (TypeError, ValueError) as error:
            raise RuntimeError("MTP2 common attention metadata cannot be specialized by layer") from error

    def begin_per_layer_step(self, scheduler_output: Any) -> None:
        enabled = self.requires_per_layer_physical_state or self._has_per_layer_rows
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
        self._per_layer_step_requires_metadata = enabled and any(int(tokens) > 0 for tokens in scheduled.values())
        if self._per_layer_step_requires_metadata:
            self._per_layer_metadata_seen = False

    def finish_per_layer_step(self) -> None:
        try:
            if self._per_layer_step_requires_metadata and not self._per_layer_metadata_seen:
                raise RuntimeError("per-layer attention metadata hook was bypassed by the host")
        finally:
            self._per_layer_step_requires_metadata = False

    def update_request_offsets(self) -> None:
        self._physical_lengths_applied = False
        if self.runner is None or self._request_offsets_cpu is None or self._request_offsets_device is None:
            return
        num_reqs = self.runner.input_batch.num_reqs
        offsets = tuple(
            self.active[request_id].removed_tokens if request_id in self.active else 0
            for request_id in self.runner.input_batch.req_ids[:num_reqs]
        )
        self._has_per_layer_rows = any(
            self.active[request_id].per_layer_physical_num_tokens is not None
            for request_id in self.runner.input_batch.req_ids[:num_reqs]
            if request_id in self.active
        )
        if offsets == self._offset_row_snapshot:
            return
        rows = max(len(self._offset_row_snapshot), num_reqs)
        self._request_offsets_cpu[:rows].zero_()
        if offsets:
            self._request_offsets_cpu[:num_reqs].copy_(torch.tensor(offsets, dtype=torch.int64))
        self._request_offsets_device[:rows].copy_(self._request_offsets_cpu[:rows], non_blocking=True)
        self._offset_row_snapshot = offsets
        self._has_active_rows = any(offsets)

    def _layer_offsets(self, request_ids: list[str]) -> list[tuple[int, ...]]:
        offsets_by_request = {
            key: dict(self.active[key].per_layer_physical_num_tokens or ()) for key in request_ids if key in self.active
        }
        return [
            tuple(
                self.active[key].semantic_anchor
                - offsets_by_request[key].get(layer.name, self.active[key].physical_anchor)
                if key in self.active
                else 0
                for key in request_ids
            )
            for layer in self.layer_caches
        ]


def _shift_positions(positions, request_indices, request_offsets, output):
    """Keep semantic RoPE positions intact; map only physical cache writes."""
    torch.sub(positions, request_offsets.index_select(0, request_indices.to(torch.long)), out=output)
    return output
