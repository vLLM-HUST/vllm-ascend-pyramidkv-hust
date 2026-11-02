# SPDX-License-Identifier: Apache-2.0
"""Metadata behavior is tested independently of worker transaction orchestration."""

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend_pyramidkv.adapter.contracts import LayerCache
from vllm_ascend_pyramidkv.adapter.metadata import AttentionMetadata
from vllm_ascend_pyramidkv.adapter.transaction import ActiveCompression


def _metadata():
    config = SimpleNamespace(speculative_config=None, cache_config=SimpleNamespace(enable_prefix_caching=False))
    metadata = AttentionMetadata(config, {}, False)
    metadata.layer_caches = (LayerCache("model.layers.0.self_attn", 0, torch.empty(0), torch.empty(0)),)
    return metadata


def test_gdn_metadata_retains_semantic_lengths_after_attention_compaction() -> None:
    @dataclass
    class Metadata:
        num_reqs: int
        seq_lens: torch.Tensor
        _seq_lens_cpu: torch.Tensor
        seq_lens_cpu_upper_bound: torch.Tensor

    provider = _metadata()
    seq_lens = torch.tensor([300], dtype=torch.int64)
    cpu_lens = torch.tensor([300], dtype=torch.int64)
    provider.runner = SimpleNamespace(
        seq_lens=seq_lens,
        optimistic_seq_lens_cpu=cpu_lens,
        input_batch=SimpleNamespace(num_reqs=1),
    )
    provider._request_offsets_cpu = torch.tensor([172], dtype=torch.int64)
    provider._request_offsets_device = torch.tensor([172], dtype=torch.int64)
    provider._semantic_seq_lens_device = torch.empty_like(seq_lens)
    provider._semantic_seq_lens_cpu = torch.empty_like(cpu_lens)
    provider._has_active_rows = True

    provider.apply_physical_attention_lengths()
    gdn = provider.semantic_gdn_metadata(Metadata(1, seq_lens, cpu_lens, cpu_lens))

    assert seq_lens.tolist() == [128]
    assert cpu_lens.tolist() == [128]
    assert gdn.seq_lens.tolist() == [300]
    assert gdn._seq_lens_cpu.tolist() == [300]
    assert gdn.seq_lens_cpu_upper_bound.tolist() == [300]


def test_physical_positions_and_lengths_preserve_semantic_rope() -> None:
    provider = _metadata()
    provider.active["compressed"] = ActiveCompression(300, 128)
    provider._request_offsets_cpu = torch.tensor([172, 0], dtype=torch.int64)
    provider._request_offsets_device = torch.tensor([172, 0], dtype=torch.int64)
    provider._physical_positions = torch.empty(2, dtype=torch.int64)
    provider._semantic_seq_lens_device = torch.empty(2, dtype=torch.int64)
    provider._semantic_seq_lens_cpu = torch.empty(2, dtype=torch.int64)
    provider._has_active_rows = True
    positions = torch.tensor([300, 20], dtype=torch.int64)
    provider.runner = SimpleNamespace(
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 1], dtype=torch.long)),
        input_batch=SimpleNamespace(num_reqs=2),
        seq_lens=torch.tensor([301, 21], dtype=torch.int64),
        optimistic_seq_lens_cpu=torch.tensor([301, 21], dtype=torch.int64),
    )

    physical = provider.physical_positions_for_slot_mapping(positions)
    provider.apply_physical_attention_lengths()

    torch.testing.assert_close(positions, torch.tensor([300, 20]))
    torch.testing.assert_close(physical, torch.tensor([128, 20]))
    torch.testing.assert_close(provider.runner.seq_lens, torch.tensor([129, 21]))
    torch.testing.assert_close(provider.runner.optimistic_seq_lens_cpu, torch.tensor([129, 21]))


def test_metadata_corrections_reuse_storage_and_replace_changed_rows() -> None:
    provider = _metadata()
    reference = torch.empty(4, dtype=torch.int64)
    original = provider._metadata_correction_tensor("layer:offsets", (172, 0), reference)
    repeated = provider._metadata_correction_tensor("layer:offsets", (172, 0), reference)
    assert repeated is original
    # Reordering, reset, batch growth/shrink must not reuse stale row values.
    for values in ((0, 172), (0, 0), (172,), (172, 0, 200)):
        current = provider._metadata_correction_tensor("layer:offsets", values, reference)
        assert current.tolist() == list(values)
        assert current is not original
        assert original.tolist() == [172, 0]  # Earlier in-flight storage is immutable.
    changed_dtype = provider._metadata_correction_tensor("layer:offsets", (172, 0, 200), reference.to(torch.int32))
    assert changed_dtype.dtype == torch.int32
    assert len(provider._metadata_tensor_cache) == 1


def test_metadata_correction_fields_do_not_alias() -> None:
    provider = _metadata()
    reference = torch.empty(1, dtype=torch.int64)
    a = provider._metadata_correction_tensor("a:delta", (10,), reference)
    b = provider._metadata_correction_tensor("b:delta", (20,), reference)
    assert a.tolist() == [10]
    assert b.tolist() == [20]
    assert a.data_ptr() != b.data_ptr()


@pytest.mark.parametrize("batched", [False, True])
def test_per_layer_eager_metadata_uses_distinct_write_slots_and_lengths(
    monkeypatch,
    batched,
) -> None:
    @dataclass
    class AscendMetadata:
        seq_lens: torch.Tensor
        seq_lens_cpu: torch.Tensor
        seq_lens_list: list[int]
        slot_mapping: torch.Tensor

    AscendMetadata.__module__ = "vllm_ascend.attention.attention_v1"

    class BlockTable:
        def __init__(self):
            self.slot_mapping = SimpleNamespace(gpu=torch.tensor([12928, 2580]))

        def native(self, num_reqs, query_start_loc, positions):
            assert num_reqs == 2
            assert query_start_loc.tolist() == [0, 1, 2]
            # Two private 128-token pages for request 0, one for request 1.
            blocks = ((100, 101), (20,))
            for token, position in enumerate(positions.tolist()):
                request = token
                self.slot_mapping.gpu[token] = blocks[request][position // 128] * 128 + position % 128

    BlockTable._ascend_pyramidkv_patch_v3_slot_mapping_original = BlockTable.native
    table = BlockTable()
    provider = _metadata()
    provider.layer_caches += (LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),)
    provider.active["compressed"] = ActiveCompression(
        300,
        128,
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 64)),
    )
    provider._has_per_layer_rows = True
    provider._physical_lengths_applied = True
    provider._request_offsets_cpu = torch.tensor([172, 0])
    provider.runner = SimpleNamespace(
        max_num_tokens=4,
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        input_batch=SimpleNamespace(
            req_ids=["compressed", "plain"],
            block_table=SimpleNamespace(block_tables=[table]),
        ),
        positions=torch.tensor([300, 20]),
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 1])),
        query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 1, 2])),
    )
    shared = AscendMetadata(
        torch.tensor([129, 21]),
        torch.tensor([129, 21]),
        [129, 21],
        table.slot_mapping.gpu,
    )
    if batched:
        table.dcp_world_size = 1
        table.block_size = table.logical_block_size = 128
        table.physical_block_size = 256
        table.blocks_per_phys_block = 2
        table.block_table = SimpleNamespace(gpu=torch.tensor([[100, 101], [20, 21]], dtype=torch.int32))
        provider._batched_slot_storage = torch.empty((2, 4), dtype=torch.int64)
        provider._per_layer_slot_buffers = {
            layer.name: provider._batched_slot_storage[index] for index, layer in enumerate(provider.layer_caches)
        }
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: shared for layer in provider.layer_caches}, "common"), 2, 2
    )
    first, second = (metadata[layer.name] for layer in provider.layer_caches)
    assert common == "common"
    assert first is not shared
    assert first.seq_lens_list == [129, 21]
    first_slot_address = first.slot_mapping.data_ptr()
    second_slot_address = second.slot_mapping.data_ptr()
    original_corrections = dict(provider._metadata_tensor_cache)
    assert second.seq_lens_list == [65, 21]
    assert second.seq_lens.tolist() == [65, 21]
    first_length_snapshot = second.seq_lens.clone()
    assert second.slot_mapping.tolist() == [100 * 128 + 64, 20 * 128 + 20]
    assert table.slot_mapping.gpu.tolist() == [12928, 2580]
    for slots in provider._per_layer_slot_buffers.values():
        assert slots[2:].tolist() == [-1, -1]
        slots[2:].fill_(777)

    provider.runner.positions[0] = 305
    table.slot_mapping.gpu[0] = 12933
    next_shared = AscendMetadata(
        torch.tensor([134, 21]),
        torch.tensor([134, 21]),
        [134, 21],
        table.slot_mapping.gpu,
    )

    def forbid_repeated_transfer(*args, **kwargs):
        raise AssertionError("Unchanged corrections must not construct another tensor")

    with monkeypatch.context() as scoped:
        scoped.setattr(torch, "tensor", forbid_repeated_transfer)
        next_metadata, _ = provider.per_layer_attention_metadata(
            ({layer.name: next_shared for layer in provider.layer_caches}, None), 2, 2
        )
    assert torch.equal(second.seq_lens, first_length_snapshot)
    assert next_metadata["model.layers.0.self_attn"].slot_mapping.data_ptr() == first_slot_address
    assert next_metadata["model.layers.1.self_attn"].slot_mapping.data_ptr() == second_slot_address
    assert next_metadata["model.layers.1.self_attn"].seq_lens_list == [70, 21]
    assert next_metadata["model.layers.1.self_attn"].slot_mapping.tolist() == [
        100 * 128 + 69,
        20 * 128 + 20,
    ]
    assert table.slot_mapping.gpu.tolist() == [12933, 2580]
    for slots in provider._per_layer_slot_buffers.values():
        assert slots[2:].tolist() == [-1, -1]

    for key, (values, tensor) in original_corrections.items():
        next_values, next_tensor = provider._metadata_tensor_cache[key]
        assert values == next_values
        assert tensor is next_tensor

    if batched:
        # Unsupported host layouts must decline before touching captured rows.
        saved_slots = provider._batched_slot_storage.clone()
        for field, unsupported in (
            ("dcp_world_size", 2),
            ("logical_block_size", 64),
            ("physical_block_size", 512),
            ("cp_kv_cache_interleave_size", 2),
        ):
            prior = getattr(table, field, 1)
            setattr(table, field, unsupported)
            declined = provider._batched_per_layer_metadata(
                {layer.name: next_shared for layer in provider.layer_caches},
                None,
                table,
                ["compressed", "plain"],
                (172, 0),
                2,
                2,
            )
            assert declined is None
            assert torch.equal(provider._batched_slot_storage, saved_slots)
            setattr(table, field, prior)

    if batched:
        # The same request IDs in a different scheduler row order must move
        # their offsets and block-table rows together, without stale cache rows.
        provider.runner.input_batch.req_ids = ["plain", "compressed"]
        provider._request_offsets_cpu = torch.tensor([0, 172])
        provider.runner.positions = torch.tensor([20, 305])
        table.block_table.gpu = table.block_table.gpu.flip(0)
        table.slot_mapping.gpu[:] = torch.tensor([2580, 12933])
        reordered = AscendMetadata(
            torch.tensor([21, 134]),
            torch.tensor([21, 134]),
            [21, 134],
            table.slot_mapping.gpu,
        )
        reordered_meta, _ = provider.per_layer_attention_metadata(
            ({layer.name: reordered for layer in provider.layer_caches}, None), 2, 2
        )
        second = reordered_meta["model.layers.1.self_attn"]
        assert second.seq_lens.tolist() == [21, 70]
        assert second.slot_mapping.tolist() == [2580, 12869]
        assert second.slot_mapping.data_ptr() == second_slot_address


def test_declared_per_layer_state_accepts_layer_aware_full_graph(
    monkeypatch,
) -> None:
    class AscendAttentionBackendImpl:
        _ascend_pyramidkv_layer_aware_graph_v1 = True

    ascend = ModuleType("vllm_ascend")
    attention = ModuleType("vllm_ascend.attention")
    attention_v1 = ModuleType("vllm_ascend.attention.attention_v1")
    attention_v1.AscendAttentionBackendImpl = AscendAttentionBackendImpl
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", attention)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention_v1", attention_v1)

    provider = _metadata()
    provider.requires_per_layer_physical_state = True
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE")),
        use_dcp=False,
        pcp_enabled=False,
        use_async_spec_decode=True,
    )
    provider.vllm_config.speculative_config = None

    provider.validate_host()


def test_per_layer_full_graph_capture_accepts_synthetic_empty_batch() -> None:
    @dataclass
    class Metadata:
        slot_mapping: torch.Tensor

    provider = _metadata()
    provider.requires_per_layer_physical_state = True
    provider.validate_host = lambda: None  # type: ignore[method-assign]
    provider.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=[], num_reqs=0),
    )
    provider.layer_caches += (LayerCache("mtp.layers.0.self_attn", 0, torch.empty(0), torch.empty(0)),)
    provider.speculative_cache_layer = provider.layer_caches[-1]
    provider._per_layer_slot_buffers = {
        layer.name: torch.empty(48, dtype=torch.int64) for layer in provider.layer_caches
    }
    shared = Metadata(torch.arange(48))
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: shared for layer in provider.layer_caches}, shared), 48, 48
    )
    captured_slots = [metadata[layer.name].slot_mapping for layer in provider.layer_caches]
    assert all(slots.data_ptr() != shared.slot_mapping.data_ptr() for slots in captured_slots)
    assert captured_slots[0].data_ptr() != captured_slots[1].data_ptr()
    assert common.slot_mapping.data_ptr() == captured_slots[1].data_ptr()
    # A captured cache-write kernel keeps this address: replay must update
    # its contents independently for unequal physical layer lengths.
    for index, layer in enumerate(provider.layer_caches):
        provider._per_layer_slot_buffers[layer.name][0] = 64 + index
    assert [int(slots[0]) for slots in captured_slots] == [64, 65]
    assert int(shared.slot_mapping[0]) == 0
    assert provider._per_layer_metadata_seen


def test_per_layer_full_graph_capture_rejects_partial_real_batch() -> None:
    provider = _metadata()
    provider.requires_per_layer_physical_state = True
    provider.validate_host = lambda: None  # type: ignore[method-assign]
    provider.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["real"], num_reqs=1),
    )

    with pytest.raises(RuntimeError, match="batch dimensions changed"):
        provider.per_layer_attention_metadata(
            ({"model.layers.0.self_attn": object()}, object()),
            48,
            48,
        )


@pytest.mark.parametrize("batched", [False, True])
def test_mtp_rejected_draft_reuses_layer_specific_physical_slots(batched) -> None:
    @dataclass
    class AscendMetadata:
        seq_lens: torch.Tensor
        seq_lens_cpu: torch.Tensor
        seq_lens_list: list[int]
        slot_mapping: torch.Tensor

    AscendMetadata.__module__ = "vllm_ascend.attention.attention_v1"

    @dataclass
    class CommonMetadata:
        seq_lens: torch.Tensor
        _seq_lens_cpu: torch.Tensor
        seq_lens_cpu_upper_bound: torch.Tensor
        slot_mapping: torch.Tensor
        seq_lens_cpu: torch.Tensor | None = None
        num_computed_tokens_cpu: torch.Tensor | None = None

    class BlockTable:
        def __init__(self) -> None:
            self.slot_mapping = SimpleNamespace(gpu=torch.tensor([128, 129, 130], dtype=torch.int64))

        def native(self, num_reqs, query_start_loc, positions):
            assert num_reqs == 1
            assert int(query_start_loc[-1]) == len(positions)
            for token, position in enumerate(positions.tolist()):
                self.slot_mapping.gpu[token] = position

    BlockTable._ascend_pyramidkv_patch_v3_slot_mapping_original = BlockTable.native
    table = BlockTable()
    provider = _metadata()
    provider.layer_caches += (LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),)
    provider.active["r"] = ActiveCompression(
        300,
        128,
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 64)),
    )
    provider._has_per_layer_rows = True
    provider._physical_lengths_applied = True
    provider._request_offsets_cpu = torch.tensor([172])
    provider.speculative_cache_layer = provider.layer_caches[1]
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        input_batch=SimpleNamespace(
            req_ids=["r"],
            block_table=SimpleNamespace(block_tables=[table]),
        ),
        positions=torch.tensor([300, 301, 302]),
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 0, 0])),
        query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 3])),
    )
    first = AscendMetadata(
        torch.tensor([131]),
        torch.tensor([131]),
        [131],
        table.slot_mapping.gpu,
    )
    first_common = CommonMetadata(
        torch.tensor([131]),
        torch.tensor([131]),
        torch.tensor([131]),
        table.slot_mapping.gpu,
        torch.tensor([131]),
        torch.tensor([300]),
    )
    if batched:
        table.dcp_world_size = 1
        table.block_size = table.logical_block_size = 128
        table.physical_block_size = 256
        table.blocks_per_phys_block = 2
        table.block_table = SimpleNamespace(gpu=torch.tensor([[0, 1]], dtype=torch.int32))
        provider._batched_slot_storage = torch.empty((2, 4), dtype=torch.int64)
        provider._per_layer_slot_buffers = {
            layer.name: provider._batched_slot_storage[index] for index, layer in enumerate(provider.layer_caches)
        }
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: first for layer in provider.layer_caches}, first_common),
        3,
        1,
    )
    assert metadata["model.layers.1.self_attn"].slot_mapping.tolist() == [64, 65, 66]
    assert common.seq_lens.tolist() == [67]
    assert common._seq_lens_cpu.tolist() == [67]
    assert common.num_computed_tokens_cpu.tolist() == [64]
    assert common.slot_mapping.tolist() == [64, 65, 66]

    # Only the first speculative token was accepted. The next iteration writes
    # semantic position 301 again and must overwrite the rejected draft slot.
    provider.runner.positions = torch.tensor([301])
    provider.runner.req_indices.gpu = torch.tensor([0])
    provider.runner.query_start_loc.gpu = torch.tensor([0, 1])
    table.slot_mapping.gpu[0] = 129
    second = AscendMetadata(
        torch.tensor([130]),
        torch.tensor([130]),
        [130],
        table.slot_mapping.gpu[:1],
    )
    second_common = CommonMetadata(
        torch.tensor([130]),
        torch.tensor([130]),
        torch.tensor([130]),
        table.slot_mapping.gpu[:1],
        torch.tensor([130]),
        torch.tensor([301]),
    )
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: second for layer in provider.layer_caches}, second_common),
        1,
        1,
    )
    assert metadata["model.layers.1.self_attn"].slot_mapping.tolist() == [65]
    assert metadata["model.layers.1.self_attn"].seq_lens_list == [66]
    assert common.seq_lens.tolist() == [66]
    assert common.num_computed_tokens_cpu.tolist() == [65]
    assert common.slot_mapping.tolist() == [65]


def test_per_layer_metadata_bypass_fails_closed() -> None:
    provider = _metadata()
    provider._has_per_layer_rows = True
    provider.begin_per_layer_step(SimpleNamespace(num_scheduled_tokens={"r": 1}))
    with pytest.raises(RuntimeError, match="metadata hook was bypassed"):
        provider.finish_per_layer_step()


def test_per_layer_metadata_allows_async_empty_execution_step() -> None:
    provider = _metadata()
    provider._has_per_layer_rows = True

    provider.begin_per_layer_step(SimpleNamespace(num_scheduled_tokens={}))
    provider.finish_per_layer_step()

    assert not provider._per_layer_step_requires_metadata
