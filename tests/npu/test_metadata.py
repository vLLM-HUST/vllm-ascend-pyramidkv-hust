"""Synthetic integer metadata comparison against the real Ascend native kernel.

No model, K/V cache or task score. The small metadata dataclasses are fixtures;
the slot-mapping oracle is the pinned host's actual BlockTable kernel.
"""

import itertools
import os
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend_pyramidkv.adapter.contracts import LayerCache
from vllm_ascend_pyramidkv.adapter.worker import ActiveCompression, PyramidKVWorker


@dataclass
class AscendMetadata:
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens_list: list
    slot_mapping: torch.Tensor


AscendMetadata.__module__ = "vllm_ascend.attention.attention_v1"


@dataclass
class Common:
    seq_lens: torch.Tensor
    _seq_lens_cpu: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor
    slot_mapping: torch.Tensor
    num_computed_tokens_cpu: torch.Tensor


@pytest.mark.skipif(os.environ.get("PYRAMIDKV_RUN_NPU_TESTS") != "1", reason="opt-in Ascend native oracle")
def test_metadata_matches_native_ascend_slots():
    pytest.importorskip("torch_npu")
    from vllm_ascend.worker.block_table import BlockTable

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    torch.manual_seed(20261028)
    max_tokens = 4096
    max_reqs = 4
    layer_count = 11
    provider = PyramidKVWorker.__new__(PyramidKVWorker)
    provider._metadata_tensor_cache = {}
    provider._batched_slot_storage = torch.empty((layer_count, max_tokens), dtype=torch.int64, device=device)
    layers = tuple(LayerCache(f"layer.{i}", i, torch.empty(0), torch.empty(0)) for i in range(layer_count))
    provider.layer_caches = layers
    provider.speculative_cache_layer = layers[-1]
    addresses = [provider._batched_slot_storage[i].data_ptr() for i in range(layer_count)]
    records = []
    for num_reqs, query_size, mode, advance, seq_device in itertools.product(
        (1, 4), (1, 3, 512), ("none", "mixed", "all"), (0, 1), ("cpu", "npu:0")
    ):
        counts = [query_size] * num_reqs
        if num_reqs == 4 and query_size == 3:
            counts = [3, 1, 2, 3]
        req_ids = [f"r{i}" for i in range(num_reqs)]
        if advance:
            req_ids.reverse()
        active = {}
        for req in req_ids:
            number = int(req[1:])
            if mode == "all" or (mode == "mixed" and number % 2 == 0):
                retained = [2048 - i * 128 for i in range(10)] + [2048]
                active[req] = ActiveCompression(
                    8192 + number * 2048,
                    2048,
                    tuple((layer.name, n) for layer, n in zip(layers, retained, strict=True)),
                )
        starts = [active[req].semantic_anchor + advance if req in active else 100 + advance for req in req_ids]
        global_offsets = tuple(active[req].removed_tokens if req in active else 0 for req in req_ids)
        positions_cpu = torch.cat(
            [torch.arange(start, start + count, dtype=torch.int64) for start, count in zip(starts, counts, strict=True)]
        )
        req_indices = torch.repeat_interleave(torch.arange(num_reqs, dtype=torch.int64), torch.tensor(counts))
        qstart = torch.tensor([0] + list(itertools.accumulate(counts)), dtype=torch.int32, device=device)
        positions = positions_cpu.to(device)
        req_device = req_indices.to(device)
        table = BlockTable.__new__(BlockTable)
        table.dcp_world_size = 1
        table.dcp_rank = 0
        table.cp_kv_cache_interleave_size = 1
        table.block_size = table.logical_block_size = 128
        table.physical_block_size = 2048
        table.blocks_per_phys_block = 16
        table.max_num_batched_tokens = max_tokens
        physical_ids = torch.randint(1, 2048, (max_reqs, 16), dtype=torch.int32)
        expanded = (physical_ids.unsqueeze(-1) * 16 + torch.arange(16, dtype=torch.int32)).reshape(max_reqs, 256)
        table.block_table = SimpleNamespace(gpu=expanded.to(device))
        table.slot_mapping = SimpleNamespace(
            gpu=torch.empty(max_tokens + 2 * max_reqs, dtype=torch.int32, device=device)
        )
        native = BlockTable.compute_slot_mapping
        global_positions = positions - torch.tensor(global_offsets, dtype=torch.int64, device=device).index_select(
            0, req_device
        )
        native(table, num_reqs, qstart, global_positions)
        original_slots = table.slot_mapping.gpu[: len(positions)].clone()
        lengths = [start + count - offset for start, count, offset in zip(starts, counts, global_offsets, strict=True)]
        padded = lengths + [0] * (max_reqs - num_reqs)
        seq = torch.tensor(padded, dtype=torch.int32, device=seq_device)
        original_seq = seq.clone()
        common_seq = torch.tensor(padded, dtype=torch.int32, device=device)
        base = AscendMetadata(seq, seq, list(padded), table.slot_mapping.gpu[: len(positions)])
        common = Common(
            common_seq,
            torch.tensor(padded, dtype=torch.int32),
            torch.tensor(padded, dtype=torch.int32),
            base.slot_mapping,
            torch.tensor(starts, dtype=torch.int32),
        )
        provider.active = active
        provider.runner = SimpleNamespace(positions=positions, req_indices=SimpleNamespace(gpu=req_device))
        provider._batched_slot_storage.fill_(123456)
        result = provider._batched_per_layer_metadata(
            {layer.name: base for layer in layers}, common, table, req_ids, global_offsets, len(positions), num_reqs
        )
        assert result is not None
        metadata, updated_common = result
        assert torch.equal(seq, original_seq)
        assert torch.equal(table.slot_mapping.gpu[: len(positions)], original_slots)
        for index, layer in enumerate(layers):
            offsets = tuple(
                active[req].semantic_anchor - dict(active[req].per_layer_physical_num_tokens)[layer.name]
                if req in active
                else 0
                for req in req_ids
            )
            expected_lengths = [
                start + count - offset for start, count, offset in zip(starts, counts, offsets, strict=True)
            ] + [0] * (max_reqs - num_reqs)
            assert metadata[layer.name].seq_lens.cpu().tolist() == expected_lengths
            assert metadata[layer.name].seq_lens_list == expected_lengths
            physical = positions - torch.tensor(offsets, dtype=torch.int64, device=device).index_select(0, req_device)
            native(table, num_reqs, qstart, physical)
            assert torch.equal(
                metadata[layer.name].slot_mapping, table.slot_mapping.gpu[: len(positions)].to(torch.int64)
            )
            assert bool((provider._batched_slot_storage[index, len(positions) :] == -1).all().item())
            assert metadata[layer.name].slot_mapping.data_ptr() == addresses[index]
        assert updated_common.num_computed_tokens_cpu.tolist() == [
            s - o for s, o in zip(starts, global_offsets, strict=True)
        ]
        assert updated_common._seq_lens_cpu.tolist() == padded
        assert updated_common.seq_lens.cpu().tolist() == padded
        assert common_seq.cpu().tolist() == padded
        records.append(
            {
                "seq_device": seq_device,
                "num_reqs": num_reqs,
                "query_size": query_size,
                "actual_counts": counts,
                "compression": mode,
                "advance": advance,
                "slots_equal_native": True,
                "lengths_equal_cpu": True,
                "padding_cleared": True,
                "captured_addresses_stable": True,
            }
        )
    torch.npu.synchronize()
    assert len(provider._metadata_tensor_cache) <= 7
    assert len(records) == 72
