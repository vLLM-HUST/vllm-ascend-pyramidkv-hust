"""Buffer ownership across cache binding and request lifecycle changes."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend_pyramidkv.adapter.contracts import LayerCache


def _runner(request_id, max_tokens):
    return SimpleNamespace(
        device=torch.device("cpu"),
        max_num_reqs=2,
        max_num_tokens=max_tokens,
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        input_batch=SimpleNamespace(num_reqs=1, req_ids=[request_id]),
        req_indices=SimpleNamespace(gpu=torch.zeros(max_tokens, dtype=torch.long)),
        seq_lens=torch.zeros(2, dtype=torch.int32),
        optimistic_seq_lens_cpu=torch.zeros(2, dtype=torch.int32),
    )


@pytest.mark.parametrize("batched", [False, True])
def test_rebind_replaces_graph_buffers_and_discards_old_step_state(monkeypatch, batched):
    from vllm_ascend_pyramidkv.adapter.metadata import AttentionMetadata
    from vllm_ascend_pyramidkv.adapter.transaction import ActiveCompression

    monkeypatch.setenv("VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA", str(int(batched)))
    active = {"old": ActiveCompression(300, 128)}
    metadata = AttentionMetadata(SimpleNamespace(speculative_config=None), active, True)
    layer = LayerCache("layer.0", 0, torch.empty(0), torch.empty(0))
    metadata.bind(_runner("old", 4), (layer,), 0, None)
    metadata.update_request_offsets()
    assert metadata.physical_positions_for_slot_mapping(torch.tensor([300])).tolist() == [128]
    old_slots = metadata._per_layer_slot_buffers[layer.name]
    correction = metadata._metadata_correction_tensor("offset", (172,), torch.empty(1, dtype=torch.int64))
    metadata.apply_physical_attention_lengths()
    metadata.begin_per_layer_step(SimpleNamespace(num_scheduled_tokens={"old": 1}))

    metadata.bind(_runner("new", 8), (layer,), 0, None)
    metadata.finish_per_layer_step()  # The old step cannot demand metadata from the new cache.
    common = object()
    assert metadata.semantic_gdn_metadata(common) is common
    positions = torch.tensor([334])
    assert metadata.physical_positions_for_slot_mapping(positions) is positions
    slots = metadata._per_layer_slot_buffers[layer.name]
    assert slots.numel() == 8 and slots.data_ptr() != old_slots.data_ptr()
    fresh = metadata._metadata_correction_tensor("offset", (0,), torch.empty(1, dtype=torch.int64))
    assert fresh.tolist() == [0] and correction.tolist() == [172]

    active["new"] = ActiveCompression(300, 128)
    metadata.update_request_offsets()
    assert metadata.physical_positions_for_slot_mapping(positions).tolist() == [162]
    active.clear()  # A worker reset must be visible without rebuilding the component.
    metadata.update_request_offsets()
    assert metadata.physical_positions_for_slot_mapping(positions) is positions
