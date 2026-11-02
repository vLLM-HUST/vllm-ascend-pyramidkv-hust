"""Runner hooks must route dynamically discovered per-layer state to its owner."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend_pyramidkv.adapter.contracts import LayerCache
from vllm_ascend_pyramidkv.adapter.metadata import AttentionMetadata
from vllm_ascend_pyramidkv.adapter.plugin import _install_runner_hooks
from vllm_ascend_pyramidkv.adapter.transaction import ActiveCompression
from vllm_ascend_pyramidkv.adapter.worker import RUNNER_PROVIDER_ATTRIBUTE, PyramidKVWorker


@dataclass
class AscendMetadata:
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens_list: list[int]
    slot_mapping: torch.Tensor


AscendMetadata.__module__ = "vllm_ascend.attention.attention_v1"


@pytest.mark.parametrize("build_metadata", [True, False])
def test_runner_routes_dynamic_layer_lengths_and_enforces_metadata_guard(build_metadata):
    class Table:
        slot_mapping = SimpleNamespace(gpu=torch.tensor([12928]))

        def native(self, num_reqs, query_start_loc, positions):
            assert num_reqs == 1 and query_start_loc.tolist() == [0, 1]
            self.slot_mapping.gpu.copy_(12800 + positions)

    Table._ascend_pyramidkv_patch_v3_slot_mapping_original = Table.native
    table = Table()

    class Runner:
        compilation_config = SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"))
        input_batch = SimpleNamespace(num_reqs=1, req_ids=["r"], block_table=SimpleNamespace(block_tables=[table]))
        positions = torch.tensor([300])
        req_indices = SimpleNamespace(gpu=torch.tensor([0]))
        query_start_loc = SimpleNamespace(gpu=torch.tensor([0, 1]))
        max_num_tokens = 4

        def load_model(self):
            pass

        def initialize_kv_cache(self, config):
            pass

        def _update_states(self, output):
            pass

        def sample_tokens(self, grammar):
            pass

        def _build_attention_metadata(self, num_tokens, num_reqs):
            base = AscendMetadata(torch.tensor([129]), torch.tensor([129]), [129], table.slot_mapping.gpu)
            return {"layer.0": base}, None

        def execute_model(self, scheduler_output):
            return self._build_attention_metadata(1, 1) if build_metadata else None

    _install_runner_hooks(Runner, object())
    runner = Runner()
    worker = PyramidKVWorker.__new__(PyramidKVWorker)
    worker.requires_per_layer_physical_state = False
    worker.query_window_tokens = 0
    worker.metadata = AttentionMetadata(
        SimpleNamespace(speculative_config=None), {"r": ActiveCompression(300, 128, (("layer.0", 64),))}, False
    )
    metadata = worker.metadata
    metadata.runner = runner
    metadata.layer_caches = (LayerCache("layer.0", 0, torch.empty(0), torch.empty(0)),)
    metadata._has_active_rows = metadata._has_per_layer_rows = metadata._physical_lengths_applied = True
    metadata._request_offsets_cpu = torch.tensor([172])
    setattr(runner, RUNNER_PROVIDER_ATTRIBUTE, worker)
    output = SimpleNamespace(num_scheduled_tokens={"r": 1})
    if not build_metadata:
        with pytest.raises(RuntimeError, match="metadata hook was bypassed"):
            runner.execute_model(output)
    else:
        result, _ = runner.execute_model(output)
        assert result["layer.0"].seq_lens.tolist() == [65]
        assert result["layer.0"].slot_mapping.tolist() == [12864]
        assert table.slot_mapping.gpu.tolist() == [12928]
