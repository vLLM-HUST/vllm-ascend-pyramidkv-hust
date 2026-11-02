# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend_pyramidkv.adapter.contracts import (
    CompressionRequest,
    CompressionResult,
    LayerCache,
    MethodRuntimeSpec,
    QueryObservation,
)
from vllm_ascend_pyramidkv.adapter.host_validation import (
    allowed_scheduler_block_sizes,
    common_compatibility_reasons,
    find_full_attention_group,
    unpack_layer_cache,
)
from vllm_ascend_pyramidkv.adapter.metadata import AttentionMetadata
from vllm_ascend_pyramidkv.adapter.transaction import PLAN_ATTRIBUTE, ActiveCompression, CompressionPlan
from vllm_ascend_pyramidkv.adapter.worker import (
    PendingCompression,
    PyramidKVWorker,
    _expand_scheduler_block_ids,
)


def _output_with_plan(
    semantic: int, source: tuple[int, ...], destination: tuple[int, ...], scheduled: int
) -> SimpleNamespace:
    output = SimpleNamespace(num_scheduled_tokens={"r": scheduled})
    setattr(
        output,
        PLAN_ATTRIBUTE,
        {"r": CompressionPlan(semantic, 128, source, destination)},
    )
    return output


class _RecordingMethod:
    name = "recording"

    def __init__(self) -> None:
        self.last_request: CompressionRequest | None = None

    def compress(self, request: CompressionRequest) -> CompressionResult:
        self.last_request = request
        return CompressionResult(physical_num_tokens=128)


class _RecordingBlockTable:
    def __init__(self) -> None:
        self.added = None

    def add_row(self, block_ids, row_idx) -> None:
        self.added = block_ids, row_idx


def test_async_scheduling_is_admitted_by_worker_compatibility(monkeypatch) -> None:
    runner_type = type("NPUModelRunner", (SimpleNamespace,), {})
    runner_type.__module__ = "vllm_ascend.worker.model_runner_v1"
    runner = runner_type(use_sparse=False, use_compress=False)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            block_size=128,
            cache_dtype="auto",
            mamba_cache_mode="none",
        ),
        speculative_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=True),
        model_config=SimpleNamespace(
            is_hybrid=False,
            use_mla=False,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(num_key_value_heads=2),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )

    assert common_compatibility_reasons(config, runner) == ()

    config.cache_config.enable_prefix_caching = True
    assert common_compatibility_reasons(config, runner) == ()

    config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=2)
    assert "speculative decoding is unqualified" in " ".join(common_compatibility_reasons(config, runner))

    config.model_config.is_hybrid = True
    config.model_config.hf_text_config.model_type = "qwen3_5_moe_text"
    config.cache_config.mamba_cache_mode = "align"
    monkeypatch.setenv("VLLM_ASCEND_PYRAMIDKV_EXPERIMENTAL_MTP2", "1")
    assert common_compatibility_reasons(config, runner) == ()


def _provider() -> PyramidKVWorker:
    provider = PyramidKVWorker.__new__(PyramidKVWorker)
    provider.method = _RecordingMethod()
    provider.requires_per_layer_physical_state = False
    provider.query_window_tokens = 0
    provider.query_layer_caches = ()
    provider.speculative_cache_layer = None
    provider._query_step_output = None
    provider._query_seen_layers = set()
    provider._query_forward_complete = False
    provider._query_tracked_ids = set()
    provider.runtime_spec = MethodRuntimeSpec(True, 256, 128, 128)
    provider.layer_caches = (LayerCache("model.layers.0.self_attn", 0, torch.empty(0), torch.empty(0)),)
    provider.pending = {}
    provider.active = {}
    provider.attention_group_index = 0
    provider.scheduler_block_size = 128
    provider.cache_block_size = 128
    provider.cache_blocks_per_scheduler_block = 1
    provider.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
    )
    provider.metadata = AttentionMetadata(provider.vllm_config, provider.active, False)
    return provider


def test_qwen35_hybrid_accepts_host_promoted_logical_block_size() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=True,
            hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        )
    )

    assert allowed_scheduler_block_sizes(config) == frozenset({128, 1152, 2048})


def test_prefix_caching_is_admitted_with_private_destination_protocol(
    monkeypatch,
) -> None:
    runner_cls = type("NPUModelRunner", (), {})
    runner_cls.__module__ = "vllm_ascend.worker.model_runner_v1"
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            block_size=128,
            cache_dtype="auto",
        ),
        speculative_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        model_config=SimpleNamespace(
            is_hybrid=False,
            use_mla=False,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(num_key_value_heads=2),
        ),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )

    assert common_compatibility_reasons(config, runner_cls()) == ()

    config.model_config.is_hybrid = True
    config.model_config.hf_text_config.model_type = "qwen3_5_moe_text"
    config.cache_config.mamba_cache_mode = "align"
    assert common_compatibility_reasons(config, runner_cls()) == ()

    config.scheduler_config.async_scheduling = True
    assert common_compatibility_reasons(config, runner_cls()) == ()

    config.speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=2,
        num_speculative_tokens_per_batch_size=None,
    )
    assert any(
        "speculative decoding is unqualified" in reason for reason in common_compatibility_reasons(config, runner_cls())
    )
    monkeypatch.setenv("VLLM_ASCEND_PYRAMIDKV_EXPERIMENTAL_MTP2", "1")
    assert common_compatibility_reasons(config, runner_cls()) == ()

    config.speculative_config.num_speculative_tokens = 3
    assert any(
        "speculative decoding is unqualified" in reason for reason in common_compatibility_reasons(config, runner_cls())
    )
    config.speculative_config.num_speculative_tokens = 2
    config.speculative_config.method = "eagle"
    assert any(
        "speculative decoding is unqualified" in reason for reason in common_compatibility_reasons(config, runner_cls())
    )
    config.speculative_config.method = "mtp"

    config.cache_config.mamba_cache_mode = "all"
    assert "hybrid compression requires mamba_cache_mode='none' or 'align'" in (
        common_compatibility_reasons(config, runner_cls())
    )


def test_scheduler_blocks_expand_to_consecutive_kernel_cache_blocks() -> None:
    actual = _expand_scheduler_block_ids((2, 0), 16, torch.device("cpu"))

    torch.testing.assert_close(
        actual,
        torch.tensor([*range(32, 48), *range(0, 16)], dtype=torch.long),
    )


def test_current_standardized_cache_uses_ascend_unpacker() -> None:
    k_cache = torch.empty(2, 128, 1, 4)
    v_cache = torch.empty_like(k_cache)

    class Impl:
        def _unpack_kv_cache(self, cache):
            assert cache == "standardized-cache"
            return k_cache, v_cache

    layer = SimpleNamespace(kv_cache=["standardized-cache"], impl=Impl())
    actual_k, actual_v = unpack_layer_cache("layer", layer)
    assert actual_k is k_cache
    assert actual_v is v_cache


def test_hybrid_group_finder_selects_only_full_attention_group() -> None:
    pytest.importorskip("vllm.v1.kv_cache_interface")
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    full_attention = object.__new__(FullAttentionSpec)
    recurrent_state = object.__new__(MambaSpec)
    groups = [
        SimpleNamespace(kv_cache_spec=recurrent_state),
        SimpleNamespace(kv_cache_spec=full_attention),
    ]

    assert find_full_attention_group(groups) == 1


def test_final_prefill_compresses_and_arms_worker_commit() -> None:
    provider = _provider()
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(device=torch.device("cpu"), requests={"r": request})

    provider.compress_scheduled_requests(_output_with_plan(300, (2, 0, 3), (50,), 44))

    assert provider.pending["r"] == PendingCompression(300, 128, (50,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.physical_num_tokens == 300
    assert provider.method.last_request.source_block_ids == ((2, 0, 3),)
    assert provider.method.last_request.destination_block_ids == ((50,),)


def test_worker_uses_scheduler_private_destination_for_prefix_caching() -> None:
    provider = _provider()
    provider.vllm_config.cache_config.enable_prefix_caching = True
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(device=torch.device("cpu"), requests={"r": request})
    provider.compress_scheduled_requests(_output_with_plan(300, (2, 0, 3), (100,), 44))

    assert provider.pending["r"] == PendingCompression(300, 128, (100,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.source_block_ids == ((2, 0, 3),)
    assert provider.method.last_request.destination_block_ids == ((100,),)


def test_prefix_worker_skips_request_without_scheduler_transaction() -> None:
    provider = _provider()
    provider.vllm_config.cache_config.enable_prefix_caching = True
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"),
        requests={
            "r": SimpleNamespace(
                num_computed_tokens=256,
                num_prompt_tokens=300,
                max_tokens=128,
                block_ids=([2, 0, 3],),
            )
        },
    )
    output = SimpleNamespace(num_scheduled_tokens={"r": 44})

    provider.compress_scheduled_requests(output)

    assert not provider.pending
    assert provider.method.last_request is None


def test_speculative_worker_requires_scheduler_transaction() -> None:
    provider = _provider()
    provider.vllm_config.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"),
        requests={
            "r": SimpleNamespace(
                num_computed_tokens=256,
                num_prompt_tokens=300,
                max_tokens=128,
                block_ids=([2, 0, 3],),
            )
        },
    )

    provider.compress_scheduled_requests(SimpleNamespace(num_scheduled_tokens={"r": 44}))

    assert not provider.pending
    assert provider.method.last_request is None


def test_worker_commit_truncates_tables_and_activates_offsets() -> None:
    provider = _provider()
    block_table = _RecordingBlockTable()
    request = SimpleNamespace(block_ids=([2, 0, 3],))
    provider.runner = SimpleNamespace(
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, block_table=block_table),
    )
    provider.pending["r"] = PendingCompression(300, 128, (50,))
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"r": 1},
    )

    provider.before_update_states(output)

    assert request.block_ids == ([50],)
    assert block_table.added == (([50],), 0)
    assert provider.active["r"] == ActiveCompression(300, 128)


def test_worker_commit_preserves_non_attention_group_tables() -> None:
    provider = _provider()
    provider.attention_group_index = 1
    block_table = _RecordingBlockTable()
    recurrent_state_blocks = [11]
    request = SimpleNamespace(block_ids=(recurrent_state_blocks, [2, 0, 3]))
    provider.runner = SimpleNamespace(
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, block_table=block_table),
    )
    provider.pending["r"] = PendingCompression(300, 128, (50,))
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"r": 1},
    )

    provider.before_update_states(output)

    assert request.block_ids == (recurrent_state_blocks, [50])
    assert block_table.added == ((recurrent_state_blocks, [50]), 0)


def test_worker_waits_for_request_reschedule_before_async_table_switch() -> None:
    provider = _provider()
    block_table = _RecordingBlockTable()
    request = SimpleNamespace(block_ids=([2, 0, 3],))
    provider.runner = SimpleNamespace(
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, block_table=block_table),
    )
    provider.pending["r"] = PendingCompression(300, 128, (50,))

    unrelated = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"other": 1},
    )
    provider.before_update_states(unrelated)
    assert request.block_ids == ([2, 0, 3],)
    assert provider.pending["r"] == PendingCompression(300, 128, (50,))
    assert block_table.added is None

    provider.before_update_states(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            num_scheduled_tokens={"r": 1},
        )
    )
    assert request.block_ids == ([50],)
    assert "r" not in provider.pending


def test_worker_discards_preempted_compression_state() -> None:
    provider = _provider()
    provider.runner = SimpleNamespace(
        requests={},
        input_batch=SimpleNamespace(req_id_to_index={}),
    )
    provider.pending["r"] = PendingCompression(300, 128, (2,))
    provider.active["r"] = ActiveCompression(300, 128)
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"r"},
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={},
    )

    provider.before_update_states(output)

    assert "r" not in provider.pending
    assert "r" not in provider.active


def test_repeated_compression_uses_current_physical_window() -> None:
    provider = _provider()
    request = SimpleNamespace(
        num_computed_tokens=427,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([5, 9],),
    )
    provider.runner = SimpleNamespace(device=torch.device("cpu"), requests={"r": request})
    provider.active["r"] = ActiveCompression(300, 128, (("model.layers.0.self_attn", 64),))

    provider.compress_scheduled_requests(_output_with_plan(428, (5, 9), (50,), 1))

    assert provider.pending["r"] == PendingCompression(428, 128, (50,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.physical_num_tokens == 256
    assert provider.method.last_request.per_layer_physical_num_tokens == (("model.layers.0.self_attn", 192),)


def test_per_layer_result_commits_only_on_acknowledged_request_step() -> None:
    provider = _provider()
    provider.layer_caches += (LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),)
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        device=torch.device("cpu"),
        requests={
            "r": SimpleNamespace(
                num_computed_tokens=256,
                num_prompt_tokens=300,
                max_tokens=128,
                block_ids=([2, 0, 3],),
            )
        },
        input_batch=SimpleNamespace(req_id_to_index={}, block_table=_RecordingBlockTable()),
    )
    provider.method.compress = lambda request: CompressionResult(
        128,
        (("model.layers.1.self_attn", 64), ("model.layers.0.self_attn", 128)),
    )
    provider.compress_scheduled_requests(_output_with_plan(300, (2, 0, 3), (50,), 44))
    expected = (
        ("model.layers.0.self_attn", 128),
        ("model.layers.1.self_attn", 64),
    )
    assert provider.pending["r"].per_layer_physical_num_tokens == expected
    other_step = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"other": 1},
    )
    provider.before_update_states(other_step)
    assert "r" in provider.pending and "r" not in provider.active

    other_step.num_scheduled_tokens = {"r": 1}
    provider.before_update_states(other_step)
    assert provider.active["r"].per_layer_physical_num_tokens == expected
    assert "r" not in provider.pending

    other_step.num_scheduled_tokens = {}
    other_step.preempted_req_ids = {"r"}
    provider.before_update_states(other_step)
    assert "r" not in provider.active


@pytest.mark.parametrize("reset_kind", ["finished", "preempted", "resumed"])
def test_per_layer_state_clears_on_request_reset(reset_kind) -> None:
    provider = _provider()
    layers = (("model.layers.0.self_attn", 64),)
    provider.pending["r"] = PendingCompression(300, 128, (50,), layers)
    provider.active["r"] = ActiveCompression(300, 128, layers)
    provider.runner = SimpleNamespace(requests={"r": object()}, input_batch=SimpleNamespace(req_id_to_index={}))
    output = SimpleNamespace(
        finished_req_ids={"r"} if reset_kind == "finished" else set(),
        preempted_req_ids={"r"} if reset_kind == "preempted" else set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"r"} if reset_kind == "resumed" else set()),
        num_scheduled_tokens={},
    )
    provider.before_update_states(output)
    assert not provider.pending and not provider.active


def test_per_layer_state_clears_if_request_disappears_after_update() -> None:
    provider = _provider()
    layers = (("model.layers.0.self_attn", 64),)
    provider.pending["r"] = PendingCompression(300, 128, (50,), layers)
    provider.active["r"] = ActiveCompression(300, 128, layers)
    provider.runner = SimpleNamespace(requests={})
    provider.after_update_states(None)
    assert not provider.pending and not provider.active


@pytest.mark.parametrize(
    "lengths",
    [
        (),
        (("model.layers.0.self_attn", 128),),
        (("model.layers.0.self_attn", 128), ("model.layers.0.self_attn", 64)),
        (("model.layers.0.self_attn", 128), ("wrong", 64)),
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 0)),
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 129)),
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", True)),
    ],
)
def test_per_layer_result_rejects_invalid_maps(lengths) -> None:
    provider = _provider()
    provider.layer_caches += (LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),)
    with pytest.raises(RuntimeError, match="per-layer physical lengths"):
        provider._validate_per_layer_lengths(lengths, 128)


def test_uniform_per_layer_result_keeps_shared_metadata_path() -> None:
    provider = _provider()
    provider.layer_caches += (LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),)
    assert (
        provider._validate_per_layer_lengths(
            (("model.layers.1.self_attn", 128), ("model.layers.0.self_attn", 128)),
            128,
        )
        is None
    )


def test_undeclared_nonuniform_lengths_fail_closed_under_graph_replay() -> None:
    provider = _provider()
    provider.layer_caches += (LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),)
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE"))
    )
    with pytest.raises(RuntimeError, match="declare per-layer physical state"):
        provider._validate_per_layer_lengths(
            (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 64)),
            128,
        )


def test_query_observation_can_exclude_auxiliary_mtp_cache() -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("mtp.layers.0.self_attn.attn", 0, torch.empty(0), torch.empty(0)),
        LayerCache("model.layers.3.self_attn", 3, torch.empty(0), torch.empty(0)),
    )
    provider.method = SimpleNamespace(query_layer_indices=(3,))
    provider.query_window_tokens = 2

    selected = provider._resolve_query_layer_caches()

    assert tuple(cache.name for cache in selected) == ("model.layers.3.self_attn",)


def test_worker_skips_short_decode_request() -> None:
    provider = _provider()
    provider.runtime_spec = MethodRuntimeSpec(True, 256, 128, 128, 64)
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        max_tokens=32,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(device=torch.device("cpu"), requests={"r": request})

    provider.compress_scheduled_requests(SimpleNamespace(num_scheduled_tokens={"r": 44}))

    assert not provider.pending
    assert provider.method.last_request is None


class _ObservingMethod(_RecordingMethod):
    def __init__(self) -> None:
        super().__init__()
        self.buffers: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.observation: QueryObservation | None = None
        self.observed_rows: torch.Tensor | None = None
        self.discarded: list[str] = []

    def capture_query(self, layer, query, spans) -> None:
        assert layer.layer_index == 0
        for span in spans:
            buffer = self.buffers.setdefault(span.request_id, torch.empty(2, 2))
            rows = query[span.start : span.end]
            count = self.counts.get(span.request_id, 0)
            if count < 2:
                copy = min(2 - count, len(rows))
                buffer[count : count + copy].copy_(rows[:copy])
                count += copy
                rows = rows[copy:]
            for row in rows:
                buffer[:-1].copy_(buffer[1:].clone())
                buffer[-1].copy_(row)
            self.counts[span.request_id] = count

    def complete_query_observation(self, observation: QueryObservation) -> None:
        assert observation.request_id == "r"
        assert observation.window_tokens == 2
        assert observation.layer_indices == (0,)
        assert self.counts["r"] == 2
        self.observation = observation
        self.observed_rows = self.buffers["r"].clone()

    def compress(self, request: CompressionRequest) -> CompressionResult:
        assert self.observation is not None
        assert request.plan is self.observation.plan
        return super().compress(request)

    def discard_query_observation(self, request_id: str) -> None:
        self.discarded.append(request_id)
        self.buffers.pop(request_id, None)
        self.counts.pop(request_id, None)


def _observing_provider() -> PyramidKVWorker:
    provider = _provider()
    provider.method = _ObservingMethod()
    provider.query_window_tokens = 2
    provider.query_layer_caches = provider.layer_caches
    request = SimpleNamespace(
        num_computed_tokens=298,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"),
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}),
        query_start_loc=SimpleNamespace(cpu=torch.tensor([0, 1])),
    )
    return provider


def test_query_observation_spans_chunked_prefill_and_matches_plan() -> None:
    provider = _observing_provider()
    first = SimpleNamespace(num_scheduled_tokens={"r": 1})

    provider.begin_query_step(first)
    provider.capture_attention_query("model.layers.0.self_attn", torch.tensor([[1.0, 2.0]]))
    provider.finish_query_step()
    stable_address = provider.method.buffers["r"].data_ptr()

    provider.runner.requests["r"].num_computed_tokens = 299
    final = _output_with_plan(300, (2, 0, 3), (50,), 1)
    provider.begin_query_step(final)
    provider.capture_attention_query("model.layers.0.self_attn", torch.tensor([[3.0, 4.0]]))
    provider.finish_query_step()
    assert provider.method.buffers["r"].data_ptr() == stable_address
    provider.compress_scheduled_requests(final)

    torch.testing.assert_close(
        provider.method.observed_rows,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert provider.method.observation.plan is getattr(final, PLAN_ATTRIBUTE)["r"]
    assert provider.method.discarded == ["r"]
    assert provider.method.buffers == {}


def test_query_observation_accepts_attention_backend_callback() -> None:
    provider = _observing_provider()
    output = SimpleNamespace(num_scheduled_tokens={"r": 1})

    provider.begin_query_step(output)
    provider.capture_attention_query("model.layers.0.self_attn", torch.tensor([[1.0, 2.0]]))
    provider.finish_query_step()

    torch.testing.assert_close(provider.method.buffers["r"][0], torch.tensor([1.0, 2.0]))
    assert provider.method.counts["r"] == 1


def test_query_observation_fails_closed_when_forward_hook_is_bypassed() -> None:
    provider = _observing_provider()
    output = SimpleNamespace(num_scheduled_tokens={"r": 1})

    provider.begin_query_step(output)
    with pytest.raises(RuntimeError, match="graph replay path is unvalidated"):
        provider.finish_query_step()
    provider.abort_query_step(output)

    assert not provider._query_forward_complete
    assert provider.method.discarded == ["r"]


def test_query_observation_is_discarded_on_preemption() -> None:
    provider = _observing_provider()
    output = SimpleNamespace(num_scheduled_tokens={"r": 1})
    provider.begin_query_step(output)
    provider.capture_attention_query("model.layers.0.self_attn", torch.tensor([[1.0, 2.0]]))
    provider.finish_query_step()

    provider.before_update_states(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids={"r"},
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            num_scheduled_tokens={},
        )
    )

    assert provider.method.buffers == {}
    assert provider.method.discarded == ["r"]


def test_worker_reads_output_limit_from_cached_request_sampling_params() -> None:
    provider = _provider()
    provider.runtime_spec = MethodRuntimeSpec(True, 256, 128, 128, 64)
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        sampling_params=SimpleNamespace(max_tokens=32),
        pooling_params=None,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(device=torch.device("cpu"), requests={"r": request})

    provider.compress_scheduled_requests(SimpleNamespace(num_scheduled_tokens={"r": 44}))

    assert not provider.pending
