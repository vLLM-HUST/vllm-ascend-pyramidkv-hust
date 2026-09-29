from types import SimpleNamespace

import pytest
import torch
from vllm_ascend_kvcompress.methods.base import (
    CompressionRequest,
    KVCompressionMethod,
    LayerCache,
    ModelShape,
    QueryBatchSpan,
    QueryObservation,
)
from vllm_ascend_kvcompress.transaction import CompressionPlan

from vllm_ascend_pyramidkv.method import (
    QWEN35_MODEL_TYPES,
    TARGET_MODEL_DISPLAY_NAME,
    TARGET_MODEL_ID,
    PyramidKVMethod,
    create_pyramidkv_method,
)


def _options() -> dict[str, object]:
    return {
        "max_capacity_prompt": 128,
        "min_compression_prompt_tokens": 4096,
        "window_size": 4,
        "kernel_size": 1,
        "pooling": "maxpool",
        "beta": 2,
        "kv_cache_granularity": "kv_head",
        "gqa_score_aggregation": "mean",
        "merge": None,
    }


def _small_shape() -> ModelShape:
    return ModelShape(
        model_type="test_hybrid",
        num_layers=8,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        rope_theta=1_000_000.0,
        has_rope_scaling=False,
        rotary_dim=4,
        attention_layer_indices=(3, 7),
    )


def _config(*, tensor_parallel_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            quantization=None,
            enforce_eager=True,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
        ),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=True,
            enable_chunked_prefill=True,
        ),
        speculative_config=None,
    )


def _bound_method() -> tuple[PyramidKVMethod, tuple[LayerCache, ...]]:
    method = PyramidKVMethod(_options(), _config(), _small_shape())
    caches = tuple(
        LayerCache(
            name=f"model.layers.{layer_index}.self_attn.attn",
            layer_index=layer_index,
            k_cache=torch.randn(49, 128, 1, 8),
            v_cache=torch.randn(49, 128, 1, 8),
        )
        for layer_index in (3, 7)
    )
    method.bind_model_runner(SimpleNamespace(max_num_reqs=2, device=torch.device("cpu")), caches)
    return method, caches


def test_factory_exposes_block_aligned_runtime_contract() -> None:
    method = create_pyramidkv_method(_options(), _config(), _small_shape())

    assert isinstance(method, KVCompressionMethod)
    assert method.name == "pyramidkv"
    assert method.query_window_tokens == 4
    assert method.runtime_spec.compression_threshold_tokens == 4097
    assert method.runtime_spec.required_recompute_tokens == 4
    assert method.runtime_spec.max_physical_num_tokens == 2048


def test_validation_target_is_the_official_qwen35_35b_model() -> None:
    assert TARGET_MODEL_ID == "Qwen/Qwen3.5-35B-A3B"
    assert TARGET_MODEL_DISPLAY_NAME == "Qwen3.5-35B"
    assert {"qwen3_5_moe_text"} == QWEN35_MODEL_TYPES


def test_query_capture_spans_chunks_and_materializes_per_layer_state() -> None:
    torch.manual_seed(11)
    method, caches = _bound_method()
    expected_tails: dict[int, torch.Tensor] = {}
    for layer in caches:
        first = torch.randn(2, 2, 8)
        final = torch.randn(3, 2, 8)
        method.capture_query(layer, first, (QueryBatchSpan("request", 0, 2),))
        method.capture_query(layer, final, (QueryBatchSpan("request", 0, 3),))
        expected_tails[layer.layer_index] = torch.cat((first, final))[-4:]

    source = tuple(range(33))
    destination = tuple(range(33, 49))
    plan = CompressionPlan(4097, 2048, source, destination)
    method.complete_query_observation(
        QueryObservation(
            request_id="request",
            plan=plan,
            semantic_num_tokens=4097,
            window_tokens=4,
            layer_indices=(3, 7),
        )
    )

    slot = method._request_slots["request"]
    for index, layer in enumerate(caches):
        assert torch.equal(method._query_buffers[slot, index], expected_tails[layer.layer_index])

    result = method.compress(
        CompressionRequest(
            request_id="request",
            semantic_num_tokens=4097,
            physical_num_tokens=4097,
            source_block_ids=(source,),
            destination_block_ids=(destination,),
            source_block_ids_device=torch.tensor(source, dtype=torch.int32),
            destination_block_ids_device=torch.tensor(destination, dtype=torch.int32),
            plan=plan,
        )
    )

    assert result.physical_num_tokens == 2048
    assert result.per_layer_physical_num_tokens == (
        ("model.layers.3.self_attn.attn", 190),
        ("model.layers.7.self_attn.attn", 66),
    )
    method.discard_query_observation("request")
    assert "request" not in method._request_slots


def test_query_completion_rejects_incomplete_window() -> None:
    method, caches = _bound_method()
    method.capture_query(
        caches[0],
        torch.randn(1, 2, 8),
        (QueryBatchSpan("request", 0, 1),),
    )
    plan = CompressionPlan(4097, 2048, tuple(range(33)), tuple(range(33, 49)))

    with pytest.raises(RuntimeError, match="complete query window"):
        method.complete_query_observation(
            QueryObservation(
                request_id="request",
                plan=plan,
                semantic_num_tokens=4097,
                window_tokens=4,
                layer_indices=(3, 7),
            )
        )


def test_target_compatibility_is_explicit_and_fail_closed(monkeypatch) -> None:
    shape = ModelShape(
        model_type="qwen3_5_moe_text",
        num_layers=40,
        num_attention_heads=16,
        num_kv_heads=2,
        head_dim=256,
        rope_theta=10_000_000.0,
        has_rope_scaling=False,
        rotary_dim=64,
        attention_layer_indices=tuple(range(3, 40, 4)),
    )
    config = _config()
    method = PyramidKVMethod(_options(), config, shape)
    backend = type("AscendAttentionBackend", (), {})
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        attn_backend=backend,
    )
    monkeypatch.setattr(torch.version, "cann", "9.1.0", raising=False)

    assert method.compatibility_reasons(runner) == ()

    config.cache_config.enable_prefix_caching = True
    assert method.compatibility_reasons(runner) == ()

    config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=2)
    runner.compilation_config.cudagraph_mode = SimpleNamespace(name="FULL_AND_PIECEWISE")
    reasons = method.compatibility_reasons(runner)

    assert not any("required_recompute_tokens" in reason for reason in reasons)
    assert any("MTP" in reason for reason in reasons)
    assert any("graph replay" in reason for reason in reasons)
