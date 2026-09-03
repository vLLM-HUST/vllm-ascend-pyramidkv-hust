import math

import pytest
import torch

from vllm_ascend_pyramidkv.legacy_contracts import KVCacheCompressionConfig
from vllm_ascend_pyramidkv.provider import (
    PyramidKVAscendConfig,
    PyramidKVAscendProvider,
    PyramidKVCapabilityContext,
    select_pyramid_kv,
)


def _small_config(**updates: object) -> PyramidKVAscendConfig:
    values = {
        "max_capacity_prompt": 12,
        "min_compression_prompt_tokens": 12,
        "window_size": 4,
        "kernel_size": 1,
        "pooling": "maxpool",
        "beta": 2,
        "kv_cache_granularity": "kv_head",
        "gqa_score_aggregation": "mean",
        "merge": None,
    }
    values.update(updates)
    return PyramidKVAscendConfig.from_dict(values)


def _core_config(
    provider_config: dict[str, object] | None = None,
) -> KVCacheCompressionConfig:
    config = provider_config or {
        "max_capacity_prompt": 512,
        "min_compression_prompt_tokens": 4096,
        "window_size": 8,
        "kernel_size": 7,
        "pooling": "maxpool",
        "beta": 20,
        "kv_cache_granularity": "kv_head",
        "gqa_score_aggregation": "mean",
        "merge": None,
    }
    return KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config=config,
    )


def _context(**updates: object) -> PyramidKVCapabilityContext:
    values = {
        "platform": "npu",
        "device_name": "Ascend910B2",
        "cann_version": "9.0",
        "use_v2_model_runner": False,
        "enforce_eager": True,
        "cudagraph_mode": "NONE",
        "pa_shape_list": (),
        "backend": "AscendAttentionBackend",
        "model_architecture": "LlamaForCausalLM",
        "dtype": "torch.bfloat16",
        "quantization": None,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 32,
        "cache_layout": "standard_bf16_paged",
        "block_size": 128,
        "hash_block_size": 128,
        "max_model_len": 8192,
        "num_kv_cache_groups": 1,
        "full_attention_only": True,
        "prefix_caching": True,
        "chunked_prefill": True,
        "sliding_window": False,
        "speculative_decoding": False,
        "kv_transfer": False,
        "kv_offload": False,
        "cache_dtype": "auto",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "async_scheduling": True,
        "balance_scheduling": False,
        "dbo_enabled": False,
        "knorm_enabled": False,
        "missing_ops": (),
    }
    values.update(updates)
    return PyramidKVCapabilityContext(**values)


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"max_capacity_prompt": 4}, "greater than window_size"),
        ({"min_compression_prompt_tokens": True}, "positive integer"),
        ({"min_compression_prompt_tokens": 11}, "greater than or equal"),
        ({"window_size": True}, "positive integer"),
        ({"kernel_size": 2}, "must be odd"),
        ({"pooling": "avgpool"}, "maxpool"),
        ({"beta": 0}, "positive integer"),
        ({"kv_cache_granularity": "query_head"}, "kv_head"),
        ({"gqa_score_aggregation": "max"}, "mean"),
        ({"merge": "pivot"}, "must be null"),
        ({"extra": 1}, "unknown PyramidKV"),
    ],
)
def test_config_rejects_unsupported_values(updates: dict[str, object], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _small_config(**updates)


def test_layer_capacity_schedule_and_admission_boundary() -> None:
    config = _small_config()

    assert config.retained_tokens(11, 0, 2) == 11
    assert config.retained_tokens(12, 0, 2) == 12
    assert config.retained_tokens(13, 0, 2) == 12
    assert config.retained_tokens(20, 0, 2) == 16
    assert config.retained_tokens(20, 1, 2) == 8


def _independent_selection(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retained_tokens: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = query.shape[1] // key.shape[1]
    repeated_key = key.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query[:, :, -window_size:, :], repeated_key.transpose(2, 3)) / math.sqrt(query.shape[-1])
    mask = torch.triu(
        torch.full(
            (window_size, window_size),
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
        ),
        diagonal=1,
    )
    scores[..., -window_size:] += mask
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    history = probabilities[..., :-window_size].sum(dim=-2)
    grouped = history.reshape(history.shape[0], key.shape[1], groups, history.shape[-1]).mean(dim=2)
    selected = grouped.topk(retained_tokens - window_size, dim=-1).indices
    gather = selected.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
    compact_key = torch.cat(
        (key[:, :, :-window_size].gather(2, gather), key[:, :, -window_size:]),
        dim=2,
    )
    compact_value = torch.cat(
        (
            value[:, :, :-window_size].gather(2, gather),
            value[:, :, -window_size:],
        ),
        dim=2,
    )
    return compact_key, compact_value, selected


@pytest.mark.parametrize(("query_heads", "kv_heads"), [(8, 2), (40, 8)])
def test_gqa_mean_selection_matches_independent_oracle(query_heads: int, kv_heads: int) -> None:
    torch.manual_seed(7)
    config = _small_config()
    query = torch.randn(1, query_heads, 20, 8, dtype=torch.float32)
    key = torch.randn(1, kv_heads, 20, 8, dtype=torch.float32)
    value = torch.randn(1, kv_heads, 20, 8, dtype=torch.float32)

    result = select_pyramid_kv(
        query,
        key,
        value,
        config,
        layer_index=1,
        num_hidden_layers=2,
    )
    expected_key, expected_value, expected_indices = _independent_selection(
        query, key, value, retained_tokens=8, window_size=4
    )

    assert result.compressed
    assert result.retained_tokens == 8
    assert torch.equal(result.selected_past_indices, expected_indices)
    assert torch.equal(result.key, expected_key)
    assert torch.equal(result.value, expected_value)


def test_below_threshold_preserves_original_tensors() -> None:
    config = _small_config()
    query = torch.randn(1, 8, 12, 8)
    key = torch.randn(1, 2, 12, 8)
    value = torch.randn(1, 2, 12, 8)

    result = select_pyramid_kv(
        query,
        key,
        value,
        config,
        layer_index=1,
        num_hidden_layers=2,
    )

    assert not result.compressed
    assert result.key is key
    assert result.value is value
    assert result.selected_past_indices is None


def test_llama_and_qwen_profiles_fail_closed_by_geometry() -> None:
    provider = PyramidKVAscendProvider.from_core_config(_core_config())

    llama = provider.compatibility_report(_core_config(), _context(), "factory")
    qwen = provider.compatibility_report(
        _core_config(),
        _context(
            model_architecture="Qwen2ForCausalLM",
            num_attention_heads=40,
            num_kv_heads=8,
            num_hidden_layers=48,
        ),
        "factory",
    )
    wrong_qwen = provider.compatibility_report(
        _core_config(),
        _context(
            model_architecture="Qwen2ForCausalLM",
            num_attention_heads=28,
            num_kv_heads=4,
            num_hidden_layers=28,
        ),
        "factory",
    )

    assert llama.supported
    assert qwen.supported
    assert not wrong_qwen.supported
    assert any("query heads must be 40" in reason for reason in wrong_qwen.reasons)
    assert any("KV heads must be 8" in reason for reason in wrong_qwen.reasons)
    assert any("hidden layers must be 48" in reason for reason in wrong_qwen.reasons)


def test_default_runtime_spec_preserves_admission_and_recompute_guards() -> None:
    config = _core_config()
    provider = PyramidKVAscendProvider.from_core_config(config)

    report = provider.compatibility_report(config, _context(), "factory")

    assert report.supported
    assert report.runtime_spec is not None
    assert report.runtime_spec.compression_threshold_tokens == 4096
    assert report.runtime_spec.required_recompute_tokens == 8
    assert report.runtime_spec.max_physical_num_tokens == 991


def test_compatibility_report_aggregates_independent_failures() -> None:
    config = _core_config()
    provider = PyramidKVAscendProvider.from_core_config(config)

    report = provider.compatibility_report(
        config,
        _context(
            platform="cuda",
            device_name="Ascend910B4",
            cann_version="8.5.1",
            chunked_prefill=True,
            async_scheduling=True,
            balance_scheduling=True,
            kv_offload=True,
            missing_ops=("npu_scatter_nd_update_",),
        ),
        "factory",
    )

    assert not report.supported
    assert any("platform must be 'npu'" in reason for reason in report.reasons)
    assert any("device must be Ascend910B2" in reason for reason in report.reasons)
    assert any("chunked prefill requires CANN 9.0" in reason for reason in report.reasons)
    assert "balance scheduling is unsupported" in report.reasons
    assert "KV offload is unsupported" in report.reasons
    assert any("npu_scatter_nd_update_" in reason for reason in report.reasons)


def test_request_plan_commit_decode_and_cleanup_lifecycle() -> None:
    provider = PyramidKVAscendProvider(_small_config())
    layer_names = (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    )
    state = provider.begin_request("request", 20, ((3,),))
    key = torch.empty(1, 2, 8, 8)
    value = torch.empty_like(key)

    for layer_name, retained in zip(layer_names, (16, 8), strict=True):
        from vllm_ascend_pyramidkv.provider import PyramidKVSelection

        provider.record_prefill_layer(
            "request",
            layer_name,
            PyramidKVSelection(
                key=key[:, :, :retained],
                value=value[:, :, :retained],
                selected_past_indices=torch.zeros(1, 2, retained - 4, dtype=torch.long),
                retained_tokens=retained,
                compressed=True,
            ),
        )

    plan = provider.finalize_plan("request", layer_names, schema_version=1)
    assert plan.semantic_num_tokens == 20
    assert plan.physical_num_tokens == 16
    assert plan.expected_block_ids == ((3,),)

    provider.mark_committed("request", ((7,),))
    provider.accept_decode_block_table("request", ((7,),))
    provider.advance_decode("request", set(layer_names))

    assert state.committed
    assert state.semantic_num_tokens == 21
    assert [layer.physical_num_tokens for layer in state.layers.values()] == [17, 9]

    provider.cleanup_request("request")
    with pytest.raises(KeyError):
        provider.get_request_state("request")


@pytest.mark.parametrize("mode", ["PIECEWISE", "FULL_DECODE_ONLY"])
def test_cann_9_graph_modes_are_retained_as_legacy_capabilities(mode: str) -> None:
    config = _core_config()
    provider = PyramidKVAscendProvider.from_core_config(config)
    report = provider.compatibility_report(
        config,
        _context(enforce_eager=False, cudagraph_mode=mode),
        "factory",
    )

    assert report.supported


def test_cann_8_graph_mode_is_rejected() -> None:
    config = _core_config()
    provider = PyramidKVAscendProvider.from_core_config(config)
    report = provider.compatibility_report(
        config,
        _context(
            cann_version="8.5.1",
            enforce_eager=False,
            cudagraph_mode="PIECEWISE",
            chunked_prefill=False,
        ),
        "factory",
    )

    assert not report.supported
    assert any("graph execution requires CANN 9.0" in reason for reason in report.reasons)


@pytest.mark.parametrize(
    ("enforce_eager", "cudagraph_mode"),
    [(True, "NONE"), (False, "PIECEWISE")],
)
def test_unvalidated_current_host_cann_9_1_fails_closed(
    enforce_eager: bool,
    cudagraph_mode: str,
) -> None:
    config = _core_config()
    provider = PyramidKVAscendProvider.from_core_config(config)
    report = provider.compatibility_report(
        config,
        _context(
            cann_version="9.1.0",
            enforce_eager=enforce_eager,
            cudagraph_mode=cudagraph_mode,
            chunked_prefill=False,
        ),
        "factory",
    )

    assert not report.supported
    assert any(
        "CANN version must start" in reason or "graph execution requires CANN 9.0" in reason
        for reason in report.reasons
    )
