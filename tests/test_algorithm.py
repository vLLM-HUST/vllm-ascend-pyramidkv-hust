import math

import pytest
import torch

from vllm_ascend_pyramidkv.algorithm import PyramidKVConfig, select_pyramid_kv


def _small_config(**updates: object) -> PyramidKVConfig:
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
    return PyramidKVConfig.from_dict(values)


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
