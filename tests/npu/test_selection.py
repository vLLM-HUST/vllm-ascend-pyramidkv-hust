# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import math
import os

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend_pyramidkv.provider import (
    PyramidKVAscendConfig,
    select_pyramid_kv,
)

pytestmark = pytest.mark.skipif(
    os.getenv("PYRAMIDKV_RUN_NPU_TESTS") != "1",
    reason="set PYRAMIDKV_RUN_NPU_TESTS=1 on an Ascend test host",
)


def _legacy_repeated_gqa_selection(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retained_tokens: int,
    window_size: int,
    kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen repeated-GQA path used only as a real-device oracle."""
    groups = query.shape[1] // key.shape[1]
    repeated_key = key.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query[:, :, -window_size:, :], repeated_key.transpose(2, 3)) / math.sqrt(query.shape[-1])
    scores[..., -window_size:] += torch.triu(
        torch.full(
            (window_size, window_size),
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        ),
        diagonal=1,
    )
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    history = probabilities[..., :-window_size].sum(dim=-2)
    history = history.reshape(query.shape[0], key.shape[1], groups, history.shape[-1]).mean(dim=2)
    pooled = F.max_pool1d(
        history,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    selected = pooled.topk(retained_tokens - window_size, dim=-1).indices
    gather_index = selected.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
    compact_key = torch.cat(
        (
            key[:, :, :-window_size, :].gather(2, gather_index),
            key[:, :, -window_size:, :],
        ),
        dim=2,
    )
    compact_value = torch.cat(
        (
            value[:, :, :-window_size, :].gather(2, gather_index),
            value[:, :, -window_size:, :],
        ),
        dim=2,
    )
    return compact_key, compact_value, selected


def test_qwen_grouped_gqa_matches_repeated_oracle_on_npu() -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")

    config = PyramidKVAscendConfig.from_dict(
        {
            "max_capacity_prompt": 512,
            "min_compression_prompt_tokens": 512,
            "window_size": 8,
            "kernel_size": 7,
            "pooling": "maxpool",
            "beta": 20,
            "kv_cache_granularity": "kv_head",
            "gqa_score_aggregation": "mean",
            "merge": None,
        }
    )
    generator = torch.Generator().manual_seed(20260903)
    query = torch.randn(1, 40, 1024, 128, generator=generator).to(device="npu", dtype=torch.bfloat16)
    key = torch.randn(1, 8, 1024, 128, generator=generator).to(device="npu", dtype=torch.bfloat16)
    value = torch.randn(1, 8, 1024, 128, generator=generator).to(device="npu", dtype=torch.bfloat16)
    retained_tokens = config.retained_tokens(1024, 47, 48)

    actual = select_pyramid_kv(
        query,
        key,
        value,
        config,
        layer_index=47,
        num_hidden_layers=48,
    )
    expected_key, expected_value, expected_indices = _legacy_repeated_gqa_selection(
        query,
        key,
        value,
        retained_tokens,
        config.window_size,
        config.kernel_size,
    )
    torch.npu.synchronize()

    assert torch.equal(actual.selected_past_indices, expected_indices)
    torch.testing.assert_close(actual.key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(actual.value, expected_value, rtol=0, atol=0)
