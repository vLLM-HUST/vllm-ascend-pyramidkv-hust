# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-independent PyramidKV budgets, GQA scoring and K/V selection.

Clean-room implementation of the observable KVCache-Factory semantics at
fc6f8f4c3d8ca7a1849a2ef67ff5fca8d285a6f0 (MIT); no host patches or custom kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PyramidKVConfig:
    """Validated provider-owned algorithm configuration."""

    max_capacity_prompt: int = 512
    min_compression_prompt_tokens: int = 4096
    window_size: int = 8
    kernel_size: int = 7
    pooling: str = "maxpool"
    beta: int = 20
    kv_cache_granularity: str = "kv_head"
    gqa_score_aggregation: str = "mean"
    merge: None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PyramidKVConfig:
        expected = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - expected)
        if unknown:
            raise ValueError("unknown PyramidKV provider_config fields: " + ", ".join(unknown))
        config = cls(**values)
        config._validate()
        return config

    def _validate(self) -> None:
        integer_fields = {
            "max_capacity_prompt": self.max_capacity_prompt,
            "min_compression_prompt_tokens": self.min_compression_prompt_tokens,
            "window_size": self.window_size,
            "kernel_size": self.kernel_size,
            "beta": self.beta,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.max_capacity_prompt <= self.window_size:
            raise ValueError(
                "max_capacity_prompt must be greater than window_size, got "
                f"{self.max_capacity_prompt} and {self.window_size}"
            )
        if self.min_compression_prompt_tokens < self.max_capacity_prompt:
            raise ValueError(
                "min_compression_prompt_tokens must be greater than or equal "
                "to max_capacity_prompt, got "
                f"{self.min_compression_prompt_tokens} and "
                f"{self.max_capacity_prompt}"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd to preserve score length, got {self.kernel_size}")
        if self.pooling != "maxpool":
            raise ValueError(f"pooling must be 'maxpool' in the first release, got {self.pooling!r}")
        if self.kv_cache_granularity != "kv_head":
            raise ValueError(
                f"kv_cache_granularity must be 'kv_head' in the first release, got {self.kv_cache_granularity!r}"
            )
        if self.gqa_score_aggregation != "mean":
            raise ValueError(
                f"gqa_score_aggregation must be 'mean' in the first release, got {self.gqa_score_aggregation!r}"
            )
        if self.merge is not None:
            raise ValueError(f"merge must be null in the first release, got {self.merge!r}")

    def retained_tokens(
        self,
        prompt_tokens: int,
        layer_index: int,
        num_hidden_layers: int,
    ) -> int:
        """Return the layer's final retained length, including recent window."""
        if prompt_tokens <= 0:
            raise ValueError(f"prompt_tokens must be positive, got {prompt_tokens}")
        if num_hidden_layers <= 1:
            raise ValueError(f"num_hidden_layers must be greater than one, got {num_hidden_layers}")
        if not 0 <= layer_index < num_hidden_layers:
            raise ValueError(f"layer_index {layer_index} is outside [0, {num_hidden_layers})")
        if prompt_tokens <= self.min_compression_prompt_tokens:
            return prompt_tokens

        past_budget = self.max_capacity_prompt - self.window_size
        if prompt_tokens < 2 * past_budget:
            return self.max_capacity_prompt

        minimum_past = past_budget // self.beta
        maximum_past = 2 * past_budget - minimum_past
        available_past = prompt_tokens - self.window_size
        if maximum_past >= available_past:
            maximum_past = available_past
            minimum_past = 2 * past_budget - maximum_past
        step = (maximum_past - minimum_past) // (num_hidden_layers - 1)
        layer_past = maximum_past - layer_index * step
        return layer_past + self.window_size


@dataclass(frozen=True)
class PyramidKVSelection:
    """One layer's selected compact K/V representation."""

    key: torch.Tensor
    value: torch.Tensor
    selected_past_indices: torch.Tensor | None
    retained_tokens: int
    compressed: bool


def select_pyramid_indices(
    query_window: torch.Tensor,
    key: torch.Tensor,
    config: PyramidKVConfig,
    *,
    layer_index: int,
    num_hidden_layers: int,
) -> tuple[torch.Tensor | None, int]:
    """Select retained history positions from the final query window."""
    if query_window.ndim != 4 or key.ndim != 4:
        raise ValueError("query_window and key must have [batch, heads, tokens, dim]")
    if query_window.shape[0] != key.shape[0] or query_window.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key batch and head dimensions must match, got {query_window.shape} and {key.shape}")
    num_query_heads = query_window.shape[1]
    num_kv_heads = key.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"query heads {num_query_heads} must be divisible by KV heads {num_kv_heads}")

    prompt_tokens = key.shape[2]
    retained_tokens = config.retained_tokens(prompt_tokens, layer_index, num_hidden_layers)
    if retained_tokens >= prompt_tokens:
        return None, prompt_tokens
    if query_window.shape[2] != config.window_size:
        raise ValueError(
            "compressed PyramidKV selection requires exactly window_size query "
            f"tokens, got {query_window.shape[2]} and {config.window_size}"
        )

    groups = num_query_heads // num_kv_heads
    grouped_query = query_window.reshape(
        query_window.shape[0],
        num_kv_heads,
        groups,
        config.window_size,
        query_window.shape[-1],
    )
    scores = torch.matmul(
        grouped_query,
        key.unsqueeze(2).transpose(-2, -1),
    ) / math.sqrt(query_window.shape[-1])
    causal_mask = torch.triu(
        torch.full(
            (config.window_size, config.window_size),
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        ),
        diagonal=1,
    )
    scores[..., -config.window_size :] += causal_mask
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_window.dtype)
    history_scores = probabilities[..., : -config.window_size].sum(dim=-2)
    history_scores = history_scores.mean(dim=2)
    pooled_scores = F.max_pool1d(
        history_scores,
        kernel_size=config.kernel_size,
        stride=1,
        padding=config.kernel_size // 2,
    )

    retained_past = retained_tokens - config.window_size
    selected = pooled_scores.topk(retained_past, dim=-1).indices
    return selected, retained_tokens


def materialize_pyramid_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    selected_past_indices: torch.Tensor,
    retained_tokens: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize compact K/V from provider-selected history positions."""
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("key and value must be matching [batch, heads, tokens, dim] tensors")
    retained_past = retained_tokens - window_size
    expected_shape = (key.shape[0], key.shape[1], retained_past)
    if tuple(selected_past_indices.shape) != expected_shape:
        raise ValueError(f"selected indices must have shape {expected_shape}, got {tuple(selected_past_indices.shape)}")
    gather_index = selected_past_indices.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
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
    return compact_key, compact_value


def select_pyramid_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: PyramidKVConfig,
    *,
    layer_index: int,
    num_hidden_layers: int,
) -> PyramidKVSelection:
    """Select history independently per KV head using GQA mean scores."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have [batch, heads, tokens, dim]")
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes must match, got {key.shape} and {value.shape}")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError(f"query/key batch, token, and head dimensions must match, got {query.shape} and {key.shape}")
    prompt_tokens = query.shape[2]
    selected, retained_tokens = select_pyramid_indices(
        query[:, :, -config.window_size :, :],
        key,
        config,
        layer_index=layer_index,
        num_hidden_layers=num_hidden_layers,
    )
    if selected is None:
        return PyramidKVSelection(
            key=key,
            value=value,
            selected_past_indices=None,
            retained_tokens=prompt_tokens,
            compressed=False,
        )
    if prompt_tokens <= config.window_size:
        raise ValueError(f"prompt length {prompt_tokens} must exceed window_size {config.window_size}")
    compact_key, compact_value = materialize_pyramid_kv(
        key,
        value,
        selected,
        retained_tokens,
        config.window_size,
    )
    return PyramidKVSelection(
        key=compact_key,
        value=compact_value,
        selected_past_indices=selected,
        retained_tokens=retained_tokens,
        compressed=True,
    )
