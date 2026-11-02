# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-HUST/vllm-ascend-kvcompress-hust, f337858 (see NOTICE).
"""Pickle-safe scheduler-to-worker KV compression transaction plan."""

from __future__ import annotations

from dataclasses import dataclass

PLAN_ATTRIBUTE = "_ascend_pyramidkv_plans_v4"


@dataclass(frozen=True)
class CompressionPlan:
    """Keep source KV immutable until the worker has written private blocks."""

    semantic_anchor: int
    physical_anchor: int
    source_block_ids: tuple[int, ...]
    destination_block_ids: tuple[int, ...]


@dataclass(frozen=True)
class ActiveCompression:
    semantic_anchor: int
    physical_anchor: int
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None

    @property
    def removed_tokens(self) -> int:
        return self.semantic_anchor - self.physical_anchor


def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size
