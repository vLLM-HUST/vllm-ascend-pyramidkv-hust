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
