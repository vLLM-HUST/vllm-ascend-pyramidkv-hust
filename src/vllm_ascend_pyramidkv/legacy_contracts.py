# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen schema-v1 values used to verify the legacy provider baseline.

These data classes mirror the provider-neutral contract merged by legacy Core
PR #232. They make the extracted provider independently importable and
testable; they are not a replacement for a current host contract. The package
manifest remains ``import_only`` until an active host exports an equivalent
versioned interface and an adapter translates these values at that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

KV_CACHE_COMPRESSION_SCHEMA_VERSION = 1


class KVCacheCompressionConfigLike(Protocol):
    """Structural subset of the historical Core configuration."""

    schema_version: int
    provider: str
    provider_config: dict[str, Any]


@dataclass(frozen=True)
class KVCacheCompressionConfig:
    """Standalone configuration used by migration tests and tooling."""

    provider: str
    provider_config: dict[str, Any]
    schema_version: int = KV_CACHE_COMPRESSION_SCHEMA_VERSION


@dataclass(frozen=True)
class KVCacheCompressionRuntimeSpec:
    """Provider-derived scheduling limits from legacy schema v1."""

    schema_version: int
    provider: str
    requires_private_destination: bool
    compression_threshold_tokens: int
    required_recompute_tokens: int
    max_physical_num_tokens: int


@dataclass(frozen=True)
class KVCacheCompressionCompatibility:
    """Serializable compatibility result from legacy schema v1."""

    schema_version: int
    provider: str
    supported: bool
    reasons: tuple[str, ...]
    platform: str
    provider_factory: str | None = None
    backend: str | None = None
    model_architecture: str | None = None
    dtype: str | None = None
    cache_layout: str | None = None
    block_size: int | None = None
    runtime_spec: KVCacheCompressionRuntimeSpec | None = None


@dataclass(frozen=True)
class KVCacheCompressionPlan:
    """Worker-to-scheduler compaction transaction from legacy schema v1."""

    schema_version: int
    provider: str
    request_id: str
    semantic_num_tokens: int
    physical_num_tokens: int
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...]
    expected_block_ids: tuple[tuple[int, ...], ...]
    kv_cache_group_id: int = 0
