# SPDX-License-Identifier: Apache-2.0
"""Resolve schema-v1 values from an active host with a standalone fallback."""

try:
    from vllm.config.kv_cache_compression import (
        KVCacheCompressionConfig as KVCacheCompressionConfigLike,
    )
    from vllm.v1.kv_cache_compression import (
        KVCacheCompressionCompatibility,
        KVCacheCompressionPlan,
        KVCacheCompressionRuntimeSpec,
    )
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("vllm"):
        raise
    from vllm_ascend_pyramidkv.legacy_contracts import (
        KVCacheCompressionCompatibility,
        KVCacheCompressionConfigLike,
        KVCacheCompressionPlan,
        KVCacheCompressionRuntimeSpec,
    )

__all__ = [
    "KVCacheCompressionCompatibility",
    "KVCacheCompressionConfigLike",
    "KVCacheCompressionPlan",
    "KVCacheCompressionRuntimeSpec",
]
