# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Lazy, side-effect-free registry for the extracted PyramidKV provider."""

from importlib import import_module
from typing import Any

from vllm_ascend_pyramidkv.legacy_contracts import (
    KV_CACHE_COMPRESSION_SCHEMA_VERSION,
    KVCacheCompressionConfigLike,
)

PYRAMIDKV_ASCEND_PROVIDER = "pyramidkv_ascend"


def get_kv_cache_compression_provider(
    config: KVCacheCompressionConfigLike,
) -> Any:
    """Construct the provider after validating its frozen schema-v1 identity."""

    if config.schema_version != KV_CACHE_COMPRESSION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported KV cache compression schema_version "
            f"{config.schema_version}; expected "
            f"{KV_CACHE_COMPRESSION_SCHEMA_VERSION}"
        )
    if config.provider != PYRAMIDKV_ASCEND_PROVIDER:
        raise ValueError(
            f"unknown Ascend KV cache compression provider {config.provider!r}; expected {PYRAMIDKV_ASCEND_PROVIDER!r}"
        )

    module = import_module("vllm_ascend_pyramidkv.provider")
    return module.PyramidKVAscendProvider.from_core_config(config)
