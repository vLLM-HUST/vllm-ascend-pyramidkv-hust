import importlib
import sys
from dataclasses import replace

import pytest

from vllm_ascend_pyramidkv import registry
from vllm_ascend_pyramidkv.legacy_contracts import KVCacheCompressionConfig

PROVIDER_MODULE = "vllm_ascend_pyramidkv.provider"


def _config() -> KVCacheCompressionConfig:
    return KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config={
            "max_capacity_prompt": 512,
            "min_compression_prompt_tokens": 4096,
            "window_size": 8,
            "kernel_size": 7,
            "pooling": "maxpool",
            "beta": 20,
            "kv_cache_granularity": "kv_head",
            "gqa_score_aggregation": "mean",
            "merge": None,
        },
    )


def test_registry_loads_known_schema_v1_provider() -> None:
    provider = registry.get_kv_cache_compression_provider(_config())

    assert type(provider).__name__ == "PyramidKVAscendProvider"
    assert provider.config.max_capacity_prompt == 512


def test_registry_rejects_unknown_schema_before_import() -> None:
    sys.modules.pop(PROVIDER_MODULE, None)
    importlib.reload(registry)

    with pytest.raises(ValueError, match="schema_version 2; expected 1"):
        registry.get_kv_cache_compression_provider(replace(_config(), schema_version=2))

    assert PROVIDER_MODULE not in sys.modules


def test_registry_rejects_unknown_provider() -> None:
    sys.modules.pop(PROVIDER_MODULE, None)
    importlib.reload(registry)

    with pytest.raises(ValueError, match="unknown Ascend"):
        registry.get_kv_cache_compression_provider(replace(_config(), provider="other"))

    assert PROVIDER_MODULE not in sys.modules


def test_provider_config_errors_remain_explicit() -> None:
    config = replace(_config(), provider_config={"unknown": 1})

    with pytest.raises(ValueError, match="unknown PyramidKV.*unknown"):
        registry.get_kv_cache_compression_provider(config)
