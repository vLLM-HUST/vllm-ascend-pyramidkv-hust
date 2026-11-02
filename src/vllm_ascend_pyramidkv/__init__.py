# SPDX-License-Identifier: Apache-2.0
"""Standalone PyramidKV plugin with lazy, default-off host activation."""

from __future__ import annotations

from typing import Any

__all__ = ["PyramidKVContractProposal", "get_provider", "register"]
__version__ = "0.3.0.dev0"


class PyramidKVContractProposal:
    """Descriptor-only proposal for the future method contract."""


def get_provider(config: Any) -> Any:
    """Lazily load the migration provider for offline verification.

    This helper is for standalone tests and does not register with a host.
    """

    from vllm_ascend_pyramidkv.registry import (
        get_kv_cache_compression_provider,
    )

    return get_kv_cache_compression_provider(config)


def register() -> None:
    """vLLM general-plugin entry point; disabled discovery needs no host imports."""
    from .adapter.config import assert_no_conflicting_adapter, extension_enabled

    if not extension_enabled():
        return
    assert_no_conflicting_adapter()
    from .adapter.plugin import register as activate

    activate()
