# SPDX-License-Identifier: Apache-2.0
"""PyramidKV Ascend migration package.

The top-level module intentionally imports neither PyTorch nor vLLM and never
activates the provider. Runtime registration remains blocked until the shared
KV-compression host accepts the required query-observation contract.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PyramidKVContractProposal", "get_provider"]
__version__ = "0.2.0.dev0"


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
