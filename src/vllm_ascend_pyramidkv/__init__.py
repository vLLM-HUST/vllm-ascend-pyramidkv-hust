# SPDX-License-Identifier: Apache-2.0
"""PyramidKV Ascend migration package.

The top-level module intentionally imports neither PyTorch nor vLLM and never
imports PyTorch or vLLM. Runtime activation is performed lazily by the Ascend
host only when Core configuration selects this provider.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PyramidKVContractProposal", "get_provider"]
__version__ = "0.2.0.dev0"


class PyramidKVContractProposal:
    """Descriptor carrier for the transactional host contract."""


def get_provider(config: Any) -> Any:
    """Lazily load the migration provider for offline verification.

    The Ascend host reaches the same factory through the provider entry point.
    """

    from vllm_ascend_pyramidkv.registry import (
        get_kv_cache_compression_provider,
    )

    return get_kv_cache_compression_provider(config)
