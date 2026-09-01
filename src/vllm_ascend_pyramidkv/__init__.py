# SPDX-License-Identifier: Apache-2.0
"""PyramidKV Ascend migration package.

The top-level module intentionally imports neither PyTorch nor vLLM and never
activates the provider. The Extension Manager can discover this descriptor but
must refuse enablement while the manifest implementation status is
``import_only``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PyramidKVContractProposal", "get_provider"]
__version__ = "0.1.0.dev0"


class PyramidKVContractProposal:
    """Metadata-only proposal; this class performs no runtime activation."""


def get_provider(config: Any) -> Any:
    """Lazily load the migration provider for offline verification.

    This helper does not register with vLLM. It exists so schema-v1 behavior
    can be tested while a current provider-neutral host contract is developed.
    """

    from vllm_ascend_pyramidkv.registry import (
        get_kv_cache_compression_provider,
    )

    return get_kv_cache_compression_provider(config)
