# SPDX-License-Identifier: Apache-2.0
"""Standalone PyramidKV plugin with lazy, default-off host activation."""

__all__ = ["register"]
__version__ = "0.4.0.dev0"


def register() -> None:
    """vLLM general-plugin entry point; disabled discovery needs no host imports."""
    from .adapter.config import assert_no_conflicting_adapter, extension_enabled

    if not extension_enabled():
        return
    assert_no_conflicting_adapter()
    from .adapter.plugin import register as activate

    activate()
