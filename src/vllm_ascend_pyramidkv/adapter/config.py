# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-HUST/vllm-ascend-kvcompress-hust, f337858 (see NOTICE).
"""Provider-level configuration and method selection."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

PROVIDER_NAME = "ascend_pyramidkv"
SUPPORTED_PROVIDER_NAMES = frozenset({PROVIDER_NAME})
DEFAULT_METHOD = "pyramidkv"
SCHEMA_VERSION = 1
ASCEND_BLOCK_SIZE = 128
EXTENSION_ID = "org.vllm-hust.ascend-pyramidkv"
ENABLE_ENV = "VLLM_ASCEND_PYRAMIDKV_ENABLED"
CONFIG_ENV = "VLLM_ASCEND_PYRAMIDKV_CONFIG"
MANAGER_ENABLED_ENV = "VLLMHUST_EXT_ENABLED_BUNDLES"

JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class ProviderSelection:
    """Validated routing information for one compression method."""

    provider_name: str
    method: str
    method_config: dict[str, JsonScalar]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProviderSelection:
        """Parse the plugin-owned configuration used by current upstream vLLM."""
        unknown = set(value) - {"schema_version", "provider", "method", "method_config"}
        if unknown:
            raise ValueError("unknown plugin configuration fields: " + ", ".join(sorted(unknown)))
        schema_version = value.get("schema_version", SCHEMA_VERSION)
        if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION}")
        provider_name = value.get("provider", PROVIDER_NAME)
        if provider_name not in SUPPORTED_PROVIDER_NAMES:
            supported = ", ".join(sorted(SUPPORTED_PROVIDER_NAMES))
            raise ValueError(f"provider {provider_name!r} is unsupported; expected one of: {supported}")
        method = value.get("method", DEFAULT_METHOD)
        if method != DEFAULT_METHOD:
            raise ValueError("only the pyramidkv method is supported")
        raw_method_config = value.get("method_config", {})
        if not isinstance(raw_method_config, Mapping):
            raise ValueError("'method_config' must be an object")
        return cls(
            provider_name=str(provider_name),
            method=method.strip().lower(),
            method_config=_copy_scalar_mapping(raw_method_config),
        )


def extension_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Return true only for explicit direct or Extension Manager activation."""
    env = os.environ if environment is None else environment
    direct = env.get(ENABLE_ENV, "").strip().lower()
    if direct in {"1", "true", "yes", "on"}:
        return True
    enabled = {item.strip() for item in env.get(MANAGER_ENABLED_ENV, "").split(",") if item.strip()}
    return EXTENSION_ID in enabled


def load_runtime_selection(
    environment: Mapping[str, str] | None = None,
) -> ProviderSelection:
    """Load direct JSON/path configuration or the manager's stored config."""
    env = os.environ if environment is None else environment
    direct = env.get(CONFIG_ENV)
    if direct:
        if direct.lstrip().startswith(("{", "[")):
            payload = json.loads(direct)
        else:
            payload = json.loads(Path(direct).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{CONFIG_ENV} must resolve to a JSON object")
        return ProviderSelection.from_mapping(payload)

    try:
        from vllm_hust_ext.config import load_config
    except ImportError as error:
        raise ValueError(f"{CONFIG_ENV} is required when Extension Manager is not installed") from error
    state = load_config().extension(EXTENSION_ID)
    if not state.enabled:
        raise ValueError(f"Extension Manager does not mark {EXTENSION_ID!r} enabled")
    return ProviderSelection.from_mapping(state.configuration)


def _copy_scalar_mapping(options: Mapping[str, Any]) -> dict[str, JsonScalar]:
    copied: dict[str, JsonScalar] = {}
    for key, value in options.items():
        if not isinstance(key, str) or not key:
            raise ValueError("provider option names must be non-empty strings")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(f"provider option {key!r} must be a JSON scalar")
        copied[key] = value
    return copied


def assert_no_conflicting_adapter(environment: Mapping[str, str] | None = None) -> None:
    """Reject competing cache mutation before installing any host hooks.

    Inspect activation state and existing modules only; never import KVCompress.
    """
    env = os.environ if environment is None else environment
    old_enabled = env.get("VLLM_ASCEND_KVCOMPRESS_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    bundles = {item.strip() for item in env.get(MANAGER_ENABLED_ENV, "").split(",")}
    old_plugin = sys.modules.get("vllm_ascend_kvcompress.plugin")
    pending_hook = getattr(old_plugin, "_runner_patch_finder", None) is not None
    installed_hook = False
    for module_name, class_name, marker in (
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "_ascend_kvcompress_patch_v3"),
        ("vllm.v1.core.sched.scheduler", "Scheduler", "_ascend_kvcompress_manager_patch_v3"),
        ("vllm.v1.core.kv_cache_manager", "KVCacheManager", "_ascend_kvcompress_manager_patch_v3"),
    ):
        module = sys.modules.get(module_name)
        installed_hook |= bool(getattr(getattr(module, class_name, None), marker, False))
    if old_enabled or "org.vllm-hust.ascend-kvcompress" in bundles or pending_hook or installed_hook:
        raise RuntimeError(
            "PyramidKV and KVCompress cannot mutate the same KV cache; "
            "disable the KVCompress bundle/environment and restart the process"
        )
