import json
import sys
from types import SimpleNamespace

import pytest

from vllm_ascend_pyramidkv.adapter.config import (
    CONFIG_ENV,
    ENABLE_ENV,
    EXTENSION_ID,
    MANAGER_ENABLED_ENV,
    ProviderSelection,
    assert_no_conflicting_adapter,
    extension_enabled,
    load_runtime_selection,
)


def test_activation_is_default_off_and_belongs_to_pyramidkv():
    assert not extension_enabled({})
    assert not extension_enabled({"VLLM_ASCEND_KVCOMPRESS_ENABLED": "1"})
    assert not extension_enabled({MANAGER_ENABLED_ENV: "org.vllm-hust.ascend-kvcompress"})
    assert extension_enabled({ENABLE_ENV: "true"})
    assert extension_enabled({MANAGER_ENABLED_ENV: EXTENSION_ID})


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": 2},
        {"provider": "ascend_kvcompress"},
        {"method": "triattention"},
        {"unknown": 1},
        {"method_config": []},
    ],
)
def test_invalid_and_old_config_is_rejected(payload):
    with pytest.raises(ValueError):
        ProviderSelection.from_mapping(payload)


def test_json_and_file_configuration_are_identical(tmp_path):
    # A long inline JSON must not be interpreted as a filesystem path.
    payload = {"provider": "ascend_pyramidkv", "method_config": {"pooling": "maxpool"}}
    raw = json.dumps(payload, indent=80)
    path = tmp_path / "config.json"
    path.write_text(raw)
    assert load_runtime_selection({CONFIG_ENV: raw}) == load_runtime_selection({CONFIG_ENV: str(path)})
    assert load_runtime_selection({CONFIG_ENV: raw}).method_config == {"pooling": "maxpool"}


@pytest.mark.parametrize(
    "environment",
    [
        {"VLLM_ASCEND_KVCOMPRESS_ENABLED": "1"},
        {MANAGER_ENABLED_ENV: "org.vllm-hust.ascend-kvcompress"},
    ],
)
def test_conflicting_activation_is_rejected(environment):
    with pytest.raises(RuntimeError, match="cannot mutate"):
        assert_no_conflicting_adapter(environment)


@pytest.mark.parametrize(
    "module_name,class_name,marker",
    [
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "_ascend_kvcompress_patch_v3"),
        ("vllm.v1.core.kv_cache_manager", "KVCacheManager", "_ascend_kvcompress_manager_patch_v3"),
    ],
)
def test_already_patched_host_is_rejected(monkeypatch, module_name, class_name, marker):
    cls = type(class_name, (), {marker: True})
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(**{class_name: cls}))
    with pytest.raises(RuntimeError, match="restart"):
        assert_no_conflicting_adapter({})


def test_pending_shared_loader_is_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend_kvcompress.plugin", SimpleNamespace(_runner_patch_finder=object()))
    with pytest.raises(RuntimeError, match="restart"):
        assert_no_conflicting_adapter({})
