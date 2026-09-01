from importlib.metadata import entry_points
from pathlib import Path

from vllm_hust_ext.manifest import activation_blocker, load_manifest

import vllm_ascend_pyramidkv


def test_descriptor_is_discoverable_but_not_activatable() -> None:
    manifest = load_manifest(Path(vllm_ascend_pyramidkv.__file__).with_name("vllm-hust-extension-v0.2.json"))

    assert manifest.bundle_id == "org.vllm-hust.ascend-pyramidkv"
    assert manifest.kind == "in_process_plugin"
    assert manifest.lifecycle_owner == "vllm"
    assert activation_blocker(manifest) == (
        "extension is descriptor-only and cannot be enabled (implementation status: import_only)"
    )
    registrations = entry_points(group="vllm_hust.extension_bundles")
    assert any(item.name == manifest.bundle_id for item in registrations)
    runtime_registrations = entry_points(group="vllm.general_plugins")
    assert not any(item.name == "vllm_ascend_pyramidkv" for item in runtime_registrations)


def test_top_level_import_has_no_runtime_side_effects() -> None:
    source = Path(vllm_ascend_pyramidkv.__file__).read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "import vllm" not in source
    assert "monkey" not in source.lower()
