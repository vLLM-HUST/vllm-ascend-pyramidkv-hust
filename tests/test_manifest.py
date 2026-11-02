from importlib.metadata import entry_points
from pathlib import Path

from vllm_hust_ext.manifest import activation_blocker, load_manifest

import vllm_ascend_pyramidkv


def test_descriptor_publishes_standalone_general_plugin():
    manifest = load_manifest(Path(vllm_ascend_pyramidkv.__file__).with_name("vllm-hust-extension-v0.3.json"))
    assert manifest.bundle_id == "org.vllm-hust.ascend-pyramidkv"
    assert manifest.kind == "in_process_plugin"
    assert manifest.lifecycle_owner == "vllm"
    assert manifest.host.name == "vllm-ascend"
    assert not manifest.requires_extensions
    assert tuple((claim.resource, claim.scope, claim.mode) for claim in manifest.resource_claims) == (
        ("vllm-ascend.kv-cache.mutation", "vllm-process", "exclusive"),
    )
    assert activation_blocker(manifest) is None
    assert tuple((item.group, item.name) for item in manifest.activation.entry_points) == (
        ("vllm.general_plugins", "ascend_pyramidkv"),
    )
    assert manifest.components[0].implementation_ref == "vllm_ascend_pyramidkv:register"
    registrations = entry_points(group="vllm_hust.extension_bundles")
    assert any(item.name == manifest.bundle_id for item in registrations)
    registrations = entry_points(group="vllm.general_plugins")
    entry = next(item for item in registrations if item.name == "ascend_pyramidkv")
    assert entry.value == "vllm_ascend_pyramidkv:register"
    assert not any(item.name == "pyramidkv" for item in entry_points(group="vllm_ascend_kvcompress.methods"))
