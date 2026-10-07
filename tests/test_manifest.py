from importlib.metadata import entry_points
from pathlib import Path

from vllm_hust_ext.manifest import activation_blocker, load_manifest

import vllm_ascend_pyramidkv


def test_distribution_requires_versioned_shared_host() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")

    assert 'dependencies = ["vllm-ascend-kvcompress-hust>=0.9,<0.10"]' in pyproject


def test_descriptor_publishes_active_external_method() -> None:
    manifest = load_manifest(Path(vllm_ascend_pyramidkv.__file__).with_name("vllm-hust-extension-v0.3.json"))

    assert manifest.bundle_id == "org.vllm-hust.ascend-pyramidkv"
    assert manifest.kind == "in_process_plugin"
    assert manifest.lifecycle_owner == "vllm"
    assert manifest.host.name == "vllm-ascend"
    assert manifest.host.version_range == (">=0.23.0.post1,<0.26,!=0.24.*,!=0.25.0.*")
    assert manifest.protocols[0].version_range is None
    assert manifest.schema_version == "0.3-experimental"
    assert tuple(
        (dependency.extension_id, dependency.version_range) for dependency in manifest.requires_extensions
    ) == (("org.vllm-hust.ascend-kvcompress", ">=0.9,<0.10"),)
    assert tuple((claim.resource, claim.scope, claim.mode) for claim in manifest.resource_claims) == (
        (
            "vllm.kv-cache.compression-method.pyramidkv",
            "vllm-process",
            "exclusive",
        ),
    )
    assert activation_blocker(manifest) is None
    assert tuple((item.group, item.name) for item in manifest.activation.entry_points) == (
        ("vllm_ascend_kvcompress.methods", "pyramidkv"),
    )
    assert manifest.components[0].implementation_ref == ("vllm_ascend_pyramidkv.method:create_pyramidkv_method")
    registrations = entry_points(group="vllm_hust.extension_bundles")
    assert any(item.name == manifest.bundle_id for item in registrations)
    provider_registrations = entry_points(group="vllm_ascend.kv_cache_compression_providers")
    assert not any(item.name == "pyramidkv_ascend" for item in provider_registrations)
    method_registrations = entry_points(group="vllm_ascend_kvcompress.methods")
    registration = next(item for item in method_registrations if item.name == "pyramidkv")
    assert registration.value == ("vllm_ascend_pyramidkv.method:create_pyramidkv_method")


def test_top_level_import_has_no_runtime_side_effects() -> None:
    source = Path(vllm_ascend_pyramidkv.__file__).read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "import vllm" not in source
    assert "monkey" not in source.lower()
