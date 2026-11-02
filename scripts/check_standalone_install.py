"""Verify an installed wheel in an environment without KVCompress or vLLM."""

import importlib.abc
import importlib.metadata
import importlib.util
import os
import sys


class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"vllm", "vllm_ascend", "vllm_ascend_kvcompress", "torch"}:
            raise AssertionError(f"disabled entry point imported {fullname}")
        return None


def main():
    assert importlib.util.find_spec("vllm_ascend_kvcompress") is None
    distribution = importlib.metadata.distribution("vllm-ascend-pyramidkv-hust")
    assert not any("kvcompress" in item.lower() for item in distribution.requires or [])
    assert not any(item.group == "vllm_ascend_kvcompress.methods" for item in distribution.entry_points)
    entry = next(item for item in distribution.entry_points if item.group == "vllm.general_plugins")
    assert entry.name == "ascend_pyramidkv"
    for name in ("VLLM_ASCEND_PYRAMIDKV_ENABLED", "VLLMHUST_EXT_ENABLED_BUNDLES"):
        os.environ.pop(name, None)
    sys.meta_path.insert(0, BlockRuntime())
    entry.load()()
    entry.load()()
    print("Standalone wheel: no shared dependency; disabled discovery imports no runtime")


if __name__ == "__main__":
    main()
