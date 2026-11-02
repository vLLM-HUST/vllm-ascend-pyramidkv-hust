"""Independence is checked in fresh processes, never hidden by sys.modules."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_isolated(code):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("VLLM_", "VLLMHUST_"))}
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_algorithm_import_with_shared_adapter_blocked():
    result = run_isolated("""
import sys
from importlib.abc import MetaPathFinder
class BlockShared(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("vllm_ascend_kvcompress"):
            raise AssertionError("forbidden shared dependency: " + fullname)
sys.meta_path.insert(0, BlockShared())
from vllm_ascend_pyramidkv.method import PyramidKVMethod
assert PyramidKVMethod.__module__ == "vllm_ascend_pyramidkv.method"
""")
    assert result.returncode == 0, result.stderr


def test_disabled_plugin_needs_no_host_or_torch():
    result = run_isolated("""
import sys
from importlib.abc import MetaPathFinder
class BlockRuntime(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "vllm", "vllm_ascend", "vllm_ascend_kvcompress"}:
            raise AssertionError("disabled plugin imported: " + fullname)
sys.meta_path.insert(0, BlockRuntime())
from vllm_ascend_pyramidkv import register
register()
register()
""")
    assert result.returncode == 0, result.stderr


def test_conflicting_activation_fails_before_importing_host():
    result = run_isolated("""
import os, sys
from importlib.abc import MetaPathFinder
class BlockHost(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "vllm", "vllm_ascend", "vllm_ascend_kvcompress"}:
            raise AssertionError("conflict imported runtime: " + fullname)
sys.meta_path.insert(0, BlockHost())
os.environ["VLLM_ASCEND_PYRAMIDKV_ENABLED"] = "1"
os.environ["VLLM_ASCEND_KVCOMPRESS_ENABLED"] = "1"
from vllm_ascend_pyramidkv import register
try:
    register()
except RuntimeError as error:
    assert "cannot mutate" in str(error)
else:
    raise AssertionError("competing adapters were allowed")
""")
    assert result.returncode == 0, result.stderr


def test_materialization_diagnostic_help_uses_local_contracts():
    script = ROOT / "scripts/qwen35/benchmark_materialization.py"
    result = run_isolated(f"""
import runpy, sys
from importlib.abc import MetaPathFinder
class NoShared(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("vllm_ascend_kvcompress"):
            raise AssertionError("diagnostic imported shared adapter")
sys.meta_path.insert(0, NoShared())
sys.argv = [{str(script)!r}, "--help"]
runpy.run_path({str(script)!r}, run_name="__main__")
""")
    assert result.returncode == 0, result.stderr
    assert "--baseline-source" in result.stdout


def test_historical_baseline_contracts_are_normalized_without_shared_import(tmp_path):
    baseline = tmp_path / "method.py"
    baseline.write_text("""
from vllm_ascend_kvcompress.methods import METHOD_API_VERSION
from vllm_ascend_kvcompress.methods.base import CompressionResult
from vllm_ascend_kvcompress.transaction import CompressionPlan
assert METHOD_API_VERSION == 1

def budget(x):
    return 2 * x - x // 2

result = CompressionResult(128, (("layer", 97),))
plan = CompressionPlan(300, 128, (1, 2, 3), (4,))
""")
    script = ROOT / "scripts/qwen35/benchmark_materialization.py"
    result = run_isolated(f"""
import runpy, sys
from pathlib import Path
from importlib.abc import MetaPathFinder
class NoShared(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("vllm_ascend_kvcompress"):
            raise AssertionError("baseline imported shared adapter")
sys.meta_path.insert(0, NoShared())
load = runpy.run_path({str(script)!r})["load_baseline"]
module = load(Path({str(baseline)!r}))
from vllm_ascend_pyramidkv.adapter.contracts import CompressionResult
from vllm_ascend_pyramidkv.adapter.transaction import CompressionPlan
assert module.result == CompressionResult(128, (("layer", 97),))
assert type(module.plan) is CompressionPlan
assert module.budget(100) == 150
""")
    assert result.returncode == 0, result.stderr
