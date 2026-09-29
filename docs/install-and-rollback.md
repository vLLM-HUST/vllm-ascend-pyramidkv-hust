# Install, inspect, and rollback

This document applies to the installable capability preview. It is not a
serving guide while exact-stack validation remains unresolved. The shared host
now provides the required eager/APC admission contract, but MTP, graph replay,
activation, serving, and rollback are not yet qualified for PyramidKV.

## Clean installation

```bash
python -m venv .venv-pyramidkv
source .venv-pyramidkv/bin/activate
python -m pip install \
  "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@9fb467447e95d753f7002b28575d6802f4347181"
python -m pip install /path/to/vllm-ascend-pyramidkv-hust
vllm-hust-ext extension inspect org.vllm-hust.ascend-pyramidkv
```

Inspection must report an `import_only` implementation and an activation
blocker. The package deliberately registers no runtime provider or method
entry point. Extension Manager enablement must fail closed.

## Offline development tests

Use an environment whose PyTorch build already matches the target host:

```bash
python -m pip install --no-deps -e /path/to/vllm-ascend-pyramidkv-hust
python -m pip install pytest ruff
pytest -q /path/to/vllm-ascend-pyramidkv-hust/tests
```

The real-device selection oracle is opt-in so ordinary package tests never
claim or consume an NPU implicitly:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 PYRAMIDKV_RUN_NPU_TESTS=1 \
  pytest -q tests/npu/test_selection.py
```

## Disable and rollback

Because this preview cannot activate, rollback consists only of uninstalling
the distribution and confirming that its Python package is absent:

```bash
python -m pip uninstall -y vllm-ascend-pyramidkv-hust
python - <<'PY'
import importlib.util

assert importlib.util.find_spec("vllm_ascend_pyramidkv") is None
PY
```

The provider owns no external service or persistent cache data. Rollback does
not require data migration.
