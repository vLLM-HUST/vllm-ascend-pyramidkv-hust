# Install, inspect, and rollback

This document applies to the active alpha and its paired schema-v1 hosts.

## Clean installation

```bash
python -m venv .venv-pyramidkv
source .venv-pyramidkv/bin/activate
python -m pip install \
  "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@9fb467447e95d753f7002b28575d6802f4347181"
python -m pip install /path/to/vllm-ascend-pyramidkv-hust
vllm-hust-ext extension inspect org.vllm-hust.ascend-pyramidkv
```

Inspection must report an `active` implementation and the
`pyramidkv_ascend` provider entry point. Activation remains explicit: add
`--kv-cache-compression-config` to a new vLLM process. Omitting that option
preserves the default host path and does not load the provider implementation.

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

Stop the serving process, remove `--kv-cache-compression-config`, uninstall the
provider distribution, and start a new process:

```bash
python -m pip uninstall -y vllm-ascend-pyramidkv-hust
python - <<'PY'
import importlib.util

assert importlib.util.find_spec("vllm_ascend_pyramidkv") is None
PY
```

The provider owns no external service or persistent cache data. Rollback does
not require data migration.
