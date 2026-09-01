# Install, inspect, and rollback

This document applies to the migration baseline. It installs an inspectable
descriptor and offline provider code; it does not enable serving.

## Clean installation

```bash
python -m venv .venv-pyramidkv
source .venv-pyramidkv/bin/activate
python -m pip install \
  "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@9fb467447e95d753f7002b28575d6802f4347181"
python -m pip install /path/to/vllm-ascend-pyramidkv-hust
vllm-hust-ext extension inspect org.vllm-hust.ascend-pyramidkv
```

Inspection must report an `import_only` implementation. Extension enablement
must be refused; a successful activation is a failure at this stage.

## Offline development tests

Use an environment whose PyTorch build already matches the target host:

```bash
python -m pip install --no-deps -e /path/to/vllm-ascend-pyramidkv-hust
python -m pip install pytest ruff
pytest -q /path/to/vllm-ascend-pyramidkv-hust/tests
```

## Disable and rollback

The migration package registers no vLLM runtime entry point and changes no host
source file, environment variable, model artifact, or cache. Consequently,
rollback consists only of removing the distribution and starting a new host
process:

```bash
python -m pip uninstall -y vllm-ascend-pyramidkv-hust
python - <<'PY'
import importlib.util

assert importlib.util.find_spec("vllm_ascend_pyramidkv") is None
PY
```

When active integration is added, this document must be expanded with an
Extension Manager disable/forget sequence, next-process fallback proof, host
configuration cleanup, and exact-version rollback procedure.
