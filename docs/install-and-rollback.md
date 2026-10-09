# Install, inspect, and rollback

This document applies to the active external method. The shared adapter owns
the scheduler/worker lifecycle, while this package supplies the `pyramidkv`
method. The exact serving profile is documented in the support matrix.

## Clean installation

On the prepared Ascend host, inherit its matched Torch/torch-npu packages into the
isolated environment. These commands do not provision CANN or the qualified core/Ascend
source stack; prepare those using the linked serving runbook first.

```bash
python -m venv --system-site-packages .venv-pyramidkv
source .venv-pyramidkv/bin/activate
python -m pip install \
  "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@d22088cf6a45aeb7c47e39101607a00d87bf2006"
python -m pip install --no-deps "vllm-ascend-kvcompress-hust @ git+https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust.git@19f322130e1b3841da953b1b421bcf3259e9b8dc"
python -m pip install --no-deps /path/to/vllm-ascend-pyramidkv-hust
vllm-hust-ext extension inspect org.vllm-hust.ascend-pyramidkv
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension enable org.vllm-hust.ascend-pyramidkv
```

Inspection must report the active
`vllm_ascend_kvcompress.methods:pyramidkv` implementation without an
activation blocker. Host compatibility checks still reject every unqualified
runtime profile. The PyramidKV package also checks shared-host method API v1
when its method entry point loads, so an older adapter cannot fail later in a
serving request.

Manifest 0.3 additionally makes the shared Bundle dependency machine-readable:
ECPA rejects PyramidKV enablement until `org.vllm-hust.ascend-kvcompress`
`>=0.9,<0.10` is installed and enabled. It also rejects another enabled Bundle
claiming the same PyramidKV method registration. These checks do not select a
method or prove it runtime-effective.

For the source-built Qwen3.5 serving profile, see the
[prepared-host runbook](qwen35-serving-smoke.md). The initial October reproduction used
direct environment activation and observed a Manager joint-plan conflict on
`vllm.environment`. The newer pinned Manager fixes that composition error; see
[the managed-launch follow-up](qwen35-managed-launch.md) for joint activation,
configuration, raw invocation evidence and shutdown verification. Enabling both
bundles alone still does not prove runtime effectiveness.

The pinned shared commit includes the FULL-graph per-layer cache-write fix and
Manifest 0.3 migration on the host-compatible validation branch. Shared
[PR #19](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/19) carries
the runtime fix against its newer default branch. The package version alone
does not distinguish these revisions. The [public 0.9.0 wheel audit](../evidence/current/2026-10-09-public-package-audit/README.md)
confirms that the downloadable wheel differs from this qualified source and still lacks the
graph-slot fix; do not replace the source pin with a version-only dependency. The
[paired protocol](issue-1-evaluation-protocol.md) reproduces the current
evaluation with an explicit `pyramidkv-aligned.json` configuration; the original
512/beta-20 smoke configuration is not interchangeable quality evidence.

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

Disable the method bundle before uninstalling it. The shared adapter may remain
enabled for other compression methods:

```bash
vllm-hust-ext extension disable org.vllm-hust.ascend-pyramidkv
python -m pip uninstall -y vllm-ascend-pyramidkv-hust
python - <<'PY'
import importlib.util

assert importlib.util.find_spec("vllm_ascend_pyramidkv") is None
PY
```

The provider owns no external service or persistent cache data. Rollback does
not require data migration.
