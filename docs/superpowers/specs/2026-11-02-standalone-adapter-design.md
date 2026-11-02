# Standalone PyramidKV Ascend adapter

## Intent and dependency audit

Make this distribution an independently activatable vLLM/Ascend plugin. The
current main (`3355b35`) installs KVCompress >=0.9,<0.10; the manifest requires
its bundle, `method.py` imports its method and transaction types, tests import
those types, CI installs its wheel, and launchers select its general plugin,
configuration and environment variables. The legacy offline provider already
owns PyramidKV selection and has no KVCompress import.

The shared runtime at `f33785844be5bff7716c9464dd8444ef0b4b05e9`
(PR 33; runtime unchanged through `aa4c1d0`) contains the scheduler transaction,
worker cache binding/query observation, immutable correction tensors, stable
per-layer graph slots, MTP corrections and guarded batched metadata fallback.
It also contains unrelated TriAttention dispatch/calibration and kernels.

## Approach and alternatives

Chosen engineering approach: take ownership of the necessary Apache-2.0 runtime
in `vllm_ascend_pyramidkv.adapter`, preserving attribution and migrating its
behavioral tests. Bind the existing PyramidKV method directly. Remove algorithm
registry dispatch, calibration and TriAttention dependencies. Keep the proven
host compatibility shims required by the pinned Qwen3.5 profile locally.

Waiting for a generic upstream interface cannot meet standalone installation.
Rewriting the complete scheduler/graph protocol would unnecessarily risk
semantic regressions. A renamed copy retaining unrelated methods is excluded.
The user authorized completion of the refactor, tests, documentation and PR;
implementation choices here are engineering decisions, not user-specified APIs.

## Boundaries

- `adapter/contracts.py` and `adapter/transaction.py`: local typed method and
  picklable transaction values; no external extension discovery.
- `adapter/config.py`: default-off activation, strict PyramidKV-only schema,
  direct JSON/path config or optional Extension Manager state.
- `adapter/plugin.py`: lazy `vllm.general_plugins:ascend_pyramidkv` registration;
  host hooks, idempotence, explicit conflict rejection for an active shared
  adapter. Importing the package or disabled entry point loads no host runtime.
- `adapter/scheduler.py`: private destination allocation, APC query suffix,
  acknowledgement barrier, deferred frees, preemption/abort cleanup.
- `adapter/worker.py`: cache binding, query observations, physical/semantic
  positions, commit/reset, per-layer and MTP metadata. Own the PR33 optimization
  with its opt-in gate and guarded fallback.
- `adapter/host_compat.py`, `adapter/qwen_gdn_ops.py`, `adapter/model.py`: exact
  supported host seams and model geometry. Unsupported geometry fails closed.
- Existing `method.py` and offline `provider.py`: keep GQA aggregation, pooling,
  layer budget schedule, strict admission > threshold, top-k order, tail window,
  auxiliary MTP selection and paged materialization semantics unchanged.

Public activation uses `VLLM_ASCEND_PYRAMIDKV_ENABLED`,
`VLLM_ASCEND_PYRAMIDKV_CONFIG`, provider `ascend_pyramidkv`, method `pyramidkv`.
Old config provider names are rejected with migration instructions. No old
runtime environment variables are consumed except to detect conflicting active
KVCompress. Manager no longer requires the shared bundle and claims the same
exclusive `vllm-ascend.kv-cache.mutation` resource as that adapter.

## Compatibility and validation

Preserve the current fail-closed Qwen/Qwen3.5-35B-A3B BF16 TP2, CANN9.1, APC,
MTP2, async, chunked-prefill, FULL_AND_PIECEWISE, mamba_cache_mode=align profile.
Host wheels remain separately provisioned; pip must not replace torch-npu.

Verify source and wheel without KVCompress installed or importable, entry-point
and default-off behavior without vLLM, strict config and conflicting activation,
algorithm regression, scheduler lifecycle and metadata equivalence (unequal
layers, row reorder, empty batches, synthetic capture, rejected drafts,
fallback and storage identity). Run full CPU suite, lint/format/build, installed
wheel/Manager lifecycle, native NPU oracle and real Qwen3.5 serving where this
host permits. Record exact limitations. Prior benchmark receipts remain
historical, not qualification for this revision. No new performance claim.

## Delivery

A reviewable PR with audit, standalone architecture/usage/rollback documentation,
updated metadata/manifest/CI/launchers, attribution, tests and validation receipt.
Preserve PR26's documentation changes by incorporating its existing commits or
merging it after review; do not rewrite historical benchmark originals.
