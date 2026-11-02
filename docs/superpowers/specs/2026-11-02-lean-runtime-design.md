# Lean standalone PyramidKV runtime

## Intent and scope

Make the independent plugin smaller and easier to maintain without changing its
algorithm, activation, supported host profile, request lifecycle or graph behavior.
The user explicitly approved removing the schema-v1 provider API and its legacy
runtime. This is a breaking cleanup, released as `0.4.0.dev0`.

## Approaches

1. Preserve the old provider behind a compatibility module: protects obsolete
   callers but retains a second runtime and most of the maintenance burden.
2. Remove the obsolete runtime and give current components clear ownership:
   selected by the user for the removal; component boundaries below are the
   implementer's choices. It preserves all current plugin behavior.
3. Rewrite host integration from scratch: unnecessary risk to APC, asynchronous
   acknowledgement and graph address stability. Not selected.

## Boundaries

- `algorithm.py` owns `PyramidKVConfig`, `PyramidKVSelection` and the three pure
  selection/materialization functions. Move their arithmetic unchanged from
  `provider.py`; neither this module nor the current method probes old host APIs.
- Delete `provider.py`, `registry.py`, root `contracts.py`, `legacy_contracts.py`,
  `get_provider` and `PyramidKVContractProposal`. Keep current adapter contracts,
  entry point, configuration schema and bundle identity. Document old import paths
  as removed rather than introducing compatibility shims.
- `adapter/host_validation.py` owns shared host/layout/runtime checks. The scheduler
  no longer imports worker implementation merely to validate host capabilities.
- `adapter/metadata.py:AttentionMetadata` owns physical offsets, saved semantic
  lengths, immutable corrections, stable per-layer slots, native fallback, graph
  capture and MTP common metadata. It receives the active-request mapping and
  configuration explicitly, and binds runner/layers/group/MTP cache together.
  No back-reference to a worker and no forwarding of arbitrary attributes.
- `PyramidKVWorker` owns binding orchestration, query observation and transactions.
  Existing host-hook methods delegate explicitly to its metadata component, keeping
  plugin hooks stable. Request resets mutate the shared active mapping in place.
- `adapter/transaction.py` owns the shared active anchor state and block-count
  helper. Worker and scheduler pending states remain distinct because they own
  different resources. Metadata paths share one per-layer offset calculation.

## Constraints and invariants

Keep strict admission, GQA mean scores, max pooling, top-k order, per-layer budget,
chronological query tail and MTP materialization arithmetic unchanged. Preserve
private destination allocation, source-page immutability, async commit/free timing,
APC recomputation and abort/preemption cleanup. Keep default-off lazy discovery,
KVCompress independence, all fail-closed host gates and opt-in batched metadata.
Graph slot addresses must stay fixed; earlier length views must stay immutable.
No host/model support expansion and no performance/quality claim.

## Validation and delivery

Retain algorithm oracle tests and every current lifecycle/metadata/host behavior
assertion. Remove only tests for explicitly retired schema-v1 behavior. Test pure
algorithm imports while old/host imports are blocked, scheduler/worker import
separation, metadata binding/reset, stable native/batched slots and corrections.
Run full suite, host-absent checks, lint, format, wheel/sdist and clean-wheel lazy
entry-point checks. Run both NPU oracles and live pinned Qwen3.5 TP2 serving with
request-matched commits/APC/MTP. Independently review the whole branch, publish a
PR, read final CI and merge under existing repository authorization when green.
Use a temporary worktree and validation files outside workspace; clean them at
completion and leave the prepared runtime pointing at the canonical main checkout.
