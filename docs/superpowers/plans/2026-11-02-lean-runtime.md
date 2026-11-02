# Lean standalone runtime implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation and a fresh whole-branch review.

**Goal:** Remove the obsolete runtime and clarify current adapter ownership.
**Architecture:** Pure algorithm, shared validation and transaction values, metadata
composition, worker orchestration and existing host hooks.
**Tech Stack:** Python 3.10+, PyTorch, vLLM/Ascend, pytest, Hatch.
**Spec:** ../specs/2026-11-02-lean-runtime-design.md

## Global Constraints

Preserve current algorithm/host semantics, default-off discovery and KVCompress
independence. Retire schema-v1 only as explicitly authorized. Release 0.4.0.dev0.
No broadened compatibility or performance claim. Keep workspace free of copies.

## Review Focus

- CPU-only algorithm import must not inspect retired vLLM contract modules.
- Async acknowledgement and request reset must update the same metadata mapping.
- Rebinding must discard corrections and slot views from an old cache allocation.
- Synthetic capture, padding and rejected MTP drafts must keep stable slot storage.
- Unsupported batched layouts must preserve native fallback and original slots.

### Task 1: Retire the legacy runtime

Files: create `algorithm.py`; delete old provider/contracts/registry; modify
`method.py`, package root, tests, manifest/version and current documentation.
Interfaces: `PyramidKVConfig`, `PyramidKVSelection`, `select_pyramid_indices`,
`materialize_pyramid_kv`, `select_pyramid_kv`; unchanged arithmetic and arguments.

- [x] Run baseline suite and preserve results.
- [x] Add a subprocess import-isolation regression; observe failure.
- [x] Move config/algorithm verbatim, remove obsolete API/runtime and only its
  behavior tests; retain all algorithm oracle assertions.
- [x] Update consumers and 0.4.0.dev0 migration documentation; run relevant tests.
- [x] Commit the independently testable cleanup.

### Task 2: Separate host validation and metadata ownership

Files: create `adapter/host_validation.py`, `adapter/metadata.py`; modify worker,
scheduler, transaction and adapter/NPU tests.
Interfaces: `AttentionMetadata(config, active, requires_per_layer_physical_state)`;
`bind(runner, layer_caches, attention_group_index, speculative_cache_layer)`;
explicit worker hook delegation; shared `ActiveCompression` and block-count helper.

- [x] Add regressions for scheduler import separation and metadata rebinding;
  observe failure before implementation.
- [x] Move shared validation out of worker; share active anchors and block math.
- [x] Extract metadata state/buffers and bind/reset logic using composition;
  consolidate per-layer offset construction for native and batched paths.
- [x] Adapt existing fixtures to component boundaries without weakening behavior
  assertions; run complete adapter/algorithm and NPU oracle tests.
- [x] Commit the independently testable ownership refactor.

### Task 3: Validate and deliver

Files: current architecture/migration docs, package/version and validation receipts
outside the source tree.

- [x] Run full suite, host-absent suite, lint, format and wheel/sdist gates.
- [ ] Verify clean-wheel discovery and live pinned TP2 serving with both metadata paths.
- [x] Request fresh whole-branch review and resolve substantive findings.
- [ ] Publish PR with exact scope, removed API list and validation limitations;
  verify final CI, integrate and clean the temporary worktree/artifacts.
