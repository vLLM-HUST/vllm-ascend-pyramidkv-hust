# Standalone PyramidKV Adapter Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove installation, import and runtime dependence on KVCompress.
**Architecture:** Own local contracts, scheduler and worker adapter; register one
PyramidKV-only general plugin. Retain selection and transaction semantics.
**Tech Stack:** Python, PyTorch, vLLM/Ascend, pytest, Hatch, Extension Manager.
**Spec:** ../specs/2026-11-02-standalone-adapter-design.md

## Global Constraints

Default off; preserve exact Qwen3.5 TP2 profile and algorithm semantics; no
KVCompress runtime or installation dependence; no unqualified performance claim.

## Review Focus

- Installed wheel discovery when both KVCompress and vLLM are absent.
- APC/chunked prefill and asynchronous commit, abort, preemption and rebind.
- Unequal layer lengths, FULL capture storage identity and MTP draft rejection.
- Opt-in batched metadata fallback on unsupported layouts/device placement.
- Manager/direct activation conflicts and configuration migration.

### Task 1: Local adapter ownership

Files: create `src/vllm_ascend_pyramidkv/adapter/{contracts,transaction,config,
model,scheduler,worker,plugin,host_compat,qwen_gdn_ops}.py`; modify `method.py`;
create `tests/test_standalone.py`, `tests/adapter/`.

- [x] Add subprocess import-blocking/default-off tests; observe baseline failure.
- [x] Port needed behavioral tests, then local runtime and contracts; remove
  registry, TriAttention calibration and kernels; keep position subtraction local.
- [x] Run algorithm and adapter tests; resolve lifecycle regressions.
- [x] Commit runtime and tests.

### Task 2: Packaging and activation

Files: pyproject, manifest, CI, launcher/configs, manifest tests.

- [x] Assert local general entry point and no required extension/distribution.
- [x] Change package/manifest/CI and launcher; default-off lazy entry function.
- [x] Build wheel and exercise clean install, import blocker and Manager lifecycle.
- [x] Commit packaging and activation.

### Task 3: Delivery and qualification

Files: README, install/rollback, host-contract, support matrix, provenance/NOTICE,
MOD metadata and migration design; NPU/integration validation scripts as needed.

- [x] Run full suite, lint, format and build on final source.
- [x] Run native metadata/selection oracle and standalone serving validation.
- [x] Update current docs; clearly label historical shared-adapter instructions.
- [x] Request fresh whole-branch review; address important findings with tests.
- [x] Publish PR and validation, read back remote checks and state.
