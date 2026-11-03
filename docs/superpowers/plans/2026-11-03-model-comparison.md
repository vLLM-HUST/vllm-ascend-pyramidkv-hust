# Model comparison implementation plan

> Execute inline with superpowers:executing-plans.

**Goal:** Compare PyramidKV OFF/ON effects on Qwen2.5-14B and Qwen3.5-35B
under the user-approved common TP2 eager/no-MTP 7168/512 C32 settings.
**Spec:** docs/superpowers/specs/2026-11-03-model-comparison-design.md.
**Architecture:** Extend two exact compatibility profiles; reuse existing
adapter/algorithm and timestamped streaming client; freeze one common prompt bank.

## Global constraints

TP1 originals immutable; no parameter tuning after formal freeze; both models
same hardware/runtime/request texts/nominal lengths; automatic pool utilization0.8;
keep failed samples and no task-quality or architecture-only causal claims.

## Review focus

- Dense TP2 local head shapes and Qwen3.5 eager/no-MTP state are valid.
- Exact original graph/MTP2 profile remains accepted; mixed unsupported modes fail.
- Same literal requests tokenize to target length under both tokenizers.
- Common host compatibility is applied identically to enabled and disabled arms.
- TP rank acknowledgement counts and cache cleanup remain verifiable.

## Task 1: Shared serving profiles and preflight

- [ ] Add failing acceptance/rejection tests; implement TP2 dense and eager hybrid.
- [ ] Add shared launch/config and freeze common-token prompt construction.
- [ ] Run focused/full tests and native OFF/ON preflights for both models.
- [ ] Review completed code and repair substantive findings with regressions.
- [ ] Freeze candidate commit/wheel, runtime, models, requests and12-run contract.

## Task 2: Paired measurement and verification

- [ ] Execute12 lifecycles, retaining all requests and failures.
- [ ] Recompute request metrics and aggregates; validate two-rank transactions,
  real lengths, identical payloads, pool/HBM samples and zero final active state.
- [ ] Compare within-model effects and explain observed cache pressure.

## Task 3: Publication and cleanup

- [ ] Deliver both TP1 and model-comparison reports with immutable artifacts,
  checksums and offline verification in benchmark repository.
- [ ] Update plugin support/docs, reviewed PRs and existing issue1.
- [ ] Complete required CI/readback, stop services and clean temporary checkouts.
