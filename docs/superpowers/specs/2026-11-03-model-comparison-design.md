# Controlled model comparison on two Ascend cards

The user explicitly requested this after the Qwen2.5 TP1 reproduction finishes:
compare the models' effect on PyramidKV using the suggested common configuration.
The completed TP1 run remains immutable under its original candidate commit.

Use Qwen2.5-14B-Instruct and Qwen3.5-35B-A3B, BF16, TP2 on the same physical
910B2 cards1/6, Eager/NONE, no MTP/speculation, APC, async/chunked prefill.
Keep max-model-len8192, max-num-seqs32, batched tokens4096, utilization0.8,
seed0, generation-config=vllm, 7168 input/512 output, concurrency32. Retain the
same 512/beta20/window8/kernel7 compression algorithm configuration. Use Mamba
align for Qwen3.5. Native hybrid logical page alignment is retained and recorded;
the shared algorithm budget does not imply identical physical allocations.
Do not reduce KV pools to introduce pressure. Actual automatic pool capacities,
weight footprint and cache pressure are part of the measured model effect.

## Product change

Extend only exact official Qwen2.5 geometry from TP1 to TP1/TP2 eager. Add an
exact Qwen3.5 TP2 eager/no-speculation profile alongside the existing TP2
FULL_AND_PIECEWISE/MTP2 profile. Retain all dtype, geometry, backend, host and
scheduler restrictions. Reject graph-without-MTP, eager-with-MTP and unqualified
TP/geometry. Reuse the current adapter and algorithm; repair any actual host
seam with a failing regression first. Common host repairs apply to OFF and ON.
Do not resurrect the removed shared adapter or change algorithm semantics.

## Inputs and sequence

The two tokenizers may tokenize the same arbitrary text differently. Generate
a fixed seed0 bank of synthetic English-word prompts from tokens that are
individually single-token words with leading spaces in both tokenizers. Verify
every whole prompt is exactly1024 (short) or7168 (long) tokens under both models,
and freeze the same literal request text for all four arms. If that construction
fails, resolve the input contract before measurement rather than silently using
different texts. This is a separate model-comparison workload; do not compare
its absolute metrics against the earlier legacy-random TP1 run as a TP effect.

Per lifecycle preserve the established short48×1024/256 at rate1, then long
32×7168/512 at rateinf/concurrency32, with one first-prompt readiness request
before each cohort, no extra warmups. The original freeze planned12 fresh lifecycles with three pairs per model.
After the first Qwen2.5 pair completed, the user explicitly reduced execution
to one OFF/ON pair per model. Retain the original contract and record this
amendment; execute only its first four lifecycles, without parameter changes. No local tests/audits/tokenization during
measured requests. Stop at the first failure, keep all incomplete data, and do
not selectively retry or tune parameters after viewing formal results.

Request timeout600s, client deadline3600s, startup deadline1200s, graceful
shutdown before forced termination. Preserve partial streams on cancellation.
Record raw SSE/timestamps, usage, pool capacity, KV usage, preemptions, HBM on
both cards, and scheduler/worker acknowledgements by TP rank. Independent
recomputation verifies all320 measured and8 readiness requests. Report the single
observation for each arm and ON/OFF changes within each model, then compare those
changes between models. Do not claim a repeatability estimate from one round.

## Verification and delivery

Add regression tests for both new accepted profiles and rejected combinations;
keep the original Qwen3.5 graph/MTP2 suite green. Run native serving preflights
for both models and both arms, including boundary/admission and C32 cleanup,
before freezing candidate/package/requests/launch identities. Use the current
NPU oracle and verified materialization contracts; no claim of live preemption
unless observed. Synthetic performance does not establish task-quality parity.

Keep originals outside workspace. Publish the TP1 evidence and this comparison
in the benchmark repository, linked from the plugin's support documentation and
existing issue1. Review code, run required CI, and integrate without new tracking
issues. Clean temporary checkouts after custody. The implementation choices
beyond the user's suggested common settings are agent decisions, not user-set
experimental claims. Do not claim this isolates architecture from every model
property: parameter count, MoE routing, hybrid cache layout and weight footprint
all belong to the model change.
