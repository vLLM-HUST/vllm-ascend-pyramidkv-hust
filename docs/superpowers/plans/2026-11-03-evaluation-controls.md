# Evaluation controls implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, followed by
> one fresh branch review. The user authorized advancing issues #31/#32/#33.

**Goal:** Execute selected frozen evaluation groups and qualified C1/C2/C4
operating points without changing the release package or historical defaults.
**Architecture:** Extend the existing preparation and execution CLI. Preserve
saved case ordering and payloads; validate plan hashes before contacting a server.
**Tech Stack:** Python standard library, pytest; existing tokenizer for preparation.
**Spec:** [Issue #31](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/31)
and the linked canonical qualification plan.

## Design and constraints

Reuse `scripts/qwen35/evaluate.py`; an independent runner would duplicate request
and evidence handling. Editing frozen JSON by hand would leave execution selection
implicit. These are implementer choices within the requested existing protocol.
Keep default preparation and execution backward compatible. No plugin, model,
algorithm, launch profile, quality gate or published wheel changes.

Add `prepare --capacity-concurrency 1 2 4` (default remains `1 4`) and
`run --group-kinds warmup performance` (default all groups). Quality remains
controlled separately by the existing `--skip-quality`. An explicit empty group
list permits quality-only execution; empty total selection is rejected. Preserve
group order from the plan and record exact selection before requests.

Forward an optional saved `cache_salt` and enforce an optional
`require_uncached` policy per case; preserve ordinary quality cache behavior.
Retain exact request JSON with responses and failures. Validate saved cases and
groups against manifest hashes, reject duplicate/unsafe IDs and missing selected
groups before network access. Capacity values are unique and limited to 1/2/4.
Actual service series and case generation rules remain in benchmark, including
their own warmup/repetition contracts. CLI support alone is not measurement.

## Review focus

- Default historical plan remains executable with unchanged cases and order.
- Group filtering cannot silently dispatch excluded cases or omit quality.
- Invalid, modified or empty plans stop before model requests.
- Cache isolation is explicit and retained in exact request evidence.
- Failed requests retain raw evidence and stop remaining groups.

### Task 1: Frozen group selection and request evidence

Files: `scripts/qwen35/evaluate.py`, `tests/test_evaluation.py`.

- [ ] Add failing tests for selection/order, invalid hashes/IDs and empty selection;
  use a local HTTP server for actual request dispatch and failure retention.
- [ ] Observe failures with `python -m pytest -q tests/test_evaluation.py`.
- [ ] Implement CLI selection, plan validation and selection receipt; retain payloads.
- [ ] Add a failing cached-response test, then implement optional salt/policy handling.
- [ ] Run targeted tests, then full project tests and lint; commit.

### Task 2: Qualified capacity inputs and documentation

Files: `scripts/qwen35/evaluate.py`, `tests/test_evaluation.py`,
`docs/evaluation-runner.md`.

- [ ] Add failing preparation tests proving C2 emits two independent requests,
  unsupported/duplicate values fail, and default input counts stay unchanged.
- [ ] Implement bounded concurrency option and document exact commands and limits.
- [ ] Run targeted and full tests and CLI checks. Obtain fresh branch review;
  fix substantive findings with regression tests and merge only after CI.

Execution progress and measured originals are tracked in the issues and benchmark,
not in this technical implementation plan.
