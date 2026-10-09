# Five-primary-dataset adaptation

Tracking: [PyramidKV #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8),
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254).

The canonical implementation and evidence are submitted in
[benchmark PR #258](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/258). This repository uses that
runner with the pinned transport and compression analyzer in `scripts/qwen35/`; it does not maintain a
second scorer or relabel existing LongBench results as primary-dataset evidence.

## Applicability and ownership

| Dataset | PyramidKV status | Reason and next step |
| --- | --- | --- |
| MMLU-Pro | `not-exercised` | All 12,032 frozen test prompts are 898–2,910 tokens with the fixed five-shot Qwen template, below the >4,096-token prefill compression threshold. The 70-task B0/B1 subset checks task-quality compatibility only. Review the named contract; full-test accuracy remains unmeasured. |
| HLE-Verified | `blocked` | Complete canonical rights review; select full vs Gold and freeze task IDs, judge model/prompts, scorer, image handling and runtime. |
| SWE-bench-Pro | `blocked` | Complete rights review and task/repository/image mapping; provision fresh isolated sandboxes and freeze agent, tools, budget and regrader. |
| FrontierScience | `blocked` | Freeze distinct Olympiad and Research task manifests, answer extraction and grading environments; do not combine their scores. |
| Terminal-Bench 2.1 | `blocked` | Provision a Harbor-compatible sandbox backend and freeze image digests, task rights, scaffold, tools, budgets and scorer for the 2.1 release. |

@Irisuko is the MOD integration owner named by #8. Benchmark #254 maintainers coordinate the canonical
rights/scorer/runtime contracts; these are coordination roles, not a statement of maintainer approval.
The full machine-readable matrix records exact source revisions and unblock conditions. The current
PyramidKV pod has no Docker executable/socket or Harbor executable, and does not mount the canonical
five-dataset source bundle. Exact MMLU-Pro files were separately downloaded and hash-verified here.

## Evidence boundary

The MMLU-Pro adapter freezes source-row hashes, all test IDs, prompt token IDs/hashes, 70 selected
cases, runtime/container/model identity, commands and scorer before model outputs. It verifies actual
server activation, includes invalid/failed answers in the accuracy denominator, preserves SSE and
resource telemetry, and checks scheduler commits/TP acknowledgements using the existing shared
analyzer. Source questions and test rationales are not substituted with synthetic text or leaked into
target prompts.

The metric is strict extracted-answer accuracy on a coverage-oriented subset: first five numeric IDs
per each of 14 categories. This is not a full or representative MMLU-Pro score. The reference A–J
extractor is pinned, but its random-guess fallback is disabled. Qwen thinking is disabled, output is
limited to 2,048 tokens, concurrency is 1, and the five-shot prompt uses a 32K context without the
reference runner's `Question:` stop or shot reduction. These differences are explicit in the named
contract.

The B0/B1 pair uses the same BF16 Qwen3.5-35B-A3B TP2 910B2 graph/MTP2 runtime and aligned
1368/beta-2 profile from the previous evaluation; only compression activation differs. Both servers
start fresh and previous workers must release HBM. Startup-only model-file cache advice is recorded
for both arms and stops before requests. This experimental v0.23 profile is separate from the
benchmark repository's official v0.18 FP16 leaderboard.

No MMLU-Pro request in this contract is compression-eligible. A successful compatibility run therefore
provides no evidence of PyramidKV optimization on this dataset. Do not pad prompts, silently lower the
threshold or transfer LongBench's KV-page reduction to these five primary datasets.

## Reproduction and review

Use the canonical report's frozen contract, input manifests, source/image hashes, reproduction
commands, per-task scores and raw archives. The shared runtime adapter must include the graph cache
write fix recorded in [shared PR #19](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/19);
this older host uses the exact backport identified by the contract.

Issue #8 remains open for matrix/contract review and the four blocked datasets. Canonical benchmark
acceptance precedes website synchronization; no website readiness or performance promotion is claimed.
