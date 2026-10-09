# Five-primary-dataset adaptation

Tracking: [PyramidKV #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8),
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254).

The canonical implementation and evidence were merged in
[benchmark PR #258](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/258). This repository uses that
runner with the pinned transport and compression analyzer in `scripts/qwen35/`; it does not maintain a
second scorer or relabel existing LongBench results as primary-dataset evidence.

## Applicability and ownership

| Dataset | PyramidKV status | Reason and next step |
| --- | --- | --- |
| MMLU-Pro | `not-exercised` | All 12,032 frozen test prompts are 898–2,910 tokens with the fixed five-shot Qwen template, below the >4,096-token prefill compression threshold. The 70-task B0/B1 subset checks task-quality compatibility only. Review the named contract; full-test accuracy remains unmeasured. |
| HLE-Verified | `blocked` | All 2,500 source rows and image modalities are audited. Complete canonical rights review; select full vs Gold and freeze judge/scorer and image-aware execution. Gold includes 93 image tasks out of 668. |
| SWE-bench-Pro | `blocked` | All 642 V2 tasks map to pinned repository/harness files and digest-verified AMD64 images. Complete task rights review; provide fresh AMD64 sandboxes and freeze agent, tools, budgets and regrader. |
| FrontierScience | `blocked` | Both input manifests and candidate solver/grader code are available. The prompts do not cross the prefill threshold. Provide a fixed judge and actual semantic calibration before solver generation; keep the two track scores separate. |
| Terminal-Bench 2.1 | `blocked` | All 89 tasks, 1,101 source files and AMD64 image digests are audited. Provide a Harbor-compatible fresh-sandbox backend and model connectivity; complete task rights and scaffold/tool/budget/verifier contracts. |

@Irisuko is the MOD integration owner named by #8. Benchmark #254 maintainers coordinate the canonical
rights/scorer/runtime contracts; these are coordination roles, not a statement of maintainer approval.
The original machine-readable matrix remains an immutable historical receipt; the newer linked
reports below record completed audits and remaining conditions. This ARM64 pod has no usable
Docker/Podman socket or Harbor backend. The required source files and image metadata were
independently retrieved and verified here; source availability no longer blocks these audits.
AMD64 sandbox execution and model connectivity remain external requirements.

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

## Matched compatibility result

The [immutable canonical report](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/b308da7eb2ecad86a9349d0dcc0f4867c197a73a/reports/pyramidkv-five-dataset-adaptation-20261009) records B0 and B1 at **55/70 correct (78.571429%)**,
with all 70 output texts and extracted predictions identical. Both arms have zero transport failures,
zero invalid answers and six output-limit terminations. All tasks remain in the denominator. The
expected zero scheduler compression commits were verified, so the status remains `not-exercised`.

Supporting telemetry also preserves the costs observed in this one pair: B1's HBM peak is 260 MiB
higher per device and its mean request completion time is about 4.9% higher; sampled peak KV usage is
unchanged. These are descriptive observations from a single sequential pair, not a general performance
estimate. No speedup, KV reduction or capacity improvement is claimed for MMLU-Pro.

The report contains raw output/archive hashes, a full 12,032-task input inventory and a CPU-only
verifier that re-scores published outputs and checks SSE correspondence. Its
[SHA256SUMS](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b308da7eb2ecad86a9349d0dcc0f4867c197a73a/reports/pyramidkv-five-dataset-adaptation-20261009/SHA256SUMS) SHA-256 is
`e581f39d0b6f87d39a3833e510074b9539e98c8591b961d70f8d551dd6b1f67b`.

## Reproduction and review

Use the canonical report's frozen contract, input manifests, source/image hashes, reproduction
commands, per-task scores and raw archives. The shared runtime adapter must include the graph cache
write fix recorded in [shared PR #19](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/19);
this older host uses the exact backport identified by the contract.

Issue #8 remains open for matrix/contract review and the four blocked datasets. Canonical benchmark
acceptance precedes website synchronization; no website readiness or performance promotion is claimed.

## FrontierScience input follow-up

Merged [benchmark PR #259](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/259)
adds [immutable input manifests and grading gaps](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/d4021107ea523093bdbfc4d70473b551cc2dd08c/reports/pyramidkv-frontierscience-inputs-20261009).
The exact frozen Olympiad and Research files have now also been downloaded and
hash-verified on this pod. The problem-only Qwen candidate has 100 Olympiad prompts
of 94–835 tokens and 60 Research prompts of 164–1,660 tokens; none crosses 4,096.
Research has only 59 unique group IDs, so track/source-row IDs preserve all 60 tasks.
The source reference answers/rubrics are kept outside solver prompts.

That original report is an input audit with no solver or judge calls. The subsequent
[benchmark PR #260](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/260) adds
[candidate solver and separate grading contracts](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/aec97b98e0b82d375fd4d113ed5f8dfc47eeefed/reports/pyramidkv-frontierscience-protocol-20261009).
Olympiad uses reference equivalence; Research uses rubric scores and its own passing threshold.
Judge identity, instructions and a checksum-bound semantic calibration receipt must be fixed before
any solver generation. Failed solver attempts remain in the denominator, while missing or failed
judge grading leaves the track score unavailable. The code and synthetic tests do not provide an
actual judge or establish semantic calibration. No real solver/judge scores have been collected.

## HLE, SWE and Terminal-Bench source follow-up

Merged [benchmark PR #261](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/261) adds
[HLE source and modality evidence](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/d48542eb97ba8c812db508d1421dd8ffc2fd7717/reports/pyramidkv-hle-source-audit-20261009)
and [SWE task/image provenance](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/d48542eb97ba8c812db508d1421dd8ffc2fd7717/reports/pyramidkv-swe-source-audit-20261009).
HLE preserves all 2,500 unique source rows, original answer types and image-task membership.
The text-only Gold subset has 575 tasks and must not be presented as the complete 668-task Gold set.
The current serving configuration disables image input, so a full score requires a separate
image-aware execution contract as well as rights and scorer review.

All 642 SWE V2 task IDs match the frozen harness, base commits and AMD64 image metadata. One Ansible
test patch differs between the dataset and harness only in CRLF sequences; verifier execution must
retain the pinned harness bytes because fixture line endings can be semantic. Root-license
provenance is not blanket clearance of task content. No image layers were downloaded and no task sandboxes were run.

Merged [benchmark PR #262](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/262) adds
[Terminal-Bench 2.1 audit and retrieval evidence](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/508d95298b2cdff53b4c879979b2cd12dff27cdc/reports/pyramidkv-terminal-source-audit-20261009).
All 89 tasks and 1,101 source files match the pinned Git tree; all image manifests/configurations
are digest-verified and AMD64. The resumable fetcher retrieves source files and image metadata by
frozen identity, without pulling image layers or executing task code. All tasks preserve their own
resource/time limits and network policy. Two complete audits are byte-identical. Cached retrieval
of 89 images and a fresh public retrieval of one task passed, as did 1,822 repository tests with four
skips and final tests/lint CI. These are source-integrity results, not Harbor task resolution scores.

The reports preserve source revisions, inventories, retrieval receipts, failure records and
checksums. Their rights, judge, sandbox and execution-contract gates remain open. No official
leaderboard, website readiness or compression-gain claim is promoted by these follow-ups.
