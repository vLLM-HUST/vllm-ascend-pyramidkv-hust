# Backend contract decision for issue #8

Status: proposal for a scope decision; no replacement contract or new measurement is approved by
this document. Checked on 2026-10-28 (Asia/Shanghai).

Tracking: [PyramidKV #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8),
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254), and the newer
[execution coordination issue #403](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/403).

## What is now available

The shared program has a reproducible ARM CPU workspace/bootstrap and an independent Docker-based
SWE runner. Its immutable
[bootstrap delivery](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/81c359a05fb11851f270bde2795dc97b1320be15)
and
[expanded V1 source lock](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/a31c820dc736db54b1ff4c88bed1b7b8b05dbb8c)
are reusable dependencies. Source-lock coverage does not prove that every image is available or that
a workspace is allocated to PyramidKV.

Terminal now also has an ARM64 derived-image route, with task-specific controls and recovery records
kept in the [canonical benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/526). These
derived images have different identities from the original AMD64 images. A control or a
reconstructed-answer grading record does not establish a successful original Agent trial.

The latest #403 coordination scope supersedes its older full-task queue: new measurements are
stopped while a bounded historical-trace replay proposal is reviewed. The older AMD64 procurement
request is not the shared program's current direction. This is a shared-resource coordination state,
not an automatic amendment of PyramidKV's frozen dataset contracts or this user's judge decision.

## Exact contract differences

| Item               | Existing PyramidKV candidate                               | Shared route                                                                                    | Required treatment                                                                                                   |
| ------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| SWE dataset        | V2 parquet and its frozen task manifest                    | V1 parquet and V1 source lock                                                                   | Name a new contract if V1 is selected; do not relabel the V2 audit or borrow V1 scores.                              |
| SWE execution      | Pinned AMD64 images; no qualified backend allocated here   | ARM64 images, independent runner and Docker workspace                                           | Freeze actual image/source identities and qualify the selected tasks on the allocated backend.                       |
| Terminal execution | Pinned AMD64 images, Harbor/Terminus candidate             | ARM64 derived images with per-task controls                                                     | Preserve task/tests/budgets; record image derivation and positive/negative controls for each selected task.          |
| Model stack        | Qualified experimental v0.23 PyramidKV host                | Shared program has a separately identified official v0.18 baseline and other development stacks | Verify method/host support before launch. Do not label a v0.23 compression ON/OFF pair as official v0.18 acceptance. |
| Judge              | Fixed identity plus semantic calibration before generation | Existing shared HLE/FrontierScience records use a substitute judge                              | Retain the user's calibration requirement; shared scores do not satisfy it automatically.                            |
| Trace replay       | No frozen PyramidKV replay contract                        | Shared proposal is awaiting its own review                                                      | Keep inference performance separate from live Agent execution and task quality.                                      |

The prior
[direct V2 input rejection probe](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/b761f7cdd89207e210f69cb973dda889e8ddde57/reports/pyramidkv-runtime-and-swe-recheck-20261022)
remains valid evidence of the runner's original V1 guard. The newer source-lock expansion does not
convert it into a V2 runner.

## Decision requested

**Recommended: prepare a separately named shared-backend contract.** Use SWE V1 and Terminal ARM64
derived images as the proposed future #8 execution route, retaining the old V2/AMD64 input audits as
historical evidence. This choice authorizes contract preparation and adapter work; actual execution
still depends on the concrete plan and resource handoff below. It does not transfer another MOD's
results or mark any blocked dataset complete.

**Alternative: retain the original V2/AMD64 execution contract.** Keep the existing inputs and
architecture as requirements. The shared V1/ARM implementation can inform a V2 adapter, but a
compatible backend, V2-specific evaluator/image qualification and resource handoff remain necessary.
This option cannot assume the shared program will provision AMD64 resources after its direction
changed.

Both choices retain HLE/FrontierScience calibration requirements. Neither choice substitutes fixed
trace replay for the five-dataset task-quality objective. A later trace-only study would be a
separate supplementary contract, with its own decision and claims.

## Concrete handoff before execution

1. Record the chosen dataset/configuration, task IDs, evaluator and runner revisions. Keep failed,
   unscored, dependency-recovered and original-scored trials distinct; do not select tasks by
   success.
1. Obtain an owner-confirmed workspace/endpoint and access method, CPU/memory/storage/network
   limits, resource window and model route. Do not treat another thread's workspace identity or
   historical running status as access granted to this task.
1. Pin selected image digests and derivation sources. Bind fresh-sandbox controls, grading files,
   fixtures and task-specific budgets. Preserve all original and derived identities separately.
1. Freeze model weights/tokenizer/template, actual host revisions, graph mode and compression
   configuration. Preflight each real input and output budget against the qualified context window;
   do not truncate, pad, alter task text or relax admission checks to obtain an eligible sample.
1. Freeze matched ON/OFF activation and lifecycle, cache policy, tool/scaffold/retry budgets, task
   quality scoring and telemetry before running. Collect request-matched scheduler/TP compression
   transactions; label an unexercised path accordingly.
1. Resolve the shared scheduling/plan-review state with #403's execution owners. Publish the
   canonical contract and later evidence in benchmark first; this MOD repository links to it.

The current proposal contains no new task execution, model response, score, performance estimate,
workspace deployment or automatic launch command.

## Independent remaining work for issue #1

The
[2026-10-27 maintainer review](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1#issuecomment-6446726027)
keeps service net cost, actual request capacity, task quality and immutable public-artifact
qualification open. It explicitly retains the completed Manager and local-copy work; these should
not be repeated as substitutes for the remaining questions.

A future service-cost study must freeze the full workload and both server lifecycles and include
compression work, prefill/decode, cache behavior and failures. Fixed trace replay may support that
study but cannot establish live task quality or maximum simultaneous request capacity. A capacity
study needs its own fixed admission/latency/failure criteria and concurrency sweep; a reduction in
KV-pool ownership alone cannot close it. Quality and resource conclusions must accompany speed
claims, including negative observations.

Public-artifact qualification remains separate: the merged shared router fix needs an immutable
published artifact and exact-artifact installation/runtime/rollback receipts. The retained repaired
candidate wheel is not the old public wheel, even though their version strings are equal. No new
release acceptance or general service-speedup claim follows from this planning change.
