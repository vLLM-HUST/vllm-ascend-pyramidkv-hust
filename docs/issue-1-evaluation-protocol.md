# Issue #1: paired evidence protocol

This protocol is fixed before inspecting the paired evaluation outputs. It
targets the remaining quality, capacity, latency, throughput, and HBM evidence
in [issue #1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1).
It does not promise that all gates will pass or authorize a public performance
claim beyond the measured configurations.

## Fixed comparison

Use the exact host/model revisions, BF16, TP2, CANN 9.1, APC, MTP2, async,
chunked prefill, graph profile, 32768 context limit, four sequence limit, and
8 GiB per-worker KV budget from the [serving runbook](qwen35-serving-smoke.md).
The only intended treatment difference is shared adapter activation:
`VLLM_ASCEND_KVCOMPRESS_ENABLED=0` versus `1`; clear
`VLLMHUST_EXT_ENABLED_BUNDLES` in both environments. Preserve the launch
environment, logs, source commits and provider configuration for both arms.

Run baseline first, then PyramidKV. Give each process two excluded warmup
requests of 8192 input / 128 output tokens. Performance cohorts use unique,
deterministically generated inputs to avoid reuse of earlier cohort prefixes.
Quality requests may benefit from APC naturally; record cached-token counts.
Use an otherwise idle server and the same saved prompt token IDs in both arms.
Repeated cohorts within one process are not independent server lifecycles.

## Quality regression

- Dataset: official `THUDM/LongBench`, revision
  `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`.
- Reference templates/metric: `THUDM/LongBench` source commit
  `2e00731f8d0bff23dc4325161044d0ed8af94c1e`, original v1 `LongBench/` directory.
- Fixed first 50 examples each from `narrativeqa`, `qasper`, and `2wikimqa`.
- Cap the pre-template instruction at 16200 tokens by keeping its first and
  last 8100 tokens; apply the model chat template with thinking disabled;
  assert the final prompt is at most 16384 tokens. Record truncation and hashes.
- Greedy generation, seed 17, reference output limits 128/128/32 respectively.
- English normalized token QA-F1, best score across supplied references,
  multiplied by 100; average examples within each task and then the three tasks.
- Reuse the historical regression gate: mean loss at most 3 F1 points and each
  task loss at most 5 points. Fix these limits before seeing candidate outputs.

This is a 150-example regression subset, not a full LongBench leaderboard
evaluation. A pass does not establish general semantic equivalence; a failure
must remain visible and blocks a positive quality-promotion statement.
Dataset content stays in the local evidence directory. Public artifacts can
carry row IDs, input hashes, predictions, scores and reproduction instructions.

## Performance, capacity, and resources

- Admission checks: exactly 1024, 4096 and 4097 prompt tokens, 32 output tokens.
  Match each request to scheduler/worker logs; disabled has no compression,
  enabled must leave 4096 unchanged and compress 4097.
- Performance: 1024, 8192 and 24576 input tokens, 128 forced output tokens,
  concurrency 1 and 4, three cohorts per configuration. Each concurrency-1
  cohort has three requests; each concurrency-4 cohort has eight requests.
- Capacity: 32704 input + 64 forced output tokens at concurrency 1 and 4.
  Success establishes a tested configured-context point, not maximum hardware
  capacity or a larger context window than the baseline.
- Use streaming completions. Retain each SSE response and token usage.
  Throughput is completed output tokens / cohort wall time; TTFT is time to
  first nonempty text chunk; E2E is request completion time. Report TPOT as
  `(last_text_time - first_text_time) / (output_tokens - 1)`, an aggregate
  client-observed measure that does not resolve individual speculative tokens.
- Sample cache utilization/running/waiting/preemptions/MTP metrics every
  approximately 0.5 seconds and `npu-smi info` every approximately 2 seconds.
  Preserve timestamps and phase labels. Sampled peaks are lower bounds on
  instantaneous peaks. Device HBM includes reserved pools and other resident
  allocations; released request KV ownership does not imply lower HBM.
- Retain every failure, warmup and request. Never silently drop failed cohorts,
  recategorize warmup after observing results, or substitute historical results.

## Commands

Download the pinned dataset `data.zip` from Hugging Face and extract the three
JSONL files. Clone the pinned reference repository. In the prepared environment:

```bash
python scripts/qwen35/evaluate.py prepare \
  --model-path "$MODEL_PATH" --longbench-repo /path/to/LongBench \
  --data /path/to/extracted/data --output /path/to/plan
python scripts/qwen35/evaluate.py run --arm baseline \
  --plan /path/to/plan --output /path/to/baseline
# Restart with the same host arguments and compression enabled.
python scripts/qwen35/evaluate.py run --arm pyramidkv \
  --plan /path/to/plan --output /path/to/pyramidkv
```

Output directories must be new. Preserve the plan, raw request/SSE results,
metrics, telemetry and each server log together. Source and evidence hashes
identify the actual evaluated implementation independently of later docs edits.
