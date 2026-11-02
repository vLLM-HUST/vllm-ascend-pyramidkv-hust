# Selecting frozen evaluation inputs

The [canonical qualification plan](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/main/docs/plans/pyramidkv-standalone-qualification-20261103.md)
defines experimental workloads, launch contracts and acceptance rules. This page
documents CLI behavior only; adding execution controls is not a measured result.

Preparation defaults preserve the historical case set. To include the qualified
two-request operating point, use the same pinned data, reference source and model:

```bash
python scripts/qwen35/evaluate.py prepare \
  --model-path /path/to/model --longbench-repo /path/to/LongBench \
  --data /path/to/data --output /path/to/new-plan \
  --samples-per-task 100 --capacity-concurrency 1 2 4
```

Only unique values 1, 2 and 4 are accepted. Each point creates that many unique
requests in a single cohort. Other prepared groups and quality cases are unchanged.
These are configured operating points; this option does not qualify larger runner
buffers, graph buckets or a higher resident-request limit.

By default, `run` executes every saved group in plan order, then quality cases.
Select kinds explicitly and omit quality with the existing flag:

```bash
python scripts/qwen35/evaluate.py run --arm baseline \
  --plan /path/to/new-plan --output /path/to/new-run \
  --group-kinds warmup capacity --skip-quality
```

Allowed kinds are `warmup`, `boundary`, `performance`, `capacity`, and
`repeatability` (for independently prepared probe groups). Selection
does not reorder the plan. Without `--skip-quality`, quality still runs after the
selected groups; `--group-kinds` with no values selects quality alone. An absent
requested kind or an empty total selection fails before network access.

The runner checks cases/groups against `manifest.json` before creating outputs or
contacting a server. It rejects duplicate/unsafe case IDs and invalid group
references. `execution.json` records the manifest hash and exact selected groups
and quality IDs. Use new output directories for every lifecycle.

Each request retains its exact `.request.json`, response `.sse`, and parsed result.
These request files can contain dataset content: keep them with restricted/local
originals under the canonical plan's custody rules, not in public source commits.
An optional frozen `cache_salt` field is sent unchanged. A case with
`require_uncached: true` fails unless usage explicitly reports zero cached tokens;
ordinary quality requests retain their original cache behavior. Failed requests
stop subsequent groups and remain in the output.

Selecting groups does not create the canonical service series' extra warmups,
repeatability probes, salts or independent process order. Prepare and freeze that
series' full case manifest under its own protocol before using these controls.
