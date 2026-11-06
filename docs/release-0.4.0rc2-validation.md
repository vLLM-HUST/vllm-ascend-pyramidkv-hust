# 0.4.0rc2.dev0 validation and release acceptance

The [published candidate receipt](releases/0.4.0rc2.dev0.md) records the completed
November 6 acceptance, exact artifacts, canonical raw evidence and remaining
qualification boundaries. The gates below remain the reproduction requirements.

This development candidate repairs an initial-compression boundary in Qwen3.5
hybrid replay. After preemption, native scheduling includes already-generated
tokens: a request with prompt length 7168, one generated token and 6144 computed
tokens can schedule 1025 tokens. Both compression participants require the
initial semantic length to equal the prompt length; 7169 skips that barrier.
The adapter now splits eligible replay at 7168 before block allocation and model
execution. Native alignment runs first; ordinary chunks, zero-token alignment,
short prompts, active compression and ordinary decode retain native behavior.
The query observation, selection algorithm, 2048-token destination and TP commit
protocol are unchanged.

## Historical evidence boundary

The frozen November 4 C128/558-block experiment has 127/128 target commits;
`long-109` is missing, although its request completed with 7168 input and 512
output tokens. Its final preemption counter is 98. The historical record lacks
per-request scheduling traces, so the exact cause of that individual omission
cannot be established retrospectively. The reproducible boundary defect above
is a supported mechanism, not proof of that historical sequence. Keep the old
failure and original acceptance status intact.

## Candidate gates

Run the [standalone validation](standalone-validation.md) package, host, device,
and lifecycle gates against the candidate source and installed wheel. Record
source commit plus any patch, wheel/sdist hashes, exact host revisions, package
versions and visible/physical device mapping. The development version string
alone does not identify an artifact.

The scheduler regression uses the pinned host's real Mamba splitting method and
checks both a running replay and a cached-prefix admission. It must observe
native scheduling of 1025, patched scheduling of 1024, and an acknowledged
7168→2048 private-block commit. Run it in the qualified host environment; a
host-dependent test skipped in generic CI does not satisfy this gate.

On the pinned core/Ascend pair, direct runner import in the generic suite can
skip the worker seam check because of upstream import ordering. Repeat that
check in a fresh process with the same plugin bootstrap as serving (not in the
generic suite, whose scheduler signature checks expect unpatched classes):

```bash
# First source the qualified host environment and select the candidate wheel.
VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_pyramidkv \
VLLM_ASCEND_PYRAMIDKV_ENABLED=1 \
VLLM_ASCEND_PYRAMIDKV_CONFIG=/absolute/evidence/pyramidkv.json \
python - <<'PY'
from vllm.plugins import load_general_plugins
load_general_plugins()
import pytest
raise SystemExit(pytest.main([
    "-q", "-rs",
    "tests/adapter/test_current_host_contract.py::test_ascend_v1_worker_seams_match_current_host",
]))
PY
```

Require this isolated check to pass rather than treating the generic-suite skip
as successful host validation. Run the four opt-in NPU oracle cases after the
serving process releases the devices.

For the high-concurrency serving gate, use the original frozen request bank:
48 short requests (1024/256), 128 long requests (7168/512), plus one readiness
request per workload. Keep TP2, eager/no-MTP, async scheduling, prefix caching,
chunked prefill, 4096 batched tokens, C128, and the native 558-block pool
(2048 logical block size, 457113 effective tokens). Freeze the launch/client
before the run, preserve all failures, and record any device substitution.

```bash
python scripts/model-comparison/audit.py \
  --requests /absolute/evidence/qwen35-requests.json \
  --run-dir /absolute/evidence/runs/qwen35-on-558
```

The command rejects missing/duplicate/unknown/malformed commits, requires one
matching acknowledgement from each TP rank, verifies the entire long request
ID set including `long-109`, checks pool geometry and request-bank identity,
checks successful completion and usage for all 178 requests, and requires zero
final running/waiting requests. Expected totals are 129 scheduler commits and
258 worker acknowledgements, including long readiness. A count-only check or
an audit accepting the observed subset is insufficient.

Additionally inspect raw SSE and retained events, serving errors, C128 arrival
coverage, replay-boundary diagnostics and process/device release. Preserve
originals outside the source repository and publish benchmark evidence through
the [original custody workflow](benchmark-originals.md) when publishing results.
The CLI is a strict transaction/completion gate, not a substitute for package,
device, error-log, concurrency or raw-stream validation.

## Promotion

Require all candidate gates and fresh evidence before release promotion. A
functional ON lifecycle does not establish throughput, latency, quality or HBM
improvements; those need a separately frozen paired benchmark. Keep
`0.4.0rc2.dev0` labeled as a development candidate, and do not reinterpret the
historical 127/128 run as passed.
