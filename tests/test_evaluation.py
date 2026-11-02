import io
import json
import runpy
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

EVAL = runpy.run_path(str(Path(__file__).parents[1] / "scripts/qwen35/evaluate.py"))
ANALYZE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/qwen35/analyze_evaluation.py"))


def saved_plan(path):
    path.mkdir()
    cases = [
        dict(
            case_id=name, kind=kind, prompt=[1, 2], prompt_tokens=2, max_tokens=2, ignore_eos=True, answers=["blue cat"]
        )
        for name, kind in [("warm", "warmup"), ("perf", "performance"), ("cap", "capacity"), ("qa", "quality")]
    ]
    cases[-1].update(task="qasper", subset="development")
    groups = [
        dict(name=row["case_id"], kind=row["kind"], concurrency=1, case_ids=[row["case_id"]]) for row in cases[:-1]
    ]
    save_plan(path, cases, groups)
    return cases, groups


def save_plan(path, cases, groups):
    for name, value in [("cases", cases), ("groups", groups)]:
        (path / f"{name}.json").write_text(json.dumps(value))
    (path / "manifest.json").write_text(
        json.dumps(
            {f"{name}_sha256": EVAL["digest"]((path / f"{name}.json").read_bytes()) for name in ["cases", "groups"]}
        )
    )


@pytest.fixture
def local_api():
    received = []
    cached = {"tokens": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            self.send_response(200)
            self.end_headers()
            event = {
                "id": f"cmpl-{len(received)}",
                "choices": [{"text": "blue cat", "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": cached["tokens"]},
                },
            }
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received, cached
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def invoke_run(plan, output, base, *options):
    return subprocess.run(
        [
            sys.executable,
            EVAL["__file__"],
            "run",
            "--plan",
            str(plan),
            "--output",
            str(output),
            "--arm",
            "baseline",
            "--base-url",
            base,
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_group_selection_dispatches_only_requested_groups_and_records_plan(tmp_path, local_api):
    base, received, _ = local_api
    saved_plan(tmp_path / "plan")
    result = invoke_run(
        tmp_path / "plan", tmp_path / "run", base, "--group-kinds", "capacity", "warmup", "--skip-quality"
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads((tmp_path / "run/results.json").read_text())
    assert [row["case_id"] for row in rows] == ["warm", "cap"]
    assert len(received) == 2
    receipt = json.loads((tmp_path / "run/execution.json").read_text())
    assert receipt["group_names"] == ["warm", "cap"]
    assert receipt["quality_case_ids"] == []
    assert receipt["plan_manifest_sha256"] == EVAL["digest"]((tmp_path / "plan/manifest.json").read_bytes())


def test_default_selection_preserves_groups_then_quality(tmp_path, local_api):
    base, received, _ = local_api
    saved_plan(tmp_path / "plan")
    result = invoke_run(tmp_path / "plan", tmp_path / "run", base)
    assert result.returncode == 0, result.stderr
    rows = json.loads((tmp_path / "run/results.json").read_text())
    assert [row["case_id"] for row in rows] == ["warm", "perf", "cap", "qa"]
    assert len(received) == 4


def test_frozen_repeatability_probes_remain_explicit_groups(tmp_path, local_api):
    base, received, _ = local_api
    plan = tmp_path / "plan"
    cases, groups = saved_plan(plan)
    cases[0]["kind"] = groups[0]["kind"] = "repeatability"
    save_plan(plan, cases, groups)
    result = invoke_run(plan, tmp_path / "run", base, "--group-kinds", "repeatability", "--skip-quality")
    assert result.returncode == 0, result.stderr
    assert len(received) == 1
    assert json.loads((tmp_path / "run/groups.json").read_text())[0]["kind"] == "repeatability"


@pytest.mark.parametrize("fault", ["cases_hash", "groups_hash", "duplicate", "unsafe", "missing", "empty"])
def test_invalid_plan_stops_before_requests(tmp_path, local_api, fault):
    base, received, _ = local_api
    plan = tmp_path / "plan"
    cases, groups = saved_plan(plan)
    options = []
    if fault.endswith("_hash"):
        name = fault.split("_")[0]
        with (plan / f"{name}.json").open("a") as stream:
            stream.write(" ")
    else:
        if fault == "duplicate":
            cases.append(cases[0])
        elif fault == "unsafe":
            cases[0]["case_id"] = "../escape"
            groups[0]["case_ids"] = ["../escape"]
        elif fault == "missing":
            options = ["--group-kinds", "boundary"]
        elif fault == "empty":
            options = ["--group-kinds", "--skip-quality"]
        save_plan(plan, cases, groups)
    result = invoke_run(plan, tmp_path / "run", base, *options)
    assert result.returncode != 0
    assert not received
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "fault", ["missing_prompt", "missing_answers", "wrong_length", "bad_token", "bad_limit", "bad_flag", "bad_salt"]
)
def test_malformed_later_case_is_rejected_before_earlier_groups(tmp_path, local_api, fault):
    base, received, _ = local_api
    plan = tmp_path / "plan"
    cases, groups = saved_plan(plan)
    row = cases[-1]
    if fault == "missing_prompt":
        del row["prompt"]
    elif fault == "missing_answers":
        del row["answers"]
    elif fault == "wrong_length":
        row["prompt_tokens"] = 3
    elif fault == "bad_token":
        row["prompt"] = [True, -1]
    elif fault == "bad_limit":
        row["max_tokens"] = 0
    elif fault == "bad_flag":
        row["ignore_eos"] = "false"
    elif fault == "bad_salt":
        row["cache_salt"] = ["invalid"]
    save_plan(plan, cases, groups)
    result = invoke_run(plan, tmp_path / "run", base)
    assert result.returncode != 0
    assert not received
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("fault", ["unknown_quality_kind", "orphan_group_case"])
def test_default_plan_cannot_silently_omit_cases(tmp_path, local_api, fault):
    base, received, _ = local_api
    plan = tmp_path / "plan"
    cases, groups = saved_plan(plan)
    if fault == "unknown_quality_kind":
        cases[-1]["kind"] = "qualty"
    else:
        groups.pop()
    save_plan(plan, cases, groups)
    result = invoke_run(plan, tmp_path / "run", base)
    assert result.returncode != 0
    assert not received
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("cached_tokens", [0, 1])
def test_saved_salt_and_uncached_policy_retain_exact_requests(tmp_path, local_api, cached_tokens):
    base, received, cached = local_api
    cached["tokens"] = cached_tokens
    plan = tmp_path / "plan"
    cases, groups = saved_plan(plan)
    cases[1].update(cache_salt="frozen-perf-salt", require_uncached=True)
    save_plan(plan, cases, groups)
    result = invoke_run(plan, tmp_path / "run", base, "--group-kinds", "performance", "--skip-quality")
    assert (result.returncode == 0) == (cached_tokens == 0), result.stderr
    assert received[0]["cache_salt"] == "frozen-perf-salt"
    assert json.loads((tmp_path / "run/requests/perf.request.json").read_text()) == received[0]
    row = json.loads((tmp_path / "run/requests/perf.json").read_text())
    assert row["success"] == (cached_tokens == 0)
    assert (tmp_path / "run/requests/perf.sse").exists()
    assert len(json.loads((tmp_path / "run/results.json").read_text())) == 1


@pytest.mark.parametrize("concurrency,expected_count", [([1, 4], 409), ([1, 2, 4], 411)])
def test_prepare_capacity_points_preserve_quality_and_add_independent_c2_cases(
    tmp_path, monkeypatch, concurrency, expected_count
):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [1, 2, 3]

        def apply_chat_template(self, messages, **kwargs):
            return "question"

    loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=loader))
    config = tmp_path / "LongBench/config"
    config.mkdir(parents=True)
    (config / "dataset2prompt.json").write_text(json.dumps(dict.fromkeys(EVAL["TASKS"], "{question}")))
    (config / "dataset2maxlen.json").write_text(json.dumps(dict.fromkeys(EVAL["TASKS"], 2)))
    for task in EVAL["TASKS"]:
        (tmp_path / f"{task}.jsonl").write_text(
            "\n".join(json.dumps({"question": str(i), "answers": ["answer"]}) for i in range(100))
        )
    args = SimpleNamespace(
        model_path="unused",
        longbench_repo=tmp_path,
        data=tmp_path,
        output=tmp_path / "plan",
        samples_per_task=100,
        capacity_concurrency=concurrency,
    )
    EVAL["prepare"](args)
    cases = json.loads((args.output / "cases.json").read_text())
    assert len(cases) == expected_count
    assert sum(row["kind"] == "quality" for row in cases) == 300
    if 2 in concurrency:
        c2 = [row for row in cases if row["case_id"].startswith("capacity-32704-c2-")]
        assert len(c2) == 2
        assert all(len(row["prompt"]) == 32704 and row["max_tokens"] == 64 for row in c2)
        assert c2[0]["prompt_sha256"] != c2[1]["prompt_sha256"]
    loaded, groups, _ = EVAL["load_plan"](args.output, ["capacity"], True)
    assert [g["concurrency"] for g in groups] == concurrency
    assert len(loaded) == expected_count


@pytest.mark.parametrize("values", [[2, 2], [8], [], [0]])
def test_capacity_rejects_unqualified_or_repeated_points_before_preparation(tmp_path, values):
    args = SimpleNamespace(output=tmp_path / "plan", capacity_concurrency=values)
    with pytest.raises(ValueError, match="capacity"):
        EVAL["prepare"](args)
    assert not args.output.exists()


@pytest.mark.parametrize(
    ("prediction", "answers", "expected"),
    [
        ("The BLUE cat!", ["blue cat"], 1.0),
        ("cat cat dog", ["cat dog"], 0.8),
        ("wrong", ["correct", "wrong"], 1.0),
        ("a", ["the"], 0.0),
        ("", ["answer"], 0.0),
        ("red", [], 0.0),
    ],
)
def test_qa_f1_reference_behavior(prediction: str, answers: list[str], expected: float) -> None:
    assert EVAL["qa_f1"](prediction, answers) == pytest.approx(expected)


@pytest.mark.parametrize("prompt_tokens", [2, 3])
def test_stream_requires_matching_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_tokens: int) -> None:
    events = [
        {"id": "cmpl-example", "choices": [{"text": "blue", "finish_reason": None}]},
        {"id": "cmpl-example", "choices": [{"text": " cat", "finish_reason": "length"}]},
        {"id": "cmpl-example", "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2}},
    ]
    data = b"".join(b"data: " + json.dumps(x).encode() + b"\n\n" for x in events) + b"data: [DONE]\n\n"

    class Response(io.BytesIO):
        status = 200

    class Client:
        def open(self, request: object, timeout: int) -> Response:
            return Response(data)

    monkeypatch.setattr(EVAL["urllib"].request, "build_opener", lambda *args: Client())
    case = {
        "case_id": "example",
        "kind": "quality",
        "prompt": [1, 2],
        "prompt_tokens": 2,
        "max_tokens": 2,
        "ignore_eos": True,
        "answers": ["blue cat"],
    }
    result = EVAL["stream_request"]("http://localhost", case, tmp_path)
    assert result["success"] == (prompt_tokens == 2)
    assert (tmp_path / "example.sse").read_bytes() == data
    if result["success"]:
        assert result["qa_f1"] == 1.0
        assert result["output"] == "blue cat"
        assert result["e2e_seconds"] >= result["ttft_seconds"] >= 0
    else:
        assert "mismatched token usage" in result["error"]


def test_telemetry_ignores_descriptions_and_unrelated_metrics() -> None:
    text = '# HELP vllm:num_preemptions_total counter\nvllm:num_preemptions_total{engine="0"} 2\nother 99\n'
    assert EVAL["metric_values"](text) == {"vllm:num_preemptions_total": 2.0}


def test_paired_transactions_enforce_admission_and_both_workers() -> None:
    requests = [{"case_id": "boundary", "success": True, "request_id": "cmpl-a", "prompt_tokens": 4097}]
    log = (
        "scheduler commit request_id=cmpl-a-deadbeef semantic_tokens=4097 physical_tokens=2048 "
        "source_blocks=3 destination_blocks=1 released_blocks=3\n"
    )
    for rank in (0, 1):
        log += (
            f"Worker_TP{rank} pid=1 worker commit acknowledged request_id=cmpl-a-deadbeef "
            "semantic_tokens=4097 physical_tokens=2048\n"
        )
    check = ANALYZE["transactions"]
    assert check(log, requests, True)["verified"]
    assert check(log.replace("cmpl-a-deadbeef", "cmpl-a-0-deadbeef"), requests, True)["verified"]
    assert not check(log.replace("cmpl-a-deadbeef", "cmpl-a-1-deadbeef"), requests, True)["verified"]
    assert not check(log, requests, False)["verified"]
    assert not check(log.replace("Worker_TP1", "Worker_TP2"), requests, True)["verified"]
    assert not check("", requests, True)["verified"]
    requests[0]["prompt_tokens"] = 4096
    assert check("", requests, True)["verified"]
    assert not check(log, requests, True)["verified"]


def test_resource_parser_keeps_device_hbm_separate_from_cache(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    row = {
        "phase": "perf",
        "metrics": {"vllm:kv_cache_usage_perc": 0.5},
        "npu_smi": "| 6 910B2 | OK | 0 / 0 |\n| 0 | 0000:82:00.0 | 0 | 0 / 0 | 47000/ 65536 |\n",
    }
    telemetry.write_text(json.dumps(row) + "\n")
    result = ANALYZE["resources"](telemetry)
    assert result["peak_kv_usage_fraction"] == 0.5
    assert result["device_hbm_mib"] == {"6": {"minimum": 47000, "peak": 47000}}


def test_quality_gate_cannot_hide_holdout_loss_with_development_gain() -> None:
    arms = {"baseline": {}, "pyramidkv": {}}
    for task in ("narrativeqa", "qasper", "2wikimqa"):
        for subset, enabled_score in (("development", 0.8), ("holdout", 0.44)):
            key = f"{task}-{subset}"
            row = {"kind": "quality", "task": task, "subset": subset, "qa_f1": 0.5}
            arms["baseline"][key] = row
            arms["pyramidkv"][key] = dict(row, qa_f1=enabled_score)
    result = ANALYZE["quality_summary"](arms)
    assert result["mean_drop_points"] < 0
    assert result["subsets"]["development"]["gate_passed"]
    assert not result["subsets"]["holdout"]["gate_passed"]
    assert not result["gate_passed"]
