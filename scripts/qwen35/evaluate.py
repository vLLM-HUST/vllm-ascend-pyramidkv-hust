"""Prepare and run paired, evidence-retaining Qwen3.5 evaluation workloads."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import statistics
import string
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

TASKS = ("narrativeqa", "qasper", "2wikimqa")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def qa_f1(prediction: str, answers: list[str]) -> float:
    """LongBench English QA F1: normalized token overlap, max over references."""

    def tokens(text: str) -> list[str]:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    pred = tokens(prediction)
    scores = []
    for answer in answers:
        gold = tokens(answer)
        overlap = sum((Counter(pred) & Counter(gold)).values())
        scores.append(2 * overlap / (len(pred) + len(gold)) if overlap else 0.0)
    return max(scores, default=0.0)


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    config = args.longbench_repo / "LongBench/config"
    templates = json.loads((config / "dataset2prompt.json").read_text())
    max_lengths = json.loads((config / "dataset2maxlen.json").read_text())
    cases = []
    inputs = {}
    # Trim the instruction's middle before applying the model's chat template.
    # Both arms consume these same saved IDs; neither independently tokenizes.
    for task in TASKS:
        path = args.data / f"{task}.jsonl"
        inputs[task] = digest(path.read_bytes())
        rows = [json.loads(line) for line in path.read_text().splitlines()][: args.samples_per_task]
        if len(rows) != args.samples_per_task:
            raise ValueError(f"Expected {args.samples_per_task} examples for {task}")
        for index, row in enumerate(rows):
            content = templates[task].format(**row)
            ids = tokenizer.encode(content, add_special_tokens=False)
            original_length = len(ids)
            if len(ids) > 16200:
                content = tokenizer.decode(ids[:8100]) + tokenizer.decode(ids[-8100:])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt = tokenizer.encode(rendered, add_special_tokens=False)
            if len(prompt) > 16384:
                raise ValueError("Chat template exceeds the declared prompt limit")
            cases.append(
                {
                    "case_id": f"{task}-{index:03d}",
                    "kind": "quality",
                    "task": task,
                    "dataset_id": row.get("_id"),
                    "subset": "development" if index < 50 else "holdout",
                    "answers": row["answers"],
                    "original_instruction_tokens": original_length,
                    "truncated": original_length > 16200,
                    "prompt": prompt,
                    "max_tokens": max_lengths[task],
                    "ignore_eos": False,
                }
            )
    # Deterministic synthetic performance inputs. No task-quality claims use them.
    pool = tokenizer.encode(
        "The garden has trees flowers and a pond. Scientific observations record light water temperature. "
        "0123456789 abcdefghijklmnopqrstuvwxyz",
        add_special_tokens=False,
    )
    groups = []

    def cohort(name: str, kind: str, length: int, output: int, concurrency: int, count: int, seed: int) -> None:
        case_ids = []
        for index in range(count):
            case_id = f"{name}-{index:02d}"
            rng = random.Random(seed + index)
            cases.append(
                {
                    "case_id": case_id,
                    "kind": kind,
                    "prompt": rng.choices(pool, k=length),
                    "max_tokens": output,
                    "ignore_eos": True,
                }
            )
            case_ids.append(case_id)
        groups.append({"name": name, "kind": kind, "concurrency": concurrency, "case_ids": case_ids})

    cohort("warmup", "warmup", 8192, 128, 1, 2, 900000)
    for length in (1024, 4096, 4097):
        cohort(f"boundary-{length}", "boundary", length, 32, 1, 1, 100000 + length)
    for length in (1024, 8192, 24576):
        for concurrency in (1, 4):
            for repeat in range(3):
                cohort(
                    f"perf-{length}-c{concurrency}-r{repeat}",
                    "performance",
                    length,
                    128,
                    concurrency,
                    3 if concurrency == 1 else 8,
                    length * 100 + concurrency * 1000 + repeat * 10,
                )
    for concurrency in (1, 4):
        cohort(f"capacity-32704-c{concurrency}", "capacity", 32704, 64, concurrency, concurrency, 800000 + concurrency)
    for case in cases:
        case["prompt_tokens"] = len(case["prompt"])
        case["prompt_sha256"] = digest(json.dumps(case["prompt"], separators=(",", ":")).encode())
    write_json(args.output / "cases.json", cases)
    write_json(args.output / "groups.json", groups)
    write_json(
        args.output / "manifest.json",
        {
            "dataset_files_sha256": inputs,
            "template_sha256": digest((config / "dataset2prompt.json").read_bytes()),
            "cases_sha256": digest((args.output / "cases.json").read_bytes()),
            "groups_sha256": digest((args.output / "groups.json").read_bytes()),
            "quality_selection": f"first {args.samples_per_task} examples per task; indices >= 50 are holdout",
            "quality_tasks": TASKS,
            "quality_gate": {"maximum_mean_f1_drop_points": 3, "maximum_task_f1_drop_points": 5},
            "generation": {"temperature": 0, "seed": 17, "thinking": False},
            "max_prompt_tokens": 16384,
            "quality_count": len(TASKS) * args.samples_per_task,
            "total_cases": len(cases),
        },
    )
    print(f"Prepared {len(cases)} cases", flush=True)


def fetch(base: str, path: str) -> str:
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open(base + path, timeout=10) as response:
        return response.read().decode()


def metric_values(text: str) -> dict[str, float]:
    result = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        key, value = line.rsplit(" ", 1)
        name = key.split("{")[0]
        if name in (
            "vllm:kv_cache_usage_perc",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:num_preemptions_total",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
        ):
            result[name] = result.get(name, 0.0) + float(value)
    return result


def stream_request(base: str, case: dict, out: Path) -> dict:
    payload = {
        "model": "qwen35-pyramidkv",
        "prompt": case["prompt"],
        "max_tokens": case["max_tokens"],
        "ignore_eos": case["ignore_eos"],
        "temperature": 0,
        "seed": 17,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        base + "/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    first = last = None
    text = ""
    usage = None
    request_id = None
    result = {k: v for k, v in case.items() if k not in ("prompt", "answers")}
    try:
        with client.open(request, timeout=600) as response, (out / f"{case['case_id']}.sse").open("wb") as raw:
            result["http_status"] = response.status
            for line in response:
                raw.write(line)
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                request_id = chunk.get("id", request_id)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("text", "")
                    if delta:
                        now = time.monotonic()
                        first = now if first is None else first
                        last = now
                        text += delta
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
        if not usage or usage["prompt_tokens"] != case["prompt_tokens"]:
            raise ValueError(f"Missing or mismatched token usage: {usage}")
        if first is None or last is None:
            raise ValueError("No output text received")
        if case["ignore_eos"] and usage["completion_tokens"] != case["max_tokens"]:
            raise ValueError("Synthetic request did not generate the fixed output length")
        result.update(
            success=True,
            request_id=request_id,
            output=text,
            usage=usage,
            ttft_seconds=first - start,
            e2e_seconds=time.monotonic() - start,
            tpot_seconds=(last - first) / max(usage["completion_tokens"] - 1, 1),
        )
        if case["kind"] == "quality":
            result["qa_f1"] = qa_f1(text, case["answers"])
    except Exception as exc:
        result.update(success=False, error=str(exc), output=text, e2e_seconds=time.monotonic() - start)
    write_json(out / f"{case['case_id']}.json", result)
    return result


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "requests"
    raw.mkdir()
    cases = json.loads((args.plan / "cases.json").read_text())
    lookup = {x["case_id"]: x for x in cases}
    groups = json.loads((args.plan / "groups.json").read_text())
    stop = threading.Event()
    phase = {"name": "initial"}

    def sample() -> None:
        last_npu = 0.0
        with (args.output / "telemetry.jsonl").open("w") as stream:
            while not stop.is_set():
                row = {"time_unix": time.time(), "phase": phase["name"]}
                try:
                    row["metrics"] = metric_values(fetch(args.base_url, "/metrics"))
                    if time.monotonic() - last_npu >= 2:
                        npu = subprocess.run(
                            ["npu-smi", "info"], capture_output=True, text=True, timeout=10, check=True
                        )
                        row["npu_smi"] = npu.stdout
                        last_npu = time.monotonic()
                except Exception as exc:
                    row["error"] = str(exc)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                stop.wait(0.5)

    thread = threading.Thread(target=sample, daemon=True)
    fetch(args.base_url, "/health")
    (args.output / "metrics-before.txt").write_text(fetch(args.base_url, "/metrics"))
    thread.start()
    all_results = []
    group_results = []
    try:
        for group in groups:
            phase["name"] = group["name"]
            print(f"{args.arm}: {group['name']}", flush=True)
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=group["concurrency"]) as pool:
                results = list(pool.map(lambda key: stream_request(args.base_url, lookup[key], raw), group["case_ids"]))
            elapsed = time.monotonic() - start
            all_results.extend(results)
            good = [x for x in results if x["success"]]
            group_results.append(
                dict(
                    group,
                    wall_seconds=elapsed,
                    success=len(good),
                    failures=len(results) - len(good),
                    output_tokens_per_second=sum(x["usage"]["completion_tokens"] for x in good) / elapsed,
                    mean_ttft_seconds=statistics.mean(x["ttft_seconds"] for x in good) if good else None,
                    mean_tpot_seconds=statistics.mean(x["tpot_seconds"] for x in good) if good else None,
                    mean_e2e_seconds=statistics.mean(x["e2e_seconds"] for x in good) if good else None,
                )
            )
            write_json(args.output / "groups.json", group_results)
            if len(good) != len(results):
                raise RuntimeError(f"Failed requests in {group['name']}; inspect retained evidence")
        if not args.skip_quality:
            quality_cases = [x for x in cases if x["kind"] == "quality"]
            for index, case in enumerate(quality_cases):
                phase["name"] = f"quality-{case['task']}"
                result = stream_request(args.base_url, case, raw)
                all_results.append(result)
                if not result["success"]:
                    raise RuntimeError(f"Failed quality request {case['case_id']}")
                if (index + 1) % 10 == 0:
                    print(f"{args.arm}: quality {index + 1}/{len(quality_cases)}", flush=True)
    finally:
        stop.set()
        thread.join(timeout=15)
        write_json(args.output / "results.json", all_results)
        (args.output / "metrics-after.txt").write_text(fetch(args.base_url, "/metrics"))
    write_json(
        args.output / "summary.json",
        {
            "arm": args.arm,
            "requests": len(all_results),
            "failures": sum(not x["success"] for x in all_results),
            "quality_f1": {
                task: 100 * statistics.mean(x["qa_f1"] for x in all_results if x.get("task") == task)
                for task in TASKS
                if any(x.get("task") == task for x in all_results)
            },
            "plan_manifest": json.loads((args.plan / "manifest.json").read_text()),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model-path", required=True)
    prep.add_argument("--longbench-repo", type=Path, required=True)
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--samples-per-task", type=int, choices=(50, 100), default=50)
    runner = sub.add_parser("run")
    runner.add_argument("--plan", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--arm", choices=("baseline", "pyramidkv"), required=True)
    runner.add_argument("--base-url", default="http://127.0.0.1:8000")
    runner.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else run(args)
