"""Offline checks for the fixed two-rank model comparison evidence."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def transactions(log, model, expected_request_ids):
    scheduler = re.findall(
        r"scheduler commit request_id=(\S+) semantic_tokens=(\d+) physical_tokens=(\d+) "
        r"source_blocks=(\d+) destination_blocks=(\d+)(?=\s|$)",
        log,
    )
    worker = re.findall(
        r"worker commit acknowledged request_id=(\S+) semantic_tokens=(\d+) physical_tokens=(\d+) "
        r"destination_blocks=(\d+) worker_rank=(\d+)(?=\s|$)",
        log,
    )
    if len(set(expected_request_ids)) != len(expected_request_ids):
        raise ValueError("Duplicate expected request ID")
    if log.count("scheduler commit request_id=") != len(scheduler):
        raise ValueError("Unparseable scheduler commit")
    if log.count("worker commit acknowledged request_id=") != len(worker):
        raise ValueError("Unparseable worker acknowledgement")
    request_ids = [row[0] for row in scheduler]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Duplicate scheduler commit")
    for expected in expected_request_ids:
        if sum(request.startswith(f"cmpl-{expected}-0-") for request in request_ids) != 1:
            raise ValueError(f"Missing or duplicate request transaction: {expected}")
    if len(scheduler) != len(expected_request_ids):
        raise ValueError("Unexpected scheduler transaction count")
    geometry = {"qwen25": ("7168", "1024", "56", "8"), "qwen35": ("7168", "2048", "4", "1")}[model]
    if any(tuple(row[1:]) != geometry for row in scheduler):
        raise ValueError("Unexpected model compression geometry")
    expected_workers = Counter(
        (request, semantic, physical, destination, rank)
        for request, semantic, physical, source, destination in scheduler
        for rank in ("0", "1")
    )
    if Counter(worker) != expected_workers:
        raise ValueError("Scheduler commit lacks exactly one matching acknowledgement from each TP rank")
    return len(scheduler)


def idle(metrics):
    """Require explicit zero activity, including every engine's metric row."""
    for name in ("num_requests_running", "num_requests_waiting"):
        values = re.findall(rf"^vllm:{name}\{{[^\n]*\}} (\S+)$", metrics, re.MULTILINE)
        if not values or any(float(value) != 0 for value in values):
            raise ValueError(f"Missing or nonzero final {name}")


def hbm(text):
    result = {}
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        card = re.match(r"\|\s*(1|6)\s+910B2\b", line)
        if card:
            used = re.search(r"(\d+)\s*/\s*65536\s*\|", lines[index + 1])
            if used:
                result[card.group(1)] = int(used.group(1))
    if result.keys() != {"1", "6"}:
        raise ValueError("Missing physical card1 or card6 HBM")
    return result


def pool(metrics):
    rows = re.findall(r"^vllm:cache_config_info\{([^\n]+)\} \S+$", metrics, re.MULTILINE)
    if len(rows) != 1:
        raise ValueError("Missing or ambiguous native cache configuration")
    labels = dict(re.findall(r'(?:^|,)(\w+)="([^"]*)"', rows[0]))
    try:
        return {
            "physical_blocks": int(labels["num_gpu_blocks"]),
            "logical_block_size": int(labels["block_size"]),
            "effective_capacity_tokens": int(labels["kv_cache_size_tokens"]),
            "max_concurrency_at_max_model_len": float(labels["kv_cache_max_concurrency"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Incomplete native cache configuration") from error


def main():
    parser = argparse.ArgumentParser(description="Strict Qwen3.5 release transaction and completion gate")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.requests.read_bytes()
    requests = json.loads(raw)
    long_ids = [row["id"] for row in requests["long"]]
    if set(long_ids) != {f"long-{i:02d}" for i in range(128)} or len(long_ids) != 128:
        raise ValueError("Release acceptance requires the complete long-00 through long-127 bank")
    count = transactions((args.run_dir / "server.log").read_text(), "qwen35", ["long-readiness", *long_ids])
    for kind, size, prompt, output in (("short", 48, 1024, 256), ("long", 128, 7168, 512)):
        rows = requests[kind]
        if len(rows) != size or {row["id"] for row in rows} != {f"{kind}-{i:02d}" for i in range(size)}:
            raise ValueError(f"Incomplete {kind} request bank")
        if any(row["input_tokens"] != prompt or row["payload"]["max_tokens"] != output for row in rows):
            raise ValueError(f"Changed {kind} workload lengths")
    metrics = (args.run_dir / "final-metrics.prom").read_text()
    idle(metrics)
    capacity = pool(metrics)
    if (capacity["physical_blocks"], capacity["logical_block_size"], capacity["effective_capacity_tokens"]) != (
        558,
        2048,
        457113,
    ):
        raise ValueError("Release acceptance requires the frozen 558-block pool")
    evaluation = args.run_dir / "evaluation"
    identity = json.loads((evaluation / "identity.json").read_text())
    if identity["requests_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Request bank identity changed")
    completed = 0
    for kind in ("short", "long"):
        rows = requests[kind]
        for row in [dict(rows[0], id=f"{kind}-readiness"), *rows]:
            result = json.loads((evaluation / kind / f"{row['id']}.result.json").read_text())
            usage = result.get("usage", {})
            if (
                result.get("success") is not True
                or result.get("id") != row["id"]
                or usage.get("prompt_tokens") != row["input_tokens"]
                or usage.get("completion_tokens") != row["payload"]["max_tokens"]
            ):
                raise ValueError(f"Incomplete or mismatched request: {row['id']}")
            completed += 1
    print(
        json.dumps(
            {
                "accepted": True,
                "target_commits": 128,
                "total_commits": count,
                "worker_acknowledgements": count * 2,
                "completed_requests": completed,
                "long109": "committed on scheduler and both TP ranks",
                "final_active_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
