"""Join paired outputs and validate compression evidence without hiding failures."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def resources(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    hbm = {}
    phases = {}
    for row in rows:
        metrics = row.get("metrics", {})
        phase = phases.setdefault(row["phase"], {"samples": 0, "peak_kv_usage_fraction": 0.0})
        phase["samples"] += 1
        phase["peak_kv_usage_fraction"] = max(
            phase["peak_kv_usage_fraction"], metrics.get("vllm:kv_cache_usage_perc", 0)
        )
        device = None
        for line in row.get("npu_smi", "").splitlines():
            match = re.search(r"\|\s*(\d+)\s+910B2", line)
            if match:
                device = match[1]
            elif device is not None:
                pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
                if pairs:
                    used, total = map(int, pairs[-1])
                    hbm.setdefault(device, []).append(used)
                    phase.setdefault("hbm_peak_mib", {})[device] = max(
                        phase.get("hbm_peak_mib", {}).get(device, 0), used
                    )
                    if total != 65536:
                        raise ValueError("Unexpected HBM total for this recorded profile")
                    device = None
    return {
        "samples": len(rows),
        "sampling_errors": sum("error" in row for row in rows),
        "device_hbm_mib": {key: {"minimum": min(values), "peak": max(values)} for key, values in hbm.items()},
        "peak_kv_usage_fraction": max(x["peak_kv_usage_fraction"] for x in phases.values()),
        "phases": phases,
    }


def transactions(log: str, results: list[dict], enabled: bool) -> dict:
    rows = []
    errors = []
    for result in results:
        if not result["success"]:
            errors.append(f"{result['case_id']}: request failed")
            continue
        identity = re.escape(result["request_id"])
        matches = re.findall(
            rf"scheduler commit request_id=({identity}(?:-[a-zA-Z0-9]+)?) "
            rf"semantic_tokens=(\d+) physical_tokens=(\d+) source_blocks=(\d+) "
            rf"destination_blocks=(\d+) released_blocks=(\d+)",
            log,
        )
        expected = enabled and result["prompt_tokens"] > 4096
        if len(matches) != int(expected):
            errors.append(f"{result['case_id']}: expected {int(expected)} commits, found {len(matches)}")
            continue
        for engine_id, semantic, physical, source, destination, released in matches:
            row = {
                "case_id": result["case_id"],
                "engine_request_id": engine_id,
                "semantic_tokens": int(semantic),
                "physical_tokens": int(physical),
                "source_blocks": int(source),
                "destination_blocks": int(destination),
                "released_blocks": int(released),
                "acknowledged_tp_ranks": [],
            }
            if int(semantic) != result["prompt_tokens"] or not 0 < int(physical) < int(semantic):
                errors.append(f"{result['case_id']}: mismatched lengths")
            for rank in (0, 1):
                acknowledgements = re.findall(
                    rf"Worker_TP{rank} [^\n]*worker commit acknowledged request_id={re.escape(engine_id)} "
                    rf"semantic_tokens=(\d+) physical_tokens=(\d+)",
                    log,
                )
                if acknowledgements == [(semantic, physical)]:
                    row["acknowledged_tp_ranks"].append(rank)
                else:
                    errors.append(f"{result['case_id']}: TP{rank} acknowledgement mismatch")
            rows.append(row)
    return {"verified": not errors, "errors": errors, "transactions": rows}


def analyze(root: Path, output: Path) -> None:
    arms = {}
    by_id = {}
    for name in ("baseline", "pyramidkv"):
        path = root / name
        results = json.loads((path / "results.json").read_text())
        by_id[name] = {x["case_id"]: x for x in results}
        groups = json.loads((path / "groups.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        arms[name] = {
            "summary": summary,
            "resources": resources(path / "telemetry.jsonl"),
            "compression": transactions((root / f"{name}-server.log").read_text(), results, name == "pyramidkv"),
            "performance": {},
            "capacity": [group for group in groups if group["kind"] == "capacity"],
            "boundary": [x for x in results if x["kind"] == "boundary"],
        }
        for length in (1024, 8192, 24576):
            for concurrency in (1, 4):
                key = f"{length}-c{concurrency}"
                cohorts = [x for x in groups if x["name"].startswith(f"perf-{key}-")]
                selected = [by_id[name][case] for group in cohorts for case in group["case_ids"]]
                good = [x for x in selected if x["success"]]
                value = {
                    "cohorts": len(cohorts),
                    "successful_requests": len(good),
                    "failed_requests": len(selected) - len(good),
                    "cohort_output_tokens_per_second": [x["output_tokens_per_second"] for x in cohorts],
                    "output_tokens_per_second": sum(x["usage"]["completion_tokens"] for x in good)
                    / sum(x["wall_seconds"] for x in cohorts),
                    "cached_prompt_tokens": sum(
                        (x["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) for x in good
                    ),
                }
                for metric in ("ttft_seconds", "tpot_seconds", "e2e_seconds"):
                    values = [x[metric] for x in good]
                    value[metric] = {
                        "mean": statistics.mean(values),
                        "p50": percentile(values, 0.5),
                        "p95": percentile(values, 0.95),
                    }
                arms[name]["performance"][key] = value
    if by_id["baseline"].keys() != by_id["pyramidkv"].keys():
        raise ValueError("Paired arms do not cover the same cases")
    for key, baseline in by_id["baseline"].items():
        enabled = by_id["pyramidkv"][key]
        if any(baseline[field] != enabled[field] for field in ("prompt_sha256", "max_tokens", "ignore_eos")):
            raise ValueError(f"Mismatched paired inputs: {key}")
    scores = {name: arm["summary"]["quality_f1"] for name, arm in arms.items()}
    drops = {task: scores["baseline"][task] - scores["pyramidkv"][task] for task in scores["baseline"]}
    quality = {
        "scores": scores,
        "baseline_minus_pyramidkv_f1_points": drops,
        "mean_drop_points": statistics.mean(drops.values()),
        "gate_passed": statistics.mean(drops.values()) <= 3 and max(drops.values()) <= 5,
        "sample_count_per_task": 50,
    }
    ratios = {}
    for key, baseline in arms["baseline"]["performance"].items():
        enabled = arms["pyramidkv"]["performance"][key]
        ratios[key] = {
            "throughput_change_percent": (
                enabled["output_tokens_per_second"] / baseline["output_tokens_per_second"] - 1
            )
            * 100
        }
        for metric in ("ttft_seconds", "tpot_seconds", "e2e_seconds"):
            ratios[key][metric + "_mean_change_percent"] = (
                enabled[metric]["mean"] / baseline[metric]["mean"] - 1
            ) * 100
    evidence = {
        "scope": "Fixed 150-example QA regression and 32K-context deployment, not full benchmark or maximum capacity",
        "quality": quality,
        "arms": arms,
        "performance_relative_changes": ratios,
        "source_artifact_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and any(parent.name in ("baseline", "pyramidkv") for parent in path.parents)
        },
    }
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)
