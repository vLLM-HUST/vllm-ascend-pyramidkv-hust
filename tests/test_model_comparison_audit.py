import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "model_comparison_audit", Path(__file__).parents[1] / "scripts/model-comparison/audit.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _log(model="qwen35"):
    physical, source, destination = (2048, 4, 1) if model == "qwen35" else (1024, 56, 8)
    return (
        f"scheduler commit request_id=cmpl-long-00-0-abcd semantic_tokens=7168 physical_tokens={physical} "
        f"source_blocks={source} destination_blocks={destination} released_blocks={source}\n"
        "(Worker_TP1 pid=20) (Worker_TP0 pid=10) "
        f"worker commit acknowledged request_id=cmpl-long-00-0-abcd semantic_tokens=7168 physical_tokens={physical} "
        f"destination_blocks={destination} worker_rank=0\n"
        f"worker commit acknowledged request_id=cmpl-long-00-0-abcd semantic_tokens=7168 physical_tokens={physical} "
        f"destination_blocks={destination} worker_rank=1\n"
    )


@pytest.mark.parametrize("model", ["qwen25", "qwen35"])
def test_two_rank_commit_uses_explicit_rank_despite_interleaved_prefixes(model):
    assert audit.transactions(_log(model), model, ["long-00"]) == 1


@pytest.mark.parametrize("mutation", ["missing_rank", "duplicate_rank", "wrong_geometry", "missing_request"])
def test_incomplete_or_mismatched_commits_fail_audit(mutation):
    log = _log()
    if mutation == "missing_rank":
        log = log.rsplit("worker commit acknowledged", 1)[0]
    elif mutation == "duplicate_rank":
        log = log.replace("worker_rank=1", "worker_rank=0")
    elif mutation == "wrong_geometry":
        log = log.replace("physical_tokens=2048", "physical_tokens=1024")
    else:
        log = log.replace("cmpl-long-00-", "cmpl-long-01-")
    with pytest.raises(ValueError):
        audit.transactions(log, "qwen35", ["long-00"])


def test_disabled_arm_rejects_any_compression():
    assert audit.transactions("native service", "qwen35", []) == 0
    with pytest.raises(ValueError):
        audit.transactions(_log(), "qwen35", [])


def test_release_coverage_reports_missing_long109():
    expected = [f"long-{i:02d}" for i in range(128)]
    log = "".join(_log().replace("long-00", key) for key in expected if key != "long-109")
    with pytest.raises(ValueError, match="long-109"):
        audit.transactions(log, "qwen35", expected)


def test_unparseable_scheduler_commit_is_not_ignored():
    with pytest.raises(ValueError, match="Unparseable scheduler"):
        audit.transactions(_log() + "scheduler commit request_id=broken\n", "qwen35", ["long-00"])


@pytest.mark.parametrize("field", ["destination_blocks=1", "worker_rank=0", "worker_rank=1"])
def test_transaction_numeric_suffixes_are_rejected(field):
    with pytest.raises(ValueError, match="Unparseable"):
        audit.transactions(_log().replace(field, field + "oops", 1), "qwen35", ["long-00"])


@pytest.mark.parametrize("ids", [["long-00", "long-00"], []])
def test_expected_ids_must_match_observed_transactions(ids):
    with pytest.raises(ValueError):
        audit.transactions(_log(), "qwen35", ids)


@pytest.mark.parametrize("value", ["1", "NaN", "-1"])
def test_release_rejects_nonzero_or_invalid_final_activity(value):
    metrics = f'vllm:num_requests_running{{engine="0"}} {value}\nvllm:num_requests_waiting{{engine="0"}} 0\n'
    with pytest.raises(ValueError):
        audit.idle(metrics)


def test_release_requires_both_final_activity_counters():
    with pytest.raises(ValueError):
        audit.idle('vllm:num_requests_running{engine="0"} 0\n')
    audit.idle('vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n')


def test_release_command_rejects_missing_long109(tmp_path):
    requests = {"long": [{"id": f"long-{i:02d}"} for i in range(128)]}
    bank = tmp_path / "requests.json"
    bank.write_text(json.dumps(requests))
    expected = ["long-readiness", *(row["id"] for row in requests["long"])]
    (tmp_path / "server.log").write_text(
        "".join(_log().replace("long-00", key) for key in expected if key != "long-109")
    )
    result = subprocess.run(
        [sys.executable, str(Path(audit.__file__)), "--requests", str(bank), "--run-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "long-109" in result.stderr


@pytest.mark.parametrize("mutation", [None, "usage", "identity", "pool", "short_bank", "long_length"])
def test_release_command_checks_complete_evidence(tmp_path, mutation):
    requests = {
        kind: [
            {"id": f"{kind}-{i:02d}", "input_tokens": tokens, "payload": {"max_tokens": output}} for i in range(count)
        ]
        for kind, count, tokens, output in [("short", 48, 1024, 256), ("long", 128, 7168, 512)]
    }
    if mutation == "short_bank":
        requests["short"].pop()
    if mutation == "long_length":
        requests["long"][109]["input_tokens"] = 512
    bank = tmp_path / "requests.json"
    bank.write_text(json.dumps(requests))
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    digest = hashlib.sha256(bank.read_bytes()).hexdigest()
    (evaluation / "identity.json").write_text(
        json.dumps({"requests_sha256": "wrong" if mutation == "identity" else digest})
    )
    for kind, rows in requests.items():
        (evaluation / kind).mkdir()
        for row in [dict(rows[0], id=f"{kind}-readiness"), *rows]:
            usage = {"prompt_tokens": row["input_tokens"], "completion_tokens": row["payload"]["max_tokens"]}
            if mutation == "usage" and row["id"] == "long-109":
                usage["completion_tokens"] = 511
            (evaluation / kind / f"{row['id']}.result.json").write_text(
                json.dumps({"id": row["id"], "success": True, "usage": usage})
            )
    (tmp_path / "server.log").write_text(
        "".join(_log().replace("long-00", key) for key in ["long-readiness", *(row["id"] for row in requests["long"])])
    )
    blocks = 557 if mutation == "pool" else 558
    (tmp_path / "final-metrics.prom").write_text(
        'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n'
        f'vllm:cache_config_info{{block_size="2048",num_gpu_blocks="{blocks}",'
        'kv_cache_size_tokens="457113",kv_cache_max_concurrency="55.8"} 1.0\n'
    )
    result = subprocess.run(
        [sys.executable, str(Path(audit.__file__)), "--requests", str(bank), "--run-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    if mutation:
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["completed_requests"] == 178


def test_hbm_reads_both_physical_cards_and_rejects_missing_card():
    text = "| 1 910B2 | OK |\n| 0 | 0 | 52100/ 65536 |\n| 6 910B2 | OK |\n| 0 | 0 | 52200/ 65536 |"
    assert audit.hbm(text) == {"1": 52100, "6": 52200}
    with pytest.raises(ValueError):
        audit.hbm(text.split("| 6 910B2")[0])


def test_hybrid_pool_uses_native_blocks_not_effective_token_capacity():
    metrics = (
        "# HELP vllm:cache_config_info cache configuration\n"
        'vllm:cache_config_info{block_size="2048",num_gpu_blocks="787",'
        'kv_cache_size_tokens="644710",kv_cache_max_concurrency="78.7"} 1.0\n'
    )
    assert audit.pool(metrics) == {
        "physical_blocks": 787,
        "logical_block_size": 2048,
        "effective_capacity_tokens": 644710,
        "max_concurrency_at_max_model_len": 78.7,
    }
    with pytest.raises(ValueError):
        audit.pool(metrics.replace('num_gpu_blocks="787",', ""))
