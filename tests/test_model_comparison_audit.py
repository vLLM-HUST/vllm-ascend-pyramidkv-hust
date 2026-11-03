import importlib.util
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


def test_hbm_reads_both_physical_cards_and_rejects_missing_card():
    text = "| 1 910B2 | OK |\n| 0 | 0 | 52100/ 65536 |\n| 6 910B2 | OK |\n| 0 | 0 | 52200/ 65536 |"
    assert audit.hbm(text) == {"1": 52100, "6": 52200}
    with pytest.raises(ValueError):
        audit.hbm(text.split("| 6 910B2")[0])
