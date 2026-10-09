"""Ensure HTTP success alone cannot be mistaken for committed compression."""

import runpy
from pathlib import Path

import pytest

SMOKE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/qwen35/smoke.py"))
REQUESTS = [{"id": "req-1", "usage": {"prompt_tokens": 5312}}]
LOG = "\n".join(
    ["scheduler commit request_id=req-1 semantic_tokens=5312 physical_tokens=2048"]
    + [
        f"(Worker_TP{rank} pid=1) worker commit acknowledged request_id=req-1 semantic_tokens=5312 physical_tokens=2048"
        for rank in (0, 1)
    ]
)


def test_requires_matching_scheduler_and_both_tp_workers() -> None:
    assert SMOKE["compression_transactions"](LOG, REQUESTS) == [
        {
            "request_id": "req-1",
            "engine_request_id": "req-1",
            "semantic_tokens": 5312,
            "physical_tokens": 2048,
            "tp_ranks": [0, 1],
        }
    ]


def test_matches_internal_engine_suffix_without_crossing_requests() -> None:
    log = LOG.replace("req-1", "req-1-a123b456")
    result = SMOKE["compression_transactions"](log, REQUESTS)
    assert result[0]["engine_request_id"] == "req-1-a123b456"
    with pytest.raises(ValueError):
        SMOKE["compression_transactions"](LOG.replace("req-1", "req-10-a123b456"), REQUESTS)
    with pytest.raises(ValueError):
        SMOKE["compression_transactions"](log + "\n" + LOG.replace("req-1", "req-1-b456a123"), REQUESTS)


@pytest.mark.parametrize(
    "log",
    [
        "",
        LOG.replace("req-1", "old-request"),
        LOG.splitlines()[0],
        "\n".join(LOG.splitlines()[:2]),
        LOG.replace("TP1", "TP0"),
        LOG.replace("physical_tokens=2048", "physical_tokens=5312"),
        LOG.replace("semantic_tokens=5312", "semantic_tokens=5313"),
        LOG + "\n" + LOG.splitlines()[-1],
        LOG.rsplit("physical_tokens=2048", 1)[0] + "physical_tokens=1024",
    ],
)
def test_rejects_missing_stale_or_inconsistent_commits(log: str) -> None:
    with pytest.raises(ValueError):
        SMOKE["compression_transactions"](log, REQUESTS)


def test_metrics_sum_series_and_require_evidence() -> None:
    metric = SMOKE["metric_total"]
    assert metric('drafts{engine="0"} 1.0\ndrafts{engine="1"} 2e1\n# drafts 99\n', "drafts") == 21
    with pytest.raises(ValueError, match="Missing metric"):
        metric("# drafts absent", "drafts")
