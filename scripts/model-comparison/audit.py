"""Offline checks for the fixed two-rank model comparison evidence."""

import re
from collections import Counter


def transactions(log, model, expected_request_ids):
    scheduler = re.findall(
        r"scheduler commit request_id=(\S+) semantic_tokens=(\d+) physical_tokens=(\d+) "
        r"source_blocks=(\d+) destination_blocks=(\d+)",
        log,
    )
    worker = re.findall(
        r"worker commit acknowledged request_id=(\S+) semantic_tokens=(\d+) physical_tokens=(\d+) "
        r"destination_blocks=(\d+) worker_rank=(\d+)",
        log,
    )
    if len(scheduler) != len(expected_request_ids):
        raise ValueError("Unexpected scheduler transaction count")
    if log.count("worker commit acknowledged request_id=") != len(worker):
        raise ValueError("Unparseable worker acknowledgement")
    request_ids = [row[0] for row in scheduler]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Duplicate scheduler commit")
    for expected in expected_request_ids:
        if sum(request.startswith(f"cmpl-{expected}-0-") for request in request_ids) != 1:
            raise ValueError(f"Missing or duplicate request transaction: {expected}")
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
