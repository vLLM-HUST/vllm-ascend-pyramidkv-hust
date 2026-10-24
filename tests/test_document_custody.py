"""Mixed-document projections and data links remain bound to immutable originals."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_document_projections_match_external_originals():
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    originals = Path(os.environ["VLLM_HUST_BENCHMARK_ORIGINALS"])
    assert len(config["document_projections"]) == 3
    for relative, row in config["document_projections"].items():
        assert row["url"].startswith("https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/")
        assert row["url"].endswith("/snapshot/" + relative)
        assert row["url"] in (ROOT / relative).read_text()
        assert hashlib.sha256((originals / relative).read_bytes()).hexdigest() == row["sha256"]
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != row["sha256"]


def test_selected_page_data_links_bind_original_bytes():
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    originals = Path(os.environ["VLLM_HUST_BENCHMARK_ORIGINALS"])
    path = "docs/selected-page-materialization.md"
    source = (ROOT / path).read_text()
    rows = config["document_data_links"][path]
    assert len(rows) == 5
    assert len({row["path"] for row in rows}) == 5
    for row in rows:
        assert row["url"].startswith("https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/")
        assert row["url"].endswith("/snapshot/" + row["path"])
        assert row["url"] in source
        assert hashlib.sha256((originals / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    assert "## Reproduction method" in source
    assert "scripts/qwen35/benchmark_materialization.py" in source
    assert "reports/pyramidkv-selected-page-abba-20261009" in source
    assert "Compression-off/on and old/new materialization are different comparisons" in source
