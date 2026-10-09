import io
import json
import runpy
from pathlib import Path

import pytest

EVAL = runpy.run_path(str(Path(__file__).parents[1] / "scripts/qwen35/evaluate.py"))


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
