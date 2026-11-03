import asyncio
import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

SPEC = importlib.util.spec_from_file_location("qwen25_measure", Path(__file__).parents[1] / "scripts/qwen25/measure.py")
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)


def _events():
    return [
        {"t": 11.0, "data": {"choices": [{"text": ""}]}},
        {"t": 13.0, "data": {"choices": [{"text": "ab"}]}},
        {"t": 14.0, "data": {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}},
        {"t": 15.0, "data": "[DONE]"},
    ]


def test_metrics_use_first_choice_and_last_choice_not_usage_or_done():
    row = measure.request_metrics(_events(), 10.0, 7, 3)
    assert row["ttft_seconds"] == 1.0
    assert row["e2e_seconds"] == 3.0
    assert row["tpot_seconds"] == 1.0
    assert row["text"] == "ab"


@pytest.mark.parametrize("mutation", ["missing_done", "wrong_input", "wrong_output", "no_tokens"])
def test_incomplete_or_wrong_workload_cannot_enter_aggregate(mutation):
    events = _events()
    if mutation == "missing_done":
        events.pop()
    elif mutation == "no_tokens":
        events = events[2:]
    else:
        events[2]["data"]["usage"]["prompt_tokens" if mutation == "wrong_input" else "completion_tokens"] += 1
    with pytest.raises(ValueError):
        measure.request_metrics(events, 10.0, 7, 3)


def test_group_metrics_use_wall_time_and_per_request_tpot():
    rows = [
        dict(ttft_seconds=1, tpot_seconds=2, e2e_seconds=5, output_tokens=3),
        dict(ttft_seconds=3, tpot_seconds=4, e2e_seconds=11, output_tokens=3),
    ]
    result = measure.group_metrics(rows, 20)
    assert result["request_throughput"] == 0.1
    assert result["output_token_throughput"] == 0.3
    assert result["mean_ttft_ms"] == 2000
    assert result["mean_tpot_ms"] == 3000
    assert result["mean_e2e_ms"] == 8000


def test_cancelled_stream_keeps_received_tokens_and_failure_receipt(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web

    async def scenario():
        received = asyncio.Event()
        release = asyncio.Event()

        async def stalled(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"text":"partial"}]}\n\n')
            received.set()
            await release.wait()
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", stalled)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        row = {"id": "stalled", "input_tokens": 7, "payload": {"max_tokens": 3}}
        try:
            async with aiohttp.ClientSession() as session:
                task = asyncio.create_task(measure.request(session, f"http://127.0.0.1:{port}", row, tmp_path))
                await asyncio.wait_for(received.wait(), 2)
                await asyncio.sleep(0.02)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            release.set()
            await runner.cleanup()
        events = json.loads((tmp_path / "stalled.events.json").read_text())
        assert events["events"][0]["data"]["choices"][0]["text"] == "partial"
        result = json.loads((tmp_path / "stalled.result.json").read_text())
        assert not result["success"]
        assert "CancelledError" in result["error"]

    asyncio.run(scenario())


async def _check_concurrency():
    client = measure
    arrived = set()
    all_arrived = asyncio.Event()
    release = asyncio.Event()
    metrics_during = asyncio.Event()

    async def completion(request):
        identity = request.headers["X-Request-Id"]
        if identity != "long-readiness":
            arrived.add(identity)
            if len(arrived) == 128:
                all_arrived.set()
            await release.wait()
        return web.Response(
            text=(
                'data: {"choices":[{"text":"a"}]}\n\n'
                'data: {"choices":[{"text":"b"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
                "data: [DONE]\n\n"
            ),
            content_type="text/event-stream",
        )

    async def metrics(request):
        if len(arrived) == 128 and not release.is_set():
            metrics_during.set()
        return web.Response(text="vllm:kv_cache_usage_perc 0\n")

    app = web.Application()
    app.router.add_post("/v1/completions", completion)
    app.router.add_get("/metrics", metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        with tempfile.TemporaryDirectory(prefix="client-protocol-fixture-") as tmp:
            root = Path(tmp)
            requests = root / "input.json"
            requests.write_text(
                json.dumps(
                    {
                        "long": [
                            {
                                "id": f"long-{i}",
                                "input_tokens": 1,
                                "delay_seconds": 0,
                                "payload": {"prompt": "fixture", "max_tokens": 2},
                            }
                            for i in range(128)
                        ]
                    }
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                task = asyncio.create_task(
                    client.run(
                        SimpleNamespace(
                            requests=requests, output=root / "output", base=f"http://127.0.0.1:{port}", kinds=["long"]
                        )
                    )
                )
                try:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(all_arrived.wait(), 3)
                    count = len(arrived)
                    if count == 128:
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(metrics_during.wait(), 3)
                    telemetry_ok = metrics_during.is_set()
                finally:
                    release.set()
                    await asyncio.wait_for(task, 15)
            assert count == 128, f"Only {count}/128 requests arrived before any completed"
            assert telemetry_ok, "Metrics were blocked while all 128 requests were in flight"
            assert len(json.loads((root / "output/long/group.json").read_text())["results"]) == 128
    finally:
        await runner.cleanup()
    print("PASS: 128 simultaneous held requests and live metrics, using real measurement client")


def test_128_inflight_requests_leave_telemetry_live(monkeypatch):
    # Hardware polling is external to this real HTTP concurrency test.
    async def completed_poll(*args, **kwargs):
        async def communicate():
            return b"", b""

        return SimpleNamespace(returncode=0, communicate=communicate)

    monkeypatch.setattr(measure.asyncio, "create_subprocess_exec", completed_poll)
    asyncio.run(_check_concurrency())
