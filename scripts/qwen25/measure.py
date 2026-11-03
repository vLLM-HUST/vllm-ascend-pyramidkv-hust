"""Frozen-request serving measurement with replayable SSE timestamps.

The completion payload, first/last-choice timing, and short-workload arrival
schedule follow vLLM 4861aab3. Preparation supplies already-tokenized identities;
no tokenization runs alongside measured requests.
"""

import argparse
import asyncio
import hashlib
import json
import signal
import statistics
import time
from contextlib import suppress
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def request_metrics(events, start, input_tokens, output_tokens):
    choices = [e for e in events if isinstance(e["data"], dict) and e["data"].get("choices")]
    usages = [e["data"]["usage"] for e in events if isinstance(e["data"], dict) and e["data"].get("usage")]
    if not choices or not usages or not events or events[-1]["data"] != "[DONE]":
        raise ValueError("Incomplete stream, missing choices, usage or DONE")
    usage = usages[-1]
    if usage.get("prompt_tokens") != input_tokens or usage.get("completion_tokens") != output_tokens:
        raise ValueError(f"Workload token mismatch: {usage}")
    ttft, e2e = choices[0]["t"] - start, choices[-1]["t"] - start
    if ttft < 0 or e2e < ttft:
        raise ValueError("Non-monotonic stream")
    return dict(
        ttft_seconds=ttft,
        e2e_seconds=e2e,
        tpot_seconds=(e2e - ttft) / (output_tokens - 1) if output_tokens > 1 else 0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage=usage,
        text="".join(e["data"]["choices"][0].get("text") or "" for e in choices),
    )


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def group_metrics(rows, wall_seconds):
    if not rows or wall_seconds <= 0:
        raise ValueError("Empty/invalid measurement")
    result = dict(
        requests=len(rows),
        wall_seconds=wall_seconds,
        request_throughput=len(rows) / wall_seconds,
        output_token_throughput=sum(r["output_tokens"] for r in rows) / wall_seconds,
    )
    for key in ("ttft", "tpot", "e2e"):
        values = [r[key + "_seconds"] * 1000 for r in rows]
        result["mean_" + key + "_ms"] = statistics.mean(values)
        result["p50_" + key + "_ms"] = percentile(values, 0.5)
        result["p95_" + key + "_ms"] = percentile(values, 0.95)
    return result


def metric_values(text):
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        if any(
            key in name
            for key in (
                "kv_cache_usage_perc",
                "num_preemptions_total",
                "num_requests_running",
                "num_requests_waiting",
                "prefix_cache",
            )
        ):
            result[name] = float(value)
    return result


async def request(session, base, row, output):
    events, lines = [], []
    start = time.perf_counter()
    status = None
    cancelled = False
    try:
        async with session.post(
            base + "/v1/completions", json=row["payload"], headers={"X-Request-Id": row["id"]}
        ) as response:
            status = response.status
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {await response.text()}")
            async for line in response.content:
                timestamp = time.perf_counter()
                lines.append(line)
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                events.append({"t": timestamp, "data": "[DONE]" if data == b"[DONE]" else json.loads(data)})
        result = request_metrics(events, start, row["input_tokens"], row["payload"]["max_tokens"])
        result.update(id=row["id"], start=start, end=time.perf_counter(), success=True)
    except (Exception, asyncio.CancelledError) as error:
        cancelled = isinstance(error, asyncio.CancelledError)
        result = dict(
            id=row["id"],
            start=start,
            end=time.perf_counter(),
            success=False,
            error=f"{type(error).__name__}: {error}",
            http_status=status,
        )
    # Defer file writes until the stream is complete.
    (output / (row["id"] + ".sse")).write_bytes(b"".join(lines))
    write(output / (row["id"] + ".events.json"), {"request": row, "start": start, "events": events})
    write(output / (row["id"] + ".result.json"), result)
    if cancelled:
        raise asyncio.CancelledError
    return result


async def run(args):
    import aiohttp

    raw = args.requests.read_bytes()
    workloads = json.loads(raw)
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "identity.json",
        {
            "requests_sha256": hashlib.sha256(raw).hexdigest(),
            "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    stop = asyncio.Event()
    stage = "startup"
    tasks = []
    main_task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, main_task.cancel)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600), connector=aiohttp.TCPConnector(limit=0)
    ) as session:

        async def telemetry():
            index = 0
            with (args.output / "telemetry.jsonl").open("w") as stream:
                while not stop.is_set():
                    row = {"at": time.time(), "t": time.perf_counter(), "stage": stage}
                    try:
                        async with session.get(
                            args.base + "/metrics", timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            row["metrics_text"] = await response.text()
                            row["metrics"] = metric_values(row["metrics_text"])
                        if index % 10 == 0:
                            proc = await asyncio.create_subprocess_exec(
                                "npu-smi", "info", stdout=asyncio.subprocess.PIPE
                            )
                            try:
                                stdout, _ = await asyncio.wait_for(proc.communicate(), 10)
                                row["npu_smi"] = stdout.decode()
                            finally:
                                if proc.returncode is None:
                                    proc.kill()
                                    await proc.wait()
                    except Exception as error:
                        row["error"] = str(error)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    index += 1
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(stop.wait(), 0.2)

        monitor = asyncio.create_task(telemetry())
        groups = []
        try:
            for kind in args.kinds:
                rows = workloads[kind]
                folder = args.output / kind
                folder.mkdir()
                # Historical bench serve sends one first-prompt readiness request
                # before each measured workload; num-warmups defaults to zero.
                stage = kind + "-readiness"
                ready = dict(rows[0], id=kind + "-readiness")
                ready_result = await request(session, args.base, ready, folder)
                if not ready_result["success"]:
                    raise RuntimeError(ready_result["error"])
                stage = kind
                start = time.perf_counter()
                tasks = []
                for row in rows:
                    await asyncio.sleep(max(0, start + row["delay_seconds"] - time.perf_counter()))
                    tasks.append(asyncio.create_task(request(session, args.base, row, folder)))
                results = await asyncio.gather(*tasks)
                end = time.perf_counter()
                write(folder / "group.json", {"start": start, "end": end, "results": results})
                if not all(r["success"] for r in results):
                    raise RuntimeError(f"{kind}: incomplete/failed requests; originals retained")
                summary = {"kind": kind, **group_metrics(results, end - start)}
                groups.append(summary)
                print(json.dumps(summary), flush=True)
                stage = kind + "-complete"
                await asyncio.sleep(1)
            write(args.output / "summary.json", groups)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            stop.set()
            await monitor
            asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8015")
    parser.add_argument("--kinds", nargs="+", default=["short", "long"])
    asyncio.run(run(parser.parse_args()))
