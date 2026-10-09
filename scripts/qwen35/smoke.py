"""Check HTTP output, request-matched TP commits, APC hits, and MTP deltas."""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


def compression_transactions(log: str, requests: list[dict]) -> list[dict]:
    transactions = []
    for request in requests:
        request_id = request["id"]
        # vLLM appends an internal engine suffix to the public completion ID.
        identity = re.escape(request_id)
        commits = re.findall(
            rf"scheduler commit request_id=({identity}(?:-[a-zA-Z0-9]+)?) "
            rf"semantic_tokens=(\d+) physical_tokens=(\d+)",
            log,
        )
        if len(commits) != 1:
            raise ValueError(f"{request_id}: expected one scheduler compression commit")
        engine_id, semantic_text, physical_text = commits[0]
        identity = re.escape(engine_id)
        semantic, physical = int(semantic_text), int(physical_text)
        if semantic != request["usage"]["prompt_tokens"] or not 0 < physical < semantic:
            raise ValueError(f"{request_id}: invalid compression lengths")
        for rank in (0, 1):
            acknowledgements = re.findall(
                rf"Worker_TP{rank} [^\n]*worker commit acknowledged request_id={identity} "
                rf"semantic_tokens=(\d+) physical_tokens=(\d+)",
                log,
            )
            if acknowledgements != [(str(semantic), str(physical))]:
                raise ValueError(f"{request_id}: TP{rank} acknowledgement missing or inconsistent")
        transactions.append(
            {
                "request_id": request_id,
                "engine_request_id": engine_id,
                "semantic_tokens": semantic,
                "physical_tokens": physical,
                "tp_ranks": [0, 1],
            }
        )
    return transactions


def metric_total(text: str, name: str) -> float:
    values = re.findall(rf"^{re.escape(name)}(?:\{{[^\n]*\}})?\s+([\d.eE+-]+)$", text, re.MULTILINE)
    if not values:
        raise ValueError(f"Missing metric: {name}")
    return sum(map(float, values))


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    # Refuse to overwrite evidence from a previous run.
    args.output.mkdir(parents=True, exist_ok=False)
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def fetch(path: str, payload: dict | None = None) -> tuple[int, str]:
        request = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with client.open(request, timeout=600) as response:
            return response.status, response.read().decode()

    def save(name: str, value: object) -> None:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        (args.output / name).write_text(text + "\n", encoding="utf-8")

    try:
        status, _ = fetch("/health")
        save("health.json", {"status": status})
        _, before = fetch("/metrics")
        save("metrics-before.txt", before)
        log_offset = args.server_log.stat().st_size
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        filler = (
            "This is background material for testing a language model. "
            "The garden has trees, flowers, and a small pond.\n"
        )
        long_prompt = filler * 230 + "\n请只输出这句话：长文本测试成功。"
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": long_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        if prompt_tokens <= 4096:
            raise ValueError(f"Prompt does not cross compression threshold: {prompt_tokens}")
        requests = []
        for name, prompt, expected in (
            ("short", "请计算 1+1，只输出数字。", "2"),
            ("long", long_prompt, "长文本测试成功。"),
            ("long-repeat", long_prompt, "长文本测试成功。"),
        ):
            payload = {
                "model": args.served_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0,
                "seed": 17,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            save(f"{name}-request.json", payload)
            start = time.monotonic()
            status, raw = fetch("/v1/chat/completions", payload)
            save(f"{name}-response.json", raw)
            response = json.loads(raw)
            if response["choices"][0]["message"]["content"].strip() != expected:
                raise ValueError(f"{name}: unexpected model output")
            requests.append(
                {
                    "name": name,
                    "id": response["id"],
                    "status": status,
                    "elapsed_seconds": time.monotonic() - start,
                    "usage": response["usage"],
                }
            )
        _, after = fetch("/metrics")
        save("metrics-after.txt", after)
        # Read only this run's logs; old successful transactions cannot pass this check.
        with args.server_log.open("rb") as stream:
            stream.seek(log_offset)
            log = stream.read().decode(errors="replace")
        save("server-excerpt.log", log)
        transactions = compression_transactions(log, requests[1:])
        cached = requests[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
        if cached <= 0:
            raise ValueError("Repeated long request did not hit the prefix cache")
        mtp = {}
        for name in ("num_draft_tokens", "num_accepted_tokens"):
            metric = f"vllm:spec_decode_{name}_total"
            mtp[name] = metric_total(after, metric) - metric_total(before, metric)
            if mtp[name] <= 0:
                raise ValueError(f"No new MTP activity: {name}")
        save(
            "summary.json",
            {
                "passed": True,
                "scope": "functional smoke only; run on an otherwise idle server",
                "requests": requests,
                "compression_transactions": transactions,
                "prefix_hit_tokens": cached,
                "mtp_delta": mtp,
            },
        )
    except Exception as exc:
        save("failure.json", {"passed": False, "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model", default="qwen35-pyramidkv")
    run(parser.parse_args())
