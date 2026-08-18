#!/usr/bin/env python3
"""
Replay a synthetic trace against a vLLM-compatible /v1/completions endpoint.

The trace CSV must contain:
  - arrival_time_s
  - input_len
  - output_len

This script emits two artifacts:
  1. a human-readable benchmark summary text file
  2. a machine-readable JSON summary with precise timing window bounds

The timing window is defined as:
  - start: immediately before the first request is sent
  - end: immediately after the last request completes
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    arrival_time_s: float
    input_len: int
    output_len: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--trace-csv", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--requests-jsonl")
    p.add_argument("--tokenizer-model")
    p.add_argument("--max-concurrency", type=int, default=64)
    p.add_argument("--num-warmups", type=int, default=0)
    p.add_argument("--request-timeout-s", type=float, default=900.0)
    p.add_argument("--fail-on-request-error", action="store_true")
    return p.parse_args()


def load_trace(path: Path) -> list[TraceRequest]:
    rows: list[TraceRequest] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"arrival_time_s", "input_len", "output_len"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"trace CSV missing required columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            rows.append(
                TraceRequest(
                    request_id=str(row.get("request_id") or i),
                    arrival_time_s=float(row["arrival_time_s"]),
                    input_len=int(row["input_len"]),
                    output_len=int(row["output_len"]),
                )
            )
    rows.sort(key=lambda r: r.arrival_time_s)
    if not rows:
        raise ValueError("trace CSV has no requests")
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def find_repeatable_chunk(tokenizer: Any) -> str:
    candidates = [
        " hello",
        " world",
        " token",
        " prompt",
        " data",
        " test",
        " sample",
        " request",
        " input",
        " trace",
        " alpha",
        " beta",
        " gamma",
        " delta",
    ]
    for piece in candidates:
        single = tokenizer.encode(piece, add_special_tokens=False)
        if len(single) != 1:
            continue
        trial = piece * 8
        if len(tokenizer.encode(trial, add_special_tokens=False)) == 8:
            return piece

    vocab_size = getattr(tokenizer, "vocab_size", 0) or 0
    for token_id in range(min(vocab_size, 10000)):
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if not piece or any(ord(c) < 32 for c in piece):
            continue
        if len(tokenizer.encode(piece, add_special_tokens=False)) != 1:
            continue
        if len(tokenizer.encode(piece * 8, add_special_tokens=False)) == 8:
            return piece

    raise RuntimeError("could not find a repeatable single-token prompt chunk")


def build_prompt_cache(tokenizer: Any, lengths: set[int]) -> dict[int, str]:
    chunk = find_repeatable_chunk(tokenizer)
    prompts: dict[int, str] = {}
    for length in sorted(lengths):
        prompt = chunk * length
        actual = len(tokenizer.encode(prompt, add_special_tokens=False))
        if actual != length:
            raise RuntimeError(
                f"failed to construct exact prompt for input_len={length}: got {actual}"
            )
        prompts[length] = prompt
    return prompts


async def consume_sse_completion(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    send_unix_s = time.time()
    send_mono = time.perf_counter()
    first_token_mono: float | None = None
    text_chunks: list[str] = []

    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            detail = await resp.text()
            return {
                "ok": False,
                "send_unix_s": send_unix_s,
                "complete_unix_s": time.time(),
                "error": f"http_{resp.status}: {detail[:300]}",
            }

        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if first_token_mono is None:
                first_token_mono = time.perf_counter()
            choices = obj.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text:
                text_chunks.append(text)

    complete_unix_s = time.time()
    complete_mono = time.perf_counter()
    full_text = "".join(text_chunks)
    completion_tokens = len(tokenizer.encode(full_text, add_special_tokens=False)) if full_text else 0

    ttft_ms = None
    tpot_ms = None
    if first_token_mono is not None:
        ttft_ms = (first_token_mono - send_mono) * 1000.0
        if completion_tokens > 1:
            tpot_ms = ((complete_mono - first_token_mono) / (completion_tokens - 1)) * 1000.0

    return {
        "ok": True,
        "send_unix_s": send_unix_s,
        "complete_unix_s": complete_unix_s,
        "e2e_latency_ms": (complete_mono - send_mono) * 1000.0,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "completion_tokens": completion_tokens,
    }


async def run_warmups(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    output_len: int,
    warmups: int,
) -> None:
    if warmups <= 0:
        return
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": True,
    }
    for _ in range(warmups):
        async with session.post(url, json=payload) as resp:
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line == "data: [DONE]":
                    break


async def replay_trace(
    requests: list[TraceRequest],
    base_url: str,
    model: str,
    prompt_cache: dict[int, str],
    tokenizer: Any,
    max_concurrency: int,
    request_timeout_s: float,
    num_warmups: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/completions"
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[dict[str, Any]] = []
    pending: set[asyncio.Task[Any]] = set()
    all_tasks: list[asyncio.Task[Any]] = []

    timeout = aiohttp.ClientTimeout(total=request_timeout_s, connect=10, sock_connect=10)
    connector = aiohttp.TCPConnector(limit=max_concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        first_req = requests[0]
        await run_warmups(
            session,
            url,
            model,
            prompt_cache[first_req.input_len],
            first_req.output_len,
            num_warmups,
        )

        replay_start_mono = time.perf_counter()

        async def one_request(req: TraceRequest) -> dict[str, Any]:
            async with semaphore:
                payload = {
                    "model": model,
                    "prompt": prompt_cache[req.input_len],
                    "max_tokens": req.output_len,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "ignore_eos": True,
                    "stream": True,
                }
                result = await consume_sse_completion(session, url, payload, tokenizer)
                result["trace_request_id"] = req.request_id
                result["arrival_time_s"] = req.arrival_time_s
                result["input_len"] = req.input_len
                result["output_len"] = req.output_len
                return result

        for req in requests:
            target = replay_start_mono + req.arrival_time_s
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            task = asyncio.create_task(one_request(req))
            pending.add(task)
            all_tasks.append(task)
            task.add_done_callback(pending.discard)

        if all_tasks:
            results = list(await asyncio.gather(*all_tasks))

    send_times = [r["send_unix_s"] for r in results if "send_unix_s" in r]
    complete_times = [r["complete_unix_s"] for r in results if "complete_unix_s" in r]
    ttfts = [float(r["ttft_ms"]) for r in results if r.get("ok") and r.get("ttft_ms") is not None]
    tpots = [float(r["tpot_ms"]) for r in results if r.get("ok") and r.get("tpot_ms") is not None]
    e2e_latencies = [
        float(r["e2e_latency_ms"])
        for r in results
        if r.get("ok") and r.get("e2e_latency_ms") is not None
    ]
    successes = sum(1 for r in results if r.get("ok"))
    failures = len(results) - successes
    input_tokens_total = sum(
        int(r["input_len"]) for r in results if r.get("ok")
    )
    requested_output_tokens_total = sum(
        int(r["output_len"]) for r in results if r.get("ok")
    )
    output_tokens_total = sum(
        int(r.get("completion_tokens") or 0) for r in results if r.get("ok")
    )

    return {
        "requests_total": len(results),
        "logical_request_count": len(results),
        "successful_requests": successes,
        "failed_requests": failures,
        "timing_start_unix_s": min(send_times) if send_times else None,
        "timing_end_unix_s": max(complete_times) if complete_times else None,
        "mean_ttft_ms": mean(ttfts),
        "p99_ttft_ms": percentile(ttfts, 99.0),
        "mean_tpot_ms": mean(tpots),
        "p99_tpot_ms": percentile(tpots, 99.0),
        "mean_e2e_latency_ms": mean(e2e_latencies),
        "p99_e2e_latency_ms": percentile(e2e_latencies, 99.0),
        "trace_duration_s": requests[-1].arrival_time_s if requests else 0.0,
        "max_concurrency": max_concurrency,
        "num_warmups": num_warmups,
        "input_tokens_total": input_tokens_total,
        "requested_output_tokens_total": requested_output_tokens_total,
        "output_tokens_total": output_tokens_total,
        "request_results": results,
    }


def format_metric(value: float) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def write_text_summary(path: Path, trace_csv: Path, summary: dict[str, Any]) -> None:
    lines = [
        "Synthetic Trace Replay Summary",
        f"Trace CSV:                               {trace_csv}",
        f"Requests total:                          {summary['requests_total']}",
        f"Successful requests:                     {summary['successful_requests']}",
        f"Failed requests:                         {summary['failed_requests']}",
        f"Mean TTFT (ms):                          {format_metric(summary['mean_ttft_ms'])}",
        f"P99 TTFT (ms):                           {format_metric(summary['p99_ttft_ms'])}",
        f"Mean TPOT (ms):                          {format_metric(summary['mean_tpot_ms'])}",
        f"P99 TPOT (ms):                           {format_metric(summary['p99_tpot_ms'])}",
        f"Mean E2E latency (ms):                   {format_metric(summary['mean_e2e_latency_ms'])}",
        f"P99 E2E latency (ms):                    {format_metric(summary['p99_e2e_latency_ms'])}",
        f"Replay trace duration (s):               {summary['trace_duration_s']:.3f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    trace_csv = Path(args.trace_csv)
    output_file = Path(args.output_file)
    summary_json = Path(args.summary_json)
    requests_jsonl = Path(args.requests_jsonl) if args.requests_jsonl else None

    requests = load_trace(trace_csv)
    tokenizer_name = args.tokenizer_model or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    prompt_cache = build_prompt_cache(tokenizer, {r.input_len for r in requests})

    try:
        summary = asyncio.run(
            replay_trace(
                requests=requests,
                base_url=args.base_url,
                model=args.model,
                prompt_cache=prompt_cache,
                tokenizer=tokenizer,
                max_concurrency=args.max_concurrency,
                request_timeout_s=args.request_timeout_s,
                num_warmups=args.num_warmups,
            )
        )
    except KeyboardInterrupt:
        return 130

    request_results = summary.pop("request_results")
    summary["trace_csv"] = trace_csv.as_posix()
    summary["model"] = args.model
    summary["tokenizer_model"] = tokenizer_name
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if requests_jsonl is not None:
        with requests_jsonl.open("w", encoding="utf-8") as stream:
            for result in request_results:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
    write_text_summary(output_file, trace_csv, summary)

    print(json.dumps({
        "timing_start_unix_s": summary["timing_start_unix_s"],
        "timing_end_unix_s": summary["timing_end_unix_s"],
        "successful_requests": summary["successful_requests"],
        "failed_requests": summary["failed_requests"],
    }))
    if summary["successful_requests"] <= 0:
        return 1
    if args.fail_on_request_error and summary["failed_requests"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
