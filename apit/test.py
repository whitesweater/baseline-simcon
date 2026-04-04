#!/usr/bin/env python3
"""Concurrent latency benchmark for OpenAI-compatible models.

- Fetches all models from /v1/models
- Sends concurrent /v1/chat/completions requests
- Stores raw and summary results to CSV/JSON/Markdown
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class AttemptResult:
    model: str
    attempt: int
    ok: bool
    http_code: int
    latency_ms: float
    error: str
    response_excerpt: str


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def http_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, Any, str]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url=url, data=data, method=method)
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        code = e.code
        body = e.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return 0, None, str(e)

    try:
        parsed = json.loads(body)
    except Exception:  # noqa: BLE001
        parsed = None
    return code, parsed, body


def fetch_models(base_url: str, api_key: str, timeout: float) -> list[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    code, payload, body = http_json("GET", f"{base_url}/v1/models", headers, None, timeout)
    if code != 200:
        raise RuntimeError(f"/v1/models failed: http={code}, body={body[:300]}")
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("/v1/models returned invalid JSON payload")

    models = []
    for item in payload.get("data", []):
        model_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(model_id, str) and model_id.strip():
            models.append(model_id)

    unique_models = sorted(set(models))
    if not unique_models:
        raise RuntimeError("No models found from /v1/models")
    return unique_models


def one_attempt(
    base_url: str,
    api_key: str,
    model: str,
    attempt: int,
    timeout: float,
    prompt: str,
    max_tokens: int,
) -> AttemptResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    code, parsed, body = http_json(
        "POST", f"{base_url}/v1/chat/completions", headers, payload, timeout
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    if code == 200 and isinstance(parsed, dict) and isinstance(parsed.get("choices"), list):
        excerpt = ""
        try:
            msg = parsed["choices"][0].get("message", {}).get("content", "")
            if isinstance(msg, str):
                excerpt = msg.strip().replace("\n", " ")[:120]
        except Exception:  # noqa: BLE001
            excerpt = ""
        return AttemptResult(model, attempt, True, 200, latency_ms, "", excerpt)

    err = "request_failed"
    if isinstance(parsed, dict):
        err_obj = parsed.get("error")
        if isinstance(err_obj, dict) and isinstance(err_obj.get("message"), str):
            err = err_obj["message"]
        elif isinstance(parsed.get("message"), str):
            err = parsed["message"]
    elif code == 0 and body:
        err = body

    return AttemptResult(model, attempt, False, code, latency_ms, err[:200], "")


def summarize(results: list[AttemptResult]) -> list[dict[str, Any]]:
    by_model: dict[str, list[AttemptResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    summary: list[dict[str, Any]] = []
    for model, rows in by_model.items():
        lats = sorted(r.latency_ms for r in rows)
        ok_rows = [r for r in rows if r.ok]
        fail_rows = [r for r in rows if not r.ok]

        row = {
            "model": model,
            "attempts": len(rows),
            "success": len(ok_rows),
            "failure": len(fail_rows),
            "success_rate": round((len(ok_rows) / len(rows)) * 100, 2),
            "latency_min_ms": round(min(lats), 2),
            "latency_p50_ms": round(percentile(lats, 0.50), 2),
            "latency_p95_ms": round(percentile(lats, 0.95), 2),
            "latency_max_ms": round(max(lats), 2),
            "latency_avg_ms": round(statistics.mean(lats), 2),
            "sample_error": fail_rows[0].error if fail_rows else "",
        }
        summary.append(row)

    summary.sort(key=lambda x: (x["success_rate"] < 100, x["latency_p50_ms"]))
    return summary


def write_outputs(
    out_dir: Path,
    stamp: str,
    base_url: str,
    workers: int,
    attempts: int,
    results: list[AttemptResult],
    summary: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    raw_csv = out_dir / f"model_latency_raw_{stamp}.csv"
    summary_csv = out_dir / f"model_latency_summary_{stamp}.csv"
    report_md = out_dir / f"model_latency_report_{stamp}.md"

    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "attempt",
                "ok",
                "http_code",
                "latency_ms",
                "error",
                "response_excerpt",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        else:
            f.write("model,attempts,success,failure,success_rate\n")

    total = len(results)
    ok = sum(1 for r in results if r.ok)
    fail = total - ok

    lines = [
        "# Model Latency Benchmark",
        "",
        f"- Generated at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Base URL: {base_url}",
        f"- Concurrency(workers): {workers}",
        f"- Attempts per model: {attempts}",
        f"- Total requests: {total}",
        f"- Success: {ok}",
        f"- Failure: {fail}",
        "",
        "## Summary",
        "",
        "| model | attempts | success | failure | success_rate(%) | p50(ms) | p95(ms) | avg(ms) | min(ms) | max(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for s in summary:
        lines.append(
            "| {model} | {attempts} | {success} | {failure} | {success_rate} | {latency_p50_ms} | {latency_p95_ms} | {latency_avg_ms} | {latency_min_ms} | {latency_max_ms} |".format(
                **s
            )
        )

    fails = [s for s in summary if s["failure"] > 0]
    if fails:
        lines.extend(["", "## Failure Samples", ""])
        for s in fails[:20]:
            lines.append(f"- {s['model']}: {s['sample_error']}")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Raw data CSV: {raw_csv.name}",
            f"- Summary CSV: {summary_csv.name}",
            f"- This report: {report_md.name}",
        ]
    )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw_csv, summary_csv, report_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all models on OpenAI-compatible endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8317", help="API base URL")
    parser.add_argument("--api-key", default=os.getenv("CLIPROXY_API_KEY", ""), help="API key")
    parser.add_argument("--workers", type=int, default=8, help="concurrency workers")
    parser.add_argument("--attempts", type=int, default=1, help="attempts per model")
    parser.add_argument("--timeout", type=float, default=60.0, help="request timeout seconds")
    parser.add_argument("--prompt", default="reply with ok", help="prompt for benchmark requests")
    parser.add_argument("--max-tokens", type=int, default=8, help="max_tokens for benchmark request")
    parser.add_argument(
        "--out-dir",
        default="/home/yhao/cliproxyapi/docs",
        help="output directory for csv/report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("ERROR: api key required. pass --api-key or set CLIPROXY_API_KEY", file=sys.stderr)
        return 2
    if args.workers < 1 or args.attempts < 1:
        print("ERROR: workers and attempts must be >= 1", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        models = fetch_models(args.base_url.rstrip("/"), args.api_key, args.timeout)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: fetch models failed: {e}", file=sys.stderr)
        return 1

    futures = []
    results: list[AttemptResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for model in models:
            for attempt in range(1, args.attempts + 1):
                futures.append(
                    pool.submit(
                        one_attempt,
                        args.base_url.rstrip("/"),
                        args.api_key,
                        model,
                        attempt,
                        args.timeout,
                        args.prompt,
                        args.max_tokens,
                    )
                )

        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: (r.model, r.attempt))
    summary = summarize(results)
    stamp = now_ts()
    raw_csv, summary_csv, report_md = write_outputs(
        out_dir=out_dir,
        stamp=stamp,
        base_url=args.base_url,
        workers=args.workers,
        attempts=args.attempts,
        results=results,
        summary=summary,
    )

    print(f"models={len(models)} total_requests={len(results)}")
    print(f"raw_csv={raw_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"report_md={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
