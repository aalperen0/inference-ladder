"""The load client: sends a workload to the server open-loop and records what it saw.

Open loop: each request is its own task, sent at its scheduled time whatever happened to the
ones before it. When the server falls behind, requests queue on the server, where TTFT
counts the wait, and not in the client, where it would be hidden. Two numbers check that the
client is not the bottleneck:
- send lag = sent_at - scheduled: the event loop was too busy to send on time.
- server_arrival - sent_at: the request waited in httpx (for a connection) or on the wire.

Times are time.perf_counter(), the server's clock too (system-wide monotonic on Linux), so
the server's prefill_start lines up with the client's sent_at.

    python3 bench/load_client.py --n 50 --rate 2      # against a running server/api.py
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt"))

import httpx
import numpy as np

from metrics import print_summary, summarize
from workload import RequestSpec, make_workload

START_DELAY = 0.1  # s: time to create every task before the first one is due


@dataclass
class ClientRecord:
    request_id: int  # the workload's id
    prompt_len: int
    max_new_tokens: int
    priority: int
    scheduled: float  # when it was due: t0 + spec.arrival
    sent_at: float | None = None
    send_lag: float | None = None  # sent_at - scheduled
    status: str = "pending"  # then finished, rejected (HTTP 400 rejected) or error
    http_status: int | None = None
    error: str | None = None
    server_id: int | None = None  # the server's request_id, from "cmpl-<id>"
    token_ids: list[int] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)  # when each token's event arrived
    finished_at: float | None = None  # when [DONE] arrived
    finish_reason: str | None = None
    # From the server's log, filled in by merge_server_log():
    server_arrival: float | None = None  # when submit() took it
    prefill_start: float | None = None
    slot: int | None = None
    server_token_times: list[float] | None = None

    @property
    def arrival(self) -> float:
        """metrics.py counts TTFT and latency from here: when the client sent the request."""
        return self.sent_at

    @property
    def tokens_generated(self) -> int:
        return len(self.token_times)


async def run_load(
    base_url: str, specs: list[RequestSpec], ignore_eos: bool = True
) -> tuple[float, list[ClientRecord]]:
    """Sends every spec at t0 + spec.arrival and waits for them all.
    Returns t0 and one record per spec, in spec order."""
    # Unlimited connections: httpx's default pool of 100 would queue a burst in the client.
    # No read timeout: under overload, waiting long for the first token is the measurement.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeout = httpx.Timeout(None, connect=10.0)
    async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=timeout) as client:
        t0 = time.perf_counter() + START_DELAY
        records = await asyncio.gather(
            *(send(client, spec, t0 + spec.arrival, ignore_eos) for spec in specs)
        )
    return t0, records


async def send(
    client: httpx.AsyncClient, spec: RequestSpec, scheduled: float, ignore_eos: bool
) -> ClientRecord:
    rec = ClientRecord(
        request_id=spec.request_id,
        prompt_len=len(spec.prompt_tokens),
        max_new_tokens=spec.max_new_tokens,
        priority=spec.priority,
        scheduled=scheduled,
    )
    body = {
        "prompt": spec.prompt_tokens,  # token ids: no tokenizer in the timing
        "max_tokens": spec.max_new_tokens,
        "stream": True,
        "ignore_eos": ignore_eos,
        "priority": spec.priority,
    }
    await asyncio.sleep(max(0.0, scheduled - time.perf_counter()))
    rec.sent_at = time.perf_counter()
    rec.send_lag = rec.sent_at - scheduled
    try:
        async with client.stream("POST", "/v1/completions", json=body) as r:
            rec.http_status = r.status_code
            if r.status_code != 200:
                await r.aread()
                body = r.json()
                kind = body.get("error", {}).get("type")
                rec.status = "rejected" if kind == "rejected" else "error"
                rec.error = r.text
                if "id" in body:  # a rejected request has one, and a server record
                    rec.server_id = int(body["id"].removeprefix("cmpl-"))
                return rec
            async for line in r.aiter_lines():
                now = time.perf_counter()  # first, before any parsing
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    rec.finished_at = now
                    rec.status = "finished"
                    break
                event = json.loads(data)
                if "error" in event:  # a step failed on the server
                    rec.status = "error"
                    rec.error = json.dumps(event["error"])
                    break
                rec.server_id = int(event["id"].removeprefix("cmpl-"))
                choice = event["choices"][0]
                if "token_id" in choice:
                    rec.token_ids.append(choice["token_id"])
                    rec.token_times.append(now)
                else:  # the last event before [DONE]
                    rec.finish_reason = choice["finish_reason"]
    except httpx.HTTPError as exc:
        rec.status = "error"
        rec.error = repr(exc)
    if rec.status == "pending":  # the stream ended without [DONE]
        rec.status = "error"
    return rec


def merge_server_log(records: list[ClientRecord], server_requests: list[dict]) -> None:
    """Adds the server's side (from /internal/log's "requests") to each client record,
    joined on the id the server assigned: its arrival, prefill_start for the queue and
    prefill times, the slot, and the server's token times for the server-side ITL."""
    by_id = {r["request_id"]: r for r in server_requests}
    for rec in records:
        server = by_id.get(rec.server_id)
        if server is not None:
            rec.server_arrival = server["arrival"]
            rec.prefill_start = server["prefill_start"]
            rec.slot = server["slot"]
            rec.server_token_times = server["token_times"]


def main():
    p = argparse.ArgumentParser(
        description="Send a Poisson workload to a running server/api.py and print what the "
        "client saw, joined with the server's log."
    )
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=50, help="requests")
    p.add_argument("--rate", type=float, default=2.0, help="λ in requests/s; inf = all at t=0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--long-frac", type=float, default=0.1, help="share of long prompts")
    args = p.parse_args()

    specs = make_workload(args.n, args.rate, args.seed, long_frac=args.long_frac)
    httpx.get(f"{args.url}/internal/log")  # drop whatever ran before this
    t0, records = asyncio.run(run_load(args.url, specs))
    merge_server_log(records, httpx.get(f"{args.url}/internal/log").json()["requests"])

    finished = [r for r in records if r.status == "finished"]
    counts = {s: sum(r.status == s for r in records) for s in ("finished", "rejected", "error")}
    print(counts)
    if finished:
        elapsed = max(r.finished_at for r in finished) - t0
        m = summarize(finished, elapsed) | {"rejected": counts["rejected"]}
        print_summary(m, elapsed, args.rate)
    lag_ms = np.array([r.send_lag for r in records]) * 1000
    print(f"  send lag (ms)  p50={np.median(lag_ms):.2f}  max={lag_ms.max():.2f}")
    if finished:
        transit_ms = np.array([r.server_arrival - r.sent_at for r in finished]) * 1000
        print(f"  sent → server (ms)  p50={np.median(transit_ms):.2f}  max={transit_ms.max():.2f}")


if __name__ == "__main__":
    main()
