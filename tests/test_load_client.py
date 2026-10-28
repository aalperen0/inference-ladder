"""
The load client must send open-loop (on schedule, never waiting for earlier requests),
record every token it was streamed, sort out how each request ended, and join its records
with the server's log.

The server is the real app under uvicorn in a background thread, with conftest.py's tiny
model, because httpx's in-process ASGI transport buffers whole responses: token times would
mean nothing.
"""

import asyncio
import socket
import threading
import time

import httpx
import pytest
import tiktoken
import uvicorn

from api import create_app
from conftest import MAX_T, V, greedy_alone
from engine import AsyncEngine, make_slot_cache
from load_client import merge_server_log, run_load
from metrics import summarize
from scheduler import Request
from workload import RequestSpec, make_workload


@pytest.fixture(scope="module")
def url(tiny):
    app = create_app(
        lambda: AsyncEngine(tiny, make_slot_cache(tiny, 2, MAX_T), 10_000),
        tiktoken.get_encoding("gpt2"),
        "tiny",
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))  # any free port
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    yield url
    server.should_exit = True
    thread.join(timeout=10)


def load(url, specs):
    httpx.get(f"{url}/internal/log")  # forget the earlier tests' requests
    return asyncio.run(asyncio.wait_for(run_load(url, specs), timeout=60))


def spec(request_id, arrival, prompt, max_new_tokens):
    return RequestSpec(request_id, prompt, max_new_tokens, arrival)


def test_every_streamed_token_is_recorded(url, tiny):
    specs = make_workload(
        8, rate=50, seed=0, vocab_size=V, short=(3, 20), long_frac=0, new_tokens=(2, 8)
    )
    _, records = load(url, specs)
    for s, rec in zip(specs, records):
        assert rec.status == "finished" and rec.http_status == 200
        expected = greedy_alone(tiny, Request(0, s.prompt_tokens, s.max_new_tokens, 0.0))
        assert rec.token_ids == expected
        assert rec.finish_reason == "length"
        assert rec.token_times == sorted(rec.token_times)
        assert rec.sent_at <= rec.token_times[0] and rec.token_times[-1] <= rec.finished_at
    assert len({rec.server_id for rec in records}) == len(records)


def test_requests_go_out_on_schedule(url):
    arrivals = [0.0, 0.05, 0.1, 0.15, 0.2]
    t0, records = load(url, [spec(i, a, [1, 2, 3], 2) for i, a in enumerate(arrivals)])
    for a, rec in zip(arrivals, records):
        assert rec.scheduled == t0 + a
        assert -0.001 < rec.send_lag < 0.02


def test_open_loop_never_waits_for_earlier_requests(url):
    """Six requests at t=0 on two slots: a closed-loop client, or one with a small
    connection pool, would get the third to the server only after one of the first two
    finished. Checked on the server's arrival stamp, since sent_at can't see a request
    waiting inside httpx."""
    _, records = load(url, [spec(i, 0.0, [i + 1] * 5, 20) for i in range(6)])
    merge_server_log(records, httpx.get(f"{url}/internal/log").json()["requests"])
    assert all(rec.status == "finished" for rec in records)
    first_done = min(rec.finished_at for rec in records)
    assert max(rec.server_arrival for rec in records) < first_done


def test_rejected_request_is_recorded_and_the_rest_still_run(url):
    too_long = spec(0, 0.0, [1] * 60, 10)  # 70 tokens > MAX_T
    _, records = load(url, [too_long, spec(1, 0.0, [2, 3], 4)])
    assert (records[0].status, records[0].http_status) == ("rejected", 400)
    assert "max_T" in records[0].error and not records[0].token_times
    statuses = {r["request_id"]: r["status"] for r in httpx.get(f"{url}/internal/log").json()["requests"]}
    assert statuses[records[0].server_id] == "rejected"
    assert records[1].status == "finished" and len(records[1].token_ids) == 4


def test_server_log_joins_and_metrics_run_on_the_records(url):
    specs = make_workload(
        6, rate=100, seed=1, vocab_size=V, short=(3, 20), long_frac=0, new_tokens=(2, 8)
    )
    t0, records = load(url, specs)
    merge_server_log(records, httpx.get(f"{url}/internal/log").json()["requests"])
    for rec in records:
        assert rec.slot in (0, 1)
        # The server stamps each step before the client reads it, on the same clock.
        assert rec.sent_at <= rec.server_arrival <= rec.prefill_start <= rec.server_token_times[0]
        assert len(rec.server_token_times) == len(rec.token_times)
        assert all(s <= c for s, c in zip(rec.server_token_times, rec.token_times))
    elapsed = max(rec.finished_at for rec in records) - t0
    m = summarize(records, elapsed)
    assert m["requests"] == 6 and 0 < m["queue"]["p50"] <= m["ttft"]["p50"]
