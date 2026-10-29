"""Metrics for one run, from the timestamps the scheduler records on each request.

Every time is in seconds since the run started, like Request.arrival.
- TTFT counts from arrival: TTFT = queue time + prefill time (no network here).
  Queue time is waiting for a slot, KV budget, or other requests' prefills;
  prefill time is the prompt's forward pass, up to its first token.
- ITL is every gap between two consecutive tokens, over all requests. The first token
  is not included (GenAI-Perf's definition; LLMPerf counts it), so say so on the page.
- TPOT is one request's mean gap: ITL's p99 finds the worst stalls, TPOT's p99 the
  slowest requests.
- Latency is end to end: from arrival to the last token.

Each r is anything with a scheduler Request's timestamp fields (arrival, prefill_start,
token_times, finished_at, tokens_generated), so a load-client record works as well.
"""

import numpy as np

PERCENTILES = (50, 90, 99)


def ttft(r) -> float:
    return r.token_times[0] - r.arrival


def queue_time(r) -> float:
    return r.prefill_start - r.arrival


def prefill_time(r) -> float:
    return r.token_times[0] - r.prefill_start


def tpot(r) -> float:
    gaps = np.diff(r.token_times)
    return float(gaps.mean()) if len(gaps) else 0.0


def latency(r) -> float:
    return r.finished_at - r.arrival


def summarize(finished: list, elapsed: float) -> dict:
    """Throughput, and p50/p90/p99 and mean of TTFT (and its queue and prefill parts),
    ITL, TPOT and latency, in seconds.
    elapsed: seconds from the start of the run to its end, as run() returns it.
    For a subset, like one priority class, the throughput numbers mean little."""
    if not finished:
        raise ValueError("no finished requests")
    out = {
        "requests": len(finished),
        "throughput_tok_s": sum(r.tokens_generated for r in finished) / elapsed,
        "requests_per_s": len(finished) / elapsed,
    }
    values = {
        "ttft": [ttft(r) for r in finished],
        "queue": [queue_time(r) for r in finished],
        "prefill": [prefill_time(r) for r in finished],
        "itl": np.concatenate([np.diff(r.token_times) for r in finished]),
        "tpot": [tpot(r) for r in finished],
        "latency": [latency(r) for r in finished],
    }
    for name, v in values.items():
        out[name] = _stats(v)
    return out


def server_itl(records: list) -> dict[str, float]:
    """ITL from the server's token stamps (load-client records after merge_server_log). The
    client's ITL minus this is what localhost HTTP and SSE add."""
    return _stats(np.concatenate([np.diff(r.server_token_times) for r in records]))


def idle_slot_share(steps: list[dict], slots: int) -> float:
    """The share of all slot-steps left empty while requests waited: capacity the batching
    threw away (E6). Counted in steps, not seconds. `waiting` is counted after admission, so
    under continuous batching a slot sits empty while requests wait only when the KV budget
    blocks the head of the queue."""
    idle = sum(slots - len(s["slots"]) for s in steps if s["waiting"])
    return idle / (slots * len(steps))


def goodput(finished: list, slo: dict[str, float]) -> float:
    """The fraction of requests that meet every SLO at once.
    slo = {"ttft": s, "tpot": s, "latency": s}, in seconds."""
    if not finished:
        raise ValueError("no finished requests")
    good = sum(
        ttft(r) <= slo["ttft"]
        and tpot(r) <= slo["tpot"]
        and latency(r) <= slo["latency"]
        for r in finished
    )
    return good / len(finished)


def print_summary(m: dict, elapsed: float, rate: float):
    """m: summarize()'s output, plus m["rejected"]."""
    print(
        f"{m['requests']} requests in {elapsed:.1f} s ({m['rejected']} rejected): "
        f"{m['throughput_tok_s']:.0f} tok/s, {m['requests_per_s']:.2f} req/s at λ = {rate}"
    )
    for name in ("ttft", "queue", "prefill", "itl", "tpot", "latency"):
        stats = "".join(f"  {k}={v * 1000:7.1f}" for k, v in m[name].items())
        print(f"  {name:8} (ms){stats}")


def _stats(values) -> dict[str, float]:
    if len(values) == 0:  # only single-token requests: no gaps between tokens
        return {f"p{p}": float("nan") for p in PERCENTILES} | {"mean": float("nan")}
    pcts = np.percentile(values, PERCENTILES)
    out = {f"p{p}": float(x) for p, x in zip(PERCENTILES, pcts)}
    out["mean"] = float(np.mean(values))
    return out
