"""Measure the HTTP server under an open-loop Poisson load, one JSON file per run.

    python3 bench/bench_serving.py --exp e5                  # λ sweep, continuous, FCFS
    python3 bench/bench_serving.py --exp e6 --rates 1 2 4    # static vs continuous vs 1 slot
    python3 bench/bench_serving.py --rates 2 --seeds 0 --n 20 --batching static --slots 4

→ results/serving/<batching>_<policy>_slots-<n>_chunk-<c>_lambda-<λ>_seed-<s>.json

Each config gets its own server/api.py process on a free port (one process per config, as
in vLLM); every λ and seed then runs against it. Flags this script doesn't know (--slots,
--max-T, --batching, …) go to every server, before the experiment's own flags, so
`--exp e6 --slots 4` runs static and continuous on 4 slots and keeps the third config on 1.
The file name comes from the server's /internal/config, so it says what actually ran.

A run: warm-up (once per server), forget the log, send the workload, join the server's log
into the client's records, summarize, save. Every request sent must show up in the
server's log as finished, rejected or aborted, or the run fails.
"""

import argparse
import asyncio
import contextlib
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

from load_client import merge_server_log, run_load
from metrics import idle_slot_share, server_itl, summarize
from results import request_rows, run_name, save_run, step_rows
from workload import RequestSpec, make_workload

SERVER = Path(__file__).resolve().parents[1] / "server" / "api.py"
INF = float("inf")

# name: (each server's own flags, default λ list). E6 uses E5's rates until E5 finds the knee;
# pass --rates to zoom in around it. E7 is added once --chunk-size exists (Phase 6).
EXPERIMENTS = {
    "e5": ([[]], [0.5, 1, 2, 3, 4, 8, INF]),
    "e6": (
        [["--batching", "static"], ["--batching", "continuous"], ["--slots", "1"]],
        [0.5, 1, 2, 3, 4, 8],
    ),
}


@contextlib.contextmanager
def server(flags: list[str], timeout: float = 180):
    """Starts server/api.py with these flags on a free port, waits for /health, yields its
    URL, and stops it afterwards."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # --port last, so a --port among the flags can't send the bench to the wrong place.
    proc = subprocess.Popen([sys.executable, str(SERVER), *flags, "--port", str(port)])
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"server/api.py {' '.join(flags)} exited with code {proc.returncode}")
            with contextlib.suppress(httpx.TransportError):
                if httpx.get(f"{url}/health").status_code == 200:
                    break
            if time.monotonic() > deadline:
                raise SystemExit(f"the server wasn't up after {timeout} s")
            time.sleep(0.2)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def warm_up(url: str, slots: int, max_T: int):
    """A full batch, one of them with a long prompt, so CUDA's one-time setup and the
    allocator's growth aren't in the first run's numbers. measure() forgets it."""
    long = RequestSpec(-1, [0] * min(512, max_T - 8), 8, arrival=0.0)
    short = [RequestSpec(-2 - i, [0] * 32, 8, arrival=0.0) for i in range(slots - 1)]
    asyncio.run(run_load(url, [long, *short]))


def measure(url: str, specs: list[RequestSpec], slots: int):
    """One run against a server that is up and idle. Returns (t0, records, steps, summary);
    t0 is the client's zero, steps the server's step log of the run."""
    httpx.get(f"{url}/internal/log")  # forget whatever ran before
    t0, records = asyncio.run(run_load(url, specs))
    log = httpx.get(f"{url}/internal/log").json()
    merge_server_log(records, log["requests"])

    client = Counter(r.status for r in records)
    ended = Counter(r["status"] for r in log["requests"])
    if sum(ended.values()) != len(specs) or ended["finished"] != client["finished"]:
        raise RuntimeError(
            f"{len(specs)} sent; the client saw {dict(client)}, the server's log {dict(ended)}"
        )
    finished = [r for r in records if r.status == "finished"]
    if not finished:
        raise RuntimeError(f"no request finished: {dict(client)}")

    elapsed = max(r.finished_at for r in finished) - t0
    summary = summarize(finished, elapsed) | {
        "n": len(specs),
        "rejected": client["rejected"],
        "errors": client["error"],
        "aborted": ended["aborted"],
        "elapsed": elapsed,
        "server_itl": server_itl(finished),
        "idle_slot_share": idle_slot_share(log["steps"], slots),
        # The client isn't the bottleneck if both stay near 0 (load_client.py).
        "send_lag_max": max(r.send_lag for r in records),
        "transit_max": max(r.server_arrival - r.sent_at for r in finished),
    }
    return t0, records, log["steps"], summary


def print_run(rate: float, seed: int, m: dict):
    def ms(stats, p):
        return f"{stats[p] * 1000:.0f}"

    print(
        f"λ={rate:<4g} seed {seed}: {m['requests']}/{m['n']} finished, {m['rejected']} rejected | "
        f"{m['requests_per_s']:.2f} req/s, {m['throughput_tok_s']:.0f} tok/s | "
        f"TTFT p50/p99 {ms(m['ttft'], 'p50')}/{ms(m['ttft'], 'p99')} ms | "
        f"ITL p50/p99 {ms(m['itl'], 'p50')}/{ms(m['itl'], 'p99')} ms "
        f"(server {ms(m['server_itl'], 'p50')}/{ms(m['server_itl'], 'p99')}) | "
        f"idle {m['idle_slot_share']:.1%} | send lag max {m['send_lag_max'] * 1000:.1f} ms",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        allow_abbrev=False,  # an abbreviation would swallow a server flag
        epilog="Any other flag goes to server/api.py; see python3 server/api.py --help.",
    )
    p.add_argument("--exp", choices=list(EXPERIMENTS), help="run an experiment's configs")
    p.add_argument("--rates", type=float, nargs="+", help="λ in requests/s; inf = all at t=0")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n", type=int, default=100, help="requests per run")
    p.add_argument("--long-frac", type=float, default=0.1, help="share of long prompts")
    args, common = p.parse_known_args()

    configs, rates = EXPERIMENTS[args.exp] if args.exp else ([[]], None)
    rates = args.rates or rates
    if rates is None:
        p.error("give --rates, or --exp for an experiment's default rates")

    for flags in configs:
        with server([*common, *flags]) as url:
            config = httpx.get(f"{url}/internal/config").json()
            print(
                f"\n{config['batching']} batching, {config['policy']}, {config['slots']} slots, "
                f"max_T {config['max_T']}, KV budget {config['kv_budget_tokens']}",
                flush=True,
            )
            warm_up(url, config["slots"], config["max_T"])
            for rate in rates:
                for seed in args.seeds:
                    specs = make_workload(args.n, rate, seed, long_frac=args.long_frac)
                    t0, records, steps, summary = measure(url, specs, config["slots"])
                    summary["offered_rate"] = rate
                    summary["offered_tok_s"] = rate * float(np.mean([s.max_new_tokens for s in specs]))
                    save_run(
                        run_name(config, rate, seed),
                        config={
                            "server": config,
                            "client": {
                                "n": args.n,
                                "rate": rate,
                                "seed": seed,
                                "long_frac": args.long_frac,
                                "ignore_eos": True,
                                "warmup": config["slots"],
                            },
                        },
                        requests=request_rows(records, t0),
                        summary=summary,
                        steps=step_rows(steps, t0),
                    )
                    print_run(rate, seed, summary)


if __name__ == "__main__":
    main()
