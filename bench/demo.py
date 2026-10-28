"""Send a handful of requests at once and show what the engine did with them: the order it
admitted them in, which slot each got and when, when each finished, and which it rejected.

    python3 server/api.py --policy priority --slots 2 --log-events   # terminal 1: live events
    python3 bench/demo.py --n 10 --too-long 2                         # terminal 2: the summary

Every request is due at t=0. With fewer slots than requests, most of them queue, and the
server's --policy picks who goes next. The --too-long ones ask for max_T new tokens, more than
a slot holds with the prompt, so the server rejects them before they queue.

With --save it records the run as S1 of docs/SCENARIOS.md (needs --n 10 --too-long 0):

    python3 server/api.py --slots 2 --policy fcfs
    python3 bench/demo.py --n 10 --too-long 0 --seed 0 --save   # results/scenarios/s1_fcfs_seed-0.json
"""

import argparse
import asyncio
import dataclasses
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt"))

import httpx

from config import GptConfig
from load_client import ClientRecord, merge_server_log, run_load
from metrics import queue_time, summarize
from results import request_rows, save_run, step_rows
from workload import RequestSpec

# Requests are named by the id the server gave them (r7), as in its --log-events lines.
# In the grid each gets one character; a run's ids are consecutive, so up to 36 are distinct.
LABELS = "0123456789abcdefghijklmnopqrstuvwxyz"


def name(r: ClientRecord) -> str:
    return "r?" if r.server_id is None else f"r{r.server_id}"


def cell(r: ClientRecord) -> str:
    return "?" if r.server_id is None else LABELS[r.server_id % len(LABELS)]


def make_requests(n: int, too_long: int, max_T: int, seed: int) -> list[RequestSpec]:
    rng = random.Random(seed)
    rejected = set(rng.sample(range(n), too_long))
    return [
        RequestSpec(
            request_id=i,
            prompt_tokens=[rng.randrange(GptConfig.vocab_size) for _ in range(rng.randint(4, 32))],
            max_new_tokens=max_T if i in rejected else rng.randint(8, 40),
            arrival=0.0,
            priority=rng.choice([0, 1, 2]),
        )
        for i in range(n)
    ]


def print_table(records: list[ClientRecord], t0: float, slots: int):
    def ms(t):
        return f"{(t - t0) * 1000:6.0f} ms" if t is not None else ""

    admitted = sorted(
        (r for r in records if r.prefill_start is not None), key=lambda r: r.prefill_start
    )
    rest = [r for r in records if r.prefill_start is None]
    print(
        f"In the order the engine admitted them. The first {slots} took free slots as they "
        "arrived; the policy chose among the rest.\n"
    )
    print(
        f"{'req':>4} {'cell':>4} {'prio':>4} {'prompt+new':>10}  {'status':9} {'slot':>4} "
        f"{'queued':>9} {'1st token':>9} {'finished':>9}"
    )
    for r in admitted + rest:
        queued = ""
        if r.prefill_start is not None:
            queued = f"{(r.prefill_start - r.server_arrival) * 1000:6.0f} ms"
        first = ms(r.token_times[0]) if r.token_times else ""
        slot = "" if r.slot is None else r.slot
        line = (
            f"{name(r):>4} {cell(r):>4} {r.priority:>4} {r.prompt_len:>4} + {r.max_new_tokens:<4} "
            f"{r.status:9} {slot:>4} {queued:>9} {first:>9} {ms(r.finished_at):>9}"
        )
        if r.status != "finished":
            line += f"  {reason(r)}"
        print(line)


def reason(r: ClientRecord) -> str:
    try:
        return json.loads(r.error)["error"]["message"]
    except (TypeError, ValueError, KeyError):
        return r.error or ""


def print_grid(steps: list[dict], records: list[ClientRecord], slots: int):
    """One column per engine step: which request each slot ran. A request's first cell is its
    prefill (reverse video in a terminal). The last row is how many were still queued."""
    label = {r.server_id: cell(r) for r in records if r.server_id is not None}
    n = len(steps)
    tty = sys.stdout.isatty()
    cells = [["."] * n for _ in range(slots)]
    for i, s in enumerate(steps):
        for slot, rid, phase in zip(s["slots"], s["request_ids"], s["phase"]):
            c = label.get(rid, "?")
            cells[slot][i] = f"\x1b[7m{c}\x1b[0m" if tty and phase == "prefill" else c
    waiting = ["+" if s["waiting"] > 9 else str(s["waiting"]) for s in steps]

    step_ms = sum(s["t_end"] - s["t_start"] for s in steps) / n * 1000
    print(f"\n{n} steps, {step_ms:.1f} ms each on average. '.' is an idle slot.\n")
    width = max(20, shutil.get_terminal_size().columns - 10)
    for start in range(0, n, width):
        end = min(n, start + width)
        print(f"{'step':9}{ruler(start, end)}")
        for slot, row in enumerate(cells):
            print(f"{f'slot {slot}':9}{''.join(row[start:end])}")
        print(f"{'waiting':9}{''.join(waiting[start:end])}\n")


def save_s1(config: dict, args, specs, t0: float, records: list[ClientRecord], steps: list[dict]):
    finished = [r for r in records if r.status == "finished"]
    elapsed = max(r.finished_at for r in finished) - t0
    by_class = {
        p: [queue_time(r) for r in finished if r.priority == p]
        for p in sorted({r.priority for r in finished})
    }
    summary = summarize(finished, elapsed) | {
        "rejected": sum(r.status == "rejected" for r in records),
        "elapsed": elapsed,
        # S1's key measure; a dict key in JSON is a string, so priority 0 is "0".
        "queue_mean_by_priority": {p: sum(q) / len(q) for p, q in by_class.items()},
        # The workload's ids, not the server's, so every config's order reads the same.
        "admission_order": [r.request_id for r in sorted(finished, key=lambda r: r.prefill_start)],
    }
    path = save_run(
        f"s1_{config['policy']}_seed-{args.seed}",
        config={
            "server": config,
            "client": {"n": args.n, "too_long": args.too_long, "ignore_eos": True, "warmup": 1},
        },
        requests=request_rows(records, t0),
        summary=summary,
        steps=step_rows(steps, t0),
        subdir="scenarios",
        scenario={
            "id": "s1",
            "name": "Priority burst",
            "seed": args.seed,
            "specs": [dataclasses.asdict(spec) for spec in specs],
        },
    )
    print(f"saved {path}")


def ruler(start: int, end: int) -> str:
    out = [" "] * (end - start)
    for i in range(start, end):
        if i % 10 == 0:
            for k, ch in enumerate(str(i)):
                if i - start + k < len(out):
                    out[i - start + k] = ch
    return "".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=10, help=f"requests, at most {len(LABELS)}")
    p.add_argument("--too-long", type=int, default=2, help="how many can never fit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--save",
        action="store_true",
        help="write the run to results/scenarios/s1_<policy>_seed-<seed>.json",
    )
    args = p.parse_args()
    if not 0 <= args.too_long <= args.n <= len(LABELS):
        p.error(f"need 0 <= --too-long <= --n <= {len(LABELS)}")
    if args.save and (args.n, args.too_long) != (10, 0):
        p.error("--save records S1 of docs/SCENARIOS.md, which is --n 10 --too-long 0")

    config = httpx.get(f"{args.url}/internal/config").json()
    specs = make_requests(args.n, args.too_long, config["max_T"], args.seed)
    # One request first, so CUDA's one-time setup on a fresh server isn't in the first steps.
    warmup = RequestSpec(request_id=-1, prompt_tokens=[0] * 8, max_new_tokens=4, arrival=0.0)
    asyncio.run(run_load(args.url, [warmup]))
    httpx.get(f"{args.url}/internal/log")  # forget whatever ran before
    t0, records = asyncio.run(asyncio.wait_for(run_load(args.url, specs), timeout=300))
    log = httpx.get(f"{args.url}/internal/log").json()
    merge_server_log(records, log["requests"])

    policy = config["policy"] + (f" ({config['max_wait']} s)" if config["max_wait"] else "")
    print(
        f"{config['model']}, policy {policy}, {config['slots']} slots of {config['max_T']} "
        f"tokens, KV budget {config['kv_budget_tokens']} tokens. Priority 0 is the highest.\n"
    )
    print_table(records, t0, config["slots"])
    if log["steps"]:
        print_grid(log["steps"], records, config["slots"])
    if args.save:
        save_s1(config, args, specs, t0, records, log["steps"])


if __name__ == "__main__":
    main()
