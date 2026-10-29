"""S5 of docs/SCENARIOS.md: 16 requests at once on 8 slots, alternating 8 and 128 new tokens,
under static and then continuous batching. Prints each run's slot grid and, with --save,
records both runs.

    python3 bench/s5_batching.py --seed 0 --save   # results/scenarios/s5_{static,continuous}_seed-0.json

It runs the engine in-process, not over HTTP. A burst sent over HTTP reaches the engine
across its first steps (S1's FCFS run started with one request queued), so static batching's
first batch would be whichever requests came first. Here all 16 are queued before the first
step, so the step counts can be checked by hand: static 256 steps with 23% of slot-steps
idle while requests wait, continuous 160 steps with none.
"""

import argparse
import asyncio
import dataclasses
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gpt"), str(ROOT / "server")]

import torch

from api import describe
from config import GptConfig
from demo import LABELS, print_grid
from engine import AsyncEngine, make_slot_cache
from load import load_gpt2
from metrics import idle_slot_share, summarize
from results import save_run, step_rows
from scheduler import Batching, Request
from workload import RequestSpec

N, SLOTS, PROMPT_LEN = 16, 8, 16
NEW_TOKENS = (8, 128)  # request i asks for NEW_TOKENS[i % 2]
MAX_T = PROMPT_LEN + max(NEW_TOKENS)  # the smallest slot that holds every request
KV_BUDGET = SLOTS * MAX_T  # never the limit: only the batching differs between the runs


def make_specs(seed: int) -> list[RequestSpec]:
    rng = random.Random(seed)
    return [
        RequestSpec(
            request_id=i,
            prompt_tokens=[rng.randrange(GptConfig.vocab_size) for _ in range(PROMPT_LEN)],
            max_new_tokens=NEW_TOKENS[i % 2],
            arrival=0.0,
        )
        for i in range(N)
    ]


async def serve(engine: AsyncEngine, requests: list[Request]) -> dict:
    """Submits every request before the engine's next step, reads every stream to the end,
    and returns the engine's log of them."""

    async def drain(r: Request):
        async for _ in engine.submit(r):
            pass

    engine.start()
    try:
        # CUDA's one-time setup, kept out of the measured steps.
        await drain(Request(-1, [0] * PROMPT_LEN, 4, arrival=0.0, ignore_eos=True))
        engine.drain_log()
        # gather starts every submit() before the engine's loop wakes, so all are queued.
        await asyncio.gather(*(drain(r) for r in requests))
        return engine.drain_log()
    finally:
        await engine.stop()


def relative(record: dict, t0: float) -> dict:
    """engine.request_record's times as seconds since t0."""
    out = record | {"token_times": [t - t0 for t in record["token_times"]]}
    for k in ("arrival", "prefill_start", "finished_at"):
        out[k] -= t0
    return out


def run(model, batching: Batching, specs: list[RequestSpec], save: bool, seed: int) -> dict:
    engine = AsyncEngine(model, make_slot_cache(model, SLOTS, MAX_T), KV_BUDGET, batching=batching)
    requests = [
        Request(s.request_id, s.prompt_tokens, s.max_new_tokens, arrival=0.0, ignore_eos=True)
        for s in specs
    ]
    log = asyncio.run(serve(engine, requests))
    steps = log["steps"]

    t0 = min(r.arrival for r in requests)
    elapsed = max(r.finished_at for r in requests) - t0
    summary = summarize(requests, elapsed) | {
        "elapsed": elapsed,
        "steps": len(steps),
        "idle_slot_share": idle_slot_share(steps, SLOTS),
    }

    print(f"\n=== {batching} batching ===")
    print_grid(steps, {r.request_id: LABELS[r.request_id] for r in requests}, SLOTS)
    if save:
        path = save_run(
            f"s5_{batching}_seed-{seed}",
            config={
                "server": {"model": "gpt2", **describe(engine)},
                "client": {"in_process": True, "ignore_eos": True, "warmup": 1},
            },
            requests=[relative(r, t0) for r in log["requests"]],
            summary=summary,
            steps=step_rows(steps, t0),
            subdir="scenarios",
            scenario={
                "id": "s5",
                "name": "Short and long outputs",
                "seed": seed,
                "specs": [dataclasses.asdict(s) for s in specs],
            },
        )
        print(f"saved {path}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0, help="the prompts' random tokens")
    p.add_argument("--save", action="store_true", help="write both runs to results/scenarios/")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_gpt2("gpt2", device)
    specs = make_specs(args.seed)
    print(
        f"gpt2 on {device}: {N} requests at once on {SLOTS} slots, prompts of {PROMPT_LEN} "
        f"tokens; even ids ask for {NEW_TOKENS[0]} new tokens, odd ids for {NEW_TOKENS[1]}."
    )
    summaries = {b: run(model, b, specs, args.save, args.seed) for b in Batching}

    print(f"\n{'batching':10} {'steps':>5} {'idle':>6} {'time':>7} {'TTFT p50':>9} {'TTFT p99':>9}")
    for b, m in summaries.items():
        print(
            f"{b:10} {m['steps']:5d} {m['idle_slot_share']:6.1%} {m['elapsed']:6.2f}s "
            f"{m['ttft']['p50'] * 1000:7.0f}ms {m['ttft']['p99'] * 1000:7.0f}ms"
        )
    print("idle: the share of slot-steps left empty while requests waited")


if __name__ == "__main__":
    main()
