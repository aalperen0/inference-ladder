from __future__ import annotations

import time
import torch
from collections import deque
from typing import TYPE_CHECKING
from scheduler import ContinousBatchScheduler, Request

if TYPE_CHECKING:
    from gpt.kv_cache import SlotKVCache
    from gpt.model import GPTModel


def run(
    model: GPTModel,
    requests: list[Request],
    cache: SlotKVCache,
    kv_budget_tokens: int,
    policy: str = "fcfs",
) -> tuple[list[Request], float]:
    sched = ContinousBatchScheduler(cache, kv_budget_tokens, policy)
    pending: deque[Request] = deque(sorted(requests, key=lambda r: r.arrival))
    start = time.perf_counter()

    def clock() -> float:
        return time.perf_counter() - start

    while pending and sched.waiting or sched.active:
        now = clock()
        while pending and pending[0].arrival <= now:
            sched.add_request(pending.popleft())

        sched.schedule(now)
        if not sched.active:
            if pending:
                time.sleep(max(0.0, pending[0].arrival - clock()))
            continue

        step(model, sched, clock)

    return sched.finished, clock()
