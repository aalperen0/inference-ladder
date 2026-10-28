"""Poisson workloads: what the load client sends, and when. Fixed by the seed, so every
side of a comparison gets the same trace.

The prompts are mostly short with a few long ones (long_frac), so chunked prefill (E7)
has something to fix.
"""

import random
from dataclasses import dataclass

from config import GptConfig


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival: float  # seconds since the run started
    priority: int = 0


def make_workload(
    n,
    rate,
    seed=0,
    vocab_size=GptConfig.vocab_size,
    short=(8, 64),
    long=(256, 512),
    long_frac=0.1,
    new_tokens=(16, 128),
) -> list[RequestSpec]:
    rng = random.Random(seed)
    specs, t = [], 0.0

    for i in range(n):
        t += rng.expovariate(rate)
        prompt_len = rng.randint(*(long if rng.random() < long_frac else short))
        specs.append(
            RequestSpec(
                request_id=i,
                prompt_tokens=[rng.randrange(vocab_size) for _ in range(prompt_len)],
                max_new_tokens=rng.randint(*new_tokens),
                arrival=t,
                priority=rng.choice([0, 1, 2]),
            )
        )

    return specs
