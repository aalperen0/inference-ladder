"""time_ms(): how long one call takes, measured the same way by every microbenchmark in bench/."""

import time
from collections.abc import Callable

import numpy as np
import torch


def time_ms(
    fn: Callable[[], object], warmup: int = 5, reps: int = 20
) -> dict[str, float]:

    sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(reps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000)

    p10, p50, p90 = np.percentile(times, (10, 50, 90))
    return {"median_ms": float(p50), "p10_ms": float(p10), "p90_ms": float(p90)}
