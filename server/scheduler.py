import time
from collections import deque
from dataclasses import dataclass, field


# MEMORY MATH -> K and V × 12 layers × 768 (heads × head_dim) × 4 B (fp32) = 72 kB per token
# at 4096 tokens: 72 kB * 4096 = 0.29 gB
# at 128K tokens: 72 kB * 131072(32 times of 4096) = 9.5 gB
KV_BYTES_PER_TOKEN = 2 * 12 * 768 * 4

POLICIES = ("fcfs", "priority", "max_wait")


@dataclass
class Request:
    request_id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival: float
    priority: int = 0
    ignore_eos: bool = (
        False  # generate all max_new_tokens even past EOS, as benchmarks do
    )
    status: str = "waiting"  # waiting → active → finished; or rejected, or aborted
    reject_reason: str | None = None
    slot: int | None = None
    tokens_generated: int = 0
    output_tokens: list[int] = field(default_factory=list)
    prefill_start: float | None = None
    token_times: list[float] = field(default_factory=list)
    finished_at: float | None = None

    @property
    def kv_tokens(self) -> int:
        """KV-cache tokens to reserve: the worst case, where it generates all max_new_tokens."""
        return len(self.prompt_tokens) + self.max_new_tokens

    def kv_bytes(self, bytes_per_token: int = KV_BYTES_PER_TOKEN) -> int:
        return self.kv_tokens * bytes_per_token

    def is_done(self) -> bool:
        return self.tokens_generated >= self.max_new_tokens


class ContinuousBatchScheduler:
    """the scheduler needs to decide whether a waiting request can be admitted, reserve its KV-cache capacity, and assign it a free slot.

    Checks a cache slot is available.
    The request fits within the per-slot context length, cache.max_T.
    The remaining KV budget can accommodate the request's reservation
    """

    def __init__(
        self, cache, kv_budget_tokens, policy="fcfs", max_wait=None, eos_id=None
    ):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
        if policy == "max_wait" and max_wait is None:
            raise ValueError("the max_wait policy needs max_wait, in seconds")
        self.cache = cache
        self.policy = policy
        self.max_wait = max_wait
        self.eos_id = eos_id
        self.max_batch_size = cache.key_cache.shape[1]  # slot
        self.max_T = cache.max_T

        self.reserved = 0
        self.kv_budget_tokens = kv_budget_tokens

        self.waiting = deque()
        self.active = []
        self.finished = []
        self.rejected = []
        self.aborted = []

    def rejection_reason(self, request: Request) -> str | None:

        if request.kv_tokens > self.max_T:
            return (
                f"request {request.request_id} needs {request.kv_tokens} tokens "
                f"(prompt + max_new_tokens), but a slot holds max_T={self.max_T}"
            )
        if request.kv_tokens > self.kv_budget_tokens:
            return (
                f"request {request.request_id} needs {request.kv_tokens} tokens, "
                f"more than the whole KV budget of {self.kv_budget_tokens}"
            )
        return None

    def add_request(self, request: Request) -> bool:
        reason = self.rejection_reason(request)
        if reason is not None:
            request.status = "rejected"
            request.reject_reason = reason
            self.rejected.append(request)
            return False

        self.waiting.append(request)
        return True

    def abort(self, request: Request) -> None:
        """Stops a request before it finishes. A waiting one leaves the queue now; an active one
        keeps its slot until the next schedule()"""
        if request.status == "waiting":
            self.waiting.remove(request)
            request.status = "aborted"
            self.aborted.append(request)
        elif request.status == "active":
            request.status = "aborted"

    def _release(self, request: Request) -> None:
        self.reserved -= request.kv_tokens
        self.cache.release(request.slot)
        (self.finished if request.status == "finished" else self.aborted).append(
            request
        )

    def schedule(self, now: float) -> list[Request]:

        remaining = []

        for req in self.active:
            if req.status in ("finished", "aborted"):
                self._release(req)
            else:
                remaining.append(req)

        self.active = remaining
        self._sort_waiting(now)

        while self.waiting and self.cache.free:
            req = self.waiting[0]

            if self.reserved + req.kv_tokens <= self.kv_budget_tokens:
                self.waiting.popleft()
                req.slot = self.cache.allocate()
                self.reserved += req.kv_tokens
                req.status = "active"
                self.active.append(req)
            else:
                break  # the first request doesn't fit yet, so the ones behind it wait

        return list(self.active)

    def _sort_waiting(self, now: float) -> None:
        """Orders `waiting` so the request the policy admits next is at the front."""
        if self.policy == "fcfs":
            key = lambda r: r.arrival
        elif self.policy == "priority":
            key = lambda r: (r.priority, r.arrival)  # ties go to whoever came first
        else:  # max_wait: priority, but past max_wait the longest-waiting goes first

            def key(r):
                if now - r.arrival > self.max_wait:
                    return (0, 0, r.arrival)
                return (1, r.priority, r.arrival)

        self.waiting = deque(sorted(self.waiting, key=key))

    def update(self, request: Request, token: int, now: float) -> None:
        request.output_tokens.append(token)
        request.tokens_generated += 1
        request.token_times.append(now)

        if request.is_done() or (token == self.eos_id and not request.ignore_eos):
            request.status = "finished"
            request.finished_at = now
