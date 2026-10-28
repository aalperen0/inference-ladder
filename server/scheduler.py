import time
from dataclasses import dataclass, field
from enum import StrEnum
from request_queue import FCFSRequestQueue, MaxWaitRequestQueue, PriorityRequestQueue


# MEMORY MATH -> K and V × 12 layers × 768 (heads × head_dim) × 4 B (fp32) = 72 kB per token
# at 4096 tokens: 72 kB * 4096 = 0.29 gB
# at 128K tokens: 72 kB * 131072(32 times of 4096) = 9.5 gB
KV_BYTES_PER_TOKEN = 2 * 12 * 768 * 4


class Policies(StrEnum):
    FCFS = "fcfs"
    PRIORITY = "priority"
    MAX_WAIT = "max_wait"


class RequestStatus(StrEnum):
    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"
    REJECTED = "rejected"
    ABORTED = "aborted"


@dataclass
class Request:
    request_id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival: float
    priority: int = 0
    ignore_eos: bool = False
    status: RequestStatus = (
        RequestStatus.WAITING
    )  # WAITING → ACTIVE → FINISHED; or REJECTED, or ABORTED
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

    def __lt__(self, other: "Request") -> bool:
        """The priority heap's order. The request_id only breaks exact ties."""
        return (self.priority, self.arrival, self.request_id) < (
            other.priority,
            other.arrival,
            other.request_id,
        )


def create_request_queue(policy: Policies, max_wait: float | None = None):
    if policy == Policies.FCFS:
        return FCFSRequestQueue()
    if policy == Policies.PRIORITY:
        return PriorityRequestQueue()
    return MaxWaitRequestQueue(max_wait)


class ContinuousBatchScheduler:
    """the scheduler needs to decide whether a waiting request can be admitted, reserve its KV-cache capacity, and assign it a free slot.

    Checks a cache slot is available.
    The request fits within the per-slot context length, cache.max_T.
    The remaining KV budget can accommodate the request's reservation
    """

    def __init__(
        self, cache, kv_budget_tokens, policy=Policies.FCFS, max_wait=None, eos_id=None
    ):
        if policy not in Policies:
            raise ValueError(
                f"unknown policy {policy!r}; choose from {', '.join(Policies)}"
            )
        if policy == Policies.MAX_WAIT and max_wait is None:
            raise ValueError("the max_wait policy needs max_wait, in seconds")
        self.cache = cache
        self.policy = Policies(policy)
        self.max_wait = max_wait
        self.eos_id = eos_id
        self.max_batch_size = cache.key_cache.shape[1]  # slot
        self.max_T = cache.max_T

        self.reserved = 0
        self.kv_budget_tokens = kv_budget_tokens

        self.waiting = create_request_queue(self.policy, max_wait)
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
        """Add request to the queue if it's not rejected"""

        reason = self.rejection_reason(request)
        if reason is not None:
            request.status = RequestStatus.REJECTED
            request.reject_reason = reason
            self.rejected.append(request)
            return False

        self.waiting.add_request(request)
        return True

    def abort(self, request: Request) -> None:
        """Stops a request before it finishes. A waiting one leaves the queue now; an active one
        keeps its slot until the next schedule()"""
        if request.status == RequestStatus.WAITING:
            self.waiting.remove_request(request)
            request.status = RequestStatus.ABORTED
            self.aborted.append(request)
        elif request.status == RequestStatus.ACTIVE:
            request.status = RequestStatus.ABORTED

    def _release(self, request: Request) -> None:
        self.reserved -= request.kv_tokens
        self.cache.release(request.slot)
        finished = request.status == RequestStatus.FINISHED
        (self.finished if finished else self.aborted).append(request)

    def schedule(self, now: float) -> list[Request]:

        remaining = []

        for req in self.active:
            if req.status in (RequestStatus.FINISHED, RequestStatus.ABORTED):
                self._release(req)
            else:
                remaining.append(req)

        self.active = remaining

        while self.waiting and self.cache.free:
            req = self.waiting.peek_request(now)
            if self.reserved + req.kv_tokens > self.kv_budget_tokens:
                break  # the next request doesn't fit yet, so the ones behind it wait

            self.waiting.pop_request(now)
            req.slot = self.cache.allocate()
            self.reserved += req.kv_tokens
            req.status = RequestStatus.ACTIVE
            self.active.append(req)

        return list(self.active)

    def update(self, request: Request, token: int, now: float) -> None:
        request.output_tokens.append(token)
        request.tokens_generated += 1
        request.token_times.append(now)

        if request.is_done() or (token == self.eos_id and not request.ignore_eos):
            request.status = RequestStatus.FINISHED
            request.finished_at = now
