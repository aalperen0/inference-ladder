"""The waiting queue, one kind per admission policy, after vLLM's v1 request_queue.py.

The scheduler only uses this interface: add_request(r), peek_request(now), pop_request(now),
remove_request(r), len() and iteration. `now` matters only to max_wait. There is no
prepend, since nothing here preempts. add_request must be called in arrival order, as the
engine does, so "first added" means "first arrived".
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduler import Request


class FCFSRequestQueue(deque):
    """First come, first served: append at the back, pop from the front."""

    def add_request(self, request: Request) -> None:
        self.append(request)

    def peek_request(self, now: float) -> Request:
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def pop_request(self, now: float) -> Request:
        return self.popleft()

    def remove_request(self, request: Request) -> None:
        self.remove(request)


class PriorityRequestQueue:
    """A heap in Request.__lt__ order: the lowest priority value first, then the earliest arrival."""

    def __init__(self):
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, request)

    def peek_request(self, now: float) -> Request:
        if not self._heap:
            raise IndexError("peek from an empty queue")
        return self._heap[0]

    def pop_request(self, now: float) -> Request:
        if not self._heap:
            raise IndexError(self._heap)
        return heapq.heappop(self._heap)

    def remove_request(self, request: Request) -> None:
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self):
        heap = self._heap[:]
        while heap:
            yield heapq.heappop(heap)


class MaxWaitRequestQueue(PriorityRequestQueue):
    """Priority order, except that requests waiting longer than max_wait go first, oldest first,
    so a low priority can't starve.
    """

    def __init__(self, max_wait: float):
        super().__init__()
        self.max_wait = max_wait
        self._by_arrival: deque[Request] = deque()

    def add_request(self, request: Request) -> None:
        super().add_request(request)
        self._by_arrival.append(request)

    def peek_request(self, now: float) -> Request:
        if self._by_arrival and now - self._by_arrival[0].arrival > self.max_wait:
            return self._by_arrival[0]
        return super().peek_request(now)

    def pop_request(self, now: float) -> Request:
        request = self.peek_request(now)
        self.remove_request(request)
        return request

    def remove_request(self, request: Request) -> None:
        super().remove_request(request)
        self._by_arrival.remove(request)
