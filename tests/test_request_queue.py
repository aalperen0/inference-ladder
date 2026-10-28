"""
Each policy's waiting queue must admit in exactly the order the scheduler used to get by
re-sorting `waiting` before every schedule(). Those old sort keys are the reference here.
"""

import random

import pytest
import torch

from kv_cache import SlotKVCache
from scheduler import ContinuousBatchScheduler, Policies, Request, create_request_queue

MAX_WAIT = 5.0


def old_sort_key(policy, now):
    """The key the old ContinuousBatchScheduler._sort_waiting(now) sorted by."""
    if policy == Policies.FCFS:
        return lambda r: r.arrival
    if policy == Policies.PRIORITY:
        return lambda r: (r.priority, r.arrival)
    return lambda r: (
        (0, 0, r.arrival) if now - r.arrival > MAX_WAIT else (1, r.priority, r.arrival)
    )


def request(request_id, arrival, priority=0):
    """8 KV tokens: a 4-token prompt and 4 new tokens."""
    return Request(request_id, [0] * 4, 4, arrival=arrival, priority=priority)


def tiny_cache(n_slots):
    return SlotKVCache(
        n_layers=1, n_slots=n_slots, n_heads=1, max_T=64, head_dim=1, device="cpu", dtype=torch.float32
    )


def finish(sched, r, now):
    for _ in range(r.max_new_tokens):
        sched.update(r, 0, now)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("policy", list(Policies))
def test_admits_in_the_old_sort_order(policy, seed):
    """Requests arrive, leave while waiting, and get admitted, in random order. Every
    admission must pick the request the old sort put first."""
    rng = random.Random(seed)
    queue = create_request_queue(policy, MAX_WAIT)
    waiting = []  # what the old scheduler kept in `waiting`, in arrival order
    now = 0.0
    for request_id in range(400):
        now += rng.expovariate(1.0)
        op = rng.random()
        if op < 0.5:  # arrivals outpace admissions, so the queue grows and some go overdue
            r = request(request_id, now, priority=rng.choice([0, 1, 2]))
            queue.add_request(r)
            waiting.append(r)
        elif op < 0.6 and waiting:  # a client leaves while waiting, from anywhere in the queue
            r = rng.choice(waiting)
            queue.remove_request(r)
            waiting.remove(r)
        elif waiting:
            expected = min(waiting, key=old_sort_key(policy, now))
            assert queue.peek_request(now) is expected
            assert queue.pop_request(now) is expected
            waiting.remove(expected)
        assert len(queue) == len(waiting)
        assert sorted(r.request_id for r in queue) == sorted(r.request_id for r in waiting)


@pytest.mark.parametrize("policy", list(Policies))
def test_empty_queue_raises(policy):
    queue = create_request_queue(policy, MAX_WAIT)
    assert not queue
    with pytest.raises(IndexError):
        queue.peek_request(0.0)
    with pytest.raises(IndexError):
        queue.pop_request(0.0)


def test_max_wait_lets_an_overdue_low_priority_request_go_first():
    queue = create_request_queue(Policies.MAX_WAIT, MAX_WAIT)
    old_low = request(0, arrival=0.0, priority=2)
    new_high = request(1, arrival=1.0, priority=0)
    queue.add_request(old_low)
    queue.add_request(new_high)
    assert queue.peek_request(now=MAX_WAIT) is new_high  # waited exactly max_wait: not over
    assert queue.pop_request(now=MAX_WAIT + 0.1) is old_low
    assert queue.pop_request(now=MAX_WAIT + 0.1) is new_high


def test_priority_scheduler_admits_the_lower_priority_value_first():
    """One slot: a busy request holds it while a priority-2 request and then a priority-0
    request queue up. The priority-0 one gets the slot next, though it arrived later."""
    sched = ContinuousBatchScheduler(tiny_cache(1), kv_budget_tokens=1_000, policy="priority")
    busy, low, high = request(0, 0.0, 1), request(1, 0.5, 2), request(2, 0.6, 0)

    sched.add_request(busy)
    assert sched.schedule(0.0) == [busy]
    sched.add_request(low)
    sched.add_request(high)
    assert sched.schedule(1.0) == [busy]  # no free slot, so both wait
    finish(sched, busy, 1.0)
    assert sched.schedule(2.0) == [high]
    finish(sched, high, 2.0)
    assert sched.schedule(3.0) == [low]


def test_request_over_the_budget_keeps_its_place():
    """Two free slots but a 12-token budget: the second request waits for the first to
    finish, and the small one behind it waits too instead of jumping the queue."""
    sched = ContinuousBatchScheduler(tiny_cache(2), kv_budget_tokens=12)
    first, second = request(0, 0.0), request(1, 0.1)
    small = Request(2, [0] * 2, 2, arrival=0.2)  # 4 KV tokens: it would fit the budget

    for r in (first, second, small):
        sched.add_request(r)
    assert sched.schedule(1.0) == [first]
    assert list(sched.waiting) == [second, small]
    finish(sched, first, 1.0)
    assert sched.schedule(2.0) == [second, small]
    assert sched.reserved == 12
