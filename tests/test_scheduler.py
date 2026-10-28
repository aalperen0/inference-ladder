"""
The async engine must serve every request exactly as if it ran alone, keep its slots and
KV budget straight, and turn every way a request can end (finished, rejected, a client
leaving, a failed step) into the right state.

The model (`tiny`) and the reference (`greedy_alone`) are in conftest.py. Each test runs
its async body with asyncio.run under a timeout (no pytest-asyncio), so an engine stuck
waiting fails the test instead of hanging it.
"""

import asyncio
import contextlib
import logging
import random
from collections import Counter

import pytest

from conftest import MAX_T, V, greedy_alone
from engine import AsyncEngine, Rejected, make_slot_cache
from scheduler import Request


def make_requests(n, seed, first_id=0, new_tokens=(2, 12)):
    rng = random.Random(seed)
    return [
        Request(
            request_id=first_id + i,
            prompt_tokens=[rng.randrange(V) for _ in range(rng.randint(3, 20))],
            max_new_tokens=rng.randint(*new_tokens),
            arrival=0.0,  # submit() stamps the real one
        )
        for i in range(n)
    ]


def run(coro, timeout=30):
    async def with_timeout():
        async with asyncio.timeout(timeout):
            return await coro

    return asyncio.run(with_timeout())


@contextlib.asynccontextmanager
async def running(model, n_slots=2, kv_budget=10_000):
    engine = AsyncEngine(model, make_slot_cache(model, n_slots, MAX_T), kv_budget)
    engine.start()
    try:
        yield engine
    finally:
        await engine.stop()


async def collect(engine, request, stop_after=None):
    """Reads the request's stream to the end, or leaves after stop_after tokens like a
    client that disconnects. Returns the tokens it read."""
    tokens = []
    async with contextlib.aclosing(engine.submit(request)) as stream:
        async for tok in stream:
            tokens.append(tok)
            if len(tokens) == stop_after:
                break
    return tokens


async def idle(engine):
    """Waits until the loop has applied every abort and released every slot."""
    while engine._inbox or engine._aborts or engine.sched.waiting or engine.sched.active:
        await asyncio.sleep(0.001)


def assert_all_freed(engine):
    assert len(engine.sched.cache.free) == engine.sched.max_batch_size
    assert engine.sched.reserved == 0
    assert not engine._streams


# --- six requests through two slots, all submitted at once ----------------------------


@pytest.fixture(scope="module")
def served(tiny):
    requests = make_requests(6, seed=0)

    async def main():
        async with running(tiny, n_slots=2) as engine:
            outputs = await asyncio.gather(*(collect(engine, r) for r in requests))
        return engine, outputs

    engine, outputs = run(main())
    return engine, requests, outputs


def test_batched_tokens_match_running_alone(tiny, served):
    """Catches rows going to the wrong slot, a token sent to the wrong stream, and a
    request stopping early or late."""
    _, requests, outputs = served
    # Distinct outputs, or a token sent to the wrong stream could go unnoticed.
    assert len({tuple(out) for out in outputs}) == len(outputs)
    for r, out in zip(requests, outputs):
        assert out == greedy_alone(tiny, r)


def test_slots_never_shared_and_reused(served):
    engine, _, _ = served
    slot_of = {}
    for s in engine.step_log:
        assert len(set(s["slots"])) == len(s["slots"])  # no slot twice in one step
        for rid, slot in zip(s["request_ids"], s["slots"]):
            assert slot_of.setdefault(rid, slot) == slot  # a request keeps its slot
    requests_per_slot = Counter(slot_of.values())
    assert set(requests_per_slot) == {0, 1}
    assert min(requests_per_slot.values()) > 1  # every slot served more than one request


def test_one_token_per_request_per_step(served):
    """_loop sends each request in the batch exactly one token per step."""
    engine, requests, _ = served
    steps_of = Counter(rid for s in engine.step_log for rid in s["request_ids"])
    assert steps_of == {r.request_id: r.max_new_tokens for r in requests}
    assert sum(s["n_prefill"] for s in engine.step_log) == len(requests)
    for s in engine.step_log:
        assert s["n_prefill"] + s["n_decode"] == len(s["slots"])
        assert s["t_start"] <= s["t_end"]


def test_step_log_fields_add_up(served):
    """The page draws its slot grid and KV reserved-vs-used curve from these fields."""
    engine, requests, _ = served
    by_id = {r.request_id: r for r in requests}
    steps_so_far = Counter()
    assert [s["step"] for s in engine.step_log] == list(range(len(engine.step_log)))
    for s in engine.step_log:
        for rid, phase, kv_len in zip(s["request_ids"], s["phase"], s["kv_len"]):
            n = steps_so_far[rid]  # the prefill writes the prompt, each decode one more token
            assert phase == ("prefill" if n == 0 else "decode")
            assert kv_len == len(by_id[rid].prompt_tokens) + n
            steps_so_far[rid] += 1
        assert s["kv_used"] == sum(s["kv_len"])
        assert s["kv_reserved"] == sum(by_id[rid].kv_tokens for rid in s["request_ids"])
        assert s["kv_used"] <= s["kv_reserved"]
    waiting = [s["waiting"] for s in engine.step_log]
    assert waiting[0] == len(requests) - 2 and waiting == sorted(waiting, reverse=True)


def test_fcfs_admits_in_arrival_order(served):
    engine, requests, _ = served
    first_seen = list(dict.fromkeys(rid for s in engine.step_log for rid in s["request_ids"]))
    assert first_seen == sorted(first_seen, key=lambda rid: requests[rid].arrival)


def test_timestamps_in_order(served):
    """metrics.py reads these: queue = prefill_start - arrival, TTFT from token_times[0]."""
    _, requests, _ = served
    for r in requests:
        assert r.status == "finished"
        assert r.arrival <= r.prefill_start <= r.token_times[0]
        assert r.token_times == sorted(r.token_times)
        assert r.finished_at == r.token_times[-1]


def test_everything_freed_after_the_run(served):
    engine, requests, _ = served
    assert_all_freed(engine)
    assert len(engine.sched.finished) == len(requests)
    assert not engine.sched.aborted and not engine.sched.rejected


# --- ways a request can end early -----------------------------------------------------


def test_client_leaving_aborts_its_request(tiny):
    """Leaving after 2 tokens frees the slot and KV budget, and the other request still
    gets every token, the same as alone."""
    leaver, stayer = make_requests(2, seed=1, new_tokens=(10, 10))

    async def main():
        async with running(tiny) as engine:
            got = await asyncio.gather(
                collect(engine, leaver, stop_after=2), collect(engine, stayer)
            )
            await idle(engine)
        return engine, got

    engine, (left, stayed) = run(main())
    assert left == greedy_alone(tiny, leaver)[:2]
    assert stayed == greedy_alone(tiny, stayer)
    assert leaver.status == "aborted" and engine.sched.aborted == [leaver]
    assert_all_freed(engine)


def test_cancelled_stream_aborts_its_request(tiny):
    """Starlette cancels the response task when a client disconnects, without closing the
    generator first: its finally must still abort the request."""
    request = make_requests(1, seed=2, new_tokens=(40, 40))[0]

    async def main():
        async with running(tiny) as engine:
            first = asyncio.Event()

            async def read():
                async for _ in engine.submit(request):
                    first.set()

            reader = asyncio.create_task(read())
            await first.wait()
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
            await idle(engine)
        return engine

    engine = run(main())
    assert request.status == "aborted" and request.tokens_generated < 40
    assert_all_freed(engine)


def test_abort_while_waiting_never_runs(tiny):
    """With the only slot busy, a waiting request whose client leaves must drop out of the
    queue without ever being prefilled."""
    busy, queued = make_requests(2, seed=3, new_tokens=(40, 40))

    async def main():
        async with running(tiny, n_slots=1) as engine:
            busy_reader = asyncio.create_task(collect(engine, busy))
            queued_reader = asyncio.create_task(collect(engine, queued))
            while not any(r is queued for r in engine.sched.waiting):
                await asyncio.sleep(0.001)
            queued_reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await queued_reader
            await busy_reader
            await idle(engine)
        return engine

    engine = run(main())
    assert queued.status == "aborted" and queued.prefill_start is None
    assert all(queued.request_id not in s["request_ids"] for s in engine.step_log)
    assert busy.status == "finished"
    assert_all_freed(engine)


@pytest.mark.parametrize(
    "prompt_len, kv_budget, why",
    [(60, 10_000, "max_T"), (40, 45, "KV budget")],
)
def test_request_that_can_never_fit_is_rejected(tiny, prompt_len, kv_budget, why):
    """Rejected comes before any token, so api.py can still answer 400, and the engine
    keeps serving the requests after it."""
    too_big = Request(
        request_id=0, prompt_tokens=[1] * prompt_len, max_new_tokens=10, arrival=0.0
    )
    fits = make_requests(1, seed=4, first_id=1)[0]

    async def main():
        async with running(tiny, kv_budget=kv_budget) as engine:
            with pytest.raises(Rejected, match=why):
                await collect(engine, too_big)
            out = await collect(engine, fits)
        return engine, out

    engine, out = run(main())
    assert too_big.status == "rejected" and why in too_big.reject_reason
    assert all(0 not in s["request_ids"] for s in engine.step_log)
    assert out == greedy_alone(tiny, fits)


def test_duplicate_request_id_is_refused(tiny):
    first, dup = make_requests(2, seed=5, new_tokens=(40, 40))
    dup.request_id = first.request_id

    async def main():
        async with running(tiny) as engine:
            async with contextlib.aclosing(engine.submit(first)) as stream:
                await anext(stream)  # first is running now
                with pytest.raises(ValueError, match="already running"):
                    await collect(engine, dup)
            await idle(engine)
        return engine

    assert_all_freed(run(main()))


def test_failed_step_fails_open_streams_and_refuses_new_requests(tiny):
    """A crash in step() (a CUDA out-of-memory, say) must reach the clients instead of
    leaving them waiting forever."""

    def crash(*args):
        raise RuntimeError("fake CUDA out of memory")

    first, later = make_requests(2, seed=6)

    async def main():
        engine = AsyncEngine(tiny, make_slot_cache(tiny, 2, MAX_T), 10_000)
        engine.step = crash
        engine.start()
        with pytest.raises(RuntimeError, match="fake CUDA"):
            await collect(engine, first)
        with pytest.raises(RuntimeError, match="engine stopped"):
            await collect(engine, later)
        with contextlib.suppress(RuntimeError):  # stop() re-raises the step's error
            await engine.stop()
        return engine

    assert isinstance(run(main()).error, RuntimeError)


# --- accounting and lifecycle ---------------------------------------------------------


def test_finished_rejected_aborted_add_up(tiny):
    """Every request ends exactly one way, and the clients and the scheduler agree on which."""
    requests = make_requests(8, seed=7, new_tokens=(5, 12))
    for r in requests[:2]:
        r.prompt_tokens = [1] * 60  # can never fit a slot
    leaving = {2, 3}

    async def main():
        async with running(tiny) as engine:

            async def client(r):
                stop_after = 1 if r.request_id in leaving else None
                try:
                    await collect(engine, r, stop_after)
                except Rejected:
                    return "rejected"
                return "aborted" if r.request_id in leaving else "finished"

            outcomes = await asyncio.gather(*(client(r) for r in requests))
            await idle(engine)
        return engine, outcomes

    engine, outcomes = run(main())
    assert Counter(outcomes) == {"rejected": 2, "aborted": 2, "finished": 4}
    assert [r.status for r in requests] == outcomes
    accepted = len(engine.sched.finished) + len(engine.sched.aborted)
    assert accepted + outcomes.count("rejected") == len(requests)
    assert_all_freed(engine)


def test_event_log_tells_each_request_story(tiny, caplog):
    """api.py --log-events prints these lines; each request's events come in order."""
    caplog.set_level(logging.INFO, logger="engine")
    fits = make_requests(1, seed=9)[0]
    too_big = Request(request_id=1, prompt_tokens=[1] * 60, max_new_tokens=10, arrival=0.0)

    async def main():
        async with running(tiny) as engine:
            await collect(engine, fits)
            with pytest.raises(Rejected):
                await collect(engine, too_big)

    run(main())
    events = [tuple(m.split()[1:3]) for m in caplog.messages]
    assert events == [
        ("r0", "queued"),
        ("r0", "admitted"),
        ("r0", "finished"),
        ("r1", "rejected"),
    ]
    assert f"{fits.max_new_tokens} tokens" in caplog.messages[2]
    assert "max_T" in caplog.messages[3]


def test_idle_engine_waits_and_stop_ends_it(tiny):
    """With nothing to do the loop must sleep on its wake event, not keep stepping, and
    stop() must end it."""

    async def main():
        async with running(tiny) as engine:
            await collect(engine, make_requests(1, seed=8)[0])
            steps = len(engine.step_log)
            await asyncio.sleep(0.05)
            assert len(engine.step_log) == steps
            task = engine._task
            assert not task.done()
        return task

    assert run(main()).cancelled()
