"""The serving engine: one continuous-batching step on GPT-2, and its command-line options.

Its modules come from gpt/, so the entry point (api.py) puts gpt/ on sys.path.
"""

import argparse
from collections.abc import Callable
import asyncio
import torch
from scheduler import Request
import time
from config import GPT2_PRESETS, gpt2_config
from kv_cache import SlotKVCache
from model import GPTModel
from scheduler import ContinuousBatchScheduler, Policies, RequestStatus


def make_slot_cache(model: GPTModel, n_slots: int, max_T: int) -> SlotKVCache:
    cfg = model.cfg
    return SlotKVCache(
        n_layers=cfg.n_layers,
        n_slots=n_slots,
        n_heads=cfg.n_heads,
        max_T=max_T,
        head_dim=cfg.embed_dim // cfg.n_heads,
        device=model.tok_embd.weight.device,
        dtype=model.tok_embd.weight.dtype,
    )


class Rejected(Exception):
    """The request can never fit (a slot's max_T, or the whole KV budget). The message says why."""


class AsyncEngine:
    def __init__(
        self,
        model,
        cache,
        kv_budget_tokens,
        policy=Policies.FCFS,
        max_wait=None,
        eos_id=None,
    ):
        self.model = model
        self.sched = ContinuousBatchScheduler(
            cache=cache,
            kv_budget_tokens=kv_budget_tokens,
            policy=policy,
            max_wait=max_wait,
            eos_id=eos_id,
        )
        self._streams: dict[int, asyncio.Queue] = {}
        self.error = None
        self._inbox = []
        self._aborts = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.step_log = []

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def submit(self, request: Request):

        if self.error is not None:
            raise RuntimeError("the engine stopped after a failed step") from self.error
        if request.request_id in self._streams:
            raise ValueError(f"request_id {request.request_id} is already running")

        reason = self.sched.rejection_reason(request)  # reads only max_T and the budget
        if reason is not None:
            request.status = RequestStatus.REJECTED
            request.reject_reason = reason
            raise Rejected(reason)

        request.arrival = time.perf_counter()
        q = self._streams[request.request_id] = asyncio.Queue()
        self._inbox.append(request)
        self._wake.set()

        try:
            while (tok := await q.get()) is not None:
                if isinstance(tok, Exception):  # a step failed
                    raise tok
                yield tok
        finally:
            del self._streams[request.request_id]
            if request.status in (RequestStatus.WAITING, RequestStatus.ACTIVE):
                self.abort(request)

    def abort(self, request: Request):
        self._aborts.append(request)
        self._send(request, None)
        self._wake.set()

    async def _loop(self):
        """The only code that touches the scheduler: everything else hands it requests and
        aborts through _inbox and _aborts. Runs as a task from start() until stop().
        A stream gets token ids, then None at the end, or the exception if a step failed."""
        clock = time.perf_counter
        while True:
            inbox, self._inbox = self._inbox, []

            for r in inbox:
                if not self.sched.add_request(r):
                    self._send(r, None)

            aborts, self._aborts = self._aborts, []

            for r in aborts:
                self.sched.abort(r)
            self.sched.schedule(clock())

            if not self.sched.active:
                self._wake.clear()
                await self._wake.wait()
                continue

            batch = list(self.sched.active)
            n_prefill = sum(r.tokens_generated == 0 for r in batch)
            t_start = clock()
            try:
                await asyncio.to_thread(self.step, self.model, self.sched, clock)
            except Exception as exc:
                self.error = exc
                for q in self._streams.values():
                    q.put_nowait(exc)
                raise

            self.step_log.append(
                {
                    "t_start": t_start,
                    "t_end": clock(),
                    "slots": [r.slot for r in batch],
                    "request_ids": [r.request_id for r in batch],
                    "n_prefill": n_prefill,
                    "n_decode": len(batch) - n_prefill,
                }
            )

            # Every request in the batch got exactly one token. Chunked prefill will change
            # that: a prefill chunk gives no token.
            for r in batch:
                self._send(r, r.output_tokens[-1])
                if r.status == RequestStatus.FINISHED:
                    self._send(r, None)

    def _send(self, r: Request, item):
        """Puts a token or None on r's stream, unless its client has already left."""
        q = self._streams.get(r.request_id)
        if q is not None:
            q.put_nowait(item)

    @staticmethod
    @torch.inference_mode
    def step(
        model: GPTModel, sched: ContinuousBatchScheduler, clock: Callable[[], float]
    ):
        """Prefill the requests admitted since the last step, then decode the others together."""
        new = [r for r in sched.active if r.tokens_generated == 0]
        running = [r for r in sched.active if r.tokens_generated > 0]
        device = model.tok_embd.weight.device

        for r in new:
            sched.cache.select([r.slot])
            r.prefill_start = clock()
            logits = model(torch.tensor([r.prompt_tokens], device=device), sched.cache)[
                :, -1
            ]
            sched.update(r, logits.argmax(-1).item(), clock())

        if running:
            sched.cache.select([r.slot for r in running])
            last = torch.tensor([[r.output_tokens[-1]] for r in running], device=device)
            tokens = model(last, sched.cache)[:, -1].argmax(-1).tolist()
            now = clock()
            for r, token in zip(running, tokens):
                sched.update(r, token, now)


def add_engine_args(p: argparse.ArgumentParser):
    """The model, scheduler and cache options, shared with the HTTP server (api.py)."""
    g = p.add_argument_group("engine")
    g.add_argument(
        "--model", default="gpt2", choices=list(GPT2_PRESETS), help="GPT-2 size"
    )
    g.add_argument(
        "--policy",
        default=Policies.FCFS,
        choices=list(Policies),
        help="admission order",
    )
    g.add_argument(
        "--max-wait",
        type=float,
        metavar="SEC",
        help="for --policy max_wait: past this wait, the longest-waiting goes first",
    )
    g.add_argument("--slots", type=int, default=8, help="requests running at once")
    g.add_argument("--max-T", type=int, default=1024, help="KV tokens per slot")
    g.add_argument("--kv-budget", type=int, default=4096, help="KV tokens, all slots")


def check_engine_args(p: argparse.ArgumentParser, args: argparse.Namespace):
    """Runs before the model loads, so a bad flag fails right away."""
    if args.policy == Policies.MAX_WAIT and args.max_wait is None:
        p.error("--policy max_wait needs --max-wait")
    context_length = gpt2_config(args.model).context_length
    if args.max_T > context_length:
        p.error(f"--max-T can be at most context_length ({context_length})")
