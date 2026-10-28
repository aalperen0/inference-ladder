"""The HTTP server: an OpenAI-style POST /v1/completions that streams over SSE, plus what
bench/ records into results/: the engine config and the step and request logs.

    python3 server/api.py --policy fcfs --slots 8 --port 8000 [--log-events]

In a browser, http://127.0.0.1:8000/docs (FastAPI's Swagger UI) sends requests and shows
/internal/config and /internal/log.

"""

import argparse
import codecs
import contextlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt"))

import tiktoken
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from engine import (
    AsyncEngine,
    Rejected,
    add_engine_args,
    check_engine_args,
    make_slot_cache,
)
from load import load_gpt2
from scheduler import Request

EOS_ID = 50256  # <|endoftext|>


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an unknown field is a 422, not ignored

    model: str | None = None
    prompt: (
        str | list[int]
    )  # text goes through the GPT-2 tokenizer; benchmarks send ids
    max_tokens: int = Field(16, ge=1)
    stream: bool = False
    temperature: Literal[0] = 0  # greedy only: step() takes the argmax
    ignore_eos: bool = False
    priority: int = 0  # lower goes first, under --policy priority or max_wait


def error(status: int, kind: str, message: str, **extra) -> JSONResponse:
    body = {**extra, "error": {"type": kind, "message": message}}
    return JSONResponse(body, status_code=status)


def describe(engine: AsyncEngine) -> dict:
    """The engine's side of /internal/config: what its numbers depend on."""
    sched, cache = engine.sched, engine.sched.cache
    weight = engine.model.tok_embd.weight
    n_layers, _, n_heads, _, head_dim = cache.key_cache.shape
    return {
        "n_params": sum(p.numel() for p in engine.model.parameters()),
        "dtype": str(weight.dtype).removeprefix("torch."),
        "device": torch.cuda.get_device_name(weight.device)
        if weight.is_cuda
        else "cpu",
        "slots": sched.max_batch_size,
        "max_T": sched.max_T,
        "kv_budget_tokens": sched.kv_budget_tokens,
        "kv_bytes_per_token": 2
        * n_layers
        * n_heads
        * head_dim
        * cache.key_cache.element_size(),
        "kv_cache_bytes": cache.key_cache.nbytes + cache.value_cache.nbytes,
        "policy": sched.policy,
        "max_wait": sched.max_wait,
        "batching": "continuous",
        "chunk_size": None,
        "eos_id": sched.eos_id,
    }


def create_app(
    make_engine, tokenizer, model_name: str, server_args: dict | None = None
):
    """make_engine() builds the AsyncEngine. It runs in the lifespan, so the engine lives on
    the server's event loop; tests pass one with a tiny model."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = make_engine()
        app.state.engine.start()
        yield
        with contextlib.suppress(Exception):  # stop() re-raises a failed step's error
            await app.state.engine.stop()

    app = FastAPI(lifespan=lifespan)
    request_ids = itertools.count()

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        engine: AsyncEngine = app.state.engine
        if body.model is not None and body.model != model_name:
            return error(404, "model_not_found", f"this server runs {model_name!r}")

        if isinstance(body.prompt, str):
            prompt = tokenizer.encode_ordinary(body.prompt)
        else:
            prompt = body.prompt
        vocab_size = engine.model.cfg.vocab_size
        if not prompt or not all(0 <= t < vocab_size for t in prompt):
            return error(
                400,
                "invalid_prompt",
                f"the prompt must be token ids in [0, {vocab_size})",
            )

        request = Request(
            request_id=next(request_ids),
            prompt_tokens=prompt,
            max_new_tokens=body.max_tokens,
            arrival=0.0,  # submit() stamps it
            priority=body.priority,
            ignore_eos=body.ignore_eos,
        )
        # Before any stream starts, while a rejection can still be an HTTP status.
        try:
            engine.check(request)
        except (
            Rejected
        ) as exc:  # the id joins it with /internal/log, where it is recorded
            return error(400, "rejected", str(exc), id=f"cmpl-{request.request_id}")
        except RuntimeError as exc:
            return error(503, "engine_failed", str(exc))

        if body.stream:
            return StreamingResponse(
                stream(engine, request), media_type="text/event-stream"
            )

        try:
            async with contextlib.aclosing(engine.submit(request)) as tokens:
                token_ids = [tok async for tok in tokens]
        except Exception as exc:  # a step failed
            return error(500, "engine_failed", str(exc))
        choice = {
            "text": tokenizer.decode(token_ids),
            "token_ids": token_ids,
            "finish_reason": finish_reason(request),
        }
        return completion(request, choice, usage=usage(request))

    async def stream(engine: AsyncEngine, request: Request):
        """One SSE event per token, then one with finish_reason and usage, then [DONE]. If the
        client leaves, Starlette cancels this, and aclosing makes submit() abort the request."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            async with contextlib.aclosing(engine.submit(request)) as tokens:
                async for tok in tokens:
                    # A character can span tokens: the decoder holds partial bytes back.
                    text = decoder.decode(tokenizer.decode_single_token_bytes(tok))
                    choice = {"text": text, "token_id": tok, "finish_reason": None}
                    yield sse(completion(request, choice))
        except Exception as exc:  # a step failed; the 200 is already sent
            yield sse({"error": {"type": "engine_failed", "message": str(exc)}})
            return
        choice = {
            "text": decoder.decode(b"", final=True),
            "finish_reason": finish_reason(request),
        }
        yield sse(completion(request, choice, usage=usage(request)))
        yield "data: [DONE]\n\n"

    def completion(request: Request, choice: dict, **extra) -> dict:
        return {
            "id": f"cmpl-{request.request_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, **choice}],
            **extra,
        }

    @app.get("/health")
    async def health():
        engine: AsyncEngine = app.state.engine
        if engine.error is not None:
            return error(503, "engine_failed", repr(engine.error))
        return {"status": "ok"}

    @app.get("/internal/config")
    async def config():
        return {"model": model_name, **describe(app.state.engine), "args": server_args}

    @app.get("/internal/log")
    async def log():
        # JSONResponse straight from json.dumps: FastAPI's encoder is slow on a long step log.
        return JSONResponse(app.state.engine.drain_log())

    return app


def finish_reason(request: Request) -> str:
    stopped = not request.ignore_eos and request.output_tokens[-1:] == [EOS_ID]
    return "stop" if stopped else "length"


def usage(request: Request) -> dict:
    prompt, completion = len(request.prompt_tokens), len(request.output_tokens)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def main():
    p = argparse.ArgumentParser(description="Serve GPT-2 over HTTP.")
    add_engine_args(p)
    g = p.add_argument_group("server")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument(
        "--log-events",
        action="store_true",
        help="print a line each time a request is queued, admitted, finished, rejected or aborted",
    )
    args = p.parse_args()
    check_engine_args(p, args)
    if args.log_events:
        logging.basicConfig(format="%(message)s")
        logging.getLogger("engine").setLevel(logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_gpt2(args.model, device)

    def make_engine():
        cache = make_slot_cache(model, args.slots, args.max_T)
        return AsyncEngine(
            model,
            cache,
            args.kv_budget,
            policy=args.policy,
            max_wait=args.max_wait,
            eos_id=EOS_ID,
        )

    app = create_app(make_engine, tiktoken.get_encoding("gpt2"), args.model, vars(args))
    # flush: uvicorn re-raises SIGTERM after shutdown, which would drop buffered output
    print(
        f"serving {args.model} on {device} at http://{args.host}:{args.port}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
