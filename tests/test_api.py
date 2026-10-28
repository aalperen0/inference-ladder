"""
The HTTP layer must pass the engine's tokens through unchanged, answer every way a request
can fail with the right status before any stream starts, and report the config and logs
that bench/ records into results/.

Starlette's TestClient runs the app's lifespan, so the engine runs on its event loop as it
would under uvicorn. The model is conftest.py's tiny one: its vocab is ids 0–99, which
tiktoken decodes as single printable characters.
"""

import json

import pytest
import tiktoken
from fastapi.testclient import TestClient

from api import create_app
from conftest import MAX_T, V, greedy_alone
from engine import AsyncEngine, make_slot_cache
from scheduler import Request

TOKENIZER = tiktoken.get_encoding("gpt2")


def make_client(model, crash=False):
    def make_engine():
        engine = AsyncEngine(model, make_slot_cache(model, 2, MAX_T), 10_000)
        if crash:
            engine.step = fake_cuda_oom
        return engine

    return TestClient(create_app(make_engine, TOKENIZER, "tiny"))


def fake_cuda_oom(*args):
    raise RuntimeError("fake CUDA out of memory")


@pytest.fixture
def client(tiny):
    with make_client(tiny) as c:
        yield c


def complete(client, **body):
    return client.post("/v1/completions", json=body)


def stream(client, **body):
    """The parsed `data:` events of a streamed completion; the last raw one must be [DONE]."""
    with client.stream("POST", "/v1/completions", json={"stream": True, **body}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        data = [line.removeprefix("data: ") for line in r.iter_lines() if line]
    assert data[-1] == "[DONE]"
    return [json.loads(d) for d in data[:-1]]


def expected_tokens(model, prompt, max_tokens):
    return greedy_alone(model, Request(0, prompt, max_tokens, arrival=0.0))


def test_streamed_tokens_match_running_alone(client, tiny):
    prompt = [5, 17, 42, 8]
    *token_events, last = stream(client, prompt=prompt, max_tokens=10)
    expected = expected_tokens(tiny, prompt, 10)

    choices = [e["choices"][0] for e in token_events]
    assert [c["token_id"] for c in choices] == expected
    assert all(c["finish_reason"] is None for c in choices)
    assert "".join(c["text"] for c in choices) == TOKENIZER.decode(expected)
    assert len({e["id"] for e in token_events + [last]}) == 1
    assert last["choices"][0]["finish_reason"] == "length"
    assert last["usage"] == {"prompt_tokens": 4, "completion_tokens": 10, "total_tokens": 14}


def test_unstreamed_completion_matches_too(client, tiny):
    r = complete(client, prompt=[3, 1, 4, 1, 5], max_tokens=7, model="tiny")
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["token_ids"] == expected_tokens(tiny, [3, 1, 4, 1, 5], 7)
    assert choice["text"] == TOKENIZER.decode(choice["token_ids"])
    assert r.json()["usage"]["completion_tokens"] == 7


def test_text_prompt_goes_through_the_tokenizer(client):
    assert TOKENIZER.encode("a") == [64]
    by_text = complete(client, prompt="a", max_tokens=5).json()["choices"][0]
    by_ids = complete(client, prompt=[64], max_tokens=5).json()["choices"][0]
    assert by_text["token_ids"] == by_ids["token_ids"]


def test_request_that_can_never_fit_is_400_and_the_server_keeps_serving(client, tiny):
    r = complete(client, prompt=[1] * 60, max_tokens=10, stream=True)
    assert r.status_code == 400
    assert r.json()["id"] == "cmpl-0"  # joins it with its record in /internal/log
    assert r.json()["error"]["type"] == "rejected"
    assert "max_T" in r.json()["error"]["message"]
    ok = complete(client, prompt=[2, 3], max_tokens=4)
    assert ok.json()["choices"][0]["token_ids"] == expected_tokens(tiny, [2, 3], 4)


@pytest.mark.parametrize(
    "body, status",
    [
        ({"prompt": [1, 2], "temperature": 0.7}, 422),  # greedy only
        ({"prompt": [1, 2], "top_p": 0.9}, 422),  # unknown fields are errors, not ignored
        ({"prompt": [1, 2], "max_tokens": 0}, 422),
        ({"prompt": [1, V]}, 400),  # outside the vocab
        ({"prompt": []}, 400),
        ({"prompt": [1, 2], "model": "gpt2-xl"}, 404),
    ],
)
def test_bad_requests_are_refused(client, body, status):
    assert complete(client, **body).status_code == status


def test_log_reports_what_ran_then_empties(client):
    stream(client, prompt=[1, 2, 3], max_tokens=4)
    complete(client, prompt=[4, 5], max_tokens=3)
    complete(client, prompt=[1] * 60, max_tokens=10)  # rejected

    log = client.get("/internal/log").json()
    statuses = {r["request_id"]: r["status"] for r in log["requests"]}
    assert statuses == {0: "finished", 1: "finished", 2: "rejected"}
    assert len(log["steps"]) == 4 + 3  # one request at a time: one step per token
    finished = [r for r in log["requests"] if r["status"] == "finished"]
    for r in finished:
        assert r["arrival"] <= r["prefill_start"] <= r["token_times"][0] <= r["finished_at"]
    assert client.get("/internal/log").json() == {"steps": [], "requests": []}


def test_config_describes_the_engine(client):
    assert client.get("/health").json() == {"status": "ok"}
    config = client.get("/internal/config").json()
    kv_bytes_per_token = 2 * 2 * 32 * 8  # K and V × layers × embed_dim × float64
    assert config["kv_bytes_per_token"] == kv_bytes_per_token
    assert config["kv_cache_bytes"] == kv_bytes_per_token * 2 * MAX_T  # 2 slots of MAX_T
    assert (config["model"], config["slots"], config["max_T"]) == ("tiny", 2, MAX_T)
    assert config["policy"] == "fcfs" and config["dtype"] == "float64"


def test_failed_step_reaches_the_client_and_the_server_reports_it(tiny):
    with make_client(tiny, crash=True) as client:
        with client.stream("POST", "/v1/completions", json={"prompt": [1, 2], "stream": True}) as r:
            events = [line for line in r.iter_lines() if line]
        assert r.status_code == 200  # sent before the step failed
        assert events == [
            'data: {"error": {"type": "engine_failed", "message": "fake CUDA out of memory"}}'
        ]
        assert client.get("/health").status_code == 503
        assert complete(client, prompt=[1, 2]).status_code == 503
