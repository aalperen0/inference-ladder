"""Rung 1: running with the KV cache must give the same result as running without it.
Every cache test runs twice: once with the cat cache, once with the static one.

Logits are compared in float64 on the CPU. In fp32 the two paths differ by up to
7.5e-3 on GPU logits of magnitude ~285 (measured), only from rounding: the cached
path multiplies matrices of different shapes, so the sums happen in a different
order. That is too close to small bugs to tell them apart. In float64 correct code
agrees to ~1e-11, so ATOL_FP64 = 1e-6 leaves no room for a bug to hide.
fp32 is still checked where it matters in practice: greedy tokens must be identical.
"""

import pytest
import tiktoken
import torch

from config import GptConfig, gpt2_config
from kv_cache import CACHE_KINDS, StaticKVCache, make_cache
from load import load_gpt2
from model import GPTModel
from sample import generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL_FP64 = 1e-6
PROMPT = "The meaning of life is"


@pytest.fixture(params=CACHE_KINDS)
def kind(request):
    """Runs each test that uses it once per cache kind: "cat" and "static"."""
    return request.param


@pytest.fixture(scope="module")
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def model32():
    return load_gpt2("gpt2", DEVICE)


@pytest.fixture(scope="module")
def model64():
    return load_gpt2("gpt2", "cpu").double()


@pytest.fixture(scope="module")
def tiny():
    """Random 2-layer model with context_length 16, to reach the context limit quickly."""
    torch.manual_seed(0)
    cfg = GptConfig(
        vocab_size=100, embed_dim=32, context_length=16, n_heads=4, n_layers=2
    )
    return GPTModel(cfg).double().eval()


def random_ids(shape, vocab_size, seed):
    return torch.randint(
        0, vocab_size, shape, generator=torch.Generator().manual_seed(seed)
    )


@torch.inference_mode()
def decode_both(model, idx, steps, kind):
    """Greedy decode `steps` tokens with the cache, one token per forward.
    At every step also run the whole sequence without a cache, for comparison.
    Returns (cached logits, uncached logits, final sequence, cache); logits are (steps+1, B, V).
    """
    cache = make_cache(kind, model, batch_size=idx.size(0), max_T=idx.size(1) + steps)
    cached = [model(idx, cache)[:, -1]]  # prefill
    uncached = [model(idx)[:, -1]]
    seq = idx
    for _ in range(steps):
        nxt = cached[-1].argmax(dim=-1, keepdim=True)  # (B, 1)
        seq = torch.cat([seq, nxt], dim=1)
        cached.append(model(nxt, cache)[:, -1])  # only the new token
        uncached.append(model(seq)[:, -1])  # the whole sequence again
    return torch.stack(cached), torch.stack(uncached), seq, cache


@torch.inference_mode()
def greedy_cached(model, idx, steps, kind):
    cache = make_cache(kind, model, batch_size=idx.size(0), max_T=idx.size(1) + steps)
    logits = model(idx, cache)[:, -1]
    seq = idx
    for _ in range(steps):
        nxt = logits.argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        logits = model(nxt, cache)[:, -1]
    return seq


# --- the model with a cache -------------------------------------------------------


def test_decode_logits_match_no_cache(model64, tokenizer, kind):
    """Every decode step: catches positions starting from 0, attending before storing the
    new k/v, and advancing the position in every layer."""
    idx = torch.tensor([tokenizer.encode(PROMPT)])
    cached, uncached, _, _ = decode_both(model64, idx, steps=20, kind=kind)
    torch.testing.assert_close(cached, uncached, atol=ATOL_FP64, rtol=0)


def test_cache_holds_every_token(model64, tokenizer, kind):
    """After prefill + n steps, every layer has stored exactly the tokens seen so far."""
    idx = torch.tensor([tokenizer.encode(PROMPT)])
    _, _, seq, cache = decode_both(model64, idx, steps=5, kind=kind)
    T, cfg = seq.size(1), model64.cfg
    head_dim = cfg.embed_dim // cfg.n_heads
    assert cache.current_pos == T
    if kind == "cat":  # grows by exactly the tokens seen
        for layer in range(cfg.n_layers):
            assert cache.key_cache[layer].shape == (1, cfg.n_heads, T, head_dim)
            assert cache.value_cache[layer].shape == (1, cfg.n_heads, T, head_dim)
    else:  # allocated once at max_T = prompt + steps, never grows
        assert cache.key_cache.shape == (cfg.n_layers, 1, cfg.n_heads, T, head_dim)
        assert cache.value_cache.shape == cache.key_cache.shape


def test_batch_rows_stay_separate(model64, kind):
    """Batch of 2: catches a cache that mixes up rows."""
    idx = random_ids((2, 8), model64.cfg.vocab_size, seed=1)
    cached, uncached, _, _ = decode_both(model64, idx, steps=10, kind=kind)
    torch.testing.assert_close(cached, uncached, atol=ATOL_FP64, rtol=0)


@torch.inference_mode()
def test_prefill_in_two_chunks(model64, kind):
    """Several new tokens with a non-empty cache: the mask must be rows pos:pos+T,
    columns :pos+T. A [:T, :T] mask fails here (rung 5's draft verification does this)."""
    idx = random_ids((1, 12), model64.cfg.vocab_size, seed=2)
    cache = make_cache(kind, model64, batch_size=1, max_T=12)
    model64(idx[:, :5], cache)
    second = model64(idx[:, 5:], cache)  # 7 new tokens after 5 cached ones
    assert cache.current_pos == 12
    torch.testing.assert_close(second, model64(idx)[:, 5:], atol=ATOL_FP64, rtol=0)


def test_greedy_tokens_match_fp32(model32, tokenizer, kind):
    """In practice (fp32, GPU): 200 greedy tokens, cached == uncached."""
    idx = torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)
    expected = generate(model32, idx, max_new_tokens=200, temperature=0)  # no cache
    assert (
        greedy_cached(model32, idx, steps=200, kind=kind).tolist() == expected.tolist()
    )


# --- edge cases on a tiny model ----------------------------------------------------


@pytest.mark.parametrize("prompt_len", [1, 10])
def test_decode_up_to_context_limit(tiny, kind, prompt_len):
    """A 1-token prompt, and decoding until exactly context_length tokens."""
    idx = random_ids((1, prompt_len), tiny.cfg.vocab_size, seed=3)
    steps = tiny.cfg.context_length - prompt_len
    cached, uncached, seq, cache = decode_both(tiny, idx, steps, kind=kind)
    assert seq.size(1) == cache.current_pos == tiny.cfg.context_length
    torch.testing.assert_close(cached, uncached, atol=ATOL_FP64, rtol=0)


@torch.inference_mode()
def test_past_context_limit_raises(tiny, kind):
    """GPT-2 has no position embedding past context_length, so the model must refuse."""
    cache = make_cache(kind, tiny, batch_size=1, max_T=tiny.cfg.context_length)
    tiny(random_ids((1, tiny.cfg.context_length), tiny.cfg.vocab_size, seed=4), cache)
    with pytest.raises(AssertionError):
        tiny(torch.zeros((1, 1), dtype=torch.long), cache)


# --- the static cache itself --------------------------------------------------------


def test_static_cache_size_matches_memory_math():
    """GPT-2 small, B=1, max_T=1024: 2 (K,V) x 12 layers x 1024 tokens x 768 x 4 B = 72 MiB,"""
    cfg = gpt2_config("gpt2")
    cache = StaticKVCache(
        n_layers=cfg.n_layers,
        batch_size=1,
        n_heads=cfg.n_heads,
        max_T=1024,
        head_dim=cfg.embed_dim // cfg.n_heads,
        device=DEVICE,
        dtype=torch.float32,
    )
    assert cache.key_cache.nbytes + cache.value_cache.nbytes == 2 * 12 * 1024 * 768 * 4
    assert cache.key_cache.device.type == torch.device(DEVICE).type
    assert cache.key_cache.dtype == torch.float32


@torch.inference_mode()
def test_static_cache_full_raises(tiny):
    """Writing past max_T must fail loudly instead of writing out of bounds."""
    cache = make_cache("static", tiny, batch_size=1, max_T=5)
    tiny(random_ids((1, 5), tiny.cfg.vocab_size, seed=5), cache)
    with pytest.raises(ValueError):
        tiny(torch.zeros((1, 1), dtype=torch.long), cache)


# --- generate(use_cache=...) ---------------------------------------------------------


def test_generate_cached_matches_uncached(model32, tokenizer, kind):
    idx = torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)
    a = generate(model32, idx, max_new_tokens=100, temperature=0, use_cache=kind)
    b = generate(model32, idx, max_new_tokens=100, temperature=0, use_cache=False)
    assert a.tolist() == b.tolist()


def test_generate_twice_gives_same_result(model32, tokenizer, kind):
    """Catches a cache left over from the previous call."""
    idx = torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)
    a = generate(model32, idx, max_new_tokens=30, temperature=0, use_cache=kind)
    b = generate(model32, idx, max_new_tokens=30, temperature=0, use_cache=kind)
    assert a.tolist() == b.tolist()


def test_generate_cached_sampling_matches_uncached(model32, tokenizer, kind):
    """Same seed, same sampling settings: the cache must not change which tokens are drawn."""
    idx = torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)
    kwargs = dict(max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9)
    torch.manual_seed(0)
    a = generate(model32, idx, use_cache=kind, **kwargs)
    torch.manual_seed(0)
    b = generate(model32, idx, use_cache=False, **kwargs)
    assert a.tolist() == b.tolist()


def test_generate_use_cache_true_means_static(model32, tokenizer):
    idx = torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)
    a = generate(model32, idx, max_new_tokens=20, temperature=0, use_cache=True)
    b = generate(model32, idx, max_new_tokens=20, temperature=0, use_cache="static")
    assert a.tolist() == b.tolist()
