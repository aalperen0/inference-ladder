"""
Running with the KV cache must give the same result as running without it.
Every cache test runs once per cache kind: cat, static and slot (one slot per batch row).
The slot cache's own tests run requests of different lengths in one batch.

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
from kv_cache import CACHE_KINDS, SlotKVCache, StaticKVCache, make_cache
from load import load_gpt2
from model import GPTModel
from sample import generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL_FP64 = 1e-6
PROMPT = "The meaning of life is"


@pytest.fixture(params=CACHE_KINDS)
def kind(request):
    """Runs each test that uses it once per cache kind: "cat", "static" and "slot"."""
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
    else:  # static and slot: allocated once at max_T = prompt + steps, never grows
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


# --- the slot cache: requests of different lengths in one batch ----------------------


def empty_slot_cache(model, n_slots, max_T):
    cfg = model.cfg
    weight = model.tok_embd.weight
    return SlotKVCache(
        n_layers=cfg.n_layers,
        n_slots=n_slots,
        n_heads=cfg.n_heads,
        max_T=max_T,
        head_dim=cfg.embed_dim // cfg.n_heads,
        device=weight.device,
        dtype=weight.dtype,
    )


@torch.inference_mode()
def start_request(model, cache, prompt):
    """Claims a slot and prefills the prompt in it alone.
    Returns (slot, seq): seq is the prompt plus its greedy next token, not yet in the cache."""
    slot = cache.allocate()
    cache.select([slot])
    nxt = model(prompt, cache)[:, -1].argmax(dim=-1, keepdim=True)
    return slot, torch.cat([prompt, nxt], dim=1)


@torch.inference_mode()
def decode_together(model, cache, seqs, steps):
    """Greedy-decodes the requests {slot: seq} in one batch, rows in the dict's order.
    Every step, each row's logits must match its own sequence run alone without a cache."""
    slots = list(seqs)
    cache.select(slots)
    for _ in range(steps):
        last = torch.cat([seqs[s][:, -1:] for s in slots])  # (B, 1)
        cached = model(last, cache)[:, -1]
        for row, s in enumerate(slots):
            uncached = model(seqs[s])[0, -1]
            torch.testing.assert_close(cached[row], uncached, atol=ATOL_FP64, rtol=0)
            seqs[s] = torch.cat([seqs[s], cached[row].argmax().view(1, 1)], dim=1)


def test_slot_requests_of_different_lengths(model64):
    """Prompts of 3, 7 and 12 tokens decoded in one batch: catches one position for the
    whole batch (each row needs its own) and a short row attending past its own length."""
    cache = empty_slot_cache(model64, n_slots=3, max_T=24)
    V = model64.cfg.vocab_size
    seqs = dict(
        start_request(model64, cache, random_ids((1, n), V, seed=10 + n))
        for n in (3, 7, 12)
    )
    decode_together(model64, cache, seqs, steps=8)


def test_slot_left_out_of_batch_is_untouched(model64):
    """Decode slots 2 and 0 (in that order) while slot 1 waits, then all three: catches
    rows going to the wrong slot, and writes or positions leaking into an unselected slot."""
    cache = empty_slot_cache(model64, n_slots=3, max_T=24)
    V = model64.cfg.vocab_size
    seqs = dict(
        start_request(model64, cache, random_ids((1, n), V, seed=20 + n))
        for n in (4, 9, 6)
    )
    batch = {2: seqs[2], 0: seqs[0]}
    decode_together(model64, cache, batch, steps=4)
    decode_together(model64, cache, {**batch, 1: seqs[1]}, steps=4)


def test_reused_slot_ignores_previous_request(model64):
    """A released slot still holds the old request's k, v. A short new request batched with
    a longer one reads columns past its own length, and they must stay masked."""
    cache = empty_slot_cache(model64, n_slots=2, max_T=24)
    V = model64.cfg.vocab_size
    old, _ = start_request(model64, cache, random_ids((1, 16), V, seed=30))
    other, other_seq = start_request(model64, cache, random_ids((1, 10), V, seed=31))
    cache.release(old)
    new, new_seq = start_request(model64, cache, random_ids((1, 3), V, seed=32))
    assert new == old
    decode_together(model64, cache, {new: new_seq, other: other_seq}, steps=5)


def test_slot_cache_bookkeeping(tiny):
    cache = empty_slot_cache(tiny, n_slots=2, max_T=8)
    a, b = cache.allocate(), cache.allocate()
    assert (a, b) == (0, 1)
    with pytest.raises(RuntimeError):
        cache.allocate()  # every slot is in use
    cache.select([b, a])
    cache.release(a)
    assert cache.slots == [b]  # a released slot leaves the batch
    with pytest.raises(ValueError):
        cache.release(a)  # already free
    with pytest.raises(ValueError):
        cache.select([a])  # not in use
    with pytest.raises(ValueError):
        cache.select([b, b])  # the same slot twice
    assert cache.allocate() == a


@torch.inference_mode()
def test_slot_cache_full_or_wrong_batch_raises(tiny):
    """A row going past max_T must fail loudly even if the others fit, and so must a batch
    whose row count is not the number of selected slots."""
    cache = empty_slot_cache(tiny, n_slots=2, max_T=5)
    V = tiny.cfg.vocab_size
    a, _ = start_request(tiny, cache, random_ids((1, 5), V, seed=40))
    b, _ = start_request(tiny, cache, random_ids((1, 2), V, seed=41))
    cache.select([a, b])
    with pytest.raises(ValueError):
        tiny(torch.zeros((2, 1), dtype=torch.long), cache)
    cache.select([b])
    with pytest.raises(ValueError):
        tiny(torch.zeros((2, 1), dtype=torch.long), cache)


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
