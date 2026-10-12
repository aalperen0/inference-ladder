"""Rung 0: our GPTModel + OpenAI weights must match Hugging Face's GPT-2.

Tolerance: fp32 rounding alone gives max |diff| ~1e-4 on logits of magnitude ~150
(measured on a GTX 1050 Ti and CPU). The subtlest real bugs measured far above it:
  LayerNorm eps 1e-6 -> 2e-2, exact GELU instead of tanh -> 1e-1,
  unbiased variance -> 2e-1, a missing transpose -> 8e+1.
ATOL = 1e-3 sits in the gap: ~5x above noise, ~20x below the smallest bug.
"""

import pytest
import tiktoken
import torch
from transformers import GPT2LMHeadModel

from load import load_gpt2
from sample import generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 1e-3
PROMPTS = ["The meaning of life is", "Hello, my name is", "In 1969, the first person to"]


@pytest.fixture(scope="module")
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def ours():
    return load_gpt2("gpt2", DEVICE)


@pytest.fixture(scope="module")
def hf():
    return GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()


def encode(tokenizer, text):
    return torch.tensor([tokenizer.encode(text)], device=DEVICE)


def test_loaded_model_is_ready_for_inference(ours):
    assert not ours.training, "dropout must be off"
    assert ours.out.weight is ours.tok_embd.weight, "output layer must be tied to wte"
    n_params = sum(p.numel() for p in ours.parameters())
    assert n_params == 124_439_808  # GPT-2 small, tied weights counted once


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_hf(ours, hf, tokenizer, prompt):
    idx = encode(tokenizer, prompt)
    with torch.inference_mode():
        a, b = ours(idx), hf(idx).logits
    assert a.shape == b.shape == (1, idx.size(1), 50257)
    torch.testing.assert_close(a, b, atol=ATOL, rtol=0)


def test_logits_match_hf_full_context(ours, hf):
    """All 1024 positions: catches position-embedding and mask bugs late in the sequence."""
    idx = torch.randint(0, 50257, (1, 1024), generator=torch.Generator().manual_seed(0)).to(DEVICE)
    with torch.inference_mode():
        torch.testing.assert_close(ours(idx), hf(idx).logits, atol=ATOL, rtol=0)


def test_logits_match_hf_batched(ours, hf):
    """Batch of 2: catches reshapes that mix up sequences in the batch."""
    idx = torch.randint(0, 50257, (2, 64), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    with torch.inference_mode():
        torch.testing.assert_close(ours(idx), hf(idx).logits, atol=ATOL, rtol=0)


def test_greedy_generation_matches_hf(ours, hf, tokenizer):
    """End to end: tokenizer + generation loop + model produce HF's exact tokens."""
    idx = encode(tokenizer, "The meaning of life is")
    a = generate(ours, idx, max_new_tokens=40, temperature=0)
    b = hf.generate(idx, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eot_token)
    assert tokenizer.decode(a[0].tolist()) == tokenizer.decode(b[0].tolist())
    assert a.tolist() == b.tolist()
