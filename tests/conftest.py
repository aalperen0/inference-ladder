"""Shared by the engine and API tests: a tiny random GPT and the reference it must match.

The model is in float64 on the CPU: fast, and batching changes the logits only by ~1e-11, so
it can never flip an argmax. The reference is plain greedy decoding without a cache, so it
shares no code with the engine or the slot cache.
"""

import pytest
import torch

from config import GptConfig
from model import GPTModel

MAX_T = 64  # the tiny model's context_length, and every slot's size
V = 100  # its vocab size


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = GptConfig(
        vocab_size=V, embed_dim=32, context_length=MAX_T, n_heads=4, n_layers=2
    )
    return GPTModel(cfg).double().eval()


@torch.inference_mode()
def greedy_alone(model, request):
    """The request's greedy tokens with no engine and no cache: the whole sequence every step."""
    seq = torch.tensor([request.prompt_tokens])
    for _ in range(request.max_new_tokens):
        nxt = model(seq)[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    return seq[0, len(request.prompt_tokens) :].tolist()
