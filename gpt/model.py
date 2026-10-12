from config import GptConfig
import torch
from torch import nn
import torch.nn.functional as F
import tiktoken
from torch.utils.data import Dataset


class GPTModel(nn.Module):
    def __init__(self, cfg: GptConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embd = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embd = nn.Embedding(cfg.context_length, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = LayerNorm(cfg.embed_dim)
        self.out = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.out.weight = self.tok_embd.weight  # weight tying, as in GPT-2

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, X, cache=None):
        B, T = X.shape
        # With a cache, the new tokens come after the ones already stored.
        pos = cache.current_pos if cache is not None else 0
        assert pos + T <= self.cfg.context_length, "sequence longer than context_length"

        tok_embd = self.tok_embd(X)
        pos_embd = self.pos_embd(torch.arange(pos, pos + T, device=X.device))

        X = tok_embd + pos_embd
        X = self.drop(X)

        for i, block in enumerate(self.trf_blocks):
            X = block(X, cache, i)
        if cache is not None:
            cache.advance(T)  # once, after every layer has stored its T new tokens

        X = self.norm(X)
        logits = self.out(X)
        return logits


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, context_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i + 1 : i + 1 + context_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, n_heads, dropout, qkv_bias=True):
        super().__init__()

        assert d_out % n_heads == 0, "d_out must be divisible"

        self.d_out = d_out
        self.num_heads = n_heads
        self.head_dim = d_out // self.num_heads
        self.query_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def split_heads(self, X):
        B, T, _ = X.shape
        return X.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, X, cache=None, layer_idx=None):
        B, T, C = X.shape
        queries = self.split_heads(self.query_proj(X))
        keys = self.split_heads(self.key_proj(X))
        values = self.split_heads(self.value_proj(X))

        pos = cache.current_pos if cache is not None else 0
        if cache is not None:
            # store the new k, v first so the new tokens also attend to themselves
            keys, values = cache.update(layer_idx, keys, values)

        # Q @ K.T
        attn_scores = queries @ keys.transpose(
            2, 3
        )  # B,H,Tq,hd @ B,H,hd,Tk -> B,H,Tq,Tk

        attn_bool = self.mask.bool()[pos : pos + T, : pos + T]

        attn_scores.masked_fill_(attn_bool, -torch.inf)

        attn_weights = F.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (
            (attn_weights @ values).transpose(1, 2).contiguous().view(B, T, C)
        )

        return self.out_proj(context_vector)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GptConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.embed_dim,
            d_out=cfg.embed_dim,
            context_length=cfg.context_length,
            n_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.embed_dim)
        self.norm2 = LayerNorm(cfg.embed_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, X, cache=None, layer_idx=None):
        shortcut = X
        X = self.norm1(X)
        X = self.att(X, cache, layer_idx)
        X = self.drop_shortcut(X)
        X = X + shortcut

        shortcut = X
        X = self.norm2(X)
        X = self.ff(X)
        X = self.drop_shortcut(X)
        X = X + shortcut
        return X


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # prevent division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, X):
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True, unbiased=False)
        normalize_x = (X - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalize_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, cfg: GptConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
        )

    def forward(self, X):
        return self.layers(X)


####################


def text_to_tokenIDS(txt, tokenizer):
    encoded = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def tokenIDS_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    decoded_text = tokenizer.decode(flat.tolist())
    return decoded_text
