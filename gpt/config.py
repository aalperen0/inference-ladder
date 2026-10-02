from dataclasses import dataclass


@dataclass
class GptConfig:
    vocab_size: int
    embed_dim: int
    context_length: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool
