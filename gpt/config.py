from dataclasses import dataclass


@dataclass
class GptConfig:
    vocab_size: int = 50257
    embed_dim: int = 768
    context_length: int = 1024
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.0
    qkv_bias: bool = True


GPT2_PRESETS = {
    "gpt2": dict(embed_dim=768, n_layers=12, n_heads=12),  # 124M
    "gpt2-medium": dict(embed_dim=1024, n_layers=24, n_heads=16),  # 355M
    "gpt2-large": dict(embed_dim=1280, n_layers=36, n_heads=20),  # 774M
    "gpt2-xl": dict(embed_dim=1600, n_layers=48, n_heads=25),  # 1558M
}


def gpt2_config(name: str = "gpt2") -> GptConfig:
    if name not in GPT2_PRESETS:
        raise ValueError(
            f"unknown GPT-2 model {name!r}; choose from {list(GPT2_PRESETS)}"
        )
    return GptConfig(**GPT2_PRESETS[name])


# The model trained on TinyStories
TINYSTORIES_30M = GptConfig(
    embed_dim=384,
    context_length=256,
    n_heads=6,
    n_layers=6,
    drop_rate=0.1,
    qkv_bias=False,
)
