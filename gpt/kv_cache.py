import torch

CACHE_KINDS = ("cat", "static")


class KV_Cache:
    def __init__(self, n_layers: int):
        # One slot per layer: each layer has its own key/value weights, so its own K and V.
        self.key_cache: list[torch.Tensor | None] = [None] * n_layers
        self.value_cache: list[torch.Tensor | None] = [None] * n_layers
        self.current_pos = 0  # tokens stored so far = position of the next token

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append this layer's new k, v and return all of its keys and values so far.
        k, v shapes: [batch_size, num_heads, seq_len, head_dim]
        """
        if self.key_cache[layer] is None:
            self.key_cache[layer], self.value_cache[layer] = k, v
        else:
            self.key_cache[layer] = torch.cat([self.key_cache[layer], k], dim=2)
            self.value_cache[layer] = torch.cat([self.value_cache[layer], v], dim=2)

        return self.key_cache[layer], self.value_cache[layer]

    def advance(self, n: int):
        self.current_pos += n

    def reset_cache(self, n_layers):
        # Reset the cache
        self.key_cache: list[torch.Tensor | None] = [None] * n_layers
        self.value_cache: list[torch.Tensor | None] = [None] * n_layers
        self.current_pos = 0


class StaticKVCache:
    """K and V for every layer in one buffer allocated up front: (n_layers, B, H, max_T, hd).
    Each step writes only the new tokens; nothing old is copied."""

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        max_T: int,
        head_dim: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ):
        self.current_pos = 0
        self.max_T = max_T
        self.key_cache = torch.empty(
            n_layers, batch_size, n_heads, max_T, head_dim, device=device, dtype=dtype
        )
        self.value_cache = torch.empty_like(self.key_cache)

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes this layer's new k, v at [pos : pos+T] and returns a view of its filled part."""
        T = k.shape[2]
        pos = self.current_pos
        if pos < 0 or T < 0 or pos + T > self.max_T:
            raise ValueError(
                f"Cache full or invalid cache range: {pos} + {T} > max_T={self.max_T}"
            )

        self.key_cache[layer, :, :, pos : pos + T] = k
        self.value_cache[layer, :, :, pos : pos + T] = v

        return (
            self.key_cache[layer, :, :, : pos + T],
            self.value_cache[layer, :, :, : pos + T],
        )

    def advance(self, n: int):
        self.current_pos += n


def make_cache(kind: str, model, batch_size: int, max_T: int):
    """A new, empty cache sized for `model`, on the model's device and dtype."""
    cfg = model.cfg
    if kind == "cat":
        return KV_Cache(cfg.n_layers)
    if kind == "static":
        weight = model.tok_embd.weight
        return StaticKVCache(
            n_layers=cfg.n_layers,
            batch_size=batch_size,
            n_heads=cfg.n_heads,
            max_T=max_T,
            head_dim=cfg.embed_dim // cfg.n_heads,
            device=weight.device,
            dtype=weight.dtype,
        )
    raise ValueError(f"unknown cache kind {kind!r}; choose from {CACHE_KINDS}")
