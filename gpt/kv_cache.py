import torch

CACHE_KINDS = ("cat", "static", "slot")


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

    def positions(self, T: int, device: str | torch.device) -> torch.Tensor:
        """(T,): the new tokens come right after the ones stored."""
        return torch.arange(self.current_pos, self.current_pos + T, device=device)

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

    def positions(self, T: int, device: str | torch.device) -> torch.Tensor:
        """(T,): the new tokens come right after the ones stored."""
        return torch.arange(self.current_pos, self.current_pos + T, device=device)


class SlotKVCache:
    """n_slots sequences in one buffer: (n_layers, n_slots, H, max_T, hd), one length per slot.
    Each slot holds one request, so requests can join and leave a running batch.

    allocate() claims a free slot and release(slot) gives it back. Before each forward,
    select(slots) picks the batch: row i of the input reads and writes slot slots[i].
    Every row of a forward stores the same T new tokens, so prefill one request at a time
    and decode many at once."""

    def __init__(
        self,
        n_layers: int,
        n_slots: int,
        n_heads: int,
        max_T: int,
        head_dim: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ):
        self.max_T = max_T
        self.key_cache = torch.zeros(
            n_layers, n_slots, n_heads, max_T, head_dim, device=device, dtype=dtype
        )
        self.value_cache = torch.zeros_like(self.key_cache)

        self.lengths = [0] * n_slots  # tokens stored in each slot
        self.free = list(range(n_slots))
        self.select([])

    def allocate(self) -> int:
        """Claims a free slot for a new request. Its old k, v stay; at length 0 nothing sees them."""
        if not self.free:
            raise RuntimeError("no free slot")
        slot = min(self.free)
        self.free.remove(slot)
        self.lengths[slot] = 0
        return slot

    def release(self, slot: int):
        """Gives a finished request's slot back. It also leaves the selected batch."""
        if not self._in_use(slot):
            raise ValueError(f"slot {slot} is not in use")
        self.free.append(slot)
        if slot in self.slots:
            self.select([s for s in self.slots if s != slot])

    def select(self, slots: list[int]):
        """The slots the next forward runs on: row i of the batch is slot slots[i]."""
        if len(set(slots)) != len(slots) or not all(self._in_use(s) for s in slots):
            raise ValueError(
                f"select needs distinct slots that are in use, got {slots}"
            )
        device = self.key_cache.device
        self.slots = list(slots)
        self._ids = torch.tensor(self.slots, dtype=torch.long, device=device)
        self._pos = torch.tensor(
            [self.lengths[s] for s in self.slots], dtype=torch.long, device=device
        )
        # The longest selected row: update() returns current_pos + T keys for every row.
        self.current_pos = max((self.lengths[s] for s in self.slots), default=0)
        # Consecutive slots are a slice, so reads are a view; any other set gathers a copy.
        first = self.slots[0] if self.slots else 0
        if self.slots == list(range(first, first + len(self.slots))):
            self._rows = slice(first, first + len(self.slots))
        else:
            self._rows = self._ids

    def positions(self, T: int, device: str | torch.device) -> torch.Tensor:
        return self._pos[:, None] + torch.arange(T, device=device)

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes row i's new k, v into slot slots[i] at [length : length+T] and returns every
        row's keys and values up to the longest row: (B, H, current_pos + T, hd). A shorter
        row's extra columns hold old or zero values; the causal mask hides them."""
        B, _, T, _ = k.shape
        if B != len(self.slots):
            raise ValueError(f"{B} rows but {len(self.slots)} slots selected")
        end = self.current_pos + T
        if end > self.max_T:
            raise ValueError(
                f"Slot full: {self.current_pos} + {T} > max_T={self.max_T}"
            )

        rows, cols = self._ids[:, None], self.positions(T, k.device)  # (B, 1), (B, T)
        # Indexing [rows, :, cols] gives (B, T, H, hd), so the new k, v go in as that.
        self.key_cache[layer][rows, :, cols] = k.transpose(1, 2)
        self.value_cache[layer][rows, :, cols] = v.transpose(1, 2)

        return (
            self.key_cache[layer, self._rows, :, :end],
            self.value_cache[layer, self._rows, :, :end],
        )

    def advance(self, n: int):
        for s in self.slots:
            self.lengths[s] += n
        self._pos = self._pos + n
        self.current_pos += n

    def _in_use(self, slot: int) -> bool:
        return 0 <= slot < len(self.lengths) and slot not in self.free


def make_cache(kind: str, model, batch_size: int, max_T: int):
    """A new, empty cache sized for `model`, on the model's device and dtype.
    "slot" gets one slot per batch row, all selected, so it works wherever "static" does."""
    cfg = model.cfg
    if kind == "cat":
        return KV_Cache(cfg.n_layers)
    weight = model.tok_embd.weight
    sizes = {
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "max_T": max_T,
        "head_dim": cfg.embed_dim // cfg.n_heads,
        "device": weight.device,
        "dtype": weight.dtype,
    }
    if kind == "static":
        return StaticKVCache(batch_size=batch_size, **sizes)
    if kind == "slot":
        cache = SlotKVCache(n_slots=batch_size, **sizes)
        cache.select([cache.allocate() for _ in range(batch_size)])
        return cache
    raise ValueError(f"unknown cache kind {kind!r}; choose from {CACHE_KINDS}")
