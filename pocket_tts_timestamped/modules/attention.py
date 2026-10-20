import torch
from torch import nn
from torch.nn import functional as F

from pocket_tts_timestamped.modules.rope import RotaryEmbedding
from pocket_tts_timestamped.modules.stateful_module import ModelState, StatefulModule
from pocket_tts_timestamped.timestamps.alignment import SelectedAttentionCapture


def complete_kv(
    cache: torch.Tensor, offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if offset.numel() > 1 and not torch.all(offset == offset.view(-1)[0]):
        raise ValueError("Linear cache offset must be identical across batch.")
    offset_value = int(offset.view(-1)[0].item())

    cache[0, :, offset_value : offset_value + k.shape[1]] = k
    cache[1, :, offset_value : offset_value + v.shape[1]] = v
    valid = cache[:, :, : offset_value + k.shape[1]]
    return valid[0], valid[1]


def _build_attention_mask(
    pos_q: torch.Tensor, pos_k: torch.Tensor, context: int | None
) -> torch.Tensor:
    delta = pos_q[:, :, None] - pos_k[:, None, :]
    mask = (pos_k[:, None, :] >= 0) & (delta >= 0)
    if context is not None:
        mask = mask & (delta < context)
    return mask[:, None]


# Stateless (training) path: positions are just arange(t) for every row, so the
# mask depends only on (t, context). Build it once at the largest t seen and
# slice — rebuilding a [B, 1, T, T] bool tensor per layer per step dominates
# the CPU cost of the forward otherwise.
_CAUSAL_MASK_CACHE: dict[tuple[int | None, torch.device], torch.Tensor] = {}


@torch.compiler.disable
def _cached_causal_mask(t: int, context: int | None, device: torch.device) -> torch.Tensor:
    key = (context, device)
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is None or cached.shape[-1] < t:
        size = max(t, cached.shape[-1] if cached is not None else 0)
        pos = torch.arange(size, device=device, dtype=torch.long)
        cached = _build_attention_mask(pos.view(1, -1), pos.view(1, -1), context)
        _CAUSAL_MASK_CACHE[key] = cached
    return cached[..., :t, :t]


# Per-layer streaming state schemas (returned by init_state and stored in model_state):
# - Linear cache (FlowLM / full causal):
#   - offset: torch.long[B]  # absolute time index for the next write / RoPE offset
#                            # (batch must be in sync)
#   - cache:  torch.[dtype][2, B, T, H, D]  # K/V stored contiguously along T (allocated capacity)


class _LinearKVCacheBackend:
    requires_state = True

    def __init__(self, num_heads: int, dim_per_head: int):
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head

    def init_state(
        self, batch_size: int, sequence_length: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        return dict(
            offset=torch.zeros(batch_size, dtype=torch.long, device=device),
            # Leading positions per row that hold padding rather than content.
            # Batched generation right-aligns rows of unequal prefix length and
            # sets this, which pushes those key positions negative so the
            # attention mask drops them (see _build_attention_mask).
            pad=torch.zeros(batch_size, dtype=torch.long, device=device),
            cache=torch.full(
                (2, batch_size, sequence_length, self.num_heads, self.dim_per_head),
                float("NaN"),
                device=device,
                dtype=dtype,
            ),
        )

    def increment_step(self, state: dict[str, torch.Tensor], increment: int):
        state["offset"] += increment

    def rope_offset(
        self, state: dict[str, torch.Tensor] | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if state is None:
            return torch.zeros((), dtype=torch.long, device=device)
        return state["offset"].view(-1)[0]

    def append_and_get(
        self, k: torch.Tensor, v: torch.Tensor, state: dict[str, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            k_attn = k.permute(0, 2, 1, 3)
            v_attn = v.permute(0, 2, 1, 3)
            pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
            pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
            offset = torch.zeros(k_attn.shape[0], device=k_attn.device, dtype=torch.long)
            return k_attn, v_attn, pos_k, offset
        cache_k, cache_v = complete_kv(state["cache"], state["offset"], k, v)
        k_attn = cache_k.permute(0, 2, 1, 3)
        v_attn = cache_v.permute(0, 2, 1, 3)
        pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
        pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
        pad = state.get("pad")
        offset = state["offset"]
        if pad is not None and bool((pad > 0).any()):
            # Padded slots get negative positions and are masked out. The query
            # offset is shifted by the same amount so query-key deltas are
            # unchanged -- otherwise a row's padding would inflate delta and
            # push real keys out of a finite `context` window. RoPE keeps using
            # the unshifted offset: it is relative, and shifting query and key
            # together leaves within-row distances identical.
            pos_k = pos_k - pad[:, None]
            offset = offset - pad
        return k_attn, v_attn, pos_k, offset


# Slots a windowed cache holds beyond its `context`. When they fill up, the keys still inside
# the window move to the front, so attention spans at most `context + _WINDOW_HEADROOM` keys.
_WINDOW_HEADROOM = 64


# - Windowed cache (finite `context`, e.g. the Mimi decoder): the linear schema plus
#   - start: torch.long[B]  # absolute time index held by cache slot 0
class _WindowedKVCacheBackend(_LinearKVCacheBackend):
    """Linear cache that drops keys no query can reach any more.

    With a finite `context`, a query at position `p` only sees keys after `p - context`.
    The linear cache keeps every key and masks the old ones out, so its memory grows with the
    sequence and each step attends over all of it. This cache stores at most
    `context + _WINDOW_HEADROOM` slots (more only while one call writes more steps than that)
    and gives the same output: keys keep their absolute positions for RoPE and the mask.
    """

    def __init__(self, num_heads: int, dim_per_head: int, context: int):
        super().__init__(num_heads, dim_per_head)
        self.context = context

    def init_state(
        self, batch_size: int, sequence_length: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        capacity = min(sequence_length, self.context + _WINDOW_HEADROOM)
        state = super().init_state(batch_size, capacity, device, dtype)
        state["start"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        return state

    def append_and_get(
        self, k: torch.Tensor, v: torch.Tensor, state: dict[str, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            return super().append_and_get(k, v, state)
        offset_tensor = state["offset"]
        if offset_tensor.numel() > 1 and not torch.all(offset_tensor == offset_tensor.view(-1)[0]):
            raise ValueError("Windowed cache offset must be identical across batch.")
        offset = int(offset_tensor.view(-1)[0].item())
        start = int(state["start"].view(-1)[0].item())
        steps = k.shape[1]
        cache = state["cache"]
        if offset - start + steps > cache.shape[2]:
            # The first new query, at `offset`, reaches back to `offset - context + 1`.
            keep = max(start, offset - self.context + 1)
            kept = offset - keep
            if kept + steps > cache.shape[2]:
                # One call writes more steps than the cache holds: grow it.
                capacity = max(kept + steps, self.context + _WINDOW_HEADROOM)
                grown = torch.full(
                    (*cache.shape[:2], capacity, *cache.shape[3:]),
                    float("NaN"),
                    device=cache.device,
                    dtype=cache.dtype,
                )
                grown[:, :, :kept] = cache[:, :, keep - start : offset - start]
                state["cache"] = cache = grown
            else:
                cache[:, :, :kept] = cache[:, :, keep - start : offset - start].clone()
            state["start"].fill_(keep)
            start = keep
        index = offset - start
        cache[0, :, index : index + steps] = k
        cache[1, :, index : index + steps] = v
        k_attn = cache[0, :, : index + steps].permute(0, 2, 1, 3)
        v_attn = cache[1, :, : index + steps].permute(0, 2, 1, 3)
        pos_k = torch.arange(start, offset + steps, device=k_attn.device, dtype=torch.long)
        pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
        pad = state.get("pad")
        if pad is not None and bool((pad > 0).any()):
            # Same shift as the linear cache: padded slots get negative positions.
            pos_k = pos_k - pad[:, None]
            offset_tensor = offset_tensor - pad
        return k_attn, v_attn, pos_k, offset_tensor


class StreamingMultiheadAttention(StatefulModule):
    """Similar to `nn.MultiheadAttention` but with support for streaming.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        context (int, optional): Number of time steps the attention can access to.
            Can access `context` time steps into the past.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: RotaryEmbedding,
        context: int | None = None,
        attn_dim: int | None = None,
        num_kv_heads: int | None = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads
        self.context = context
        # Queries and the attention output are attn_dim wide (embed_dim by default). Keys and
        # values have num_kv_heads heads (num_heads by default), each shared by
        # num_heads // num_kv_heads query heads (grouped-query attention); the cache holds those.
        self.attn_dim = attn_dim or embed_dim
        self.num_kv_heads = num_kv_heads or num_heads
        assert self.attn_dim % num_heads == 0 and num_heads % self.num_kv_heads == 0
        self.dim_per_head = self.attn_dim // num_heads
        self._cache_backend = (
            _LinearKVCacheBackend(self.num_kv_heads, self.dim_per_head)
            if context is None
            else _WindowedKVCacheBackend(self.num_kv_heads, self.dim_per_head, context)
        )

        kv_dim = self.dim_per_head * self.num_kv_heads
        self.in_proj = nn.Linear(embed_dim, self.attn_dim + 2 * kv_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, embed_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        weight = self.in_proj.weight
        if isinstance(weight, torch.Tensor):
            device = weight.device
            dtype = weight.dtype
        else:
            # torch.ao dynamic-quantized Linear: weight is a method returning a
            # qint8 tensor, but activations stay float32 — that's what the cache holds.
            device = weight().device
            dtype = torch.float32
        return self._cache_backend.init_state(batch_size, sequence_length, device, dtype)

    def increment_step(self, state: dict[str, torch.Tensor], increment: int = 1):
        self._cache_backend.increment_step(state, increment)

    def forward(
        self,
        query: torch.Tensor,
        model_state: ModelState | None,
        attn_mask: torch.Tensor | None = None,
        attention_capture: SelectedAttentionCapture | None = None,
        layer_index: int | None = None,
    ) -> torch.Tensor:
        state = None if model_state is None else self.get_state(model_state)

        projected = self.in_proj(query)
        # Split (b, t, q + 2 * kv) into q (b, t, h, d) and k, v (b, t, h_kv, d)
        b, t, _ = projected.shape
        d = self.dim_per_head
        q = projected[..., : self.attn_dim].view(b, t, self.num_heads, d)
        kv = projected[..., self.attn_dim :].view(b, t, 2, self.num_kv_heads, d)
        k, v = torch.unbind(kv, dim=2)
        rope_offset = self._cache_backend.rope_offset(state, b, q.device)
        q, k = self.rope(q, k, offset=rope_offset)
        q = q.transpose(1, 2)

        k_attn, v_attn, pos_k, offset = self._cache_backend.append_and_get(k, v, state)
        if self.num_kv_heads != self.num_heads:
            k_attn = k_attn.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v_attn = v_attn.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        if attn_mask is not None:
            pass  # precomputed by the caller (stateless path)
        elif state is None:
            attn_mask = _cached_causal_mask(t, self.context, q.device)
        else:
            pos_q = offset.view(-1, 1) + torch.arange(t, device=q.device, dtype=torch.long).view(
                1, -1
            )
            attn_mask = _build_attention_mask(pos_q, pos_k, self.context)

        x = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask, dropout_p=0.0)
        x = x.transpose(1, 2)
        # Reshape from (b, t, h, d) to (b, t, h*d)
        b, t, h, d = x.shape
        x = x.reshape(b, t, h * d)
        x = self.out_proj(x)

        if attention_capture is not None and layer_index is not None:
            attention_capture.capture_attention(layer_index, q, k_attn)

        return x
