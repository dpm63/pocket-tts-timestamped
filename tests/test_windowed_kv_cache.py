"""The windowed KV cache of finite-context attention matches the linear cache."""

import pytest
import torch

from pocket_tts_timestamped.modules.attention import (
    _WINDOW_HEADROOM,
    StreamingMultiheadAttention,
    _LinearKVCacheBackend,
    _WindowedKVCacheBackend,
)
from pocket_tts_timestamped.modules.stateful_module import (
    ModelState,
    StatefulModule,
    increment_steps,
    init_states,
)
from pocket_tts_timestamped.modules.transformer import StreamingTransformer

CONTEXT = 8
CAPACITY = CONTEXT + _WINDOW_HEADROOM


def build(linear: bool) -> StreamingTransformer:
    """A small finite-context transformer; `linear` swaps in the linear cache as reference."""
    torch.manual_seed(0)
    transformer = StreamingTransformer(
        d_model=32, num_heads=4, num_layers=2, dim_feedforward=64, context=CONTEXT
    )
    for name, module in transformer.named_modules():
        if isinstance(module, StatefulModule):
            module._module_absolute_name = name
        if isinstance(module, StreamingMultiheadAttention):
            assert isinstance(module._cache_backend, _WindowedKVCacheBackend)
            if linear:
                module._cache_backend = _LinearKVCacheBackend(
                    module.num_kv_heads, module.dim_per_head
                )
    return transformer.eval()


@torch.no_grad()
def run(
    transformer: StreamingTransformer, x: torch.Tensor, call_sizes: list[int], pad: int = 0
) -> tuple[torch.Tensor, ModelState]:
    """Stream `x` through `transformer` in calls of `call_sizes` steps."""
    assert sum(call_sizes) == x.shape[1]
    state = init_states(transformer, x.shape[0], x.shape[1])
    for module_state in state.values():
        module_state["pad"].fill_(pad)
    outputs, position = [], 0
    for size in call_sizes:
        outputs.append(transformer(x[:, position : position + size], state))
        increment_steps(transformer, state, size)
        position += size
    return torch.cat(outputs, dim=1), state


@pytest.mark.parametrize(
    "call_sizes",
    [
        [1] * (3 * CAPACITY),  # compacts several times
        [3 * CAPACITY],  # one call larger than the cache: grows on the first call
        [5, 1, 1, CAPACITY + 7, 2, 1, 40, 1, 1, CAPACITY // 2, 3],  # compaction and growth mixed
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_windowed_cache_matches_linear(call_sizes: list[int], batch_size: int):
    torch.manual_seed(1)
    x = torch.randn(batch_size, sum(call_sizes), 32)
    expected, _ = run(build(linear=True), x, call_sizes)
    actual, _ = run(build(linear=False), x, call_sizes)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_windowed_cache_matches_linear_with_padding():
    """Right-aligned padding keeps its meaning after the padded slots are dropped."""
    torch.manual_seed(1)
    pad = 6
    x = torch.cat([torch.zeros(1, pad, 32), torch.randn(1, 2 * CAPACITY, 32)], dim=1)
    call_sizes = [pad + 4] + [1] * (2 * CAPACITY - 4)
    expected, _ = run(build(linear=True), x, call_sizes, pad=pad)
    actual, _ = run(build(linear=False), x, call_sizes, pad=pad)
    torch.testing.assert_close(actual[:, pad:], expected[:, pad:], atol=1e-5, rtol=1e-5)


def test_windowed_cache_shrinks_back_after_a_large_call():
    """A batched decode grows the cache once; later single steps must not attend over it."""
    torch.manual_seed(1)
    call_sizes = [3 * CAPACITY] + [1] * 5
    x = torch.randn(1, sum(call_sizes), 32)
    expected, _ = run(build(linear=True), x, call_sizes)
    actual, state = run(build(linear=False), x, call_sizes)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for module_state in state.values():
        assert module_state["cache"].shape[2] == CAPACITY


def test_windowed_cache_stays_bounded():
    torch.manual_seed(1)
    x = torch.randn(1, 10 * CAPACITY, 32)
    _, state = run(build(linear=False), x, [1] * x.shape[1])
    for module_state in state.values():
        assert module_state["cache"].shape[2] <= CAPACITY
        assert int(module_state["offset"]) - int(module_state["start"]) <= CAPACITY
