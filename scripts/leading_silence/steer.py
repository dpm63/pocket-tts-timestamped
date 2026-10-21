"""The trainable form of the leading-silence vectors: hooks on a frozen FlowLM.

One vector per residual-stream index i = 0..num_layers, added to the input of layer i (and,
for i = num_layers, to the transformer output before `out_norm`) at positions marked BOS.
`TTSModel.load_model(skip_leading_silence=...)` applies the exported vectors natively.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from pocket_tts_timestamped.models.flow_lm import FlowLMModel

# Position kinds of a training sequence; only BOS positions are steered.
OTHER, BOS, AUDIO = 0, 1, 2


class Steer(nn.Module):
    """theta is [num_layers + 1, dim], each row in units of `scale` (the RMS of the
    activations it is added to), so one learning rate fits every layer."""

    scale: torch.Tensor

    def __init__(self, flow_lm: FlowLMModel, scale: list[float]) -> None:
        super().__init__()
        count = len(flow_lm.transformer.layers) + 1
        if len(scale) != count:
            raise ValueError(f"Expected {count} scales, got {len(scale)}")
        self.theta = nn.Parameter(torch.zeros(count, flow_lm.dim))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        # Position kinds [B, S] of the current transformer call; None steers nothing.
        self.kinds: torch.Tensor | None = None
        self._handles: list[RemovableHandle] = []

    def vectors(self) -> torch.Tensor:
        """The vectors as added, [num_layers + 1, dim]: the exported form."""
        return self.theta * self.scale[:, None]

    def add(self, x: torch.Tensor, index: int) -> torch.Tensor:
        if self.kinds is None:
            return x
        mask = (self.kinds == BOS).to(x.dtype)[..., None]
        return x + mask * (self.theta[index] * self.scale[index]).to(x.dtype)

    def attach(self, flow_lm: FlowLMModel) -> None:
        layers = flow_lm.transformer.layers
        for index, layer in enumerate(layers):
            self._handles.append(
                layer.register_forward_pre_hook(
                    lambda module, args, index=index: (self.add(args[0], index), *args[1:])
                )
            )
        self._handles.append(
            flow_lm.transformer.register_forward_hook(
                lambda module, args, output: self.add(output, len(layers))
            )
        )

    def detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
