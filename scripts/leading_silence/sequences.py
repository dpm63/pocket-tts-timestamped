"""FlowLM sequences for training: batched teacher rollouts and the cached student forward.

A row is [BOS_BEFORE_VOICE][VOICE x Tv][TEXT x L][BOS][AUDIO ...]. Rows of different lengths
are right-aligned (left-padded) and the padding is masked through each attention state's
`pad`. The prefix keys/values of each row are kept at unpadded positions, so training runs
only BOS and the teacher-forced frames over them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from pocket_tts_timestamped import TTSModel
from pocket_tts_timestamped.data.audio import audio_read
from pocket_tts_timestamped.data.audio_utils import convert_audio, end_on_pause
from pocket_tts_timestamped.models import flow_lm as flow_lm_module
from pocket_tts_timestamped.models.flow_lm import FlowLMModel
from pocket_tts_timestamped.models.text_chunking import prepare_text_prompt
from pocket_tts_timestamped.modules.stateful_module import ModelState, increment_steps, init_states
from pocket_tts_timestamped.utils.utils import download_if_necessary
from scripts.leading_silence.steer import AUDIO, BOS, Steer

FRAME_SAMPLES = 1920  # 80 ms at 24 kHz
# Flow head type -> its sampler in pocket_tts_timestamped.models.flow_lm.
_DECODERS = {
    "lsd": "lsd_decode",
    "flow_matching": "ot_decode",
    "drifting": "drifting_decode",
    "pointwise": "pointwise_decode",
}
KeyValues = dict[str, torch.Tensor]


def head_z(flow_lm: FlowLMModel, hidden: torch.Tensor) -> torch.Tensor:
    """The flow and EOS heads' input ("z"): out_norm, then z_proj where the model has one."""
    z = flow_lm.out_norm(hidden)
    return z if flow_lm.z_proj is None else flow_lm.z_proj(z)


def sample_latent(model: TTSModel, z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """The latent the flow head makes from z with a given noise draw (the model's sampler)."""
    decode = getattr(flow_lm_module, _DECODERS[model.flow_lm.flow_type])
    return decode(partial(model.flow_lm.flow_net, z), noise, model.sampler_decode_steps)


@torch.no_grad()
def encode_voice(model: TTSModel, source: str | Path) -> torch.Tensor:
    """Raw Mimi latents [Tv, C] of a voice prompt, as `get_state_for_audio_prompt` makes them."""
    path = download_if_necessary(str(source)) if "://" in str(source) else Path(source)
    audio, sample_rate = audio_read(path)
    audio = convert_audio(audio, sample_rate, model.config.mimi.sample_rate, 1)
    audio = end_on_pause(audio, model.config.mimi.sample_rate)
    return model.mimi.encode_to_latent(audio.unsqueeze(0).to(model.device))[0].float()


def frames_after_eos(model: TTSModel, text: str) -> int:
    """Frames the model generates after EOS for `text` (its guess + 2)."""
    return _prepare(model, text)[1] + 2


def tokenize(model: TTSModel, text: str) -> torch.Tensor:
    """Token ids [L] of `text`, prepared as the model prepares a whole text."""
    return model.flow_lm.conditioner.prepare(_prepare(model, text)[0])[0]


def _prepare(model: TTSModel, text: str) -> tuple[str, int]:
    return prepare_text_prompt(
        text,
        model.pad_with_spaces_for_short_inputs,
        model.remove_semicolons,
        model.append_terminal_punctuation,
        model.capitalize_first_letter,
        model.replace_characters,
    )


def prefix_embeddings(model: TTSModel, voice: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """[BOS_BEFORE_VOICE, VOICE x Tv, TEXT x L] input embeddings of one row."""
    flow_lm = model.flow_lm
    parts = [
        F.linear(voice.float(), flow_lm.speaker_proj_weight),
        flow_lm.conditioner(tokens[None].to(voice.device))[0],
    ]
    if flow_lm.insert_bos_before_voice:
        parts.insert(0, flow_lm.bos_before_voice[0])
    return torch.cat(parts, dim=0)


def _scrub_nan(module: torch.nn.Module, args: object, output: torch.Tensor) -> torch.Tensor:
    # Left-padded positions attend to nothing and come out NaN; zeroing them keeps their
    # (masked) keys and values finite in later layers.
    return output.nan_to_num(0.0)


def _streaming_state(
    model: TTSModel, batch: int, width: int, extra: int, pad: torch.Tensor
) -> ModelState:
    state = init_states(model.flow_lm, batch, width + extra)
    for module_state in state.values():
        if "pad" in module_state:
            module_state["pad"].copy_(pad)
    return state


@torch.no_grad()
def prefill(
    model: TTSModel, prefixes: list[torch.Tensor], extra: int
) -> tuple[ModelState, torch.Tensor, int]:
    """Run right-aligned prefixes through the streaming FlowLM: (state with room for `extra`
    more steps, per-row left padding, width)."""
    flow_lm = model.flow_lm
    device = flow_lm.bos_emb.device
    batch, width = len(prefixes), max(prefix.shape[0] for prefix in prefixes)
    x = torch.zeros(batch, width, flow_lm.dim, device=device)
    pad = torch.tensor([width - prefix.shape[0] for prefix in prefixes], device=device)
    for row, prefix in enumerate(prefixes):
        x[row, width - prefix.shape[0] :] = prefix
    state = _streaming_state(model, batch, width, extra, pad)
    handles = [layer.register_forward_hook(_scrub_nan) for layer in flow_lm.transformer.layers]
    try:
        flow_lm.transformer(x, state)
    finally:
        for handle in handles:
            handle.remove()
    increment_steps(flow_lm, state, width)
    return state, pad, width


def _shift_keys(model: TTSModel, keys: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Rotate cached keys [B, P, H, D] by `shifts[b]` positions. Keys are stored after RoPE,
    which is a pure rotation, so this moves them between padded and unpadded positions."""
    dim = keys.shape[-1]
    steps = torch.arange(dim // 2, device=keys.device, dtype=torch.float32)
    max_period = model.flow_lm.transformer.rope.max_period
    frequencies = torch.exp(steps * (-math.log(max_period) * 2 / dim))
    angle = shifts.to(torch.float32).view(-1, 1, 1, 1) * frequencies
    pairs = keys.view(*keys.shape[:-1], dim // 2, 2).float()
    real, imaginary = pairs[..., 0], pairs[..., 1]
    cos, sin = torch.cos(angle), torch.sin(angle)
    rotated = torch.stack([real * cos - imaginary * sin, real * sin + imaginary * cos], dim=-1)
    return rotated.view(keys.shape).to(keys.dtype)


def row_key_values(
    model: TTSModel, state: ModelState, row: int, pad: torch.Tensor, width: int
) -> KeyValues:
    """One row's prefix keys/values {module: [2, P, H, D]} at unpadded positions."""
    out = {}
    for name, module_state in state.items():
        if "cache" in module_state:
            kv = module_state["cache"][:, row, int(pad[row]) : width].clone()
            kv[0] = _shift_keys(model, kv[0][None], -pad[row : row + 1])[0]
            out[name] = kv
    return out


@dataclass
class Rollout:
    prefix_kv: list[KeyValues]  # per row
    z: torch.Tensor  # [B, T, Dz] head input per step
    latent: torch.Tensor  # [B, T, C] normalized, as fed back
    eos_logit: torch.Tensor  # [B, T]


@torch.no_grad()
def rollout(
    model: TTSModel, prefixes: list[torch.Tensor], frames: int, generator: torch.Generator
) -> Rollout:
    """Batched free-running generation of `frames` latents from BOS (EOS ignored).

    Per step: z (head input) [B, T, Dz], latent (normalized, as fed back) [B, T, C] and EOS
    logit [B, T]; plus each row's prefix keys/values.
    """
    flow_lm = model.flow_lm
    batch = len(prefixes)
    state, pad, width = prefill(model, prefixes, frames)
    key_values = [row_key_values(model, state, row, pad, width) for row in range(batch)]
    inputs = flow_lm.bos_emb.view(1, 1, -1).expand(batch, 1, -1)
    zs, latents, eos = [], [], []
    for _ in range(frames):
        z = head_z(flow_lm, flow_lm.transformer(flow_lm.input_linear(inputs), state))[:, -1]
        z = z.float()
        increment_steps(flow_lm, state, 1)
        noise = (
            torch.randn(batch, flow_lm.ldim, device=z.device, generator=generator) * model.temp**0.5
        )
        latent = sample_latent(model, z, noise)
        zs.append(z)
        latents.append(latent)
        eos.append(flow_lm.out_eos(z)[:, 0])
        inputs = latent[:, None].to(flow_lm.dtype)
    return Rollout(key_values, torch.stack(zs, 1), torch.stack(latents, 1), torch.stack(eos, 1))


@torch.no_grad()
def decode_latents(model: TTSModel, latents: torch.Tensor) -> torch.Tensor:
    """Normalized latents [B, T, C] -> PCM [B, T * FRAME_SAMPLES], from a cold decoder."""
    flow_lm, mimi = model.flow_lm, model.mimi
    steps = int(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, batch_size=latents.shape[0], sequence_length=latents.shape[1] * steps)
    pcm = mimi.decode_from_latent(latents * flow_lm.emb_std + flow_lm.emb_mean, state)
    return pcm[:, 0, : latents.shape[1] * FRAME_SAMPLES]


def student_forward(
    model: TTSModel, steer: Steer, key_values: list[KeyValues], after: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """z at BOS [B, Dz] and after each teacher-forced latent of `after` [B, m, C] ([B, m, Dz]),
    over the rows' cached prefix keys/values. The steer must be attached."""
    flow_lm = model.flow_lm
    device = flow_lm.bos_emb.device
    lengths = [next(iter(kv.values())).shape[1] for kv in key_values]
    batch, width = len(key_values), max(lengths)
    pad = torch.tensor([width - length for length in lengths], device=device)
    state = _streaming_state(model, batch, width, 1 + after.shape[1], pad)
    for name, module_state in state.items():
        if "cache" in module_state:
            cache = module_state["cache"]
            cache[:, :, :width] = 0.0  # padded slots are masked; keep them finite
            for row, kv in enumerate(key_values):
                cache[:, row, width - lengths[row] : width] = kv[name]
            cache[0, :, :width] = _shift_keys(model, cache[0, :, :width], pad)
    increment_steps(flow_lm, state, width)
    bos = flow_lm.input_linear(flow_lm.bos_emb).view(1, 1, -1).expand(batch, 1, -1)
    x = torch.cat([bos, flow_lm.input_linear(after.to(flow_lm.dtype))], dim=1)
    kinds = torch.tensor([BOS] + [AUDIO] * after.shape[1], device=device)
    steer.kinds = kinds[None].expand(batch, -1)
    try:
        z = head_z(flow_lm, flow_lm.transformer(x, state)).float()
    finally:
        steer.kinds = None
    return z[:, 0], z[:, 1:]
