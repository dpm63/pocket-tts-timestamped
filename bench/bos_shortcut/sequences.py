"""FlowLM stream assembly on top of a frozen pocket_tts TTSModel.

- teacher rollouts: batched streaming generation from [BOS_BEFORE_VOICE, VOICE, TEXT]
  prefixes, recording every step's backbone output z, noise and latent
- student forward: one stateless pass over [prefix, (SOFT), BOS, first latent]
  with the steer vector applied, returning z at BOS and at the next position
"""

import contextlib
import math
from functools import partial

import torch
import torch.nn.functional as F
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.models.flow_lm import lsd_decode
from pocket_tts.modules.stateful_module import increment_steps, init_states
from pocket_tts.utils.utils import download_if_necessary

from .sites import AUDIO, BOS, BOS_BEFORE_VOICE, PAD, SOFT, TEXT, VOICE, Steer

try:  # pocket-tts >= 3.1 (needed for the newer language models, e.g. french 6L)
    from pocket_tts.models.text_chunking import prepare_text_prompt

    TokenizedText = None
except ImportError:  # pocket-tts 3.0
    from pocket_tts.conditioners.base import TokenizedText
    from pocket_tts.models.tts_model import prepare_text_prompt

FRAME_SAMPLES = 1920  # 80 ms Mimi frame at 24 kHz


@torch.no_grad()
def encode_voice(model, source: str, max_frames: int | None = None) -> torch.Tensor:
    """Raw Mimi latents [Tv, C] of a voice prompt, as get_state_for_audio_prompt computes them."""
    path = download_if_necessary(source) if isinstance(source, str) and "://" in source else source
    audio, sr = audio_read(path)
    audio = convert_audio(audio, sr, model.config.mimi.sample_rate, 1)
    if max_frames is not None:
        audio = audio[..., : max_frames * FRAME_SAMPLES]
    latents = model.mimi.encode_to_latent(audio.unsqueeze(0).to(model.device))
    return latents[0].float()


def prepare_text(model, text: str) -> tuple[str, int]:
    """(prompt text, frames-after-EOS guess) exactly as TTSModel prepares it."""
    options = {}
    if hasattr(model, "replace_characters"):  # pocket-tts >= 3.1 language options
        options = dict(append_terminal_punctuation=model.append_terminal_punctuation,
                       capitalize_first_letter=model.capitalize_first_letter, replace_characters=model.replace_characters)  # fmt: skip
    return prepare_text_prompt(
        text, model.pad_with_spaces_for_short_inputs, model.remove_semicolons, **options
    )


def text_tokens(model, prepared_text: str) -> torch.Tensor:
    """Token ids [1, L] of an already prepared prompt."""
    out = model.flow_lm.conditioner.prepare(prepared_text)
    return out if TokenizedText is None else out.tokens


def embed_tokens(model, tokens: torch.Tensor) -> torch.Tensor:
    """LUT embeddings [B, L, D] of token ids [B, L]."""
    conditioner = model.flow_lm.conditioner
    return conditioner(tokens) if TokenizedText is None else conditioner(TokenizedText(tokens))


def tokenize(model, text: str) -> torch.Tensor:
    return text_tokens(model, prepare_text(model, text)[0])[0]


def prefix_embeddings(
    model, voice_latents: torch.Tensor, tokens: torch.Tensor
) -> tuple[torch.Tensor, list[int]]:
    """[BOS_BEFORE_VOICE, VOICE x Tv, TEXT x L] embeddings and kinds of one row."""
    fl = model.flow_lm
    voice = F.linear(voice_latents.float(), fl.speaker_proj_weight)
    text = embed_tokens(model, tokens[None].to(voice.device))[0]
    parts, kinds = [voice, text], [VOICE] * voice.shape[0] + [TEXT] * text.shape[0]
    if fl.insert_bos_before_voice:
        parts.insert(0, fl.bos_before_voice[0])
        kinds.insert(0, BOS_BEFORE_VOICE)
    return torch.cat(parts, dim=0), kinds


def _scrub_nan(module, args, out):
    # Left-padded positions attend to nothing and come out NaN; zero them so
    # their (masked) keys and values stay finite in the next layers.
    return out.nan_to_num(0.0)


def _streaming_state(
    model, batch: int, width: int, extra_positions: int, pad: torch.Tensor
) -> dict:
    state = init_states(model.flow_lm, batch, width + extra_positions)
    for module_state in state.values():
        if "pad" in module_state:
            module_state["pad"].copy_(pad)
    return state


@torch.no_grad()
def prefill(
    model, prefixes: list[torch.Tensor], extra_positions: int
) -> tuple[dict, torch.Tensor, int]:
    """Run right-aligned prefixes through the streaming FlowLM.

    Returns (state with room for `extra_positions` more steps, per-row left
    padding, width). The left padding is masked via the state's per-row `pad`.
    """
    fl = model.flow_lm
    device = fl.bos_emb.device
    B, width = len(prefixes), max(p.shape[0] for p in prefixes)
    x = torch.zeros(B, width, fl.dim, device=device)
    pad = torch.tensor([width - p.shape[0] for p in prefixes], device=device)
    for b, p in enumerate(prefixes):
        x[b, width - p.shape[0] :] = p
    state = _streaming_state(model, B, width, extra_positions, pad)
    handles = [layer.register_forward_hook(_scrub_nan) for layer in fl.transformer.layers]
    try:
        fl.transformer(x, state)
    finally:
        for h in handles:
            h.remove()
    increment_steps(fl, state, width)
    return state, pad, width


def _shift_keys(model, keys: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Rotate cached keys [B, P, H, D] by shifts[b] positions (RoPE is a pure rotation).

    Keys are stored after RoPE, so they encode their absolute position in the
    batch they were computed in, which depends on that batch's left padding.
    Same frequencies as pocket_tts.modules.rope.apply_rope.
    """
    D = keys.shape[-1]
    ds = torch.arange(D // 2, device=keys.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(model.flow_lm.transformer.rope.max_period) * 2 / D))
    angle = shifts.to(torch.float32).view(-1, 1, 1, 1) * freqs  # [B, 1, 1, D/2]
    k = keys.view(*keys.shape[:-1], D // 2, 2).float()
    kr, ki = k[..., 0], k[..., 1]
    cos, sin = torch.cos(angle), torch.sin(angle)
    return (
        torch.stack([kr * cos - ki * sin, kr * sin + ki * cos], dim=-1)
        .view(keys.shape)
        .to(keys.dtype)
    )


def row_prefix_kv(
    model, state: dict, b: int, pad: torch.Tensor, width: int
) -> dict[str, torch.Tensor]:
    """One row's prefix keys/values {attention module: [2, P, H, D]} from a prefilled state,
    with keys moved to the row's unpadded positions 0..P-1."""
    out = {}
    for name, st in state.items():
        if "cache" in st:
            kv = st["cache"][:, b, int(pad[b]) : width].clone()
            kv[0] = _shift_keys(model, kv[0][None], -pad[b : b + 1])[0]
            out[name] = kv
    return out


@torch.no_grad()
def prefix_kv(model, prefixes: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    state, pad, width = prefill(model, prefixes, 0)
    return [row_prefix_kv(model, state, b, pad, width) for b in range(len(prefixes))]


@torch.no_grad()
def rollout(
    model,
    prefixes: list[torch.Tensor],
    n_frames: int,
    generator: torch.Generator | None = None,
    keep_prefix_kv: bool = False,
) -> dict:
    """Batched free-running generation of `n_frames` latents (EOS ignored).

    Returns per step: z (out_norm output) [B,T,D], eps (flow noise) [B,T,C],
    latent (normalized, as fed back) [B,T,C], eos_logit [B,T]; with
    keep_prefix_kv also each row's prefix keys/values (for cached training).
    """
    fl = model.flow_lm
    device = fl.bos_emb.device
    B = len(prefixes)
    state, pad, width = prefill(model, prefixes, n_frames)
    out = (
        {"prefix_kv": [row_prefix_kv(model, state, b, pad, width) for b in range(B)]}
        if keep_prefix_kv
        else {}
    )
    inp = fl.bos_emb.view(1, 1, -1).expand(B, 1, -1)
    zs, epss, lats, eos = [], [], [], []
    for _ in range(n_frames):
        z = fl.out_norm(fl.transformer(fl.input_linear(inp), state))[:, -1].float()
        increment_steps(fl, state, 1)
        eps = torch.randn(B, fl.ldim, device=device, generator=generator) * model.temp**0.5
        lat = lsd_decode(partial(fl.flow_net, z), eps, model.sampler_decode_steps)
        zs.append(z)
        epss.append(eps)
        lats.append(lat)
        eos.append(fl.out_eos(z)[:, 0])
        inp = lat[:, None].to(fl.dtype)
    out.update(
        z=torch.stack(zs, 1),
        eps=torch.stack(epss, 1),
        latent=torch.stack(lats, 1),
        eos_logit=torch.stack(eos, 1),
    )
    return out


@torch.no_grad()
def decode_latents(model, latents: torch.Tensor) -> torch.Tensor:
    """Normalized latents [B, T, C] -> PCM [B, T * FRAME_SAMPLES]."""
    fl, mimi = model.flow_lm, model.mimi
    steps_per_latent = int(mimi.encoder_frame_rate / mimi.frame_rate)
    mimi_state = init_states(
        mimi, batch_size=latents.shape[0], sequence_length=latents.shape[1] * steps_per_latent
    )
    pcm = mimi.decode_from_latent(latents * fl.emb_std + fl.emb_mean, mimi_state)
    return pcm[:, 0, : latents.shape[1] * FRAME_SAMPLES]


@contextlib.contextmanager
def _steer_applied(steer: Steer | None, flow_lm, kinds: torch.Tensor):
    """Set the steer's position kinds for one forward; attach its hooks unless
    the caller (training) keeps them attached."""
    if steer is None:
        yield
        return
    attach_here = not steer.attached
    if attach_here:
        steer.attach(flow_lm)
    steer.kinds = kinds
    try:
        yield
    finally:
        steer.kinds = None
        if attach_here:
            steer.detach_hooks()


def student_forward_cached(
    model,
    steer: Steer | None,
    kv_rows: list[dict[str, torch.Tensor]],
    first_latent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """student_forward for sites that only act from BOS on, over cached prefix keys/values.

    Only BOS (and `first_latent`, if given) go through the network, so a step
    costs one or two positions per row instead of the whole prefix. Equal to
    student_forward for those sites (the prefix does not depend on the steer).
    """
    fl = model.flow_lm
    device = fl.bos_emb.device
    assert steer is None or steer.site.prefix_independent, steer.site
    lengths = [next(iter(kv.values())).shape[1] for kv in kv_rows]
    B, width = len(kv_rows), max(lengths)
    pad = torch.tensor([width - n for n in lengths], device=device)
    state = _streaming_state(model, B, width, 2, pad)
    for name, st in state.items():
        if "cache" in st:
            st["cache"][:, :, :width] = 0.0  # padded slots are masked; keep them finite
            for b, kv in enumerate(kv_rows):
                st["cache"][:, b, width - lengths[b] : width] = kv[name]
            # Back from unpadded positions to this batch's padded ones.
            st["cache"][0, :, :width] = _shift_keys(model, st["cache"][0, :, :width], pad)
    increment_steps(fl, state, width)
    x = fl.input_linear(fl.bos_emb).view(1, 1, -1).expand(B, 1, -1)
    kinds = [BOS]
    if first_latent is not None:
        x = torch.cat([x, fl.input_linear(first_latent.to(fl.dtype))[:, None]], dim=1)
        kinds.append(AUDIO)
    kind_t = torch.tensor(kinds, device=device)[None].expand(B, -1)
    with _steer_applied(steer, fl, kind_t):
        h = fl.out_norm(fl.transformer(x, state)).float()
    return h[:, 0], (h[:, 1] if first_latent is not None else None)


def student_forward(
    model,
    steer: Steer | None,
    prefixes: list[torch.Tensor],
    prefix_kinds: list[list[int]],
    first_latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """z at BOS and at the following position (fed `first_latent`) for each row.

    Rows are [prefix, (SOFT), BOS, first_latent], right-padded; the stateless
    causal transformer never lets padding reach valid positions.
    """
    fl = model.flow_lm
    device = fl.bos_emb.device
    bos_in = fl.input_linear(fl.bos_emb)
    next_in = fl.input_linear(first_latent.to(fl.dtype))
    soft = steer.vector() if steer is not None and steer.site.inserts else None
    rows, kinds = [], []
    for b, (p, k) in enumerate(zip(prefixes, prefix_kinds, strict=True)):
        tail, tail_kinds = [bos_in[None], next_in[b][None]], [BOS, AUDIO]
        if soft is not None:
            tail.insert(0, soft[None])
            tail_kinds.insert(0, SOFT)
        rows.append(torch.cat([p, *tail], dim=0))
        kinds.append(k + tail_kinds)
    lengths = torch.tensor([r.shape[0] for r in rows], device=device)
    width = int(lengths.max())
    x = torch.zeros(len(rows), width, fl.dim, device=device)
    kind_t = torch.full((len(rows), width), PAD, device=device, dtype=torch.long)
    for b, (r, k) in enumerate(zip(rows, kinds, strict=True)):
        x[b, : r.shape[0]] = r
        kind_t[b, : r.shape[0]] = torch.tensor(k, device=device)
    with _steer_applied(steer, fl, kind_t):
        h = fl.out_norm(fl.transformer(x, None)).float()
    rows_idx = torch.arange(len(rows), device=device)
    return h[rows_idx, lengths - 2], h[rows_idx, lengths - 1]


def sample_latent(model, z: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """The latent the flow head produces from z with a given noise draw."""
    return lsd_decode(partial(model.flow_lm.flow_net, z), eps, model.sampler_decode_steps)
