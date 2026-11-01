"""Teacher rollouts and training (the bos_shortcut recipe, handoff section 2, plus the
whole-sequence rows and EOS loss of its section 7).

At BOS the student (frozen model + vectors) should produce the z the teacher had at its
speech onset k, and then continue like the teacher did:

- z-MSE at BOS against the teacher's z at k;
- z-MSE over the next `TRAJECTORY` frames, teacher-forced with the teacher's latents;
- z-MSE over a few whole sequences per step (onset to past EOS, text without its final
  punctuation), so frames long after BOS aren't pushed off course;
- an EOS loss on every teacher-forced frame, so the EOS head decides as the teacher's did.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pocket_tts_timestamped import TTSModel
from pocket_tts_timestamped.models.flow_lm import FlowLMModel
from scripts.leading_silence.onsets import frame_db, speech_start
from scripts.leading_silence.sequences import (
    PrefixKV,
    VoiceKeyValues,
    decode_latents,
    frames_after_eos,
    prefix_embeddings,
    rollout,
    student_forward,
    tokenize,
)
from scripts.leading_silence.steer import BOS, Steer

ROLLOUT_FRAMES = 32  # onsets are looked for in the first 32 frames
TRAJECTORY = 4


@dataclass(frozen=True)
class Setting:
    rows: int
    lr: float
    steps: int
    batch: int = 64
    # Whole-sequence rows per step, and the EOS loss weight.
    full_rows: int = 7
    eos_weight: float = 1.0
    full_pairs: int = 256
    full_max_frames: int = 200
    # Rows per whole-sequence rollout batch; 24-layer rollouts to 200 frames need smaller ones.
    full_batch: int = 32


def default_setting(num_layers: int) -> Setting:
    """One setting per model size (adopted 2026-10-21); 6L also covers the 8-layer models."""
    if num_layers >= 24:
        return Setting(rows=128, lr=0.1, steps=50, full_batch=16)
    return Setting(rows=256, lr=0.3, steps=100)


@dataclass
class Rows:
    """Teacher rollouts that reached speech at frame k >= 1."""

    voice: list[int]
    text: list[str]
    k: list[int]
    prefix_kv: list[PrefixKV]
    z_onset: torch.Tensor  # [N, Dz] the teacher's z at k
    eos_onset: torch.Tensor  # [N] its EOS logit there
    latents: torch.Tensor  # [N, m, C] the teacher's latents k..k+m-1
    z_after: torch.Tensor  # [N, m, Dz] the z after each
    eos_after: torch.Tensor  # [N, m]


@dataclass
class FullRows:
    """Teacher rollouts from the onset to the end of the EOS tail, one length per row."""

    k: list[int]
    prefix_kv: list[PrefixKV]
    latents: list[torch.Tensor]  # [L, C] each
    z_after: list[torch.Tensor]  # [L, Dz]
    eos_after: list[torch.Tensor]  # [L]


Pairs = list[tuple[int, str]]
Log = Callable[[str], None]


def _batches(
    model: TTSModel, voices: dict[int, torch.Tensor], pairs: Pairs, size: int
) -> list[tuple[Pairs, list[torch.Tensor]]]:
    out = []
    for start in range(0, len(pairs), size):
        chunk = pairs[start : start + size]
        prefixes = [prefix_embeddings(model, voices[v], tokenize(model, text)) for v, text in chunk]
        out.append((chunk, prefixes))
    return out


@torch.no_grad()
def make_rows(
    model: TTSModel,
    voices: dict[int, torch.Tensor],
    pairs: Pairs,
    wanted: int,
    batch: int,
    seed: int,
    store: VoiceKeyValues,
    log: Log = print,
) -> Rows:
    """Roll out `pairs` in order until `wanted` rows have an onset at frame 1 or later. Rows
    already speaking at frame 0 are left out: they are usually abrupt prompt continuations
    and would teach the vectors to keep them. Prefix keys/values are kept through `store`."""
    generator = torch.Generator(device=model.flow_lm.bos_emb.device).manual_seed(seed)
    voice, text, onsets, key_values, tensors = [], [], [], [], []
    seen = 0
    for chunk, prefixes in _batches(model, voices, pairs, batch):
        result = rollout(model, prefixes, ROLLOUT_FRAMES, generator)
        pcm = decode_latents(model, result.latent).cpu().numpy()
        for row, (voice_index, row_text) in enumerate(chunk):
            k = speech_start(frame_db(pcm[row]))
            if k < 1 or k + TRAJECTORY >= ROLLOUT_FRAMES:
                continue
            voice.append(voice_index)
            text.append(row_text)
            onsets.append(k)
            key_values.append(
                store.split(voice_index, voices[voice_index].shape[0], result.prefix_kv[row])
            )
            after = slice(k + 1, k + 1 + TRAJECTORY)
            tensors.append(
                (
                    result.z[row, k],
                    result.eos_logit[row, k],
                    result.latent[row, k : k + TRAJECTORY],
                    result.z[row, after],
                    result.eos_logit[row, after],
                )
            )
        seen += len(chunk)
        log(f"rollouts: {seen} done, {len(onsets)} rows kept")
        if len(onsets) >= wanted:
            break
    if len(onsets) < wanted:
        raise RuntimeError(f"Only {len(onsets)} of {wanted} rollouts reached speech")
    z_onset, eos_onset, latents, z_after, eos_after = (
        torch.stack(column) for column in zip(*tensors[:wanted], strict=True)
    )
    return Rows(
        voice[:wanted],
        text[:wanted],
        onsets[:wanted],
        key_values[:wanted],
        z_onset,
        eos_onset,
        latents,
        z_after,
        eos_after,
    )


@torch.no_grad()
def make_full_rows(
    model: TTSModel,
    voices: dict[int, torch.Tensor],
    pairs: Pairs,
    batch: int,
    max_frames: int,
    seed: int,
    store: VoiceKeyValues,
    log: Log = print,
) -> FullRows:
    """Rollouts to past EOS. The tail is the text's frames-after-EOS after the first EOS past
    the onset; rows without an onset in the first frames, or not done by `max_frames`, drop.
    Callers pass texts without terminal punctuation and the model set not to add one."""
    generator = torch.Generator(device=model.flow_lm.bos_emb.device).manual_seed(seed)
    out = FullRows([], [], [], [], [])
    for chunk, prefixes in _batches(model, voices, pairs, batch):
        result = rollout(model, prefixes, max_frames, generator)
        pcm = decode_latents(model, result.latent[:, :ROLLOUT_FRAMES]).cpu().numpy()
        for row, (voice_index, text) in enumerate(chunk):
            k = speech_start(frame_db(pcm[row]))
            fired = (result.eos_logit[row] > model.eos_threshold).nonzero()[:, 0]
            fired = fired[fired > k]
            if k < 1 or not len(fired):
                continue
            end = int(fired[0]) + frames_after_eos(model, text)
            if end >= max_frames:
                continue
            out.k.append(k)
            out.prefix_kv.append(
                store.split(voice_index, voices[voice_index].shape[0], result.prefix_kv[row])
            )
            out.latents.append(result.latent[row, k:end])
            out.z_after.append(result.z[row, k + 1 : end + 1])
            out.eos_after.append(result.eos_logit[row, k + 1 : end + 1])
        log(f"full rollouts: {len(out.k)} rows kept")
    return out


def eos_loss(
    model: TTSModel, z: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Cross-entropy between the student's and the teacher's EOS decisions, as probabilities
    centred on the threshold: far-negative logits barely count, crossings do."""
    head = model.flow_lm.out_eos
    student = head(z.to(head.weight.dtype))[..., 0].float() - model.eos_threshold
    target = torch.sigmoid(teacher.float() - model.eos_threshold)
    loss = F.binary_cross_entropy_with_logits(student, target, reduction="none")
    return loss.mean() if mask is None else (loss * mask).sum() / mask.sum()


class _Probe(Steer):
    """Records what each vector would be added to, adding nothing."""

    def __init__(self, flow_lm: FlowLMModel) -> None:
        super().__init__(flow_lm, [1.0] * (len(flow_lm.transformer.layers) + 1))
        self.seen: dict[int, list[torch.Tensor]] = {}

    def add(self, x: torch.Tensor, index: int) -> torch.Tensor:
        if self.kinds is not None:
            self.seen.setdefault(index, []).append(x[self.kinds == BOS].detach())
        return x


def scales(model: TTSModel, rows: Rows, count: int = 64) -> list[float]:
    """RMS of the activations each vector is added to, at BOS: the vectors' units."""
    probe = _Probe(model.flow_lm)
    probe.attach(model.flow_lm)
    try:
        with torch.no_grad():
            indices = list(range(min(count, len(rows.k))))
            student_forward(
                model, probe, [rows.prefix_kv[i] for i in indices], rows.latents[indices]
            )
    finally:
        probe.detach_hooks()
    return [float(torch.cat(probe.seen[i]).pow(2).mean().sqrt()) for i in sorted(probe.seen)]


def train(
    model: TTSModel, rows: Rows, full: FullRows | None, setting: Setting, seed: int = 0
) -> Steer:
    """Adam, linear warm-up over the first 5% of steps, then cosine to 0."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.flow_lm.bos_emb.device
    steer = Steer(model.flow_lm, scales(model, rows)).to(device)
    steer.attach(model.flow_lm)
    optimizer = torch.optim.Adam([steer.theta], lr=setting.lr)
    warmup = max(1, setting.steps // 20)

    def factor(step: int) -> float:
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, step / setting.steps)))
        return min(1.0, (step + 1) / warmup) * cosine

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    rng, full_rng = random.Random(seed), random.Random(seed + 1)
    try:
        for _ in range(setting.steps):
            index = rng.sample(range(len(rows.k)), min(setting.batch, len(rows.k)))
            z_bos, z_after = student_forward(
                model, steer, [rows.prefix_kv[i] for i in index], rows.latents[index].to(device)
            )
            loss = F.mse_loss(z_bos, rows.z_onset[index].to(device))
            loss = loss + F.mse_loss(z_after, rows.z_after[index].to(device))
            if setting.eos_weight:
                loss = loss + setting.eos_weight * (
                    eos_loss(model, z_bos, rows.eos_onset[index].to(device))
                    + eos_loss(model, z_after, rows.eos_after[index].to(device))
                )
            if full is not None and setting.full_rows:
                pick = full_rng.sample(range(len(full.k)), min(setting.full_rows, len(full.k)))
                loss = loss + _full_loss(model, steer, full, pick, setting.eos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
    finally:
        steer.detach_hooks()
    return steer


def _full_loss(
    model: TTSModel, steer: Steer, full: FullRows, pick: list[int], eos_weight: float
) -> torch.Tensor:
    device = model.flow_lm.bos_emb.device

    def padded(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(values, batch_first=True).to(device)

    lengths = torch.tensor([len(full.latents[i]) for i in pick])
    mask = (torch.arange(int(lengths.max()))[None] < lengths[:, None]).float().to(device)
    _, z = student_forward(
        model, steer, [full.prefix_kv[i] for i in pick], padded([full.latents[i] for i in pick])
    )
    target = padded([full.z_after[i] for i in pick])
    loss = (((z - target) ** 2).mean(-1) * mask).sum() / mask.sum()
    if eos_weight:
        loss = loss + eos_weight * eos_loss(
            model, z, padded([full.eos_after[i] for i in pick]), mask
        )
    return loss
