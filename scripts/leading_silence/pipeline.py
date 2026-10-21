"""Train, export and check leading-silence vectors for one checkpoint (one workflow job)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from pocket_tts_timestamped import TTSModel
from scripts.leading_silence import data, evaluate
from scripts.leading_silence.sequences import FRAME_SAMPLES, encode_voice
from scripts.leading_silence.steer import Steer
from scripts.leading_silence.train import (
    FullRows,
    Rows,
    Setting,
    default_setting,
    make_full_rows,
    make_rows,
    train,
)

WARMUP_FRAMES = 8  # the decoder warms up on this much encoded silence
ROLLOUT_BATCH = 32
VECTORS_FILE = "vectors.safetensors"
REPORT_FILE = "report.json"
Phase = Callable[[str, Callable[[], Any]], Any]


def silence_latents(model: TTSModel, frames: int) -> torch.Tensor:
    """Normalized latents [frames, ldim] of Mimi-encoded digital silence."""
    flow_lm = model.flow_lm
    zeros = torch.zeros(1, 1, frames * FRAME_SAMPLES, device=flow_lm.bos_emb.device)
    raw = model.mimi.encode_to_latent(zeros)[0]
    if raw.shape[-1] != flow_lm.ldim:
        # English 2026-01: its encoder output isn't the FlowLM latent. The research warmed it
        # on the model's own generated silence (bos_shortcut generated_silence_latents).
        raise NotImplementedError("This checkpoint can't encode silence into its latent space")
    return ((raw - flow_lm.emb_mean) / flow_lm.emb_std).float().contiguous()


def _train(model: TTSModel, folder: Path, setting: Setting, phase: Phase) -> Steer:
    """Teacher rollouts on the train split, then training. The rollouts (with every row's
    prefix keys/values, several GB) are freed on return."""
    catalog = data.voices(folder, "train")
    pool = data.texts(folder, "train")
    with torch.no_grad():
        latents = phase(
            "encode_voices",
            lambda: {index: encode_voice(model, v.source) for index, v in enumerate(catalog)},
        )
    pairs = data.sample_pairs(catalog, pool, 4 * setting.rows, seed=1)
    rows: Rows = phase(
        "rollouts", lambda: make_rows(model, latents, pairs, setting.rows, ROLLOUT_BATCH, seed=1)
    )
    full: FullRows | None = None
    if setting.full_rows:
        full_pairs = [
            (voice, evaluate.bare(text))
            for voice, text in data.sample_pairs(catalog, pool, setting.full_pairs, seed=3)
        ]
        model.append_terminal_punctuation = False  # these rows end without a period
        try:
            full = phase(
                "full_rollouts",
                lambda: make_full_rows(
                    model, latents, full_pairs, ROLLOUT_BATCH, setting.full_max_frames, seed=3
                ),
            )
        finally:
            model.append_terminal_punctuation = model.config.append_terminal_punctuation
    return phase("train", lambda: train(model, rows, full, setting))


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    times: dict[str, float] = {}

    def phase(name: str, call: Callable[[], Any]) -> Any:  # noqa: ANN401
        started = time.perf_counter()
        result = call()
        times[name] = round(time.perf_counter() - started, 1)
        print(f"[{name}] {times[name]:.1f} s", flush=True)
        return result

    model = phase("load", lambda: TTSModel.load_model(language=args.config))
    if not model.has_voice_cloning:
        raise SystemExit("Training needs the voice-cloning weights (set HF_TOKEN)")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    folder = phase("data", lambda: data.fetch(data.language_of(args.config)))
    setting = default_setting(len(model.flow_lm.transformer.layers))
    if args.smoke:
        setting = dataclasses.replace(setting, rows=16, steps=4, full_pairs=16, full_max_frames=120)
    steer = _train(model, folder, setting, phase)

    path = out / VECTORS_FILE
    vectors = steer.vectors().detach().float().contiguous()
    with torch.no_grad():
        warmup = silence_latents(model, WARMUP_FRAMES)
    safetensors.torch.save_file(
        {"vectors": vectors.cpu(), "warmup_latents": warmup.cpu()},
        str(path),
        metadata={"config": args.config, "setting": json.dumps(dataclasses.asdict(setting))},
    )

    test_voices = data.voices(folder, "test")
    test_texts = data.texts(folder, "test")[: args.eval_texts]
    with torch.no_grad():
        states = [model.get_state_for_audio_prompt(voice.source) for voice in test_voices]
    baseline = phase("check_baseline", lambda: evaluate.measure(model, states, test_texts))
    model._load_leading_silence_vectors(path)
    steered = phase("check_vectors", lambda: evaluate.measure(model, states, test_texts))
    report = {
        "config": args.config,
        "setting": dataclasses.asdict(setting),
        "smoke": args.smoke,
        "test_voices": [voice.id for voice in test_voices],
        "test_texts": len(test_texts),
        "baseline": baseline,
        "vectors": steered,
        "problems": evaluate.problems(baseline, steered),
        "times_s": times,
        "torch_threads": torch.get_num_threads(),
    }
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
