"""A listening set for one checkpoint's checks: the trials a check flags with the vectors but
not without them, as WAV pairs (same seed), to tell real harm from a false negative before
`publish --accept`.

The trials are the checks' own (`evaluate`: same test voices, texts and seeds), so the set
shows what the workflow measured. Checks:

- `tail_period`, `tail_no_period`: speech after EOS (> -35 dB) or EOS on the last loud frame,
  on whole generations with and without the final period;
- `first_word`: Whisper misses the first word (the Whisper check's generations);
- `abrupt`: the onset's first 30 ms louder than -45 dB (onset trials).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile
import torch
from numpy.typing import NDArray

from pocket_tts_timestamped import TTSModel
from scripts.leading_silence import data, evaluate, transcribe
from scripts.leading_silence.onsets import ABRUPT_DB, frame_db

CHECKS = ("tail_period", "tail_no_period", "first_word", "abrupt")


def _generate(
    model: TTSModel, trials: list[tuple[Any, Any, int, str]], checks: set[str]
) -> dict[str, list[Any]]:
    """Each needed kind of trial, in trial order."""
    out: dict[str, list[Any]] = {}
    with torch.no_grad():
        if checks & {"tail_period", "first_word"}:
            out["period"] = [
                evaluate.tail_trial(model, state, text, index, True)
                for _, state, index, text in trials
            ]
        if "tail_no_period" in checks:
            out["no_period"] = [
                evaluate.tail_trial(model, state, text, index, False)
                for _, state, index, text in trials
            ]
        if "abrupt" in checks:
            out["onset"] = [
                evaluate.onset_trial(model, state, text, index) for _, state, index, text in trials
            ]
    return out


def _tail_flagged(pcm: NDArray[np.float32], eos: int | None) -> list[str]:
    flags = evaluate.tail_flags(pcm, eos)
    if flags is None:
        return ["no EOS"]
    names = {"speech": "speech after EOS", "early_eos": "EOS on the last loud frame"}
    return [label for key, label in names.items() if flags[key]]


def _tail_entry(pcm: NDArray[np.float32], eos: int | None) -> dict[str, Any]:
    db = frame_db(pcm)
    tail = db[eos + 1 :] if eos is not None else db[:0]
    return {
        "eos_frame": eos,
        "frames": len(db),
        "tail_peak_db": round(float(tail.max()), 1) if len(tail) else None,
        "flags": _tail_flagged(pcm, eos),
    }


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = {name.strip() for name in args.checks.split(",") if name.strip()}
    if checks - set(CHECKS):
        raise SystemExit(f"Unknown checks: {sorted(checks - set(CHECKS))}; choose from {CHECKS}")
    torch.set_num_threads(args.threads)
    model = TTSModel.load_model(language=args.config)
    language = data.language_of(args.config)
    folder = data.fetch(language)
    voices = data.voices(folder, "test")
    texts = data.texts(folder, "test")[: args.eval_texts]
    with torch.no_grad():
        states = [model.get_state_for_audio_prompt(voice.source) for voice in voices]
    trials = [
        (voice, state, index, text)
        for voice, state in zip(voices, states, strict=True)
        for index, text in enumerate(texts)
    ]
    print(f"{args.config}: {len(trials)} trials per kind, checks {sorted(checks)}", flush=True)
    sides = {"without": _generate(model, trials, checks)}
    model._load_leading_silence_vectors(Path(args.vectors))
    sides["with"] = _generate(model, trials, checks)

    selected: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = {}
    for check, kind in (("tail_period", "period"), ("tail_no_period", "no_period")):
        if check in checks:
            selected[check] = []
            for number in range(len(trials)):
                without = _tail_entry(*sides["without"][kind][number][:2])
                with_ = _tail_entry(*sides["with"][kind][number][:2])
                if set(with_["flags"]) - set(without["flags"]):
                    selected[check].append((number, without, with_))
    if "first_word" in checks:
        whisper = transcribe.load_whisper(args.threads)
        selected["first_word"] = []
        for number in evaluate.whisper_trial_numbers(len(trials), args.whisper_trials):
            heard = {
                side: transcribe.transcribe_clip(
                    whisper,
                    sides[side]["period"][number][0],
                    sides[side]["period"][number][2],
                    model.sample_rate,
                    language,
                )
                for side in ("without", "with")
            }
            if heard["without"]["first_word_right"] and not heard["with"]["first_word_right"]:
                selected["first_word"].append((number, heard["without"], heard["with"]))
    if "abrupt" in checks:
        selected["abrupt"] = []
        for number in range(len(trials)):
            heads = {
                side: evaluate.onset_measures(sides[side]["onset"][number], model.sample_rate)
                for side in ("without", "with")
            }
            abrupt = {
                side: heads[side]["head_db"] is not None and heads[side]["head_db"] > ABRUPT_DB
                for side in heads
            }
            if abrupt["with"] and not abrupt["without"]:
                selected["abrupt"].append((number, heads["without"], heads["with"]))

    kinds = {"tail_period": "period", "tail_no_period": "no_period", "first_word": "period"}
    manifest: dict[str, Any] = {
        "config": args.config,
        "vectors": str(args.vectors),
        "trials": len(trials),
        "checks": {},
    }
    for check, found in selected.items():
        kind = kinds.get(check, "onset")
        entries = []
        for number, without, with_ in found[: args.most]:
            voice, _, index, text = trials[number]
            stem = f"{check}-{voice.id}-t{index}"
            for side in ("without", "with"):
                trial = sides[side][kind][number]
                pcm = trial if kind == "onset" else trial[0]
                scipy.io.wavfile.write(
                    out / f"{stem}-{side}-vectors.wav", model.sample_rate, pcm.astype(np.float32)
                )
            spoken = text if kind == "onset" else sides["with"][kind][number][2]
            entries.append(
                {"file": stem, "voice": voice.id, "text": spoken, "without": without, "with": with_}
            )
        manifest["checks"][check] = {
            "flagged": len(found),
            "saved": len(entries),
            "trials": entries,
        }
        print(f"{check}: {len(found)} trials flagged with vectors only, {len(entries)} saved")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
