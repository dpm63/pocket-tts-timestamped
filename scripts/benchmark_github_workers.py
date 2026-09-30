"""Benchmark checkpoint evaluation inference on a GitHub-hosted CPU runner.

The input set comes from .timestamp-eval's existing English verification samples.
Run via the manual benchmark workflow; warmup samples are excluded from rates.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

PROMPTS = Path(__file__).with_name("benchmark-data") / "checkpoint-prompts.json"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def summary(
    mode: str,
    model: str,
    backend: str | None,
    load_s: float,
    setup_s: float,
    rows: list[dict],
    warmup: int,
    output: Path,
) -> None:
    measured = rows[warmup:]
    sample_count = sum(row["count"] for row in measured)
    inference_s = sum(row["seconds"] for row in measured)
    audio_s = sum(row["audio_seconds"] for row in measured)
    result = {
        "mode": mode,
        "model": model,
        "backend": backend,
        "runner": os.environ.get("RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS", platform.system()),
        "cpu_count": os.cpu_count(),
        "warmup_samples": sum(row["count"] for row in rows[:warmup]),
        "measured_samples": sample_count,
        "measured_audio_seconds": audio_s,
        "model_load_seconds": load_s,
        "other_setup_seconds": setup_s,
        "inference_seconds": inference_s,
        "seconds_per_sample": inference_s / sample_count,
        "estimated_500_inference_seconds": inference_s * 500 / sample_count,
        "samples": rows,
    }
    save_json(output, result)
    print(
        json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2),
        flush=True,
    )


def tts(model_id: str, output: Path, audio_dir: Path | None) -> None:
    import scipy.io.wavfile
    import torch

    from pocket_tts_timestamped import TTSModel

    fixture = json.loads(PROMPTS.read_text(encoding="utf-8"))
    samples = fixture["samples"]
    started = time.perf_counter()
    model = TTSModel.load_model(language=model_id).eval()
    load_s = time.perf_counter() - started
    states = {}
    rows = []
    setup_s = 0.0
    if audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples):
        voice = sample["voice"]
        if voice not in states:
            started = time.perf_counter()
            states[voice] = model.get_state_for_audio_prompt(voice)
            setup_s += time.perf_counter() - started
        torch.manual_seed(sample["seed"])
        started = time.perf_counter()
        result = model.generate_audio_with_timestamps(
            states[voice], sample["text"], copy_state=True
        )
        elapsed = time.perf_counter() - started
        audio = result.audio.detach().float().cpu().numpy()
        if audio_dir is not None:
            scipy.io.wavfile.write(audio_dir / f"{index:03d}.wav", model.sample_rate, audio)
        rows.append(
            {
                "index": index,
                "count": 1,
                "seconds": elapsed,
                "audio_seconds": len(audio) / model.sample_rate,
                "characters": len(sample["text"]),
            }
        )
        print(f"{model_id} {index + 1}/{len(samples)}: {elapsed:.2f}s", flush=True)
    summary("tts", model_id, None, load_s, setup_s, rows, fixture["warmup"], output)


def stt(model_id: str, backend: str, audio_dir: Path, output: Path, batch_size: int) -> None:
    import scipy.io.wavfile
    import torch

    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    fixture = json.loads(PROMPTS.read_text(encoding="utf-8"))
    paths = [audio_dir / f"{index:03d}.wav" for index in range(len(fixture["samples"]))]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("The TTS audio artifact is incomplete")
    started = time.perf_counter()
    if model_id == "parakeet-tdt-0.6b-v3":
        from nemo.collections.asr.models import ASRModel

        model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3").to("cpu").eval()
    else:
        from crisperwhisper import CrisperWhisperModel

        hf_id = "nyralabs/CrisperWhisper2.0_" + model_id.rsplit("-", 1)[-1]
        model = CrisperWhisperModel(
            hf_id,
            backend=backend,
            device="cpu",
            compute_type="int8" if backend == "ct2" else "float32",
        )
    load_s = time.perf_counter() - started
    rows = []
    warmup_count = fixture["warmup"]
    chunks = [paths[:warmup_count]] + [
        paths[offset : offset + batch_size]
        for offset in range(warmup_count, len(paths), batch_size)
    ]
    for index, chunk in enumerate(chunks):
        started = time.perf_counter()
        if model_id == "parakeet-tdt-0.6b-v3":
            result = model.transcribe(
                [str(path) for path in chunk], batch_size=len(chunk), timestamps=True
            )
            if isinstance(result, tuple):
                result = result[0]
            outputs = list(result)
            if any(
                not isinstance(item.timestamp, dict) or "word" not in item.timestamp
                for item in outputs
            ):
                raise RuntimeError("Parakeet did not return word timestamps")
        elif backend == "transformers" and len(chunk) > 1:
            from benchmark_crisper_batch_adapter import transcribe_short_batch

            outputs = transcribe_short_batch(
                model, [(path, {"_evaluation_language": "en"}) for path in chunk]
            )
        else:
            outputs = [
                model.transcribe(str(path), language="en", mode="intended", word_timestamps=True)
                for path in chunk
            ]
        elapsed = time.perf_counter() - started
        if len(outputs) != len(chunk):
            raise RuntimeError("Transcription output count differs from input count")
        if model_id != "parakeet-tdt-0.6b-v3" and any(
            getattr(item, "words", None) is None for item in outputs
        ):
            raise RuntimeError("CrisperWhisper did not return word timestamps")
        audio_s = 0.0
        for path in chunk:
            sample_rate, audio = scipy.io.wavfile.read(path, mmap=True)
            audio_s += audio.shape[-1] / sample_rate
        rows.append(
            {"batch": index, "count": len(chunk), "seconds": elapsed, "audio_seconds": audio_s}
        )
        print(
            f"{model_id}/{backend} {sum(row['count'] for row in rows)}/{len(paths)}: "
            f"{elapsed:.2f}s",
            flush=True,
        )
    summary("stt", model_id, backend, load_s, 0.0, rows, 1, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tts", "stt"))
    parser.add_argument("model")
    parser.add_argument("--backend", choices=("ct2", "transformers"), default="ct2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "tts":
        tts(args.model, args.output, args.audio_dir)
    else:
        if args.audio_dir is None:
            parser.error("STT requires --audio-dir")
        stt(args.model, args.backend, args.audio_dir, args.output, args.batch_size)


if __name__ == "__main__":
    main()
