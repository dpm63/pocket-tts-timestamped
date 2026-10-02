"""Benchmark checkpoint evaluation inference on a GitHub-hosted CPU runner.

The input set comes from .timestamp-eval's existing English verification samples.
Run via the manual benchmark workflow; warmup samples are excluded from rates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
    metadata: dict | None = None,
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
    result.update(metadata or {})
    if mode == "stt":
        result["outputs_with_word_timestamps"] = sum(
            row.get("timestamped_outputs", 0) for row in measured
        )
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


def stt(
    model_id: str,
    backend: str,
    audio_dir: Path,
    output: Path,
    batch_size: int,
    compute_type: str | None = None,
    model_revision: str | None = None,
) -> None:
    import scipy.io.wavfile
    import torch

    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    fixture = json.loads(PROMPTS.read_text(encoding="utf-8"))
    paths = [audio_dir / f"{index:03d}.wav" for index in range(len(fixture["samples"]))]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("The TTS audio artifact is incomplete")
    metadata = {}
    started = time.perf_counter()
    if model_id == "parakeet-tdt-0.6b-v3":
        from nemo.collections.asr.models import ASRModel

        model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3").to("cpu").eval()
    else:
        from crisperwhisper import CrisperWhisperModel

        hf_id = "nyralabs/CrisperWhisper2.0_" + model_id.rsplit("-", 1)[-1]
        if model_revision:
            from huggingface_hub import snapshot_download

            hf_id = snapshot_download(hf_id, revision=model_revision)
        compute_type = compute_type or ("int8" if backend == "ct2" else "float32")
        model = CrisperWhisperModel(hf_id, backend=backend, device="cpu", compute_type=compute_type)
    load_s = time.perf_counter() - started
    if backend == "ct2" and model_id != "parakeet-tdt-0.6b-v3":
        effective = model._engine.model.compute_type
        allowed = {"int8", "int8_float32"} if compute_type == "int8" else {compute_type}
        if effective not in allowed:
            raise RuntimeError(f"Requested {compute_type}, engine uses {effective}")
        metadata = {
            "compute_type": compute_type,
            "effective_compute_type": effective,
            "model_path": str(model._engine.model_path),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "crisperwhisper",
                    "ctranslate2-crisperwhisper",
                    "torch",
                    "transformers",
                )
            },
            "audio_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
            },
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                None,
            ),
            "threads_per_engine": 4,
        }
    if backend == "transformers" and model_id != "parakeet-tdt-0.6b-v3":
        effective = str(next(model._engine.model.parameters()).dtype)
        if compute_type == "float32" and effective != "torch.float32":
            raise RuntimeError(f"Float32 requested, model uses {effective}")
        metadata = {
            "compute_type": compute_type,
            "effective_compute_type": effective,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("crisperwhisper", "torch", "transformers")
            },
            "audio_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
            },
        }
    metadata["model_revision"] = model_revision
    metadata["batch_size"] = batch_size
    rows = []
    warmup_count = fixture["warmup"]
    warmup_chunks = [
        paths[offset : min(offset + batch_size, warmup_count)]
        for offset in range(0, warmup_count, batch_size)
    ]
    chunks = warmup_chunks + [
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
        timestamped_outputs = (
            len(outputs)
            if model_id == "parakeet-tdt-0.6b-v3"
            else sum(getattr(item, "words", None) is not None for item in outputs)
        )
        audio_s = 0.0
        for path in chunk:
            sample_rate, audio = scipy.io.wavfile.read(path, mmap=True)
            audio_s += audio.shape[-1] / sample_rate
        rows.append(
            {
                "batch": index,
                "count": len(chunk),
                "seconds": elapsed,
                "audio_seconds": audio_s,
                "timestamped_outputs": timestamped_outputs,
            }
        )
        print(
            f"{model_id}/{backend} {sum(row['count'] for row in rows)}/{len(paths)}: "
            f"{elapsed:.2f}s",
            flush=True,
        )
    summary("stt", model_id, backend, load_s, 0.0, rows, len(warmup_chunks), output, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tts", "stt"))
    parser.add_argument("model")
    parser.add_argument("--backend", choices=("ct2", "transformers"), default="ct2")
    parser.add_argument("--model-revision")
    parser.add_argument("--compute-type", choices=("int8", "float32"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "tts":
        tts(args.model, args.output, args.audio_dir)
    else:
        if args.audio_dir is None:
            parser.error("STT requires --audio-dir")
        stt(
            args.model,
            args.backend,
            args.audio_dir,
            args.output,
            args.batch_size,
            args.compute_type,
            args.model_revision,
        )


if __name__ == "__main__":
    main()
