#!/usr/bin/env python3
"""Compare the Rust aligner with the Python implementation, without downloading a model."""

import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START = time.monotonic()


def log(message: str) -> None:
    print(
        f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} {message} elapsedMs={int((time.monotonic() - START) * 1000)}",
        flush=True,
    )


def voiced_fixtures(
    alignment: types.ModuleType, torch: types.ModuleType
) -> list[dict[str, object]]:
    """Frames around the silence threshold, labelled by Python's `is_voiced`.

    Frames within 0.1% of the threshold are left out: there, the summation order of the energy
    (BLAS in Python, sequential in Rust) can decide, and neither answer is wrong.
    """
    randomizer = random.Random(44)
    threshold = alignment._SILENCE_RMS_THRESHOLD
    frames: list[dict[str, object]] = [
        {"pcm": pcm, "voiced": alignment.is_voiced(torch.tensor(pcm, dtype=torch.float32))}
        for pcm in ([], [0.0] * 1920)
    ]
    while len(frames) < 100:
        rms = 10 ** randomizer.uniform(-4, -2)
        pcm = torch.tensor([randomizer.gauss(0, rms) for _ in range(1920)], dtype=torch.float32)
        energy = torch.dot(pcm, pcm).item()
        if abs(energy / (threshold**2 * pcm.numel()) - 1) < 1e-3:
            continue
        frames.append({"pcm": pcm.tolist(), "voiced": alignment.is_voiced(pcm)})
    return frames


def main() -> None:
    log("START Python/Rust timestamp parity timeoutSeconds=600")
    # Import the algorithm without loading TTSModel or touching checkpoints.
    package = types.ModuleType("pocket_tts_timestamped")
    package.__path__ = [str(ROOT.parent / "pocket_tts_timestamped")]
    sys.modules["pocket_tts_timestamped"] = package
    alignment = importlib.import_module("pocket_tts_timestamped.timestamps.alignment")
    text = importlib.import_module("pocket_tts_timestamped.timestamps.text")
    import torch

    randomizer = random.Random(43)
    cases = []
    for number in range(200):
        units = []
        synthetic = number % 2 == 0
        for i in range(5):
            word = text._SourceWord(f"word{i}", i, i * 6, i * 6 + 5) if i % 2 == 0 else None
            units.append(
                text._TextUnit(
                    f"unit{i}", i * 6, i * 6 + 5, word, synthetic=word is None and synthetic
                )
            )
        align = alignment.WordAlignment(tuple(units))
        frames = []
        expected = []
        for i in range(40):
            scores = [randomizer.random() for _ in units]
            if i % 9 == 0:
                scores[0] = 0.00001
            voiced = randomizer.random() > 0.3
            frame = {"scores": scores, "voiced": voiced, "sample": i * 1920}
            frames.append(frame)
            for e in align.process_frame(
                torch.tensor(scores, dtype=torch.float32), voiced, i * 0.08
            ):
                expected.append(
                    {
                        "start": e.word_index * 6,
                        "length": 5,
                        "wordIndex": e.word_index,
                        "sampleOffset": round(
                            (e.end_time if isinstance(e, alignment.WordEnd) else e.start_time)
                            * 24000
                        ),
                        "isEnd": isinstance(e, alignment.WordEnd),
                    }
                )
        for e in align.finish(40 * 0.08):
            expected.append(
                {
                    "start": e.word_index * 6,
                    "length": 5,
                    "wordIndex": e.word_index,
                    "sampleOffset": round(e.end_time * 24000),
                    "isEnd": True,
                }
            )
        layout = {
            "units": [
                {
                    "start": i * 6 if u.is_word else 0,
                    "length": 5 if u.is_word else 0,
                    "wordIndex": u.word.word_index if u.word is not None else -1,
                    "synthetic": u.synthetic,
                }
                for i, u in enumerate(units)
            ],
            "tokenToUnit": [[float(i == j) for j in range(5)] for i in range(5)],
        }
        # f32 input scores must be identical in both implementations.
        for f in frames:
            f["scores"] = torch.tensor(f["scores"], dtype=torch.float32).tolist()
        cases.append(
            {
                "name": f"random-{number}",
                "layout": layout,
                "frames": frames,
                "end": 40 * 1920,
                "expected": expected,
            }
        )
    voiced_frames = voiced_fixtures(alignment, torch)
    with tempfile.TemporaryDirectory(prefix="pocket-timestamp-parity-") as folder:
        fixture = Path(folder) / "cases.json"
        fixture.write_text(json.dumps({"alignment": cases, "voiced": voiced_frames}))
        cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
        log(f"PHASE Rust reference fixtures count=200 voicedFrames={len(voiced_frames)}")
        subprocess.run(
            [
                cargo,
                "run",
                "--release",
                "--manifest-path",
                str(ROOT / "Cargo.toml"),
                "--example",
                "align_reference",
                "--",
                str(fixture),
            ],
            check=True,
            timeout=600,
            env=os.environ,
        )
    log("FINISH Python/Rust timestamp parity cases=200")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAIL Python/Rust timestamp parity")
        raise
