"""Training voices and texts, from the dataset repo on the Hugging Face Hub.

Per language, `<language>/voices.csv` lists voices (id, base, split, source) and
`<language>/texts.csv` texts (text, split); splits are train / val / test, by voice and by
text. A source is a built-in voice name (its clip is encoded) or a file in the dataset.
Built with `scripts/leading_silence/build_dataset.py`.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

from pocket_tts_timestamped.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

DATASET = "geli95us/pocket-tts-leading-silence"
# Pinned for reproducible training; update it with the dataset.
REVISION = "54ed03388f4e6059a493e4fb23c8cf348a53ed8d"
LANGUAGES = ("english", "french", "german", "spanish", "italian", "portuguese", "dutch")


@dataclass(frozen=True)
class Voice:
    id: str
    base: str
    split: str
    # An hf:// clip or a local file.
    source: str


def language_of(config: str) -> str:
    language = next((name for name in LANGUAGES if config.startswith(name)), None)
    if language is None:
        raise ValueError(f"No training data language for config {config!r}")
    return language


def fetch(language: str) -> Path:
    """The dataset's files for `language`, downloaded (cached) from the Hub, or from the
    folder in `LEADING_SILENCE_DATA` (a local copy, such as build_dataset's output)."""
    local = os.environ.get("LEADING_SILENCE_DATA")
    if local:
        root = local
    else:
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            DATASET, repo_type="dataset", revision=REVISION, allow_patterns=[f"{language}/*"]
        )
    folder = Path(root) / language
    if not (folder / "voices.csv").exists():
        raise FileNotFoundError(f"The dataset {DATASET} has no data for {language} yet")
    return folder


def voices(folder: Path, split: str) -> list[Voice]:
    with (folder / "voices.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == split]
    return [
        Voice(
            id=row["id"],
            base=row["base"],
            split=row["split"],
            source=_ORIGINS_OF_PREDEFINED_VOICES.get(row["source"]) or str(folder / row["source"]),
        )
        for row in rows
    ]


def texts(folder: Path, split: str) -> list[str]:
    with (folder / "texts.csv").open(newline="", encoding="utf-8") as stream:
        return [row["text"] for row in csv.DictReader(stream) if row["split"] == split]


def sample_pairs(
    catalog: list[Voice], pool: list[str], count: int, seed: int
) -> list[tuple[int, str]]:
    """(voice index, text) pairs: a base voice uniformly, then one of its variants."""
    rng = random.Random(seed)
    by_base: dict[str, list[int]] = {}
    for index, voice in enumerate(catalog):
        by_base.setdefault(voice.base, []).append(index)
    bases = sorted(by_base)
    return [(rng.choice(by_base[rng.choice(bases)]), rng.choice(pool)) for _ in range(count)]
