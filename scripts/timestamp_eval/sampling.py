"""Freeze a balanced prompt/voice order before collecting or judging audio."""

from __future__ import annotations

import hashlib
import random
import unicodedata
from pathlib import Path
from typing import Any

import requests

from scripts.timestamp_eval.common import lexical_words, read, write

BANDS = ((4, 7), (8, 12), (13, 18))


def clean(text: str) -> bool:
    return (
        bool(lexical_words(text))
        and not any(c.isdigit() for c in text)
        and not any(c in text for c in "@/\\")
        and all(not c.isalpha() or unicodedata.name(c, "").startswith("LATIN") for c in text)
    )


def fetch(language: str, count: int, excluded: set[str], seed: int) -> list[list[dict[str, Any]]]:
    session = requests.Session()
    bands = []
    for band, (minimum, maximum) in enumerate(BANDS):
        accepted: dict[str, dict[str, Any]] = {}
        for _ in range(40):
            response = session.get(
                "https://api.tatoeba.org/v1/sentences",
                params={
                    "lang": language,
                    "word_count": f"{minimum}-{maximum}",
                    "is_native": "yes",
                    "is_orphan": "no",
                    "is_unapproved": "no",
                    "sort": "random",
                    "showtrans": "none",
                    "limit": min(1000, max(100, count * 2)),
                },
                timeout=90,
            )
            response.raise_for_status()
            for sentence in response.json().get("data", []):
                text = str(sentence["text"]).strip()
                key = str(sentence["id"])
                if (
                    key not in excluded
                    and clean(text)
                    and minimum <= len(lexical_words(text)) <= maximum
                ):
                    accepted[key] = {"sentence_id": key, "text": text, "band": band}
            if len(accepted) >= count:
                break
        if len(accepted) < count:
            raise RuntimeError(
                f"Insufficient clean {language} prompts in band {minimum}-{maximum}: {len(accepted)}/{count}"
            )
        values = sorted(accepted.values(), key=lambda s: s["sentence_id"])
        random.Random(seed + band).shuffle(values)
        bands.append(values[:count])
    return bands


def extend(item: dict[str, Any], path: Path, target: int) -> list[dict[str, Any]]:
    samples = read(path) if path.exists() else []
    if len(samples) >= target:
        return samples
    excluded = {s["sentence_id"] for s in samples}
    # Reserve alternate prompts for tokenizer chunk limits, without selecting on ASR success.
    count = (target - len(samples) + 2) // 3 + 2 * len(item["voices"])
    seed = int(item["checkpoint_sha256"][:8], 16) + len(samples)
    pools = fetch(item["tatoeba_language"], count, excluded, seed)
    offsets = [0, 0, 0]
    while len(samples) < target:
        index = len(samples)
        voices = item["voices"]
        band = (index % len(voices) + index // len(voices)) % len(BANDS)
        prompt = pools[band][offsets[band]]
        offsets[band] += 1
        digest = hashlib.sha256(f"{item['checkpoint_sha256']}:{index}".encode()).digest()
        samples.append(
            {
                **prompt,
                "index": index,
                "voice": voices[index % len(voices)],
                "seed": int.from_bytes(digest[:4], "little") & 0x7FFFFFFF,
            }
        )
    write(path, samples)
    return samples
