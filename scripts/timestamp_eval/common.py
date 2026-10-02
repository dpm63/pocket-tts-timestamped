"""Protocol, atomic records, configuration discovery, and content identity."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROTOCOL = 1
REFERENCE = {
    "repo": "nyralabs/CrisperWhisper2.0_medium",
    "revision": "f011e4a7869860de1ce596391d53c18bf8853e77",
    "backend": "ct2",
    "compute_type": "float32",
}
LANGUAGES = {
    "english": ("eng", "en"),
    "french": ("fra", "fr"),
    "german": ("deu", "de"),
    "spanish": ("spa", "es"),
    "italian": ("ita", "it"),
    "portuguese": ("por", "pt"),
    "dutch": ("nld", "nl"),
}
ENGLISH_VOICES = (
    "alba",
    "marius",
    "cosette",
    "javert",
    "jean",
    "anna",
    "vera",
    "fantine",
    "charles",
    "paul",
    "eponine",
    "azelma",
    "george",
    "mary",
    "jane",
    "michael",
    "eve",
)
NATIVE_VOICES = {
    "french": "estelle",
    "german": "juergen",
    "spanish": "lola",
    "italian": "giovanni",
    "portuguese": "rafael",
    "dutch": "daan",
}


def read(path: Path) -> Any:  # noqa: ANN401
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_identity(config: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    # Context changes intentionally do not invalidate a known checkpoint.
    return (
        config.get("weights_path"),
        config.get("weights_path_without_voice_cloning"),
        config.get("flow_lm", {}).get("weights_path"),
    )


def discover(config_dir: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(config_dir.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        prefix = next((p for p in LANGUAGES if path.stem.startswith(p)), None)
        if prefix is None:
            raise ValueError(f"Define evaluation language and voices for new config {path}")
        if not config.get("weights_path") and not config.get("flow_lm", {}).get("weights_path"):
            raise ValueError(f"No checkpoint in {path}")
        result.append(
            {
                "id": path.stem,
                "config_path": path.as_posix(),
                "config": config,
                "tatoeba_language": LANGUAGES[prefix][0],
                "asr_language": LANGUAGES[prefix][1],
                "voices": list(ENGLISH_VOICES if prefix == "english" else (NATIVE_VOICES[prefix],)),
                "baseline": [
                    [h["layer"], h["head"]] for h in (config.get("timestamp_heads") or [])
                ],
            }
        )
    return result


def resolve_checkpoint(item: dict[str, Any]) -> dict[str, Any]:
    from pocket_tts_timestamped.utils.utils import download_if_necessary

    config = item["config"]
    sources = [p for p in config_identity(config)[:2] if p]
    if not sources:
        sources = [config["flow_lm"]["weights_path"]]
    errors = []
    for source in sources:
        try:
            path = download_if_necessary(source)
            digest = file_hash(path)
            cache = os.environ.get("TIMESTAMP_HASH_CACHE")
            if cache:
                target = path.resolve()
                if not target.is_relative_to(Path(cache).resolve()):
                    raise ValueError("Checkpoint hash cleanup escapes the job cache")
                target.unlink()
            return {**item, "checkpoint_sha256": digest, "effective_weights": source}
        except Exception as error:
            errors.append(f"{source}: {type(error).__name__}: {error}")
    raise RuntimeError("No available checkpoint: " + "; ".join(errors))


def head_label(heads: list[list[int]]) -> str:
    return "+".join(f"L{layer}H{head}" for layer, head in heads)


def lexical_words(text: str) -> list[str]:
    from pocket_tts_timestamped.timestamps.text import _lexical_word_spans

    return [text[span.begin : span.end] for span in _lexical_word_spans(text)]


def normalize(word: str) -> str:
    import unicodedata

    normalized = (
        unicodedata.normalize("NFKC", word)
        .casefold()
        .replace("’", "'")
        .replace("‐", "-")
        .replace("‑", "-")
    )
    return "".join(c for c in normalized if c.isalnum() or c in "'-")


def strict_match(words: list[dict[str, Any]], expected: list[str]) -> bool:
    import math

    return (
        bool(words)
        and len(words) == len(expected)
        and all(
            normalize(w["word"]) == normalize(e)
            and math.isfinite(w["start"])
            and math.isfinite(w["end"])
            and 0 <= w["start"] <= w["end"]
            for w, e in zip(words, expected)
        )
    )
