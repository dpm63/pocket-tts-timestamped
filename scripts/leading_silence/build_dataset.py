"""Build the training dataset folder (see data.py) from the research workspace's files.

English: the built-in voices and Kokoro voice variants of
`multivoice_transition/voice_split.csv`, and LibriTTS texts of
`libritts_balanced/libritts_balanced_50000.csv` (3-30 words; row % 50: 0 val, 1 test, rest
train). These are the voices, texts and splits the vectors were developed on.

usage: python -m scripts.leading_silence.build_dataset RESEARCH_DIR OUT_DIR
Then upload OUT_DIR as the dataset repo `data.DATASET`.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

TEXT_SPLIT_EVERY = 50
MIN_WORDS, MAX_WORDS = 3, 30

CARD = """---
license: cc-by-4.0
language: [en]
pretty_name: Pocket TTS leading-silence training data
---

# Pocket TTS leading-silence training data

Voices and texts for training the leading-silence vectors of
[pocket-tts-timestamped](https://github.com/dpm63/pocket-tts-timestamped)
(`scripts/leading_silence`). Splits are by voice and by text: train, val, test.

## english

- `voices.csv`: `id, base, split, source`. A source is either a Pocket TTS built-in voice name
  (its clip comes from [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices)) or a
  file here: `kokoro/*.wav`, variants (pitch and speed shifts) of
  [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) voices (Apache-2.0).
- `texts.csv`: `id, text, split`. Sentences from
  [LibriTTS](https://www.openslr.org/60/) (CC BY 4.0), balanced by initial phone; `id` is the
  LibriTTS utterance id.
"""


def build_english(research: Path, out: Path) -> None:
    folder = out / "english"
    (folder / "kokoro").mkdir(parents=True, exist_ok=True)
    voices = []
    with (research / "multivoice_transition" / "voice_split.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["family"] == "pocket":
                name = row["base_voice"]
                voices.append((name, name, row["split"], name))
                continue
            for wav in row["variants"].split(";"):
                shutil.copyfile(research / "kokoro_voices" / wav, folder / "kokoro" / wav)
                voices.append((Path(wav).stem, row["base_voice"], row["split"], f"kokoro/{wav}"))
    with (folder / "voices.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "base", "split", "source"])
        writer.writerows(voices)

    texts = []
    source = research / "libritts_balanced" / "libritts_balanced_50000.csv"
    with source.open(newline="", encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f)):
            if not MIN_WORDS <= int(row["word_count"]) <= MAX_WORDS:
                continue
            split = {0: "val", 1: "test"}.get(index % TEXT_SPLIT_EVERY, "train")
            texts.append((row["id"], row["text"], split))
    with (folder / "texts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "text", "split"])
        writer.writerows(texts)
    print(f"english: {len(voices)} voices, {len(texts)} texts")


def main() -> None:
    research, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(CARD, encoding="utf-8")
    build_english(research, out)


if __name__ == "__main__":
    main()
