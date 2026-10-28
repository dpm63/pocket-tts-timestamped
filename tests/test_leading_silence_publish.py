import argparse
import json
from pathlib import Path

import pytest
import safetensors.torch
import torch

from scripts.leading_silence import publish
from scripts.leading_silence.publish import _configs, checkpoint_name, stale_vectors
from scripts.leading_silence.weights import weights_sha256


def test_vectors_are_named_after_the_checkpoint_not_the_language_alias() -> None:
    assert checkpoint_name(["english", "english_2026-10"]) == "english_2026-10"
    assert checkpoint_name(["french"]) == "french"
    assert checkpoint_name(["english_2026-09", "english_2026-09_2"]) == "english_2026-09"


def test_every_shipped_vectors_file_fits_its_configs_weights() -> None:
    # Upstream moves aliases such as `english` to new weights: a config left pointing at the
    # old file would load vectors trained on another model.
    configs = {
        name: config for name, config in _configs().items() if config.get("leading_silence_vectors")
    }
    assert configs
    for name, config in configs.items():
        reason = stale_vectors(config, weights_sha256(config["weights_path"]))
        assert reason is None, f"{name}: {reason}"


def _trained_run(tmp_path: Path, problems: list[str]) -> tuple[Path, Path]:
    configs = tmp_path / "configs"
    configs.mkdir()
    for alias in ("dutch", "dutch_2026-10"):
        (configs / f"{alias}.yaml").write_text("weights_path: x\n\nflow_lm:\n  dim: 1\n")
    work = tmp_path / "work"
    folder = work / "results" / "abc"
    folder.mkdir(parents=True)
    plan = {"key": "abc", "config": "dutch_2026-10", "aliases": ["dutch", "dutch_2026-10"]}
    plan["weights_sha256"] = "f" * 64
    (work / "plan.json").write_text(json.dumps({"matrix": [plan]}))
    safetensors.torch.save_file(
        {"vectors": torch.zeros(2, 1), "warmup_latents": torch.zeros(1, 1)},
        str(folder / "vectors.safetensors"),
        metadata={"weights_sha256": "f" * 64},
    )
    section = {"speech_by_frame_1": 0.9, "onset_p95": 1, "abrupt_rate": 0.1}
    tail = {"tail_speech_rate": 0.0, "early_eos_rate": 0.0}
    side = {"onsets": section, "tail_with_period": tail, "tail_without_period": tail}
    report = {"smoke": False, "baseline": side, "vectors": side, "problems": problems}
    (folder / "report.json").write_text(json.dumps(report))
    return configs, work


def test_publish_accepts_named_false_negatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs, work = _trained_run(tmp_path, ["Whisper WER is 9.0% (baseline 4.0%)"])
    monkeypatch.setattr(publish, "CONFIGS_DIR", configs)
    args = argparse.Namespace(work=str(work), open_pr=False, from_run="", accept="")
    publish.publish(args)
    assert not (configs / "leading_silence").exists()

    publish.publish(argparse.Namespace(**{**vars(args), "accept": "dutch_2026-10"}))
    assert (configs / "leading_silence" / "dutch_2026-10.safetensors").exists()
    for alias in ("dutch", "dutch_2026-10"):
        text = (configs / f"{alias}.yaml").read_text()
        assert "leading_silence_vectors: leading_silence/dutch_2026-10.safetensors" in text
    body = (work / "pr-body.md").read_text()
    assert "Published despite failed checks" in body and "WER is 9.0%" in body
