"""Regression checks for the automated checkpoint-selection protocol."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from pocket_tts_timestamped.timestamps import WordAlignment, WordEnd
from pocket_tts_timestamped.timestamps.text import _SourceWord, _TextUnit
from scripts.timestamp_eval.artifacts import Artifacts
from scripts.timestamp_eval.common import config_identity, strict_match
from scripts.timestamp_eval.publish import update_config, update_readme
from scripts.timestamp_eval.scoring import candidates, rank, summarize


def row(heads: list[list[int]], mae: float, skipped: int = 0) -> dict[str, Any]:
    return summarize(
        heads,
        np.asarray([1000, 1000 - skipped, 0, 0], dtype=float),
        np.asarray([100, 100, mae * 100, mae * 100]),
    )


def test_penalty_units_skip_gate_and_rescoring_preserve_raw_metrics() -> None:
    raw = {"rows": [row([[0, 0]], 50, 1), row([[0, 1], [1, 0]], 49.6, 0), row([[0, 2]], 30, 6)]}
    original = repr(raw)
    scoring = rank(raw)
    assert scoring["winner"]["heads"] == [[0, 1], [1, 0]]
    assert scoring["winner"]["score_ms"] == 50.6
    assert scoring["best_mae"][0]["eligible"] is False
    changed = rank(raw, skip_penalty=0, head_penalty=2)
    assert changed["winner"]["heads"] == [[0, 0]]
    assert repr(raw) == original
    assert rank({"rows": [row([[0, 0]], 10, 6)]})["winner"] is None
    with pytest.raises(ValueError):
        rank(raw, head_penalty=-1)


def test_candidate_pool_keeps_incomplete_partner_and_baseline_outside_top_ten() -> None:
    singles = [row([[i, 0]], 10 + i, 900 if i == 1 else 0) for i in range(12)]
    baseline = [[11, 0], [10, 0]]
    groups = candidates(singles, [baseline])
    assert len(groups) == 638
    assert [[0, 0], [1, 0]] in groups
    assert baseline in groups
    assert [[11, 0]] not in groups


def test_content_identity_ignores_heads_and_generation_context() -> None:
    config = {
        "weights_path": "model",
        "timestamp_heads": [{"layer": 1, "head": 2}],
        "default_temperature": 0.3,
    }
    changed = {**config, "timestamp_heads": [{"layer": 3, "head": 4}], "default_temperature": 0.5}
    assert config_identity(config) == config_identity(changed)
    assert config_identity(config) != config_identity({**config, "weights_path": "new-model"})


def test_strict_matching_rejects_segmentation_and_missing_timing() -> None:
    assert strict_match([{"word": "L’été,", "start": 0.0, "end": 1.0}], ["l’été"])
    assert not strict_match([{"word": "word", "start": float("nan"), "end": 1.0}], ["word"])
    assert not strict_match([{"word": "one two", "start": 0.0, "end": 1.0}], ["one", "two"])
    assert not strict_match([], ["word"])


def test_config_patch_preserves_unrelated_yaml_and_supports_unconfigured_models() -> None:
    text = "# note\nweights_path: hf://model\ntimestamp_heads:\n- layer: 3\n  head: 8\n# temperature note\ndefault_temperature: 0.3\nflow_lm: {}\n"
    replacement = update_config(text, [[1, 2], [3, 4]])
    assert "# note" in replacement and "# temperature note" in replacement
    assert yaml.safe_load(replacement) == {
        **yaml.safe_load(text),
        "timestamp_heads": [{"layer": 1, "head": 2}, {"layer": 3, "head": 4}],
    }
    missing = "weights_path: hf://model\n\nflow_lm: {}\n"
    assert yaml.safe_load(update_config(missing, [[0, 0]]))["timestamp_heads"] == [
        {"layer": 0, "head": 0}
    ]


def test_readme_patch_separates_cohorts_and_retains_legacy_reference() -> None:
    text = "Evaluation results, MAE is measured against CrisperWhisper 2.0 large:\n| Checkpoint | Head | Samples | Words | Skip rate | Start/end MAE |\n|---|---|---|---|---|---|\n| English 2026-09 | L3H8 | 438 | 3,903 | 0% | 60 ms |\n| German | L3H6 | 309 | 2,610 | 0% | 64 ms |\n\n# Other section\n"
    candidate = row([[1, 2]], 12)
    output = update_readme(
        text,
        {
            "english_2026-09": {
                "result": {"skip_samples": 1500, "mae_samples": 500},
                "candidate": candidate,
            }
        },
    )
    assert "1500 | 1,000 |" in output
    assert "500 | 100 |" in output
    assert "Medium CT2 float32" in output
    assert "Large (legacy)" in output
    assert output.endswith("# Other section\n")


@pytest.mark.parametrize("seed", range(5))
def test_compiled_alignment_matches_production_state_transitions(seed: int) -> None:
    pytest.importorskip("numba")
    from scripts.timestamp_eval.alignment import predict

    units = []
    word_units = []
    for index in range(6):
        word_units.append(len(units))
        units.append(
            _TextUnit(
                f"w{index}", index, index + 1, _SourceWord(f"w{index}", index, index, index + 1)
            )
        )
        if index in (1, 3, 5):
            units.append(_TextUnit(".", index, index + 1, None, synthetic=index == 5))
    rng = np.random.default_rng(seed)
    values = rng.random((15, 80, len(units)), dtype=np.float32)
    values[::2] *= 0.001
    voiced = rng.random(80) > 0.4
    punctuation = np.asarray(
        [any(not u.is_word and not u.synthetic for u in units[i + 1 :]) for i in word_units]
    )
    expected = np.full((len(values), 6, 2), np.nan)
    for group, scores in enumerate(values):
        alignment = WordAlignment(tuple(units))
        events = []
        for frame, score in enumerate(scores):
            events.extend(
                alignment.process_frame(torch.from_numpy(score), bool(voiced[frame]), frame / 12.5)
            )
        events.extend(alignment.finish(6.4))
        for event in events:
            if isinstance(event, WordEnd):
                expected[group, event.word_index] = [event.start_time, event.end_time]
    actual = predict(values, voiced, np.asarray(word_units, dtype=np.int64), punctuation, 12.5, 6.4)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)


def test_artifact_archive_cannot_escape_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("../outside.json", "{}")
    response = SimpleNamespace(content=content.getvalue(), raise_for_status=lambda: None)
    transport = Artifacts("owner/repo", "fake-token")
    monkeypatch.setattr(transport.session, "get", lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="escapes"):
        transport.download(
            {"archive_download_url": "https://api.github.com/archive"}, tmp_path / "artifacts"
        )
    assert not (tmp_path / "outside.json").exists()


def test_sampling_interleaves_bands_and_preserves_order_on_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.timestamp_eval import sampling

    def fetch(
        language: str, count: int, excluded: set[str], seed: int
    ) -> list[list[dict[str, Any]]]:
        return [
            [
                {
                    "sentence_id": f"{band}:{seed}:{i}",
                    "text": "Some frozen prompt words",
                    "band": band,
                }
                for i in range(count)
            ]
            for band in range(3)
        ]

    monkeypatch.setattr(sampling, "fetch", fetch)
    item = {
        "checkpoint_sha256": "a" * 64,
        "voices": [f"v{i}" for i in range(17)],
        "tatoeba_language": "eng",
    }
    path = tmp_path / "prompts.json"
    first = sampling.extend(item, path, 60)
    extended = sampling.extend(item, path, 100)
    assert extended[:60] == first
    assert [s["band"] for s in first[:3]] == [0, 1, 2]
    assert {s["band"] for s in extended if s["voice"] == "v0"} == {0, 1, 2}
    assert len({s["sentence_id"] for s in extended}) == 100
    assert [s["index"] for s in extended] == list(range(100))


def test_reference_resume_does_not_retry_rejections_and_extends_only_mae_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    pytest.importorskip("numba")
    from scripts.timestamp_eval import capture, runner
    from scripts.timestamp_eval.common import REFERENCE, file_hash, read, write

    heads = [[0, 0], [0, 1], [0, 2]]
    samples = [{"index": i, "text": "word", "seed": i, "voice": "alba"} for i in range(102)]
    item = {
        "key": "test",
        "id": "english_2026-09",
        "samples": samples[:2],
        "asr_language": "en",
        "baselines": [],
    }
    plan = {
        "active": [item],
        "skip_samples": 2,
        "mae_samples": 2,
        "shards": 1,
        "previous_run_id": 1,
        "run_id": 2,
        "capture_run_id": 1,
    }
    write(tmp_path / "plan.json", plan)

    def generate_sample(index: int, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "audio.wav").write_bytes(b"audio")
        write(
            directory / "sample.json",
            {"words": ["word"], "heads": heads, "audio_sha256": file_hash(directory / "audio.wav")},
        )
        np.savez(directory / "singles.npz", bounds=np.tile(np.asarray([[[0.1, 0.4]]]), (3, 1, 1)))

    for index in range(2):
        generate_sample(index, tmp_path / "audio/samples" / f"{index:06d}")
    write(tmp_path / "reference/reference.json", {"0": {"usable": False}})
    called = []
    generated = []

    class Model:
        def __init__(self, path: str, **kwargs: object):
            assert kwargs == {"backend": "ct2", "device": "cpu", "compute_type": "float32"}
            self._engine = SimpleNamespace(model=SimpleNamespace(compute_type="float32"))

        def transcribe(self, path: str, **kwargs: object) -> SimpleNamespace:
            called.append(int(Path(path).parent.name))
            assert kwargs["language"] == "en"
            return SimpleNamespace(
                text="word", words=[SimpleNamespace(word="word", start=0.1, end=0.4)]
            )

    class Generator:
        def __init__(self, source: dict[str, Any]):
            assert source is item or source == item

        def generate(self, sample: dict[str, Any], directory: Path) -> None:
            generated.append(sample["index"])
            generate_sample(sample["index"], directory)

    monkeypatch.setitem(sys.modules, "crisperwhisper", SimpleNamespace(CrisperWhisperModel=Model))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo, revision: "model")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-version")
    monkeypatch.setattr(runner.Artifacts, "__init__", lambda self: None)
    monkeypatch.setattr(runner.Artifacts, "named", lambda *args, **kwargs: False)
    monkeypatch.setattr(runner, "extend", lambda item, path, target: samples)
    monkeypatch.setattr(capture, "Generator", Generator)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    runner.reference_run(argparse.Namespace(work=str(tmp_path), key="test", budget_seconds=60))
    assert called == [1, 2]
    assert generated == [2]
    status = read(tmp_path / "reference/status.json")
    assert status == {"complete": True, "attempts": 3, "matches": 2, "selected_ids": [1, 2]}
    with np.load(tmp_path / "reference/single-metrics.npz") as saved:
        assert saved["skip"].shape == (2, 3, 4)
        assert saved["mae"].shape == (2, 3, 4)
    assert read(tmp_path / "reference/runtime.json")["repo"] == REFERENCE["repo"]


def test_rescore_plan_uses_saved_metrics_and_schedules_no_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.timestamp_eval import runner
    from scripts.timestamp_eval.common import read, write

    plan = {
        "active": [{"key": "test"}],
        "cached": [],
        "publish": True,
        "skip_samples": 60,
        "mae_samples": 20,
    }
    write(tmp_path / "plan.json", plan)
    result = {"rows": [row([[0, 0]], 10)]}
    write(tmp_path / "rescore/test/result.json", result)
    calls = []

    def named(
        self: Artifacts, run_id: int, name: str, destination: Path, required: bool = True
    ) -> bool:
        calls.append(name)
        return True

    monkeypatch.setattr(runner.Artifacts, "__init__", lambda self: None)
    monkeypatch.setattr(runner.Artifacts, "named", named)
    monkeypatch.setenv("GITHUB_RUN_ID", "2")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    runner.plan_run(
        argparse.Namespace(
            work=str(tmp_path),
            resume_run=None,
            rescore_run=1,
            publish=False,
            skip_penalty=20,
            head_penalty=1,
            skip_limit=0.5,
        )
    )
    rescored = read(tmp_path / "plan.json")
    assert rescored["active"] == []
    assert rescored["cached"][0]["result"] == result
    assert rescored["publish"] is False
    assert calls == ["timestamp-plan", "timestamp-v1-test"]


def test_capture_trims_non_emitted_eos_frame_from_each_chunk(tmp_path: Path) -> None:
    pytest.importorskip("numba")
    from collections.abc import Generator as EventGenerator

    from pocket_tts_timestamped.timestamps import (
        AudioChunk,
        TimestampedAudio,
        TimestampEvent,
        TimestampTextChunk,
    )
    from scripts.timestamp_eval import capture
    from scripts.timestamp_eval.capture import Generator, replay

    chunks = [
        TimestampTextChunk(
            text=word,
            units=(_TextUnit(word, 0, len(word), _SourceWord(word, index, 0, len(word))),),
            token_to_unit=torch.ones(1, 1),
        )
        for index, word in enumerate(("one", "two"))
    ]

    class Model:
        sample_rate = 24000
        config = SimpleNamespace(mimi=SimpleNamespace(frame_rate=12.5))

        def get_state_for_audio_prompt(self, voice: str) -> dict[str, Any]:
            return {}

        def _prepare_timestamp_text_chunk(self, chunk: TimestampTextChunk) -> torch.Tensor:
            return torch.ones(1, 1)

        def _generate_audio_with_timestamps_short_text(
            self,
            model_state: dict[str, Any],
            timestamp_chunk: TimestampTextChunk,
            frames_after_eos: int,
            copy_state: bool,
            time_offset: float,
        ) -> EventGenerator[TimestampEvent, None, float]:
            captured = capture.model_module.SelectedAttentionCapture(
                ((0, 0),), 0, 1, timestamp_chunk.token_to_unit
            )
            # One emitted latent followed by a captured, non-emitted terminal step.
            captured.record(0, (0,), torch.ones(1, 1, 1))
            captured.record(0, (0,), torch.ones(1, 1, 1))
            yield AudioChunk(torch.full((1920,), 0.1), time_offset, time_offset + 0.08)
            return time_offset + 0.08

        def generate_audio_with_timestamps(
            self, state: dict[str, Any], text: str, copy_state: bool
        ) -> TimestampedAudio:
            audio = []
            for index, chunk in enumerate(chunks):
                self._prepare_timestamp_text_chunk(chunk)
                for event in self._generate_audio_with_timestamps_short_text(
                    model_state=state,
                    timestamp_chunk=chunk,
                    frames_after_eos=2,
                    copy_state=True,
                    time_offset=index * 0.08,
                ):
                    assert isinstance(event, AudioChunk)
                    audio.append(event.audio)
            return TimestampedAudio(torch.cat(audio), ())

    generator = object.__new__(Generator)
    generator.model = Model()  # ty: ignore[invalid-assignment]
    generator.item = {"checkpoint_sha256": "test", "baseline": []}
    generator.heads = [[0, 0]]
    generator.states = {}
    metadata = generator.generate({"voice": "alba", "seed": 1, "text": "one two"}, tmp_path)
    assert [c["offset_seconds"] for c in metadata["chunks"]] == [0, 0.08]
    assert [c["duration"] for c in metadata["chunks"]] == [0.08, 0.08]
    with np.load(tmp_path / "units.npz") as data:
        assert data["scores_0"].shape[1] == data["scores_1"].shape[1] == 1
    np.testing.assert_allclose(replay(tmp_path, [[[0, 0]]])[0], [[0, 0.08], [0.08, 0.16]])
