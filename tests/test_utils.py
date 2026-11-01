from pathlib import Path

import pytest

from pocket_tts_timestamped.utils import utils


def test_download_http_cache_suffix_ignores_query_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class Response:
        content = b"data"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(utils, "make_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(utils.requests, "get", lambda url: Response())

    cached_file = utils.download_if_necessary("https://example.com/audio.wav?download=true")

    assert cached_file.parent == tmp_path
    assert cached_file.suffix == ".wav"
    assert "?" not in cached_file.name
    assert cached_file.read_bytes() == b"data"


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).parent.parent / "pocket_tts_timestamped" / "config").glob("*.yaml")),
)
def test_shipped_configs_sample_at_the_tuned_temperature(config_path: Path):
    # Configs without an explicit value used to fall back to 0.7 silently (#322).
    from pocket_tts_timestamped.utils.config import load_config

    assert load_config(config_path).default_temperature == 0.3


@pytest.mark.parametrize(
    "language, revision",
    [
        ("english", "a7056f505a4a8ca72d8b5fb1440071d6f1a35739"),
        ("english_2026-09_2", "a7056f505a4a8ca72d8b5fb1440071d6f1a35739"),
        ("english_2026-10", "a7056f505a4a8ca72d8b5fb1440071d6f1a35739"),
        ("french", "a7056f505a4a8ca72d8b5fb1440071d6f1a35739"),
    ],
)
def test_predefined_voices_match_checkpoint_revisions(language: str, revision: str):
    assert utils.get_predefined_voice(language, "alba") == (
        "hf://kyutai/pocket-tts-without-voice-cloning/"
        f"languages/{language}/embeddings/alba.safetensors@{revision}"
    )


def test_default_english_matches_calibrated_checkpoint():
    from pocket_tts_timestamped.default_parameters import DEFAULT_LANGUAGE
    from pocket_tts_timestamped.utils.config import CONFIGS_DIR, load_config

    default = load_config(CONFIGS_DIR / f"{DEFAULT_LANGUAGE}.yaml").model_dump()
    calibrated = load_config(CONFIGS_DIR / "english_2026-10.yaml").model_dump()
    assert calibrated["timestamp_heads"]
    for key in ("weights_path", "weights_path_without_voice_cloning"):
        assert default.pop(key) == calibrated.pop(key).replace(
            "/languages/english_2026-10/", "/languages/english/"
        )
    # Upstream reads the same tokenizer revision through the `english` directory.
    tokenizers = [c["flow_lm"]["lookup_table"].pop("tokenizer_path") for c in (default, calibrated)]
    assert tokenizers[0] == tokenizers[1].replace(
        "/languages/english_2026-09/", "/languages/english/"
    )
    assert default == calibrated


def test_eos_threshold_defaults_to_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """load_model takes the EOS threshold from the config unless the caller passes one."""
    import yaml

    from pocket_tts_timestamped import TTSModel
    from pocket_tts_timestamped.default_parameters import DEFAULT_EOS_THRESHOLD

    config = Path(__file__).parent.parent / "pocket_tts_timestamped" / "config" / "english.yaml"
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump({**yaml.safe_load(config.read_text()), "default_eos_threshold": -1.0})
    )
    seen = []
    monkeypatch.setattr(
        TTSModel,
        "_from_pydantic_config_with_weights",
        classmethod(lambda cls, cfg, temp, steps, clamp, eos, origin: seen.append(eos)),
    )
    TTSModel.load_model(config=config)
    TTSModel.load_model(config=custom)
    TTSModel.load_model(config=custom, eos_threshold=-2.0)
    assert seen == [DEFAULT_EOS_THRESHOLD, -1.0, -2.0]
