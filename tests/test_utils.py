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
        ("english", "575d14372df89a4534bcb989870a34e77544b09f"),
        ("english_2026-09_2", "575d14372df89a4534bcb989870a34e77544b09f"),
        ("french", "575d14372df89a4534bcb989870a34e77544b09f"),
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
    calibrated = load_config(CONFIGS_DIR / "english_2026-09_2.yaml").model_dump()
    assert calibrated["timestamp_heads"]
    for key in ("weights_path", "weights_path_without_voice_cloning"):
        assert default.pop(key) == calibrated.pop(key).replace(
            "/languages/english_2026-09_2/", "/languages/english/"
        )
    assert default == calibrated
