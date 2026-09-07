from scripts.normalize_upstream_namespace import _normalize_text


def test_upstream_namespace_normalization_is_idempotent() -> None:
    upstream = """\
from pocket_tts import TTSModel
from pocket_tts.models.tts_model import TTSModel as TTSModelImpl

pocket-tts = "pocket_tts.main:cli_app"
pocket-tts generate
"""
    expected = """\
from pocket_tts_timestamped import TTSModel
from pocket_tts_timestamped.models.tts_model import TTSModel as TTSModelImpl

pocket-tts-timestamped = "pocket_tts_timestamped.main:cli_app"
pocket-tts-timestamped generate
"""

    assert _normalize_text(upstream) == expected
    assert _normalize_text(expected) == expected


def test_upstream_namespace_normalization_preserves_shared_resources() -> None:
    shared_resources = """\
https://raw.githubusercontent.com/kyutai-labs/pocket-tts/refs/heads/main/pocket_tts/config/english.yaml
cache_dir = Path.home() / ".cache" / "pocket_tts"
"""

    assert _normalize_text(shared_resources) == shared_resources
