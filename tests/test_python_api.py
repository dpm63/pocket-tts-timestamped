"""Tests for the public Python API surface."""

import pocket_tts_timestamped
from pocket_tts_timestamped import TTSModel
from pocket_tts_timestamped.models.tts_model import TTSModel as TTSModelImpl


def test_public_api_exports_expected_symbols():
    expected_symbols = [
        "AudioChunk",
        "TimestampedAudio",
        "TTSModel",
        "WordEnd",
        "WordStart",
        "WordTimestamp",
        "export_model_state",
    ]
    assert pocket_tts_timestamped.__all__ == expected_symbols
    assert all(hasattr(pocket_tts_timestamped, symbol) for symbol in expected_symbols)


def test_public_api_tts_model_points_to_implementation():
    assert TTSModel is TTSModelImpl


def test_public_api_expected_methods_and_properties():
    for method_name in (
        "load_model",
        "generate_audio",
        "generate_audio_stream",
        "generate_audio_with_timestamps",
        "generate_audio_with_timestamps_stream",
        "get_state_for_audio_prompt",
    ):
        assert callable(getattr(TTSModel, method_name))

    for property_name in ("device", "sample_rate"):
        assert isinstance(getattr(TTSModel, property_name), property)
