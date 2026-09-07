from pocket_tts_timestamped.models.model_state import export_model_state
from pocket_tts_timestamped.models.tts_model import TTSModel
from pocket_tts_timestamped.timestamps import (
    AudioChunk,
    TimestampedAudio,
    WordEnd,
    WordStart,
    WordTimestamp,
)

# Public methods:
# TTSModel.device
# TTSModel.sample_rate
# TTSModel.load_model
# TTSModel.generate_audio
# TTSModel.generate_audio_stream
# TTSModel.generate_audio_with_timestamps
# TTSModel.generate_audio_with_timestamps_stream
# TTSModel.get_state_for_audio_prompt

__all__ = [
    "AudioChunk",
    "TimestampedAudio",
    "TTSModel",
    "WordEnd",
    "WordStart",
    "WordTimestamp",
    "export_model_state",
]
