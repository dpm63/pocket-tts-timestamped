"""Session: frame-by-frame timestamped generation with replaceable text, on a real model."""

from typing import Any

import pytest
import torch

from pocket_tts_timestamped import AudioChunk, TTSModel, WordEnd, WordStart
from pocket_tts_timestamped.modules.stateful_module import ModelState

ModelAndVoice = tuple[TTSModel, ModelState]

TEXT = "El café está aquí, la casa está allí y el jardín es muy bonito."


@pytest.fixture(scope="module")
def model_and_voice() -> ModelAndVoice:
    model = TTSModel.load_model(language="spanish")
    # A built-in voice works with the weights without voice cloning too (CI has no HF_TOKEN).
    # A sine-tone prompt doesn't: that model then stops after a few words.
    return model, model.get_state_for_audio_prompt("lola")


def test_replay_continues_like_incremental_generation(model_and_voice: ModelAndVoice) -> None:
    """Re-reading the retained audio in one pass reproduces the incremental FlowLM state."""
    model, voice = model_and_voice
    session = model.start_session(voice)
    window = session.prepare_text(TEXT)
    session.set_text(window)
    torch.manual_seed(0)
    for _ in range(8):
        session.generate_frame()
    rng = torch.get_rng_state()
    expected = session.generate_frame()

    session.discard(expected)
    session.set_text(window, history_from=0)
    torch.set_rng_state(rng)
    actual = session.generate_frame()

    assert actual.index == expected.index
    torch.testing.assert_close(actual.latent, expected.latent, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual.scores, expected.scores, atol=1e-4, rtol=1e-4)


def test_eos_gate_counts_replayed_audio(
    model_and_voice: ModelAndVoice, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EOS is ignored until the FlowLM has read 6 frames of audio, replayed ones included."""
    model, voice = model_and_voice
    run_flow_lm = model._run_flow_lm_and_increment_step

    def always_eos(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401
        latent, _ = run_flow_lm(**kwargs)
        return latent, torch.tensor([True])

    monkeypatch.setattr(model, "_run_flow_lm_and_increment_step", always_eos)
    session = model.start_session(voice)
    window = session.prepare_text(TEXT)
    session.set_text(window)
    assert [session.generate_frame().eos for _ in range(7)] == [False] * 6 + [True]

    session.set_text(window, history_from=4)  # replays 3 frames
    assert [session.generate_frame().eos for _ in range(4)] == [False] * 3 + [True]

    session.set_text(window)  # a new utterance from BOS
    assert not any(session.generate_frame().eos for _ in range(6))


def test_window_reports_positions_in_the_whole_text(model_and_voice: ModelAndVoice) -> None:
    model, voice = model_and_voice
    session = model.start_session(voice)
    start = TEXT.index("la casa")
    window = session.prepare_text(TEXT, start)
    words = window.chunk.words
    assert words[0].text == "la" and words[0].word_index == 4
    assert all(TEXT[word.begin : word.end] == word.text for word in words)
    # A window that starts mid-text continues a sentence: no capital, unlike the text start.
    assert window.chunk.text.startswith("la casa")
    assert session.prepare_text("el café").chunk.text.startswith("El café")


def test_text_switch_keeps_one_word_sequence(model_and_voice: ModelAndVoice) -> None:
    """Expose a prefix, switch to the full text mid-word, replaying the audio: one stream of
    word events over the whole text, in order and located in it."""
    model, voice = model_and_voice
    session = model.start_session(voice)
    prefix_end = TEXT.index(" la casa")
    session.set_text(session.prepare_text(TEXT[:prefix_end], append_terminal_punctuation=False))
    torch.manual_seed(1)
    events = []
    switched = False
    for _ in range(200):
        frame = session.generate_frame()
        events += session.decode([frame])
        if not switched and any(
            isinstance(event, WordEnd) and event.word == "está" for event in events
        ):
            # Like a held-back switch: the prefix's last word ("aquí,") is still unspoken.
            session.set_text(session.prepare_text(TEXT), history_from=0)
            switched = True
        elif switched and frame.eos:
            break
    events += session.finish()

    assert switched
    starts = [event for event in events if isinstance(event, WordStart)]
    ends = [event for event in events if isinstance(event, WordEnd)]
    assert [event.word_index for event in starts] == sorted({e.word_index for e in starts})
    assert [event.word_index for event in ends] == [event.word_index for event in starts]
    assert max(event.word_index for event in ends) >= 6  # well past the prefix
    for event in starts + ends:
        assert TEXT[event.char_start : event.char_end] == event.word
    audio = [event for event in events if isinstance(event, AudioChunk)]
    assert [event.start_time for event in audio[1:]] == [event.end_time for event in audio[:-1]]
