"""The training path (hooks over cached, left-padded prefixes) matches the native vectors."""

from pathlib import Path

import safetensors.torch
import scipy.io.wavfile
import torch

from pocket_tts_timestamped import TTSModel
from scripts.leading_silence.pipeline import silence_latents
from scripts.leading_silence.sequences import (
    encode_voice,
    prefill,
    prefix_embeddings,
    row_key_values,
    student_forward,
    tokenize,
)
from scripts.leading_silence.steer import Steer

TEXTS = ["Hello there.", "The keeper opened the gate and invited us inside."]


def test_training_forward_matches_native_vectors(tmp_path: Path) -> None:
    model = TTSModel.load_model(language="spanish")
    positions = torch.arange(model.sample_rate, dtype=torch.float32)
    prompt = 0.1 * torch.sin(2 * torch.pi * 220 * positions / model.sample_rate)
    voice_file = tmp_path / "voice.wav"
    scipy.io.wavfile.write(voice_file, model.sample_rate, (prompt * 32767).short().numpy())

    flow_lm = model.flow_lm
    steer = Steer(flow_lm, [1.0] * (len(flow_lm.transformer.layers) + 1))
    torch.manual_seed(0)
    steer.theta.data.normal_(0, 0.05)
    with torch.no_grad():
        voice = encode_voice(model, voice_file)
        # Two rows of different lengths: the shorter one is left-padded in the batch.
        prefixes = [prefix_embeddings(model, voice, tokenize(model, text)) for text in TEXTS]
        state, pad, width = prefill(model, prefixes, 0)
        key_values = [row_key_values(model, state, row, pad, width) for row in range(2)]
        steer.attach(flow_lm)
        try:
            z_training, _ = student_forward(
                model, steer, key_values, torch.zeros(2, 0, flow_lm.ldim)
            )
        finally:
            steer.detach_hooks()

    vectors = tmp_path / "vectors.safetensors"
    tensors = {"vectors": steer.vectors().detach(), "warmup_latents": silence_latents(model, 8)}
    safetensors.torch.save_file(tensors, str(vectors))
    model._load_leading_silence_vectors(vectors)
    state_for_voice = model.get_state_for_audio_prompt(voice_file)
    captured: list[torch.Tensor] = []
    handle = flow_lm.out_norm.register_forward_hook(lambda m, a, out: captured.append(out[0, -1]))
    try:
        for row, text in enumerate(TEXTS):
            session = model.start_session(state_for_voice)
            session.set_text(session.prepare_text(text))
            captured.clear()
            session.generate_frame()
            torch.testing.assert_close(captured[-1].float(), z_training[row], atol=1e-4, rtol=1e-4)
    finally:
        handle.remove()
