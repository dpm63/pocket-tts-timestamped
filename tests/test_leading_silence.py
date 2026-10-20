"""Leading-silence vectors: applied on a lone BOS step only, loaded with a warm Mimi decoder."""

from pathlib import Path

import pytest
import safetensors.torch
import torch

from pocket_tts_timestamped import AudioChunk, TTSModel
from pocket_tts_timestamped.models.flow_lm import FlowLMModel
from pocket_tts_timestamped.modules.stateful_module import (
    ModelState,
    StatefulModule,
    increment_steps,
    init_states,
)
from pocket_tts_timestamped.utils.config import CONFIGS_DIR, FlowLMConfig, load_config


def small_flow_lm() -> FlowLMModel:
    torch.manual_seed(0)
    config = load_config(CONFIGS_DIR / "english.yaml").flow_lm.model_dump()
    config["transformer"].update(d_model=64, num_heads=4, num_layers=2)
    config["lookup_table"]["dim"] = 64
    config["flow"].update(dim=32, depth=1)
    flow_lm = FlowLMModel.from_pydantic_config(
        FlowLMConfig(**config), latent_dim=32, insert_bos_before_voice=True
    )
    for name, module in flow_lm.named_modules():
        if isinstance(module, StatefulModule):
            module._module_absolute_name = name
    return flow_lm.eval()


@torch.no_grad()
def run_steps(flow_lm: FlowLMModel, vectors: torch.Tensor | None) -> list[torch.Tensor]:
    """Text prefill, a lone BOS step, one more step, then a replay (text + BOS + a fixed
    latent) in a fresh state."""
    flow_lm.register_buffer("leading_silence_vectors", vectors, persistent=False)
    text = flow_lm.conditioner(flow_lm.conditioner.prepare("Hello world."))
    empty_text = text[:, :0]
    bos = torch.full((1, 1, flow_lm.ldim), float("nan"))
    state = init_states(flow_lm, batch_size=1, sequence_length=64)

    def step(sequence: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(1)
        latent, _ = flow_lm(sequence, text_embeddings, state, 1, 0.3, None, -4.0)
        increment_steps(flow_lm, state, text_embeddings.shape[1] + sequence.shape[1])
        return latent[:, None]

    prefill = step(bos[:, :0], text)
    first = step(bos, empty_text)
    second = step(first, empty_text)
    replay_state: ModelState = init_states(flow_lm, batch_size=1, sequence_length=64)
    state = replay_state
    replay = step(torch.cat([bos, torch.zeros_like(first)], dim=1), text)
    return [prefill, first, second, replay]


def test_vectors_apply_only_on_a_lone_bos_step() -> None:
    flow_lm = small_flow_lm()
    vectors = 0.5 * torch.randn(3, 64)
    baseline = run_steps(flow_lm, None)
    steered = run_steps(flow_lm, vectors)
    zero = run_steps(flow_lm, torch.zeros(3, 64))

    assert torch.equal(steered[0], baseline[0])  # text prefill
    assert not torch.allclose(steered[1], baseline[1])  # the BOS step
    assert torch.equal(steered[3], baseline[3])  # BOS inside a replay
    for zero_step, baseline_step in zip(zero, baseline, strict=True):
        torch.testing.assert_close(zero_step, baseline_step)


@pytest.fixture(scope="module")
def spanish() -> tuple[TTSModel, ModelState]:
    model = TTSModel.load_model(language="spanish")
    positions = torch.arange(model.sample_rate, dtype=torch.float32)
    prompt = (0.1 * torch.sin(2 * torch.pi * 220 * positions / model.sample_rate))[None]
    return model, model.get_state_for_audio_prompt(prompt)


def write_vectors(model: TTSModel, path: Path, vectors: torch.Tensor) -> Path:
    """A vector file whose warm-up latents are Mimi-encoded digital silence."""
    silence = torch.zeros(1, 1, 8 * model.mimi.frame_size)
    raw = model.mimi.encode_to_latent(silence)[0]
    warmup = ((raw - model.flow_lm.emb_mean) / model.flow_lm.emb_std).contiguous()
    safetensors.torch.save_file({"vectors": vectors, "warmup_latents": warmup}, path)
    return path


@torch.no_grad()
def first_frames(model: TTSModel, voice: ModelState) -> tuple[torch.Tensor, torch.Tensor]:
    """Latents and audio of the first 3 frames of a session, with a fixed seed."""
    session = model.start_session(voice)
    session.set_text(session.prepare_text("Hola, buenos días."))
    torch.manual_seed(3)
    frames = [session.generate_frame() for _ in range(3)]
    audio = [event.audio for event in session.decode(frames) if isinstance(event, AudioChunk)]
    return torch.cat([frame.latent for frame in frames]), torch.cat(audio)


def test_loaded_vectors_steer_the_first_frame_and_warm_the_decoder(
    spanish: tuple[TTSModel, ModelState], tmp_path: Path
) -> None:
    baseline_model, voice = spanish
    shape = (baseline_model.config.flow_lm.transformer.num_layers + 1, baseline_model.flow_lm.dim)
    baseline_latents, baseline_audio = first_frames(baseline_model, voice)
    assert baseline_audio[0] == 0  # a cold decoder fades the first frame in

    zero_path = write_vectors(baseline_model, tmp_path / "zero.safetensors", torch.zeros(shape))
    zero_model = TTSModel.load_model(language="spanish", skip_leading_silence=zero_path)
    zero_latents, zero_audio = first_frames(zero_model, voice)
    assert torch.equal(zero_latents, baseline_latents)  # zero vectors change no latent
    assert zero_audio[0] != 0  # a warm decoder needs no fade

    torch.manual_seed(0)
    steer_path = write_vectors(baseline_model, tmp_path / "s.safetensors", torch.randn(shape))
    steered_model = TTSModel.load_model(language="spanish", skip_leading_silence=steer_path)
    steered_latents, _ = first_frames(steered_model, voice)
    assert not torch.allclose(steered_latents[0], baseline_latents[0])


def test_vector_files_are_validated(spanish: tuple[TTSModel, ModelState], tmp_path: Path) -> None:
    model, _ = spanish
    wrong = write_vectors(model, tmp_path / "wrong.safetensors", torch.zeros(2, 3))
    with pytest.raises(ValueError, match="Leading-silence vectors have shape"):
        model._load_leading_silence_vectors(wrong)
    assert model.flow_lm.leading_silence_vectors is None


def test_checkpoint_without_vectors_fails_before_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("weights were loaded")

    monkeypatch.setattr(TTSModel, "_from_pydantic_config_with_weights", fail)
    with pytest.raises(ValueError, match="no leading-silence vectors"):
        TTSModel.load_model(language="spanish", skip_leading_silence=True)
