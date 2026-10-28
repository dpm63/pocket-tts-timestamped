import copy
import logging
import math
import os
import queue
import statistics
import threading
import time
from collections.abc import Generator, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urljoin

import safetensors
import safetensors.torch
import scipy.io.wavfile
import torch
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self

from pocket_tts_timestamped.data.audio import audio_read
from pocket_tts_timestamped.data.audio_utils import convert_audio, end_on_pause, fade_in_audio
from pocket_tts_timestamped.default_parameters import (
    DEFAULT_LANGUAGE,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_SAMPLER_DECODE_STEPS,
    MAX_TOKEN_PER_CHUNK,
    MAX_TOKEN_PER_CHUNK_TEACHER_FORCED,
)
from pocket_tts_timestamped.models.flow_lm import FlowLMModel
from pocket_tts_timestamped.models.mimi import build_mimi
from pocket_tts_timestamped.models.model_state import _import_model_state, _is_safetensors_source
from pocket_tts_timestamped.models.session import Frame, Session, TextWindow
from pocket_tts_timestamped.models.text_chunking import (
    prepare_text_prompt,
    split_into_best_sentences,
)
from pocket_tts_timestamped.modules.attention import StreamingMultiheadAttention
from pocket_tts_timestamped.modules.stateful_module import (
    ModelState,
    StatefulModule,
    increment_steps,
    init_states,
)
from pocket_tts_timestamped.quantization import RECOMMENDED_CONFIG, apply_dynamic_int8
from pocket_tts_timestamped.timestamps.alignment import SelectedAttentionCapture
from pocket_tts_timestamped.timestamps.records import (
    AudioChunk,
    TimestampedAudio,
    TimestampEvent,
    WordEnd,
    WordTimestamp,
)
from pocket_tts_timestamped.timestamps.text import (
    TimestampTextChunk,
    _continuation_timestamp_text_chunk,
    _iter_timestamp_text_chunks,
)
from pocket_tts_timestamped.utils.config import CONFIGS_DIR, Config, load_config
from pocket_tts_timestamped.utils.utils import (
    _ORIGINS_OF_PREDEFINED_VOICES,
    DEBUG_MIMI,
    display_execution_time,
    download_if_necessary,
    get_predefined_voice,
    size_of_dict,
)
from pocket_tts_timestamped.utils.weights_loading import (
    get_flow_lm_state_dict,
    get_mimi_state_dict,
    get_training_checkpoint_state_dicts,
)

torch.set_num_threads(1)
logger = logging.getLogger(__name__)

# Generated latents handed to a Mimi decoder thread; timestamp generation hands over
# session frames instead. None closes either stream.
LatentItem = torch.Tensor | Frame
LatentQueue = queue.Queue[LatentItem | None]
# Decoder thread -> generator: ("chunk", audio), ("done", None) or ("error", exception).
ResultQueue = queue.Queue[tuple[str, Any]]


class _PreparedTextChunk(NamedTuple):
    # The chunk on its own, as the model reads it.
    text: str
    frames_after_eos: int
    # With teacher_force_previous_chunk, the previous chunk's text and this one's, prepared
    # together; the chunk is generated from it when the previous chunk's frames are forced.
    continuation_text: str | None = None


def _iter_latent_batches(
    latents_queue: LatentQueue, max_frames: int, cancel_event: threading.Event | None = None
) -> Iterator[list[LatentItem]]:
    """Decode queued frames together without waiting for another frame to arrive."""
    while cancel_event is None or not cancel_event.is_set():
        item = latents_queue.get()
        if item is None or (cancel_event is not None and cancel_event.is_set()):
            return
        batch = [item]
        finished = False
        while max_frames <= 0 or len(batch) < max_frames:
            try:
                item = latents_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                finished = True
                break
            batch.append(item)
        yield batch
        if finished:
            return


def _beside_config(config_path: str, relative: str) -> str:
    """`relative` resolved against the directory of `config_path` (local, hf:// or http(s))."""
    if config_path.startswith("hf://"):
        path, _, revision = config_path.partition("@")
        return f"{path.rpartition('/')[0]}/{relative}" + (f"@{revision}" if revision else "")
    if config_path.startswith(("http://", "https://")):
        return urljoin(config_path, relative)
    return str(Path(config_path).parent / relative)


def stamp_state_names(tts_model: "TTSModel") -> None:
    """StatefulModules find their slice of the state dict by absolute name."""
    for top_module in (tts_model.flow_lm, tts_model.mimi):
        for module_name, module in top_module.named_modules():
            if isinstance(module, StatefulModule):
                module._module_absolute_name = module_name


VOICE_CLONING_UNSUPPORTED = (
    f"We could not download the weights for the model with voice cloning, "
    f"but you're trying to use voice cloning. "
    f"Without voice cloning, you can use our catalog of voices {list(_ORIGINS_OF_PREDEFINED_VOICES.keys())}. "
    f"If you want access to the model with voice cloning, go to "
    f"https://huggingface.co/kyutai/pocket-tts and accept the terms, "
    f"then make sure you're logged in locally with `uvx hf auth login`."
)


class TTSModel(nn.Module):
    _TOKENS_PER_SECOND_ESTIMATE = 3.0
    _GEN_SECONDS_PADDING = 2.0
    # EOS is ignored on the first frames: before speech starts, the EOS logit of some voices can
    # cross the threshold, and a short text then ends before the word is spoken.
    _MIN_FRAMES_BEFORE_EOS = 6
    # ((device, dtype), decoder state) after the leading-silence warm-up; see _new_mimi_state.
    _warm_mimi_cache: tuple[tuple[torch.device, torch.dtype], ModelState] | None = None
    # teacher_force_previous_chunk: the Mimi decoder of a chunk first runs over the last forced
    # frames (about 1 s), so that its audio follows on from the previous chunk's.
    _DECODER_WARMUP_FRAMES = 12

    def __init__(
        self,
        flow_lm: FlowLMModel,
        temp: float,
        sampler_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        config: Config,
        origin: Path | None = None,
        pad_with_spaces_for_short_inputs: bool = False,
        model_recommended_frames_after_eos: int | None = None,
        remove_semicolons: bool = False,
        append_terminal_punctuation: bool = True,
        capitalize_first_letter: bool = True,
        replace_characters: dict[str, str] | None = None,
    ):
        super().__init__()
        self.flow_lm = flow_lm
        # 0 = decode every queued frame in one call; 1 = one frame per call
        self.max_decoder_frames_per_call = 0
        self.temp = temp
        self.sampler_decode_steps = sampler_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.has_voice_cloning = True
        self.origin = origin
        self.pad_with_spaces_for_short_inputs: bool = pad_with_spaces_for_short_inputs
        self.model_recommended_frames_after_eos = model_recommended_frames_after_eos
        self.remove_semicolons = remove_semicolons
        self.append_terminal_punctuation = append_terminal_punctuation
        self.capitalize_first_letter = capitalize_first_letter
        self.replace_characters = replace_characters or {}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def sample_rate(self) -> int:
        return self.config.mimi.sample_rate

    @classmethod
    def _from_pydantic_config(
        cls,
        config: Config,
        temp: float,
        sampler_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        origin: Path | None,
    ) -> Self:
        flow_lm = FlowLMModel.from_pydantic_config(
            config.flow_lm,
            latent_dim=config.mimi.quantizer.dimension,
            insert_bos_before_voice=config.flow_lm.insert_bos_before_voice,
        )
        tts_model = cls(
            flow_lm,
            temp,
            sampler_decode_steps,
            noise_clamp,
            eos_threshold,
            config,
            origin=origin,
            pad_with_spaces_for_short_inputs=config.pad_with_spaces_for_short_inputs,
            model_recommended_frames_after_eos=config.model_recommended_frames_after_eos,
            remove_semicolons=config.remove_semicolons,
            append_terminal_punctuation=config.append_terminal_punctuation,
            capitalize_first_letter=config.capitalize_first_letter,
            replace_characters=config.replace_characters,
        )
        return tts_model

    has_custom_weights: bool = False

    def load_training_checkpoint(self, path: str | Path, use_ema: bool = True):
        """Load weights from a training checkpoint (.pt) into a config-built model."""
        self.has_custom_weights = True
        flow_lm_state, mimi_state = get_training_checkpoint_state_dicts(Path(path), use_ema)
        config = self.config
        self.flow_lm.speaker_proj_weight = torch.nn.Parameter(
            torch.zeros(
                (
                    config.flow_lm.transformer.d_model,
                    config.mimi.inner_dim or config.mimi.seanet.dimension,
                ),
                dtype=torch.float32,
            )
        )
        self.flow_lm.load_state_dict(flow_lm_state, strict=True)
        self.mimi = build_mimi(config.mimi).to(device="cpu")
        # Training freezes Mimi and does not checkpoint it, so its weights come
        # from the config -- either its own safetensors or the packaged bundle.
        if mimi_state:
            self.mimi.load_state_dict(mimi_state, strict=False)
        elif config.mimi.weights_path is not None:
            self.mimi.load_state_dict(
                get_mimi_state_dict(download_if_necessary(config.mimi.weights_path)), strict=True
            )
        else:
            if config.weights_path is None:
                raise ValueError("no Mimi weights: set mimi.weights_path or weights_path")
            try:
                bundle = download_if_necessary(config.weights_path)
            except Exception:
                if config.weights_path_without_voice_cloning is None:
                    raise
                self.has_voice_cloning = False
                bundle = download_if_necessary(config.weights_path_without_voice_cloning)
            bundled = safetensors.torch.load_file(bundle)
            self.mimi.load_state_dict(
                {k.removeprefix("mimi."): v for k, v in bundled.items() if k.startswith("mimi.")},
                strict=True,
            )
        self.mimi.eval()
        stamp_state_names(self)

    @classmethod
    def _from_pydantic_config_with_weights(
        cls,
        config: Config,
        temp: float,
        sampler_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        origin: Path | None = None,
    ) -> Self:
        tts_model = cls._from_pydantic_config(
            config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
        )
        tts_model.flow_lm.speaker_proj_weight = torch.nn.Parameter(
            torch.zeros(
                (
                    config.flow_lm.transformer.d_model,
                    config.mimi.inner_dim or config.mimi.seanet.dimension,
                ),
                dtype=torch.float32,
            )
        )
        if config.flow_lm.weights_path is not None:
            if config.mimi.weights_path is None:
                raise ValueError(
                    "If you specify flow_lm.weights_path you should specify mimi.weights_path"
                )
            logger.info(f"Loading FlowLM weights from {config.flow_lm.weights_path}")
            state_dict_flowlm = get_flow_lm_state_dict(
                download_if_necessary(config.flow_lm.weights_path)
            )
            tts_model.flow_lm.load_state_dict(state_dict_flowlm, strict=True)

        tts_model.mimi = build_mimi(config.mimi).to(device="cpu")

        # Load mimi weights from the config safetensors file with complete mapping for strict loading

        if config.mimi.weights_path is not None:
            if config.flow_lm.weights_path is None and config.weights_path is None:
                raise ValueError(
                    "If you specify mimi.weights_path you should specify flow_lm.weights_path "
                    "or weights_path"
                )
            logger.info(f"Loading Mimi weights from {config.mimi.weights_path}")
            mimi_state = get_mimi_state_dict(download_if_necessary(config.mimi.weights_path))
            tts_model.mimi.load_state_dict(mimi_state, strict=True)

        tts_model.mimi.eval()

        if config.weights_path is not None:
            logger.info(f"Loading TTSModel weights from {config.weights_path}")
            try:
                weights_file = download_if_necessary(config.weights_path)
            except Exception:
                if config.weights_path_without_voice_cloning is None:
                    raise
                tts_model.has_voice_cloning = False
                weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

            state_dict = safetensors.torch.load_file(weights_file)
            if config.mimi.weights_path is not None:
                # mimi.weights_path, loaded above, replaces the Mimi weights of the bundle.
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mimi.")}
                state_dict.update({f"mimi.{k}": v for k, v in tts_model.mimi.state_dict().items()})
            tts_model.load_state_dict(state_dict, strict=True)

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning(
                "No weights_path specified for FlowLM or TTSModel, model is uninitialized!"
            )
        size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6
        if os.environ.get("POCKET_TTS_SAVE_WEIGHTS", "0") == "1":
            save_path = "./model.safetensors"
            safetensors.torch.save_file(tts_model.state_dict(), save_path)
            logger.info(f"Saved TTSModel weights to {save_path}")
        logging.info(f"TTS Model loaded successfully. Its size is {size_in_mb} MB")

        stamp_state_names(tts_model)
        return tts_model

    @classmethod
    def load_model(
        cls,
        language: str | None = None,
        config: str | Path | None = None,
        temp: float | None = None,
        sampler_decode_steps: int = DEFAULT_SAMPLER_DECODE_STEPS,
        noise_clamp: float | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float | None = None,
        quantize: bool = False,
        checkpoint: str | Path | None = None,
        lsd_decode_steps: int | None = None,
        skip_leading_silence: bool | str | Path = False,
    ) -> Self:
        """Load a pre-trained TTS model with specified configuration.

        This class method loads a complete TTS model including the flow language model
        and Mimi compression model from pre-trained weights. The model is initialized
        with the specified generation parameters and ready for inference.

        Args:
            language: Optional language identifier to select a predefined config. Incompatible with
                the `config` argument. Available options
                are `"english_2026-01"`, `"english_2026-04"`, `"english_2026-04_24l"`,
                `"english_2026-09"`, `"english_2026-09_2"`, `"english_2026-09_24l"`,
                `"english_2026-10"`, `"english_drifting_26-09"`, `"english_18m_2026-10"`,
                `"english_18m_refl_2026-10"`, `"english_7m_2026-10"`, `"english_7m_refl_2026-10"`,
                `"english_2m_2026-10"`, `"english_2m_refl_2026-10"`, `"english"`, `"french"`,
                `"french_24l"`, `"dutch"`, `"dutch_24l"`, `"german"`, `"german_24l"`,
                `"italian"`, `"italian_24l"`, `"portuguese"`, `"portuguese_24l"`, `"spanish"`,
                `"spanish_24l"`.
                If neither `config` nor `language` is provided, defaults to `"english"`,
                the same model as `"english_2026-10"`.
            config: A path to a custom YAML config file: a local path, an `https://`
                URL, or an `hf://` path (e.g. `"hf://<repo_id>/<path>[@revision]"`).
            temp: Sampling temperature for generation. Higher values produce more
                diverse but potentially lower quality output. If None, defaults to
                the model's recommended value from its config file
                (``default_temperature``, 0.3).
            sampler_decode_steps: Number of steps for Lagrangian Self Distillation
                decoding. More steps can improve quality but increase computation.
            noise_clamp: Maximum value for noise sampling. If None, no clamping
                is applied. Helps prevent extreme values in generation.
            eos_threshold: Threshold for end-of-sequence detection. Higher values
                make the model more likely to continue generating. If None, defaults to
                the model's value from its config file (``default_eos_threshold``, -4.0).
            checkpoint: Optional training checkpoint (.pt) to load instead of the
                config's safetensors weights, so any step of a run can be played
                without exporting first. EMA weights are used when the checkpoint
                carries them.
            quantize: If True, apply dynamic int8 quantization to the transformer's
                attention and FFN layers. Reduces runtime memory by ~48% and improves
                inference speed by ~27% on x86 (FBGEMM).
                No measurable impact on speech quality (WER unchanged).
                For optimized performance, install torchao:
                ``pip install pocket-tts-timestamped[quantize]``
            skip_leading_silence: Start speaking on the first frame instead of after the
                0.5-1 s of silence the model usually generates first. True uses the vectors
                that ship with the checkpoint (an error if it has none); a path loads your own
                (safetensors with `vectors` and `warmup_latents`). Off by default, which keeps
                generation identical to upstream pocket-tts.

        Returns:
            TTSModel: Fully initialized model with loaded weights on cpu, ready for
                text-to-speech generation.

        Raises:
            FileNotFoundError: If the specified config file or model weights
                are not found.
            ValueError: If the configuration is invalid or incompatible.

        Example:
            ```python
            from pocket_tts_timestamped import TTSModel

            # Load with default settings
            model = TTSModel.load_model()

            # Load with int8 quantization
            model = TTSModel.load_model(quantize=True)
            ```
        """
        if config is not None and language is not None:
            raise ValueError(
                "Cannot specify both config and language, please choose one or the other."
            )
        if config is None and language is None:
            language = DEFAULT_LANGUAGE
        if language is not None:
            config = CONFIGS_DIR / f"{language}.yaml"
        if lsd_decode_steps is not None:
            logger.warning("lsd_decode_steps is deprecated, use sampler_decode_steps")
            sampler_decode_steps = lsd_decode_steps

        config_path = str(config)
        # hf:// paths may carry a "@revision" suffix, strip it before checking the extension.
        suffix_source = (
            config_path.split("@")[0] if config_path.startswith("hf://") else config_path
        )
        if Path(suffix_source).suffix not in (".yaml", ".yml"):
            raise ValueError("Config should be a path to a YAML file ending with .yaml")
        model_config = load_config(config_path)
        if temp is None:
            temp = model_config.default_temperature
        if eos_threshold is None:
            eos_threshold = model_config.default_eos_threshold
        leading_silence_source = None
        if skip_leading_silence is True:
            leading_silence_source = model_config.leading_silence_vectors
            if leading_silence_source is None:
                raise ValueError(
                    f"The config at {config_path} has no leading-silence vectors; pass a path "
                    "to skip_leading_silence to use your own."
                )
            if not leading_silence_source.startswith(("hf://", "http://", "https://")):
                leading_silence_source = _beside_config(config_path, leading_silence_source)
        elif skip_leading_silence is not False:
            leading_silence_source = str(skip_leading_silence)
        logger.info(f"Loading model from config at {config_path}...")

        origin = Path(config_path)
        if checkpoint is not None:
            tts_model = cls._from_pydantic_config(
                model_config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
            )
            tts_model.load_training_checkpoint(checkpoint)
        else:
            tts_model = cls._from_pydantic_config_with_weights(
                model_config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
            )

        if quantize:
            apply_dynamic_int8(tts_model.flow_lm, RECOMMENDED_CONFIG)

        if leading_silence_source is not None:
            tts_model._load_leading_silence_vectors(download_if_necessary(leading_silence_source))

        return tts_model

    def _load_leading_silence_vectors(self, path: Path) -> None:
        """Load leading-silence vectors and warm up the Mimi decoder they start into.

        The file holds `vectors` ([num_layers + 1, d_model], added to the FlowLM residual
        stream on a lone BOS step) and `warmup_latents` ([frames, ldim], normalized latents of
        silence). A cold decoder makes an instant start too loud, so every decoder state starts
        from one that has decoded the warm-up latents.
        """
        tensors = safetensors.torch.load_file(str(path))
        vectors, warmup = tensors["vectors"], tensors["warmup_latents"]
        transformer = self.config.flow_lm.transformer
        expected = (transformer.num_layers + 1, transformer.d_model)
        if tuple(vectors.shape) != expected:
            raise ValueError(
                f"Leading-silence vectors have shape {tuple(vectors.shape)}, expected {expected}"
            )
        if warmup.dim() != 2 or warmup.shape[1] != self.flow_lm.ldim:
            raise ValueError(
                f"Warm-up latents have shape {tuple(warmup.shape)}, expected [frames, "
                f"{self.flow_lm.ldim}]"
            )
        decoder_attention = [
            module
            for module in self.mimi.decoder_transformer.modules()
            if isinstance(module, StreamingMultiheadAttention)
        ]
        if any(module.context is None for module in decoder_attention):
            # The warm state is copied per utterance, so its caches must be able to grow.
            raise NotImplementedError("Leading-silence warm-up needs a finite-context decoder")
        device = self.flow_lm.bos_emb.device
        self.flow_lm.register_buffer(
            "leading_silence_vectors", vectors.to(device), persistent=False
        )
        self.flow_lm.register_buffer("leading_silence_warmup", warmup.to(device), persistent=False)
        self._new_mimi_state(0)  # warm up now rather than on the first utterance

    def _new_mimi_state(
        self, sequence_length: int, continue_from: torch.Tensor | None = None
    ) -> tuple[ModelState, bool]:
        """A decoder state for new audio, and whether it is warm.

        `continue_from` ([1, frames, ldim]) is the end of the audio played before
        (teacher_force_previous_chunk): a fresh decoder runs over it, so the new audio follows
        on. Otherwise, with leading-silence vectors it is a copy of a decoder that has decoded
        the warm-up latents, built once per device and dtype; without, a fresh state, whose
        first frame needs fading in.
        """
        if continue_from is not None:
            steps_per_latent = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
            steps = continue_from.shape[1] * steps_per_latent
            state = init_states(self.mimi, batch_size=1, sequence_length=sequence_length + steps)
            self._warm_up_decoder(continue_from, state)
            return state, True
        warmup = self.flow_lm.leading_silence_warmup
        if warmup is None:
            return init_states(self.mimi, batch_size=1, sequence_length=sequence_length), False
        weight = next(self.mimi.parameters())
        key = (weight.device, weight.dtype)
        if self._warm_mimi_cache is None or self._warm_mimi_cache[0] != key:
            steps_per_latent = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
            steps = warmup.shape[0] * steps_per_latent
            state = init_states(self.mimi, batch_size=1, sequence_length=steps)
            self._warm_up_decoder(warmup[None], state)
            self._warm_mimi_cache = (key, state)
        return copy.deepcopy(self._warm_mimi_cache[1]), True

    @torch.no_grad
    def _warm_up_decoder(self, latents: torch.Tensor, mimi_state: ModelState) -> None:
        """Decode normalized `latents` ([1, frames, ldim]) into `mimi_state`, dropping the audio."""
        self.mimi.decode_from_latent(
            latents * self.flow_lm.emb_std + self.flow_lm.emb_mean, mimi_state
        )
        steps_per_latent = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        increment_steps(self.mimi, mimi_state, increment=steps_per_latent * latents.shape[1])

    def _run_flow_lm_and_increment_step(
        self,
        model_state: ModelState,
        text_tokens: torch.Tensor | None = None,
        backbone_input_latents: torch.Tensor | None = None,
        audio_conditioning: torch.Tensor | None = None,
        attention_capture: SelectedAttentionCapture | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First one is the backbone output, second one is the audio decoding output."""
        if text_tokens is None:
            text_tokens = torch.zeros((1, 0), dtype=torch.int64, device=self.flow_lm.device)
        if backbone_input_latents is None:
            backbone_input_latents = torch.empty(
                (1, 0, self.flow_lm.ldim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )
        if audio_conditioning is None:
            audio_conditioning = torch.empty(
                (1, 0, self.flow_lm.dim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )

        output = self._run_flow_lm(
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            model_state=model_state,
            audio_conditioning=audio_conditioning,
            attention_capture=attention_capture,
        )
        increment_by = (
            text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
        )
        increment_steps(self.flow_lm, model_state, increment=increment_by)
        return output

    def _run_flow_lm(
        self,
        model_state: ModelState,
        text_tokens: torch.Tensor,
        backbone_input_latents: torch.Tensor,
        audio_conditioning: torch.Tensor,
        attention_capture: SelectedAttentionCapture | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embeddings = self.flow_lm.conditioner(text_tokens)
        text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)

        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            model_state=model_state,
            sampler_decode_steps=self.sampler_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
            attention_capture=attention_capture,
        )
        return output_embeddings[:, None, :], is_eos

    def _decode_and_dump(self, encoded: torch.Tensor, filename: str):
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=10000)
        resored_audio = self.mimi.decode_from_latent(encoded, mimi_state)
        scipy.io.wavfile.write(filename, self.sample_rate, resored_audio.numpy())
        logger.info("Saved restored audio from Mimi encoding to %s for debugging", filename)

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        encoded = self.mimi.encode_to_latent(audio)

        if DEBUG_MIMI:
            # sanity check
            self._decode_and_dump(encoded, "debug_encoded_latent_decoded.wav")

        latents = encoded.to(torch.float32)
        conditioning = F.linear(latents, self.flow_lm.speaker_proj_weight)
        return conditioning

    def _expand_kv_cache(self, model_state: ModelState, sequence_length: int):
        """Expand KV cache back to full sequence_length for generation.

        When a model state is retrieved from cache with sliced KV caches,
        this method expands them back to the full size needed for generation.

        Args:
            model_state: The model state dict containing potentially sliced KV caches
            sequence_length: Target sequence length to expand caches to
        """
        for module_name, module_state in model_state.items():
            if "cache" in module_state:
                cache = module_state["cache"]
                # KV cache has shape [2, batch_size, current_length, num_kv_heads, dim_per_head]
                current_length = cache.shape[2]
                if current_length < sequence_length:
                    # Create expanded cache filled with NaN for unused positions
                    expanded_cache = torch.full(
                        (
                            cache.shape[0],
                            cache.shape[1],
                            sequence_length,
                            cache.shape[3],
                            cache.shape[4],
                        ),
                        float("NaN"),
                        device=cache.device,
                        dtype=cache.dtype,
                    )
                    # Copy existing data to the beginning
                    expanded_cache[:, :, :current_length, :, :] = cache
                    module_state["cache"] = expanded_cache

    def _flow_lm_current_end(self, model_state: ModelState) -> int:
        for module_state in model_state.values():
            offset = module_state.get("offset")
            if offset is not None:
                return int(offset.view(-1)[0].item())
        raise ValueError(
            "Could not find offset in model state, please open an issue "
            "at https://github.com/kyutai-labs/pocket-tts/issues"
        )

    @torch.no_grad
    def _decode_audio_worker(
        self,
        latents_queue: LatentQueue,
        result_queue: ResultQueue,
        mimi_sequence_length: int,
        mimi_steps_per_latent: int,
        warmup_latents: torch.Tensor | None = None,
    ) -> None:
        """Worker thread function for decoding audio latents from queue with immediate streaming."""
        try:
            audio_chunks = []
            mimi_state, warm = self._new_mimi_state(mimi_sequence_length, warmup_latents)
            # A fresh Mimi decoder state puts a small step (about -41 dBFS) in its first samples, heard
            # as a click at the start of every chunk: fade the first 5 ms in. (The longer burst some
            # voices produced came from prompts ending on speech, handled by end_on_pause.) A warm
            # decoder (leading-silence vectors, or continuing the previous chunk) needs no fade.
            first_batch = not warm
            for batch in _iter_latent_batches(latents_queue, self.max_decoder_frames_per_call):
                latents = []
                for item in batch:
                    if not isinstance(item, torch.Tensor):
                        raise TypeError(
                            "Ordinary audio decoding received timestamp attention scores"
                        )
                    latents.append(item)
                mimi_decoding_input = (
                    torch.cat(latents, dim=1) * self.flow_lm.emb_std + self.flow_lm.emb_mean
                )

                t = time.monotonic()
                audio_frame = self.mimi.decode_from_latent(mimi_decoding_input, mimi_state)
                if first_batch:
                    fade_in_audio(audio_frame, self.sample_rate)
                    first_batch = False
                increment_steps(
                    self.mimi, mimi_state, increment=mimi_steps_per_latent * len(latents)
                )
                audio_frame_duration = audio_frame.shape[2] / self.config.mimi.sample_rate
                logger.debug(
                    " " * 30 + "Decoded %d ms of audio (%d frames) with mimi in %d ms",
                    int(audio_frame_duration * 1000),
                    len(latents),
                    int((time.monotonic() - t) * 1000),
                )
                audio_chunks.append(audio_frame)

                result_queue.put(("chunk", audio_frame))

                for _ in latents:
                    latents_queue.task_done()

            # Signal completion
            result_queue.put(("done", None))

        except Exception as e:
            # Put error in result queue
            result_queue.put(("error", e))

    @torch.no_grad
    def _decode_timestamped_audio_worker(
        self,
        frames_queue: LatentQueue,
        result_queue: ResultQueue,
        session: Session,
        cancel_event: threading.Event,
    ) -> None:
        try:
            for batch in _iter_latent_batches(
                frames_queue, self.max_decoder_frames_per_call, cancel_event
            ):
                frames = []
                for item in batch:
                    if not isinstance(item, Frame):
                        raise TypeError("Timestamp audio decoding requires session frames")
                    frames.append(item)
                for event in session.decode(frames):
                    if cancel_event.is_set():
                        return
                    result_queue.put(("event", event))
                    if isinstance(event, AudioChunk):
                        frames_queue.task_done()

            if cancel_event.is_set():
                return

            for event in session.finish():
                result_queue.put(("event", event))
            result_queue.put(("done", session.time))
        except Exception as error:
            result_queue.put(("error", error))

    @torch.no_grad
    def generate_audio(
        self,
        model_state: ModelState,
        text_to_generate: str,
        max_tokens: int | None = None,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        teacher_force_previous_chunk: bool = False,
    ) -> torch.Tensor:
        """Generate complete audio tensor from text input.

        This method generates the full audio output for the given text prompt
        and returns it as a single tensor. It internally uses the streaming
        generation method but collects all chunks before returning.

        This method is NOT thread-safe; separate model instances should be used
        for concurrent generation.

        Args:
            model_state: Model state dictionary containing hidden states and
                positional information. Can be obtained from get_state_for_audio_prompt()
                or init_states(). The state may be modified during generation.
            text_to_generate: Input text to convert to speech. The text will be
                automatically formatted (capitalization, punctuation) for optimal
                generation quality.
            frames_after_eos: Number of additional frames to generate after
                detecting end-of-sequence. If None, automatically determined
                based on text length (1-3 frames).
            copy_state: Whether to create a deep copy of the model state before
                generation. If True, preserves the original state for reuse.
                If False, modifies the input state in-place. Defaults to True.
            max_tokens: Long texts are generated in chunks of at most this many tokens.
                Defaults to 50, or 25 with `teacher_force_previous_chunk`.
            teacher_force_previous_chunk: When True, each chunk after the first is generated
                from the voice prompt with the text of the previous chunk and of this one, the
                previous chunk's generated latents being teacher-forced as the first frames, so
                that the new chunk continues the previous one instead of starting afresh. Costs
                one prefill of the previous chunk's frames and text per chunk. Forces
                `copy_state=True`. Defaults to False.

        Returns:
            torch.Tensor: Generated audio tensor with shape [channels, samples]
                at the model's sample rate (typically 24kHz). The audio is
                normalized and ready for playback or saving.
                You can get the sample rate from the `sample_rate` attribute.

        Raises:
            ValueError: If text_to_generate is empty or invalid.
            RuntimeError: If generation fails due to model errors.

        Example:
            ```python
            from pocket_tts_timestamped import TTSModel

            model = TTSModel.load_model()

            voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

            # Generate audio
            audio = model.generate_audio(voice_state, "Hello world!", frames_after_eos=2, copy_state=True)

            print(f"Generated audio shape: {audio.shape}")
            print(f"Audio duration: {audio.shape[-1] / model.sample_rate:.2f} seconds")
            ```
        """
        audio_chunks = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            max_tokens=max_tokens,
            teacher_force_previous_chunk=teacher_force_previous_chunk,
        ):
            audio_chunks.append(chunk)
        return torch.cat(audio_chunks, dim=0)

    @torch.no_grad
    def generate_audio_stream(
        self,
        model_state: ModelState,
        text_to_generate: str,
        max_tokens: int | None = None,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        stop: threading.Event | None = None,
        teacher_force_previous_chunk: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Generate audio streaming chunks from text input.

        This method generates audio from text and yields chunks as they become
        available, enabling real-time playback or processing. It uses multithreading
        to parallelize generation and decoding for optimal performance.
        This method is NOT thread-safe; separate model instances should be used
        for concurrent generation.

        Args:
            model_state: Model state dictionary containing hidden states and
                positional information. Can be obtained from get_state_for_audio_prompt()
                or init_states(). The state may be modified during generation.
            text_to_generate: Input text to convert to speech. The text will be
                automatically formatted (capitalization, punctuation) for optimal
                generation quality.
            frames_after_eos: Number of additional frames to generate after
                detecting end-of-sequence. If None, automatically determined
                based on text length (1-3 frames). Defaults to None.
            copy_state: Whether to create a deep copy of the model state before
                generation. If True, preserves the original state for reuse.
                If False, modifies the input state in-place. Defaults to True.
            max_tokens: Long texts are generated in chunks of at most this many tokens.
                Defaults to 50, or 25 with `teacher_force_previous_chunk`.
            teacher_force_previous_chunk: When True, each chunk after the first is generated
                from the voice prompt with the text of the previous chunk and of this one, the
                previous chunk's generated latents being teacher-forced as the first frames, so
                that the new chunk continues the previous one instead of starting afresh. Costs
                one prefill of the previous chunk's frames and text per chunk. Forces
                `copy_state=True`. Defaults to False.
            stop: Optional event that ends generation after the current frame.

        Yields:
            torch.Tensor: Audio chunks with shape [samples] at the model's
                sample rate (typically 24kHz). Chunks are yielded as soon as
                they are decoded, enabling real-time streaming.

        Raises:
            ValueError: If text_to_generate is empty or invalid.
            RuntimeError: If generation fails due to model errors or threading issues.

        Example:
            ```python
            from pocket_tts_timestamped import TTSModel

            model = TTSModel.load_model()

            voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")
            # Stream generation
            for chunk in model.generate_audio_stream(voice_state, "Long text content..."):
                # Process each chunk as it's generated
                print(f"Generated chunk: {chunk.shape[0]} samples")
                # Could save chunks to file or play in real-time
            ```

        Note:
            This method uses multithreading to parallelize latent generation
            and audio decoding. Generation performance is logged including
            real-time factor (RTF) metrics.
        """
        if frames_after_eos is None:
            frames_after_eos = self.model_recommended_frames_after_eos
        if stop is None:
            stop = threading.Event()

        prev_latents: list[torch.Tensor] = []
        for chunk in self._prepare_text_chunks(
            text_to_generate, max_tokens, frames_after_eos, teacher_force_previous_chunk
        ):
            if stop.is_set():
                break
            forced = None
            text = chunk.text
            if chunk.continuation_text is not None and prev_latents:
                forced = torch.cat(prev_latents, dim=1)
                text = chunk.continuation_text
            prev_latents = []
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=text,
                frames_after_eos=chunk.frames_after_eos,
                copy_state=copy_state or teacher_force_previous_chunk,
                stop=stop,
                latents_out=prev_latents if teacher_force_previous_chunk else None,
                forced_latents=forced,
            )

    def _prepare_text_prompt(self, text: str) -> tuple[str, int]:
        return prepare_text_prompt(
            text,
            self.pad_with_spaces_for_short_inputs,
            self.remove_semicolons,
            self.append_terminal_punctuation,
            self.capitalize_first_letter,
            self.replace_characters,
        )

    def _prepare_text_chunks(
        self,
        text: str,
        max_tokens: int | None,
        frames_after_eos: int | None,
        teacher_force_previous_chunk: bool = False,
    ) -> list[_PreparedTextChunk]:
        """Prepare each synthesis prompt before tokenization or timestamp mapping."""
        if max_tokens is None:
            max_tokens = (
                MAX_TOKEN_PER_CHUNK_TEACHER_FORCED
                if teacher_force_previous_chunk
                else MAX_TOKEN_PER_CHUNK
            )
        # Long texts are generated chunk by chunk. By default every chunk starts afresh from the
        # voice prompt; with teacher_force_previous_chunk it continues the previous one.
        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer,
            text,
            max_tokens,
            self.pad_with_spaces_for_short_inputs,
            remove_semicolons=self.remove_semicolons,
            append_terminal_punctuation=self.append_terminal_punctuation,
            capitalize_first_letter=self.capitalize_first_letter,
            replace_characters=self.replace_characters,
        )
        prepared_chunks = []
        prev_chunk: str | None = None
        for chunk in chunks:
            # The EOS delay depends on the new chunk only.
            prepared, frames_guess = self._prepare_text_prompt(chunk)
            continuation = None
            if teacher_force_previous_chunk and prev_chunk is not None:
                continuation, _ = self._prepare_text_prompt(f"{prev_chunk} {chunk}")
            effective_frames = (
                frames_after_eos if frames_after_eos is not None else frames_guess + 2
            )
            prepared_chunks.append(_PreparedTextChunk(prepared, effective_frames, continuation))
            prev_chunk = chunk
        return prepared_chunks

    def generate_audio_with_timestamps(
        self,
        model_state: ModelState,
        text_to_generate: str,
        max_tokens: int | None = None,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        teacher_force_previous_chunk: bool = False,
    ) -> TimestampedAudio:
        """Generate audio and completed word-level timestamps.

        Takes the options of `generate_audio()` and produces the same audio.
        """

        stream = self.generate_audio_with_timestamps_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            max_tokens=max_tokens,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            teacher_force_previous_chunk=teacher_force_previous_chunk,
        )
        audio_chunks = []
        words = []
        for event in stream:
            if isinstance(event, AudioChunk):
                audio_chunks.append(event.audio)
            elif isinstance(event, WordEnd):
                words.append(
                    WordTimestamp(
                        event.word,
                        event.word_index,
                        event.start_time,
                        event.end_time,
                        event.char_start,
                        event.char_end,
                    )
                )
        audio = (
            torch.cat(audio_chunks, dim=0)
            if audio_chunks
            else torch.empty(0, dtype=torch.float32, device=self.device)
        )
        return TimestampedAudio(audio=audio, words=tuple(words))

    def generate_audio_with_timestamps_stream(
        self,
        model_state: ModelState,
        text_to_generate: str,
        max_tokens: int | None = None,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        stop: threading.Event | None = None,
        teacher_force_previous_chunk: bool = False,
    ) -> Generator[TimestampEvent, None, None]:
        """Stream audio and word events; `stop` ends generation after the current frame.

        Takes the options of `generate_audio_stream()` and produces the same audio.
        Queued audio is drained and an open word is closed before the stream ends.
        Closing the iterator instead immediately cancels and joins its workers.
        """

        if self.config.timestamp_heads is None:
            raise ValueError(
                "Word timestamps are not supported by this model config because "
                "timestamp_heads is absent. Use generate_audio() or "
                "generate_audio_stream() for ordinary generation."
            )
        return self._generate_audio_with_timestamps_events(
            model_state=model_state,
            text_to_generate=text_to_generate,
            max_tokens=max_tokens,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            stop=stop,
            teacher_force_previous_chunk=teacher_force_previous_chunk,
        )

    @torch.no_grad
    def _generate_audio_with_timestamps_events(
        self,
        model_state: ModelState,
        text_to_generate: str,
        max_tokens: int | None,
        frames_after_eos: int | None,
        copy_state: bool,
        stop: threading.Event | None = None,
        teacher_force_previous_chunk: bool = False,
    ) -> Generator[TimestampEvent, None, None]:
        if frames_after_eos is None:
            frames_after_eos = self.model_recommended_frames_after_eos
        if stop is None:
            stop = threading.Event()
        prepared_chunks = self._prepare_text_chunks(
            text_to_generate, max_tokens, frames_after_eos, teacher_force_previous_chunk
        )
        tokenizer = self.flow_lm.conditioner.tokenizer
        timestamp_chunks = _iter_timestamp_text_chunks(
            text_to_generate, (chunk.text for chunk in prepared_chunks), tokenizer, best_effort=True
        )
        time_offset = 0.0
        prev_latents: list[torch.Tensor] = []
        for timestamp_chunk, chunk in zip(timestamp_chunks, prepared_chunks, strict=True):
            if stop.is_set():
                break
            forced = None
            if chunk.continuation_text is not None and prev_latents:
                forced = torch.cat(prev_latents, dim=1)
                # Only this chunk's words are timed; the previous chunk's text is context.
                timestamp_chunk = _continuation_timestamp_text_chunk(
                    timestamp_chunk, chunk.continuation_text, tokenizer
                )
            prev_latents = []
            time_offset = yield from self._generate_audio_with_timestamps_short_text(
                model_state=model_state,
                timestamp_chunk=timestamp_chunk,
                frames_after_eos=chunk.frames_after_eos,
                copy_state=copy_state or teacher_force_previous_chunk,
                time_offset=time_offset,
                stop=stop,
                latents_out=prev_latents if teacher_force_previous_chunk else None,
                forced_latents=forced,
            )

    @torch.no_grad
    def _generate_audio_with_timestamps_short_text(
        self,
        model_state: ModelState,
        timestamp_chunk: TimestampTextChunk,
        frames_after_eos: int,
        copy_state: bool,
        time_offset: float,
        stop: threading.Event | None = None,
        latents_out: list[torch.Tensor] | None = None,
        forced_latents: torch.Tensor | None = None,
    ) -> Generator[TimestampEvent, None, float]:
        prepared = self._prepare_timestamp_text_chunk(timestamp_chunk)
        # With forced frames the text also holds the previous chunk's: a looser upper bound.
        max_gen_len = self._estimate_max_gen_len(prepared.shape[1])
        session = Session(
            self,
            model_state,
            copy_state=copy_state,
            time_offset=time_offset,
            decoder_frames=max_gen_len,
            history=forced_latents,
        )
        window = TextWindow(timestamp_chunk, prepared, frames_after_eos)

        frames_queue: LatentQueue = queue.Queue()
        result_queue: ResultQueue = queue.Queue()
        cancel_event = threading.Event()
        decoder_thread = threading.Thread(
            target=self._decode_timestamped_audio_worker,
            args=(frames_queue, result_queue, session, cancel_event),
            daemon=True,
        )
        decoder_thread.start()
        generation_thread: threading.Thread | None = None
        audio_end = time_offset
        try:
            with display_execution_time("Prompting text"):
                # Teacher forcing replays the previous chunk's frames after BOS and the text.
                history_from = None if forced_latents is None else 0
                session.set_text(window, history_from=history_from, reserve_frames=max_gen_len)
            generation_thread = threading.Thread(
                target=self._generate_session_frames,
                args=(session, max_gen_len, frames_after_eos, frames_queue, result_queue),
                kwargs={"cancel_event": cancel_event, "stop": stop, "latents_out": latents_out},
                daemon=True,
            )
            generation_thread.start()

            while True:
                kind, value = result_queue.get()
                if kind == "event":
                    yield value
                elif kind == "done":
                    audio_end = value
                    break
                elif kind == "error":
                    raise value
        finally:
            cancel_event.set()
            frames_queue.put(None)
            if generation_thread is not None:
                generation_thread.join()
            decoder_thread.join()
        return audio_end

    @torch.no_grad
    def _generate_session_frames(
        self,
        session: Session,
        max_gen_len: int,
        frames_after_eos: int,
        frames_queue: LatentQueue,
        result_queue: ResultQueue,
        cancel_event: threading.Event,
        stop: threading.Event | None = None,
        latents_out: list[torch.Tensor] | None = None,
    ) -> None:
        """Generate one text's frames into `frames_queue`, ending `frames_after_eos` after EOS.
        The latents of the queued frames are also appended to `latents_out`.

        EOS is gated as in `_autoregressive_generation`, on frames generated since `set_text`,
        so the audio is the same as ordinary generation's when it replays a previous chunk.
        """

        def stopped() -> bool:
            return cancel_event.is_set() or (stop is not None and stop.is_set())

        steps_times: list[float] = []
        eos_step = None
        try:
            for generation_step in range(max_gen_len):
                if stopped():
                    break
                with display_execution_time("Generating latent", print_output=False) as timer:
                    frame = session.generate_frame()
                    if stopped():
                        break
                    if (
                        frame.eos
                        and eos_step is None
                        and generation_step >= self._MIN_FRAMES_BEFORE_EOS
                    ):
                        eos_step = generation_step
                    if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                        break
                    if latents_out is not None:
                        latents_out.append(frame.latent)
                    frames_queue.put(frame)
                assert timer.elapsed_time_ms is not None
                steps_times.append(timer.elapsed_time_ms)
            else:
                if os.environ.get("KPOCKET_TTS_ERROR_WITHOUT_EOS", "0") == "1":
                    raise RuntimeError("Generation reached maximum length without EOS!")
                logger.warning(
                    "Maximum generation length reached without EOS, this very often indicates an error."
                )
        except Exception as error:
            logger.error("Error in autoregressive generation: %s", error)
            # Report the generation error before stopping the decoder. Otherwise
            # the decoder can publish "done" first and hide the exception.
            result_queue.put(("error", error))
        finally:
            # Wake the decoder on normal completion, cancellation, and generation errors.
            frames_queue.put(None)
        if steps_times:
            logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)))

    def start_session(self, voice_state: ModelState) -> Session:
        """Start frame-by-frame timestamped generation in `voice_state`'s voice.

        The session's text can be replaced mid-utterance; see `Session`. `voice_state` is
        copied, not modified.
        """
        if self.config.timestamp_heads is None:
            raise ValueError(
                "Word timestamps are not supported by this model config because "
                "timestamp_heads is absent."
            )
        return Session(self, voice_state)

    def _prepare_timestamp_text_chunk(self, timestamp_chunk: TimestampTextChunk) -> torch.Tensor:
        if timestamp_chunk.prepared_tokens is None:
            return self.flow_lm.conditioner.prepare(timestamp_chunk.text)
        return timestamp_chunk.prepared_tokens.to(self.flow_lm.device)

    @torch.no_grad
    def _generate_audio_stream_short_text(
        self,
        model_state: ModelState,
        text_to_generate: str,
        frames_after_eos: int,
        copy_state: bool,
        stop: threading.Event,
        latents_out: list[torch.Tensor] | None = None,
        forced_latents: torch.Tensor | None = None,
    ) -> Iterator[torch.Tensor]:
        if copy_state:
            model_state = copy.deepcopy(model_state)

        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        token_count = prepared.shape[1]
        # With forced frames the text also holds the previous chunk's: a looser upper bound.
        max_gen_len = self._estimate_max_gen_len(token_count)
        mimi_steps_per_latent = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        warmup = None
        if forced_latents is not None:
            warmup = forced_latents[:, -self._DECODER_WARMUP_FRAMES :]
        # _new_mimi_state makes room for the warm-up frames.
        mimi_sequence_length = max_gen_len * mimi_steps_per_latent

        # Set up multithreaded generation and decoding
        latents_queue: LatentQueue = queue.Queue()
        result_queue: ResultQueue = queue.Queue()

        # Start decoder worker thread
        decoder_thread = threading.Thread(
            target=self._decode_audio_worker,
            args=(latents_queue, result_queue, mimi_sequence_length, mimi_steps_per_latent, warmup),
            daemon=True,
        )
        logger.info("starting timer now!")
        t_generating = time.monotonic()
        decoder_thread.start()

        # Generate latents and add them to queue (decoder processes them in parallel)
        self._generate(
            model_state=model_state,
            prepared=prepared,
            max_gen_len=max_gen_len,
            frames_after_eos=frames_after_eos,
            latents_queue=latents_queue,
            result_queue=result_queue,
            stop=stop,
            latents_out=latents_out,
            forced_latents=forced_latents,
        )

        # Stream audio chunks as they become available
        total_generated_samples = 0
        while True:
            result = result_queue.get()
            if result[0] == "chunk":
                # Audio chunk available immediately for streaming/playback
                audio_chunk = result[1]
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk[0, 0]  # Remove batch, channel
            elif result[0] == "done":
                # Generation complete
                break
            elif result[0] == "error":
                # Wait for decoder thread to finish cleanly before propagating error
                with display_execution_time("Waiting for mimi decoder to finish"):
                    decoder_thread.join()
                # Propagate error
                raise result[1]

        # Wait for decoder thread to finish cleanly
        with display_execution_time("Waiting for mimi decoder to finish"):
            decoder_thread.join()

        # Print timing information
        duration_generated_audio = int(
            total_generated_samples * 1000 / self.config.mimi.sample_rate
        )
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / generation_time

        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    @torch.no_grad
    def _generate(
        self,
        model_state: ModelState,
        prepared: torch.Tensor,
        max_gen_len: int,
        frames_after_eos: int,
        latents_queue: LatentQueue,
        result_queue: ResultQueue,
        stop: threading.Event | None = None,
        latents_out: list[torch.Tensor] | None = None,
        forced_latents: torch.Tensor | None = None,
    ) -> threading.Thread:
        token_count = prepared.shape[1]
        n_forced = 0 if forced_latents is None else forced_latents.shape[1]
        current_end = self._flow_lm_current_end(model_state)
        required_len = current_end + token_count + n_forced + max_gen_len
        self._expand_kv_cache(model_state, sequence_length=required_len)

        first_input = None
        with display_execution_time("Prompting text"):
            if forced_latents is None or n_forced == 0:
                self._run_flow_lm_and_increment_step(model_state=model_state, text_tokens=prepared)
            else:
                # Teacher forcing: BOS then every forced frame but the last, which is the input
                # of the first generated step. The outputs of this prefill, EOS included, are
                # discarded: EOS can only fire on new frames.
                bos = torch.full_like(forced_latents[:, :1], float("NaN"))
                self._run_flow_lm_and_increment_step(
                    model_state=model_state,
                    text_tokens=prepared,
                    backbone_input_latents=torch.cat([bos, forced_latents[:, :-1]], dim=1),
                )
                first_input = forced_latents[:, -1:]

        def run_generation() -> None:
            try:
                self._autoregressive_generation(
                    model_state,
                    max_gen_len,
                    frames_after_eos,
                    latents_queue,
                    stop=stop,
                    latents_out=latents_out,
                    first_input=first_input,
                )
            except Exception as error:
                logger.error("Error in autoregressive generation: %s", error)
                # Report the generation error before stopping the decoder. Otherwise
                # the decoder can publish "done" first and hide the exception.
                result_queue.put(("error", error))
                # Signal decoder to stop by putting None (completion sentinel)
                latents_queue.put(None)

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()
        return generation_thread

    @torch.no_grad
    def _autoregressive_generation(
        self,
        model_state: ModelState,
        max_gen_len: int,
        frames_after_eos: int,
        latents_queue: LatentQueue,
        stop: threading.Event | None = None,
        latents_out: list[torch.Tensor] | None = None,
        first_input: torch.Tensor | None = None,
    ) -> None:
        backbone_input = first_input
        if backbone_input is None:
            backbone_input = torch.full(
                (1, 1, self.flow_lm.ldim),
                fill_value=float("NaN"),
                device=next(iter(self.flow_lm.parameters())).device,
                dtype=self.flow_lm.dtype,
            )
        steps_times: list[float] = []
        eos_step = None
        for generation_step in range(max_gen_len):
            if stop is not None and stop.is_set():
                break
            with display_execution_time("Generating latent", print_output=False) as timer:
                next_latent, is_eos = self._run_flow_lm_and_increment_step(
                    model_state=model_state, backbone_input_latents=backbone_input
                )
                if stop is not None and stop.is_set():
                    break
                if (
                    is_eos.item()
                    and eos_step is None
                    and generation_step >= self._MIN_FRAMES_BEFORE_EOS
                ):
                    eos_step = generation_step
                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break

                # Add generated latent to queue for immediate decoding.
                if latents_out is not None:
                    latents_out.append(next_latent)
                latents_queue.put(next_latent)
                backbone_input = next_latent
            assert timer.elapsed_time_ms is not None
            steps_times.append(timer.elapsed_time_ms)
        else:
            if os.environ.get("KPOCKET_TTS_ERROR_WITHOUT_EOS", "0") == "1":
                raise RuntimeError("Generation reached maximum length without EOS!")
            logger.warning(
                "Maximum generation length reached without EOS, this very often indicates an error."
            )
        # On normal completion and cancellation. A generation error propagates instead, and
        # the caller reports it before waking the decoder.
        latents_queue.put(None)
        if steps_times:
            logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)))

    @lru_cache(maxsize=2)
    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> ModelState:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    @torch.no_grad
    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> ModelState:
        """Create model state conditioned on audio prompt for continuation.

        This method processes an audio prompt and creates a model state that
        captures the acoustic characteristics (speaker voice, style, prosody)
        for use in subsequent text-to-speech generation. The resulting state
        enables voice cloning and audio continuation with speaker consistency.

        Args:
            audio_conditioning: Audio prompt to condition (or .safetensors to load). Can be:
                - Path: Local file path to audio file (or .safetensors)
                - str: URL to download audio file (or .safetensors) from
                - torch.Tensor: Pre-loaded audio tensor with shape [channels, samples]
            truncate: Whether to truncate long audio prompts to 30 seconds.
                Helps prevent memory issues with very long inputs. Defaults to False.

        Returns:
            dict: Model state dictionary containing hidden states and positional
                information conditioned on the audio prompt. This state can be
                passed to `generate_audio()` or `generate_audio_stream()` for
                voice-consistent generation.

        Raises:
            FileNotFoundError: If audio file path doesn't exist.
            ValueError: If audio tensor is invalid or empty.
            RuntimeError: If audio processing or encoding fails.

        Example:
            ```python
            from pocket_tts_timestamped import TTSModel

            model = TTSModel.load_model()
            # From HuggingFace URL
            voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

            # From local file
            voice_state = model.get_state_for_audio_prompt("./my_voice.wav")

            # Reload state from a .safetensors file (much faster than extracting from an audio file)
            voice_state = model.get_state_for_audio_prompt("./my_voices.safetensors")

            # From HTTP URL
            voice_state = model.get_state_for_audio_prompt(
                "https://huggingface.co/kyutai/tts-voices/resolve"
                "/main/expresso/ex01-ex02_default_001_channel1_168s.wav"
            )
            ```

        Note:
            - Audio is automatically resampled to the model's sample rate (24kHz)
            - The audio is encoded using the Mimi compression model and projected
              to the flow model's latent space
            - Processing time is logged for performance monitoring
            - The state preserves speaker characteristics for voice cloning
        """
        if isinstance(audio_conditioning, (str, Path)) and _is_safetensors_source(
            audio_conditioning
        ):
            if isinstance(audio_conditioning, str):
                audio_conditioning = download_if_necessary(audio_conditioning)

            return _import_model_state(audio_conditioning, self.device)

        elif (
            isinstance(audio_conditioning, str)
            and audio_conditioning in _ORIGINS_OF_PREDEFINED_VOICES
        ):
            # We get the audio conditioning directly from the safetensors file.
            if self.has_custom_weights:
                raise ValueError(
                    f"Predefined voice {audio_conditioning!r} is a state precomputed with the "
                    "released weights; feeding it to other weights leaves the model out of "
                    "distribution (it typically never emits EOS). Pass the voice as an audio "
                    "file instead."
                )
            if self.origin is None or not self.origin.is_relative_to(CONFIGS_DIR):
                raise ValueError(
                    f"Cannot use predefined voices when the model "
                    f"is not loaded from a config associated with a language."
                    f"Here the origin is {self.origin}"
                )
            return _import_model_state(
                download_if_necessary(
                    get_predefined_voice(language=self.origin.stem, name=audio_conditioning)
                ),
                self.device,
            )

        if not self.has_voice_cloning and isinstance(audio_conditioning, (str, Path)):
            raise ValueError(VOICE_CLONING_UNSUPPORTED)

        if isinstance(audio_conditioning, str):
            audio_conditioning = download_if_necessary(audio_conditioning)

        if isinstance(audio_conditioning, Path):
            audio, conditioning_sample_rate = audio_read(audio_conditioning)

            if truncate:
                max_samples = int(30 * conditioning_sample_rate)  # 30 seconds of audio
                if audio.shape[-1] > max_samples:
                    audio = audio[..., :max_samples]
                    logger.info(f"Audio truncated to first 30 seconds ({max_samples} samples)")

            audio_conditioning = convert_audio(
                audio, conditioning_sample_rate, self.config.mimi.sample_rate, 1
            )

        audio_conditioning = end_on_pause(audio_conditioning, self.config.mimi.sample_rate)
        with display_execution_time("Encoding audio prompt"):
            prompt = self._encode_audio(audio_conditioning.unsqueeze(0).to(self.device))

        if self.flow_lm.insert_bos_before_voice:
            prompt = torch.cat([self.flow_lm.bos_before_voice, prompt], dim=1)

        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=prompt.shape[1])

        with display_execution_time("Prompting audio"):
            self._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)

        logger.info(
            "Size of the model state for audio prompt: %d MB", size_of_dict(model_state) // 1e6
        )

        return model_state

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self._TOKENS_PER_SECOND_ESTIMATE + self._GEN_SECONDS_PADDING
        frame_rate = self.config.mimi.frame_rate
        return math.ceil(gen_len_sec * frame_rate)
