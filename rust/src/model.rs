//! Port of `TTSModel` from `pocket_tts/models/tts_model.py`, on the streaming primitives of
//! xn-ptts: loading a config's weights, conditioning on a voice, and generating audio chunk by
//! chunk while a second thread decodes it.

use crate::config::{Config, FlowType, Origin};
use crate::defaults;
use crate::rng::NoiseRng;
use crate::text::{TextOptions, prepare_text_prompt, split_into_best_sentences};
use crate::tokenizer::{SharedTokenizer, Tokenizer};
use crate::weights::{SafetensorsFile, WeightFiles};
use anyhow::{Context, Result, bail};
use ptts::flow_lm::{FlowLMState, StepInput};
use ptts::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use ptts::tts_model::{MimiEnc, TTSModel, TTSState};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use xn::{BackendQ, Tensor, WithDTypeF};

/// The backends this binary runs on: the CPU, with f32 or int8 weights.
struct GeneratedFrame<T: WithDTypeF, B: xn::Backend> {
    latent: Tensor<T, B>,
    scores: Option<Vec<f32>>,
}

pub trait CpuBackend: BackendQ<B = xn::CpuDevice> {}
impl<Q: BackendQ<B = xn::CpuDevice>> CpuBackend for Q {}

const TOKENS_PER_SECOND_ESTIMATE: f64 = 3.0;
const GEN_SECONDS_PADDING: f64 = 2.0;
/// EOS is ignored on the first frames: before speech starts, the EOS logit of some voices can
/// cross the threshold, and a short text then ends before the word is spoken.
const MIN_FRAMES_BEFORE_EOS: usize = 6;
// Mimi emits 80 ms per frame. Bound both producer backlog and individual decode
// work: a blocked PCM consumer must not turn the next decode into seconds of audio.
const LATENT_QUEUE_CAPACITY: usize = 4;
const MAX_DECODE_FRAMES: usize = 4;
// Preserve the former ONNX ceiling margin independently of the requested
// post-EOS tail (already supplied by the app as six/eight frames).
const COMPLETION_MARGIN_FRAMES: usize = 3;

fn completion_budget(estimated: usize) -> Result<usize> {
    estimated
        .checked_add(COMPLETION_MARGIN_FRAMES)
        .context("Pocket generation budget overflow")
}

fn timestamp_frame_size(sample_rate: usize, frame_rate: f64) -> Result<usize> {
    let samples = sample_rate as f64 / frame_rate;
    anyhow::ensure!(
        frame_rate.is_finite()
            && frame_rate > 0.
            && samples.is_finite()
            && samples >= 1.
            && samples <= usize::MAX as f64
            && (samples - samples.round()).abs() < 1e-6,
        "Invalid timestamp frame rate"
    );
    Ok(samples.round() as usize)
}

// Preserve the standard CLI warning policy unless its caller or environment
// explicitly requires a completed EOS tail. Cancellation is not exhaustion.
fn check_generation_completion(
    reached_max_len: bool,
    require_eos: bool,
    environment: Option<&str>,
) -> Result<()> {
    if reached_max_len {
        if require_eos || environment == Some("1") {
            bail!("Generation reached maximum length without EOS");
        }
        log::warn!(
            "Maximum generation length reached without EOS, this very often indicates an error."
        );
    }
    Ok(())
}

pub fn voice_cloning_unsupported() -> String {
    format!(
        "We could not download the weights for the model with voice cloning, but you're trying to use voice cloning. Without voice cloning, you can use our catalog of voices {:?}. If you want access to the model with voice cloning, go to https://huggingface.co/kyutai/pocket-tts and accept the terms, then make sure you're logged in locally with `uvx hf auth login`.",
        defaults::predefined_voice_names()
    )
}

/// What `load_model` takes besides the config.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub temperature: Option<f32>,
    pub sampler_decode_steps: usize,
    pub noise_clamp: Option<f32>,
    /// The config's `default_eos_threshold` when `None`.
    pub eos_threshold: Option<f32>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            sampler_decode_steps: defaults::DEFAULT_SAMPLER_DECODE_STEPS,
            noise_clamp: None,
            eos_threshold: None,
        }
    }
}

/// The flow LM state after a voice prompt, which every generation starts from.
pub struct VoiceState<Q: CpuBackend> {
    transformer_state: StreamingTransformerState<Q::T, Q::B>,
}

impl<Q: CpuBackend> VoiceState<Q> {
    /// Positions the prompt fills.
    pub fn len(&self) -> usize {
        match self.transformer_state.layer_states.first() {
            Some(LayerAttentionState::FlowLm(state)) => state.current_end,
            _ => 0,
        }
    }

    /// Writes the state as the Python package's `export_model_state` does: per attention layer,
    /// `{module}/cache` of shape `[2, batch, time, kv_heads, dim]` and `{module}/offset`.
    pub fn export(&self, path: &Path) -> Result<()> {
        let mut tensors: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
        for (i, layer) in self.transformer_state.layer_states.iter().enumerate() {
            let LayerAttentionState::FlowLm(state) = layer else {
                bail!("layer {i} has no flow LM attention state")
            };
            let (b, _, h, d) = state.k_cache.dims4()?;
            let t = state.current_end;
            let mut bytes = Vec::with_capacity(2 * b * t * h * d * 4);
            for cache in [&state.k_cache, &state.v_cache] {
                let values = cache.narrow(1, 0..t)?.contiguous()?.to_vec()?;
                bytes.extend(values.iter().flat_map(|v| v.to_f32().to_le_bytes()));
            }
            let module = format!("transformer.layers.{i}.self_attn");
            tensors.push((
                format!("{module}/cache"),
                safetensors::Dtype::F32,
                vec![2, b, t, h, d],
                bytes,
            ));
            let offset: Vec<u8> = (0..b).flat_map(|_| (t as i64).to_le_bytes()).collect();
            tensors.push((
                format!("{module}/offset"),
                safetensors::Dtype::I64,
                vec![b],
                offset,
            ));
        }
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                Ok((
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes)?,
                ))
            })
            .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
        safetensors::tensor::serialize_to_file(views, None, path)
            .with_context(|| format!("cannot write {}", path.display()))?;
        Ok(())
    }
}

pub struct TtsModel<Q: CpuBackend> {
    pub config: Config,
    pub origin: Origin,
    model: TTSModel<Q>,
    encoder: Option<MimiEnc<Q>>,
    tokenizer: Arc<Tokenizer>,
    text_options: TextOptions,
    has_voice_cloning: bool,
    rng: Mutex<NoiseRng>,
    require_eos: bool,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<Q: CpuBackend> TtsModel<Q> {
    /// Port of `TTSModel.load_model`, for the config already read.
    pub fn load(config: Config, origin: Origin, options: &LoadOptions) -> Result<Self> {
        let mut weight_files = WeightFiles::default();
        let mut has_voice_cloning = true;
        if let Some(mimi_path) = config.mimi.weights_path.as_deref() {
            if config.flow_lm.weights_path.is_none() && config.weights_path.is_none() {
                bail!(
                    "If you specify mimi.weights_path you should specify flow_lm.weights_path or weights_path"
                )
            }
            log::info!("Loading Mimi weights from {mimi_path}");
            weight_files.mimi = Some(crate::download::download_if_necessary(mimi_path)?);
        }
        if let Some(weights_path) = config.weights_path.as_deref() {
            log::info!("Loading TTSModel weights from {weights_path}");
            let path = match crate::download::download_if_necessary(weights_path) {
                Ok(path) => path,
                Err(e) => {
                    let Some(fallback) = config.weights_path_without_voice_cloning.as_deref()
                    else {
                        return Err(e);
                    };
                    log::debug!("falling back to the weights without voice cloning: {e:#}");
                    has_voice_cloning = false;
                    crate::download::download_if_necessary(fallback)?
                }
            };
            weight_files.bundle = Some(path);
        } else if let Some(flow_lm_path) = config.flow_lm.weights_path.as_deref() {
            if weight_files.mimi.is_none() {
                bail!("flow_lm.weights_path needs mimi.weights_path, for the Mimi weights")
            }
            log::info!("Loading FlowLM weights from {flow_lm_path}");
            weight_files.flow_lm = Some(crate::download::download_if_necessary(flow_lm_path)?);
        } else {
            bail!("No weights_path specified for the TTSModel, the model would be uninitialized")
        }

        let lookup_table = &config.flow_lm.lookup_table;
        let tokenizer = Arc::new(Tokenizer::load(
            lookup_table.n_bins,
            &lookup_table.tokenizer_path,
            &lookup_table.tokenizer,
        )?);

        // The drifting and pointwise heads map the noise in one step, whatever the number of
        // steps asked.
        let decode_steps = match config.flow_type() {
            FlowType::Drifting | FlowType::Pointwise => 1,
            FlowType::Lsd | FlowType::FlowMatching => options.sampler_decode_steps,
        };
        if decode_steps == 0 {
            bail!("the number of sampler decode steps must be positive")
        }
        let eos_threshold = options
            .eos_threshold
            .unwrap_or(config.default_eos_threshold);
        let ptts_config = config.to_ptts(decode_steps, eos_threshold)?;
        let weights = crate::weights::load(&weight_files)?;
        let model = TTSModel::<Q>::load(
            &weights.vb,
            Box::new(SharedTokenizer(tokenizer.clone())),
            &ptts_config,
            &HashMap::new(),
        )?;
        let has_encoder = weights.vb.contains("mimi.encoder.model.0.conv.weight")
            && weights.vb.contains(ptts::loader::SPEAKER_PROJ_WEIGHT);
        let encoder = if has_encoder {
            Some(MimiEnc::<Q>::load(&weights.vb, &ptts_config)?)
        } else {
            None
        };
        drop(weights);

        let temperature = options.temperature.unwrap_or(config.default_temperature);
        let transformer = &config.flow_lm.transformer;
        Ok(Self {
            num_layers: transformer.num_layers,
            num_kv_heads: transformer.num_kv_heads(),
            head_dim: transformer.head_dim(),
            text_options: TextOptions::from_config(&config),
            rng: Mutex::new(NoiseRng::new(temperature, options.noise_clamp)?),
            require_eos: false,
            config,
            origin,
            model,
            encoder,
            tokenizer,
            has_voice_cloning,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.config.mimi.sample_rate
    }

    /// Port of `get_state_for_audio_prompt`: `source` is a predefined voice name, an audio file
    /// or a `.safetensors` state, as a local path, an `http(s)://` URL or an `hf://` path.
    /// `truncate` keeps the first 30 seconds of an audio file.
    pub fn get_state_for_audio_prompt(
        &self,
        source: &str,
        truncate: bool,
    ) -> Result<VoiceState<Q>> {
        if is_safetensors_source(source) {
            let path = crate::download::download_if_necessary(source)?;
            return self.import_state(&path);
        }
        if defaults::is_predefined_voice(source) {
            let Origin::Language(language) = &self.origin else {
                bail!(
                    "Cannot use predefined voices when the model is not loaded from a config associated with a language.Here the origin is {}",
                    self.origin.display_path()
                )
            };
            let voice = defaults::get_predefined_voice(language, source);
            let path = crate::download::download_if_necessary(&voice)?;
            return self.import_state(&path);
        }
        if !self.has_voice_cloning || self.encoder.is_none() {
            bail!(voice_cloning_unsupported())
        }

        let path = crate::download::download_if_necessary(source)?;
        let (mut audio, sample_rate) = crate::audio::audio_read(&path)?;
        if truncate {
            let max_samples = 30 * sample_rate as usize;
            if audio.len() > max_samples {
                audio.truncate(max_samples);
                log::info!("Audio truncated to first 30 seconds ({max_samples} samples)");
            }
        }
        let audio = ptts::audio::resample(audio, sample_rate as usize, self.sample_rate())?;
        let audio = crate::audio::end_on_pause(&audio, self.sample_rate());
        self.state_for_audio(&audio)
    }

    /// Encodes mono audio at the model's sample rate and prompts the flow LM with it.
    fn state_for_audio(&self, audio: &[f32]) -> Result<VoiceState<Q>> {
        let encoder = self
            .encoder
            .as_ref()
            .with_context(voice_cloning_unsupported)?;
        let start = Instant::now();
        let data = audio.iter().map(|&v| Q::T::from_f32(v)).collect();
        let audio = Tensor::from_vec(data, (1, 1, audio.len()), self.model.device())?;
        let conditioning = encoder.encode_audio(&audio)?;
        log::info!(
            "Encoding audio prompt took {} ms",
            start.elapsed().as_millis()
        );

        let start = Instant::now();
        let frames = conditioning.dim(1usize)?;
        let mut state = self
            .model
            .init_flow_lm_state(1, self.model.voice_prompt_len(frames))?;
        self.model.prompt_audio(&mut state, &conditioning)?;
        log::info!("Prompting audio took {} ms", start.elapsed().as_millis());
        let voice = VoiceState {
            transformer_state: state.flow_lm_state.transformer_state,
        };
        let kv_dim = self.num_kv_heads * self.head_dim;
        let size = 2 * self.num_layers * voice.len() * kv_dim * std::mem::size_of::<f32>();
        log::info!(
            "Size of the model state for audio prompt: {} MB",
            size / 1_000_000
        );
        Ok(voice)
    }

    /// Reads a state written by `export-voice`, from this binary or the Python package.
    pub fn import_state(&self, path: &Path) -> Result<VoiceState<Q>> {
        let file = SafetensorsFile::open(path)?;
        let dim_per_head = self.head_dim;
        let mut layer_states = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let module = format!("transformer.layers.{i}.self_attn");
            let cache_name = format!("{module}/cache");
            if !file.tensors.contains_key(&cache_name) {
                bail!(
                    "{} is not a voice state of this model: it has no {cache_name}",
                    path.display()
                )
            }
            let (cache, shape) = file.read_f32(&cache_name)?;
            let [2, b, capacity, h, d] = shape[..] else {
                bail!("{cache_name} has shape {shape:?}, expected [2, batch, time, heads, dim]")
            };
            if b != 1 || h != self.num_kv_heads || d != dim_per_head {
                bail!(
                    "{cache_name} has shape {shape:?}, this model needs [2, 1, _, {}, {dim_per_head}]",
                    self.num_kv_heads
                )
            }
            // States written before the offset was stored hold the position in the length of
            // `current_end`.
            let offset_name = format!("{module}/offset");
            let current_end_name = format!("{module}/current_end");
            let used = if file.tensors.contains_key(&offset_name) {
                file.read_i64(&offset_name)?.first().copied().unwrap_or(0) as usize
            } else if let Some(info) = file.tensors.get(&current_end_name) {
                info.shape.first().copied().unwrap_or(0)
            } else {
                capacity
            };
            if used > capacity {
                bail!("{offset_name} is {used}, past the {capacity} cached positions")
            }
            let per_position = h * d;
            let plane = capacity * per_position;
            let take = |start: usize| -> Result<Tensor<Q::T, Q::B>> {
                let data = cache[start..start + used * per_position]
                    .iter()
                    .map(|&v| Q::T::from_f32(v))
                    .collect();
                Ok(Tensor::from_vec(
                    data,
                    (1, used, h, d),
                    self.model.device(),
                )?)
            };
            layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState {
                k_cache: take(0)?,
                v_cache: take(plane)?,
                current_end: used,
            }));
        }
        Ok(VoiceState {
            transformer_state: StreamingTransformerState {
                layer_states,
                attention_capture: None,
            },
        })
    }

    fn estimate_max_gen_len(&self, token_count: usize) -> usize {
        let gen_len_sec = token_count as f64 / TOKENS_PER_SECOND_ESTIMATE + GEN_SECONDS_PADDING;
        (gen_len_sec * self.config.mimi.frame_rate).ceil() as usize
    }

    /// Port of `generate_audio_stream`: splits the text into chunks and generates them one after
    /// the other from the voice state, calling `on_chunk` with the audio as it is decoded.
    /// Setting `stop` ends the generation early.
    pub fn generate_audio_stream(
        &self,
        voice: &VoiceState<Q>,
        text_to_generate: &str,
        max_tokens: usize,
        frames_after_eos: Option<usize>,
        stop: &AtomicBool,
        on_chunk: &mut (dyn FnMut(&[f32]) -> Result<()> + Send),
    ) -> Result<()> {
        let frames_after_eos = frames_after_eos.or(self.config.model_recommended_frames_after_eos);
        // This is a very simplistic way of handling long texts, as in the Python package.
        let chunks = split_into_best_sentences(
            self.tokenizer.as_ref(),
            text_to_generate,
            max_tokens,
            &self.text_options,
        )?;
        for chunk in chunks {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let (text, frames_after_eos_guess) = prepare_text_prompt(&chunk, &self.text_options)?;
            let frames = frames_after_eos.unwrap_or(frames_after_eos_guess + 2);
            self.generate_short_text(voice, &text, frames, stop, on_chunk)?;
        }
        Ok(())
    }

    fn generate_short_text(
        &self,
        voice: &VoiceState<Q>,
        text: &str,
        frames_after_eos: usize,
        stop: &AtomicBool,
        on_chunk: &mut (dyn FnMut(&[f32]) -> Result<()> + Send),
    ) -> Result<()> {
        let tokens = self.tokenizer.encode(text)?;
        self.generate_prepared_tokens(voice, &tokens, frames_after_eos, stop, on_chunk)
    }

    /// Opt into an error when generation exhausts its budget before the EOS tail.
    /// The default preserves the standard CLI warning; the upstream environment
    /// override `KPOCKET_TTS_ERROR_WITHOUT_EOS=1` also requires an error.
    pub fn set_require_eos(&mut self, required: bool) {
        self.require_eos = required;
    }

    pub fn set_seed(&self, seed: u64) {
        self.rng
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reseed(seed);
    }

    pub fn generate_prepared_tokens(
        &self,
        voice: &VoiceState<Q>,
        tokens: &[u32],
        frames_after_eos: usize,
        stop: &AtomicBool,
        on_chunk: &mut (dyn FnMut(&[f32]) -> Result<()> + Send),
    ) -> Result<()> {
        self.generate_tokens_internal(voice, tokens, frames_after_eos, stop, on_chunk, None, None)
    }

    pub fn generate_tokens_timestamped(
        &self,
        voice: &VoiceState<Q>,
        tokens: &[u32],
        frames_after_eos: usize,
        stop: &AtomicBool,
        on_chunk: &mut (dyn FnMut(&[f32]) -> Result<()> + Send),
        layout: crate::timestamps::Layout,
        on_word: &mut (dyn FnMut(crate::timestamps::WordEvent) -> Result<()> + Send),
    ) -> Result<()> {
        layout.validate(tokens.len())?;
        self.generate_tokens_internal(
            voice,
            tokens,
            frames_after_eos,
            stop,
            on_chunk,
            Some(layout),
            Some(on_word),
        )
    }

    fn generate_tokens_internal(
        &self,
        voice: &VoiceState<Q>,
        tokens: &[u32],
        frames_after_eos: usize,
        stop: &AtomicBool,
        on_chunk: &mut (dyn FnMut(&[f32]) -> Result<()> + Send),
        layout: Option<crate::timestamps::Layout>,
        on_word: Option<&mut (dyn FnMut(crate::timestamps::WordEvent) -> Result<()> + Send)>,
    ) -> Result<()> {
        let max_gen_len = completion_budget(self.estimate_max_gen_len(tokens.len()))?;
        let seq_budget = voice.len() + tokens.len() + max_gen_len;
        let transformer_state = voice.transformer_state.with_seq_budget(seq_budget)?;
        let mut state = TTSState {
            flow_lm_state: FlowLMState {
                transformer_state,
                conditions: None,
            },
        };

        let t_generating = Instant::now();
        let start = Instant::now();
        self.model.prompt_text(&mut state, tokens)?;
        log::info!("Prompting text took {} ms", start.elapsed().as_millis());

        if let Some(layout) = &layout {
            let heads = self
                .config
                .timestamp_heads
                .as_ref()
                .context("No timestamp heads for this checkpoint")?;
            anyhow::ensure!(
                heads.iter().all(|h| h.layer < self.num_layers
                    && h.head < self.config.flow_lm.transformer.num_heads),
                "Timestamp head outside model"
            );
            state.flow_lm_state.transformer_state.attention_capture =
                Some(ptts::timestamp_attention::AttentionCapture::new(
                    heads.iter().map(|h| (h.layer, h.head)).collect(),
                    voice.len(),
                    layout.token_to_unit.clone(),
                )?);
        }
        let sample_rate = self.sample_rate();
        let (latent_tx, latent_rx) =
            std::sync::mpsc::sync_channel::<GeneratedFrame<Q::T, Q::B>>(LATENT_QUEUE_CAPACITY);
        let (generated, decoded) = std::thread::scope(|s| {
            let decoder = s.spawn(move || {
                self.decode_worker(
                    latent_rx,
                    on_chunk,
                    layout.map(|l| crate::timestamps::Alignment::new(l.units)),
                    on_word,
                )
            });
            let generated = self.autoregressive_generation(
                &mut state,
                max_gen_len,
                frames_after_eos,
                &latent_tx,
                stop,
            );
            // Disconnect rather than enqueue an EOF sentinel: closing must never
            // wait for capacity, including after cancellation or a decoder error.
            drop(latent_tx);
            let decoded = decoder
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("the decoder panicked")));
            (generated, decoded)
        });
        generated?;
        let total_samples = decoded?;

        let audio_ms = total_samples * 1000 / sample_rate;
        let generation_ms = t_generating.elapsed().as_millis().max(1) as usize;
        log::info!(
            "Generated: {audio_ms} ms of audio in {generation_ms} ms so {:.2}x faster than real-time",
            audio_ms as f64 / generation_ms as f64
        );
        Ok(())
    }

    fn autoregressive_generation(
        &self,
        state: &mut TTSState<Q>,
        max_gen_len: usize,
        frames_after_eos: usize,
        latent_tx: &std::sync::mpsc::SyncSender<GeneratedFrame<Q::T, Q::B>>,
        stop: &AtomicBool,
    ) -> Result<()> {
        let mut previous: Option<Tensor<Q::T, Q::B>> = None;
        let mut eos_step = None;
        let mut step_times = Vec::new();
        let mut reached_max_len = true;
        for generation_step in 0..max_gen_len {
            if stop.load(Ordering::Relaxed) {
                reached_max_len = false;
                break;
            }
            let start = Instant::now();
            let input = match &previous {
                None => StepInput::Bos { batch: 1 },
                Some(latent) => StepInput::Latent(latent),
            };
            if let Some(capture) = state
                .flow_lm_state
                .transformer_state
                .attention_capture
                .as_mut()
            {
                capture.begin_frame();
            }
            let (latent, is_eos) = {
                let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
                self.model.generate_step(state, input, &mut *rng)?
            };
            if is_eos && eos_step.is_none() && generation_step >= MIN_FRAMES_BEFORE_EOS {
                eos_step = Some(generation_step);
            }
            if eos_step.is_some_and(|eos_step| generation_step >= eos_step + frames_after_eos) {
                reached_max_len = false;
                break;
            }
            // A closed channel means the decoder stopped on an error, which it reports.
            if latent_tx
                .send(GeneratedFrame {
                    latent: latent.clone(),
                    scores: state
                        .flow_lm_state
                        .transformer_state
                        .attention_capture
                        .as_ref()
                        .map(|c| c.finish_frame())
                        .transpose()?,
                })
                .is_err()
            {
                reached_max_len = false;
                break;
            }
            previous = Some(latent);
            step_times.push(start.elapsed().as_millis());
        }
        check_generation_completion(
            reached_max_len,
            self.require_eos,
            std::env::var("KPOCKET_TTS_ERROR_WITHOUT_EOS")
                .ok()
                .as_deref(),
        )?;
        if !step_times.is_empty() {
            let mean = step_times.iter().sum::<u128>() / step_times.len() as u128;
            log::info!("Average generation step time: {mean} ms");
        }
        Ok(())
    }

    /// Decodes at most four queued latents per call, and hands the audio
    /// to `on_chunk`. Returns the number of samples decoded.
    fn decode_worker(
        &self,
        latent_rx: std::sync::mpsc::Receiver<GeneratedFrame<Q::T, Q::B>>,
        on_chunk: &mut (dyn FnMut(&[f32]) -> Result<()> + Send),
        mut alignment: Option<crate::timestamps::Alignment>,
        mut on_word: Option<&mut (dyn FnMut(crate::timestamps::WordEvent) -> Result<()> + Send)>,
    ) -> Result<usize> {
        let mut mimi_state = self.model.init_mimi_state(1)?;
        // A fresh Mimi decoder state puts a small step in its first samples, heard as a click at
        // the start of every chunk: fade the first 5 ms in.
        let fade_len = self.sample_rate() / 200;
        let mut fade_in = true;
        let mut total = 0;
        while let Ok(latent) = latent_rx.recv() {
            let mut latents = vec![latent];
            let mut finished = false;
            while latents.len() < MAX_DECODE_FRAMES {
                match latent_rx.try_recv() {
                    Ok(latent) => latents.push(latent),
                    Err(TryRecvError::Disconnected) => {
                        finished = true;
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                }
            }
            let refs: Vec<&Tensor<Q::T, Q::B>> = latents.iter().map(|f| &f.latent).collect();
            let batch = Tensor::cat(&refs, 1)?;
            let audio = self.model.decode_latent(&batch, &mut mimi_state)?;
            let mut pcm = audio.to_vec()?;
            if fade_in {
                let n = fade_len.min(pcm.len());
                for (i, sample) in pcm[..n].iter_mut().enumerate() {
                    *sample *= if fade_len > 1 {
                        i as f32 / (fade_len - 1) as f32
                    } else {
                        0.0
                    };
                }
                fade_in = false;
            }
            if let (Some(alignment), Some(callback)) = (alignment.as_mut(), on_word.as_mut()) {
                let frame_size =
                    timestamp_frame_size(self.sample_rate(), self.config.mimi.frame_rate)?;
                anyhow::ensure!(
                    pcm.len() == latents.len() * frame_size,
                    "Timestamp frame/audio count mismatch"
                );
                for (frame, audio) in latents.iter().zip(pcm.chunks_exact(frame_size)) {
                    let scores = frame
                        .scores
                        .as_deref()
                        .context("Timestamp frame missing attention")?;
                    for event in
                        alignment.process(scores, crate::timestamps::is_voiced(audio), total)?
                    {
                        callback(event)?;
                    }
                    // Decode in batches as before, but deliver each timed frame at its
                    // real PCM boundary: consumers hold one pending word marker.
                    on_chunk(audio)?;
                    total += audio.len();
                }
            } else {
                on_chunk(&pcm)?;
                total += pcm.len();
            }
            if finished {
                break;
            }
        }
        if let (Some(alignment), Some(callback)) = (alignment.as_mut(), on_word.as_mut()) {
            for event in alignment.finish(total) {
                callback(event)?;
            }
        }
        Ok(total)
    }
}

/// Whether a voice source names a `.safetensors` state rather than audio.
pub fn is_safetensors_source(source: &str) -> bool {
    let path = if source.starts_with("http://") || source.starts_with("https://") {
        let rest = source.split(['?', '#']).next().unwrap_or(source);
        rest.splitn(4, '/').nth(3).unwrap_or("").to_string()
    } else if source.starts_with("hf://") {
        source
            .rsplit_once('@')
            .map_or(source, |(path, _)| path)
            .to_string()
    } else {
        source.to_string()
    };
    path.ends_with(".safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_generation_warns_without_eos_unless_environment_requires_it() {
        // No process-global environment mutation: tests can run concurrently.
        for environment in [None, Some(""), Some("0"), Some("true"), Some("2")] {
            assert!(check_generation_completion(true, false, environment).is_ok());
        }
        let error = check_generation_completion(true, false, Some("1")).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Generation reached maximum length without EOS"
        );
    }

    #[test]
    fn explicit_eos_requirement_does_not_turn_completion_or_cancellation_into_errors() {
        for environment in [None, Some("0"), Some("1")] {
            assert!(check_generation_completion(true, true, environment).is_err());
            assert!(check_generation_completion(false, true, environment).is_ok());
            assert!(check_generation_completion(false, false, environment).is_ok());
        }
    }

    #[test]
    fn timestamp_frames_keep_fractional_frame_rates() {
        assert_eq!(timestamp_frame_size(24000, 12.5).unwrap(), 1920);
        assert!(timestamp_frame_size(24000, 0.).is_err());
        assert!(timestamp_frame_size(24000, f64::NAN).is_err());
    }
    #[test]
    fn late_eos_retains_the_onnx_completion_margin() {
        // 45 tokens: ceil((45 / 3 + 2) * 12.5) = 213. EOS at 207 needs
        // a loop reaching 213 to finish a six-frame tail, not a failure.
        let estimated = 213;
        let budget = completion_budget(estimated).unwrap();
        assert_eq!(budget, 216);
        assert!(!(0..estimated).any(|frame| frame >= 207 + 6));
        assert!((0..budget).any(|frame| frame >= 207 + 6));
        // Also preserve a short input's eight-frame tail at the boundary.
        assert!((0..budget).any(|frame| frame >= 205 + 8));
        assert!(completion_budget(usize::MAX).is_err());
    }

    #[test]
    fn safetensors_sources_are_recognized() {
        assert!(is_safetensors_source("voice.safetensors"));
        assert!(is_safetensors_source(
            "https://example.com/v.safetensors?download=1"
        ));
        assert!(is_safetensors_source("hf://a/b/v.safetensors@main"));
        assert!(!is_safetensors_source("hf://a/b/v.wav"));
        assert!(!is_safetensors_source("alba"));
    }
}
