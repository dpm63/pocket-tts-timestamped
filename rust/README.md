# pocket-tts, the standalone binary

A port of the `pocket-tts` CLI (`generate`, `serve`, `export-voice`) that runs without Python,
built on the [xn-ptts](https://github.com/gradium-ai/xn-ptts) implementation of the model,
through [a fork](https://github.com/gabrieldemarmiesse/xn-ptts/tree/pocket-tts-small-models)
that adds what pocket-tts models need. Only source builds are provided, not prebuilt release
artifacts.

```bash
# From rust/: prepare the pinned, patched model dependency first.
python3 prepare.py
cargo build --release
./target/release/pocket-tts generate --text "Hello world."
```

Building needs CMake and a C++ compiler, for the SentencePiece tokenizer that most community
models use. x86 builds target x86-64-v3 (AVX2), see `.cargo/config.toml`.

## Differences with the Python CLI

- Runs on the CPU only (`--device cpu`), and `--quantize` uses xn's int8 weights.

- `serve --reload` is accepted and ignored. The server answers `/`, `/health` and `/tts` as the
  Python one does, without FastAPI's `/docs`.
- The noise is drawn from a different generator, so the audio differs sample by sample from the
  Python package's, as two runs with different seeds would. The seed is fixed, so a command
  always gives the same audio; `POCKET_TTS_SEED` sets another one.
- The tensor operations use 4 threads by default; set `RAYON_NUM_THREADS` to change it.

## Word timestamps

The Rust **library** streams word starts and ends alongside the audio; the `generate` and
`serve` commands do not expose them. The alignment is a port of the Python package's
`pocket_tts_timestamped/timestamps/alignment.py`, and the attention heads it reads are each
config's `timestamp_heads`. The configs are embedded from `pocket_tts_timestamped/config`, the
same files the Python package reads, so heads and weights revisions cannot drift between the two.

The synthesis sources come from Pocket TTS 4.0.1 (`a922b912dfa366975294afb568d815b305acde5d`),
except `src/text.rs`, which follows this repository's `main`; see [`source.json`](source.json).

Only the selected FlowLM heads are captured, and only their text keys are read; with timing
disabled, nothing is read back. Timed and untimed calls use the same weights, sampler, decoder,
EOS policy and tokens. No model conversion or voice download is required by timing.

### Build and checks

Prerequisites: Python 3, Git, Rust 1.96, CMake and a C++ compiler. PyTorch is
needed only for comparison with the reference Python algorithm.

Run from the repository root:

```sh
python3 rust/prepare.py
cargo test --locked --release --manifest-path rust/Cargo.toml --lib
cargo build --locked --release --manifest-path rust/Cargo.toml
python3 rust/test_prepare.py
# Run in a Python environment containing PyTorch:
python3 rust/check-parity.py
```

`prepare.py` materializes the pinned xn-ptts base `7efa2697` under the ignored
`rust/.deps/`, applies the versioned selected-head patch and verifies the base,
patch digest, patched file hashes and tracked/untracked changes. It refuses a
modified cache; use a fresh destination to investigate a mismatch. Git commands
use LF checkouts regardless of Windows/global autocrlf settings, without changing
the user’s Git configuration.

`check-parity.py` runs the Python aligner and silence detection on generated
fixtures and checks that Rust gives the same word events and voiced/silent
decisions. `.github/workflows/rust-timestamps.yml` runs all of the above on pull
requests that touch `rust/` or the files it embeds.

### Library contract

`TtsModel::generate_tokens_timestamped` accepts the already prepared token IDs,
a `timestamps::Layout`, a PCM callback and a `WordEvent` callback. The layout
contains source text spans, word indices, synthetic-punctuation flags and
fractional token-to-unit overlap rows. This keeps text preparation and original
text mapping in the caller, avoiding a second set of text substitutions.
`Layout::validate` rejects inconsistent dimensions, values and word spans.
Call `generate_prepared_tokens` when timing is not needed.

```rust,ignore
model.generate_tokens_timestamped(
    &voice, &tokens, frames_after_eos, &stop,
    &mut |pcm| { /* enqueue source-rate PCM */ Ok(()) },
    layout,
    &mut |event| { /* enqueue word marker at this PCM boundary */ Ok(()) },
)?;
```

`WordEvent` has caller-owned `start`, `length`, `word_index`, a source PCM
`sample_offset`, and `is_end`. The coordinates are not UTF-8/UTF-16 conversions
inside Rust: the caller chooses its text coordinate system consistently.
Offsets restart at zero for each token request; add the audio already emitted
when concatenating requests. Start events precede their PCM frame. End events
can occur on silence, a word transition, or at the end of the request. Word
boundaries are estimates on the generated audio's timeline, not proof that every
input word was pronounced, and do not account for playback latency.

The decoder computes up to four frames together. With timing enabled it
delivers PCM **one frame at a time** (1,920 samples at 24 kHz/12.5 Hz) so consumers
with one pending word marker cannot overwrite or anticipate events. This changes
callback granularity, not waveform samples. Callback errors stop generation and
join the decoder; callbacks must not reenter the model. A loaded model supports
one synthesis operation at a time.

At most four generated frames wait for the decoder, so a slow PCM consumer holds
generation back instead of building up a backlog. The generation budget has three
frames of margin beyond the estimate, so that an EOS near the estimate still gets
its post-EOS tail. `set_seed` reseeds the noise generator.

Running out of budget before EOS is a warning, as in the CLI, and an error only
when `KPOCKET_TTS_ERROR_WITHOUT_EOS=1`. Callers that must reject a possibly
incomplete result can call `model.set_require_eos(true)` after loading; this
applies to ordinary and timestamped generation. A normal EOS-tail completion or
a requested cancellation is never a budget-exhaustion error.

### Calibrations for other weights

Callers that load other weights than those pinned in these configs, with a config
of their own, can find the heads to put in its `timestamp_heads` with
`timestamps::calibrated_heads`. It reads `calibrations.json`, where the weights'
SHA-256, language and model dimensions must all match. Unknown weights get no
heads, so no timestamps, rather than the heads of another checkpoint of the same
language. The September and
October Italian checkpoints use different heads, as in the Python reference
history.
