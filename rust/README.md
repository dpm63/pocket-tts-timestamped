# pocket-tts, the standalone binary

A port of the `pocket-tts` CLI (`generate`, `serve`, `export-voice`) that runs without Python,
built on the [xn-ptts](https://github.com/gradium-ai/xn-ptts) implementation of the model,
through [a fork](https://github.com/gabrieldemarmiesse/xn-ptts/tree/pocket-tts-small-models)
that adds what pocket-tts models need. This branch provides source builds;
it does not add prebuilt release artifacts.

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

## Word timestamps on this branch

This `rust` branch adds a Rust **library** interface for streaming word
starts and ends. The Python package keeps its own version and implementation.
The synthesis modules and configuration values of the Rust engine are frozen
to **Pocket TTS 4.0.1**, commit `a922b912dfa366975294afb568d815b305acde5d`;
this is not an update to upstream `main`. See [`source.json`](source.json).
The old Rust package label `3.3.0` at that upstream tag is corrected to `4.0.1`.

The algorithm and calibrated attention heads come from this repository's Python
implementation at `813d892a64a69894750d7f42bab129ea5804a654`. Only selected
FlowLM query/key heads are captured. The existing `sdpa_decode` calculation is
retained; disabled timing performs no attention readback. Normal and timed calls
use the same weights, sampler, decoder, EOS policy and text/token inputs.
Compatibility fixes retained from the application port: a bounded four-frame
latent queue, decoder batches of at most four frames, completion margin and
explicit reseeding. No model conversion or voice download is required by timing.

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

`prepare.py` materializes the existing XN base `7efa2697` under ignored
`rust/.deps/`, applies the versioned selected-head patch and verifies the base,
patch digest, patched file hashes and tracked/untracked changes. It refuses a
modified cache; use a fresh destination to investigate a mismatch. It never
changes the separate developer checkout in `Libraries/xn-ptts`.
Dependency setup errors retain their original Git output and cause.

### Library contract

`TtsModel::generate_tokens_timestamped` accepts the already prepared token IDs,
a `timestamps::Layout`, a PCM callback and a `WordEvent` callback. The layout
contains source text spans, word indices, synthetic-punctuation flags and
fractional token-to-unit overlap rows. This keeps text preparation and original
text mapping in the caller, avoiding a second set of text substitutions.
`Layout::validate` rejects inconsistent dimensions, values and word spans.
Call `generate_prepared_tokens` when timing is not needed.

Generation-budget exhaustion preserves the standard Rust CLI behavior: a warning,
with an error only when `KPOCKET_TTS_ERROR_WITHOUT_EOS=1`. Library callers that
must reject a potentially incomplete result can call `model.set_require_eos(true)`
after loading. This applies to ordinary and timestamped generation without
changing the CLI default. A normal EOS-tail completion or requested cancellation
does not become a budget-exhaustion error. The MAUI application bridge opts in
explicitly; the timestamp algorithm itself does not require strict completion.

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
boundaries are estimates, not proof that every input word was pronounced.

The decoder still computes up to four frames together. With timing enabled it
delivers PCM **one frame at a time** (1,920 samples at 24 kHz/12.5 Hz) so consumers
with one pending word marker cannot overwrite or anticipate events. This changes
callback granularity, not waveform samples or synthesis grouping. No wall-clock
timers or synthesized pauses are introduced. Callback errors stop generation and
join the decoder; callbacks must not reenter the model. A loaded model supports
one synthesis operation at a time.

The existing `generate`/`serve` command line remains an ordinary audio interface;
no timestamp CLI switch is claimed here. Library integration is the implemented
path. The app bridge is tracked separately in the Leggio repository; it snapshots
these sources and patch files, so an installed app never reads developer folders.

### Verification and limits

The unchanged Python aligner is compared with Rust on 200 generated fixtures by
`check-parity.py`. Native application tests cover real catalog checkpoints,
waveform parity with timing disabled/enabled, source word positions, frame
boundaries, callback errors, Stop and background timing deactivation. See the
application's `Tests/PocketRustIntegrationTests.cs` for device-independent probes.
Verified results are recorded below; host tests
do not certify rendering, battery usage or performance on a physical phone.

## Stable baseline

Rust synthesis sources/configs: Pocket TTS 4.0.1 (`a922b912dfa366975294afb568d815b305acde5d`).
The first local commit imports this baseline; the following commit adds timestamps.

Verified locally on 23 November 2026:

- Separate base import (`3076881`): 16 upstream Rust tests passed; two reviews
  confirmed stable-source/config parity and absence of timestamp code.
- Rust library: 26 tests passed, including standard/explicit EOS-exhaustion
  policies and normal completion/cancellation. The standalone CLI also passed
  `cargo check --locked --release --manifest-path rust/Cargo.toml --bin pocket-tts`.
- Rust/Python alignment: 200 deterministic reference fixtures passed.
- Dependency materializer: three offline regressions passed, including injected
  Git/timeout diagnostics, unexpected build scripts and out-of-hunk edits.
- Application native integration: all catalog checkpoints across seven languages
  generated audio with word markers; the Italian fixed-seed comparison had
  identical PCM (maximum sample delta 0). The real portable speech consumer,
  cancellation, callback error identity/stack and 320 ms decoder bound passed.

These automated results do not qualify physical iPhone playback. A separate
MAUI application integration was also tested by a user on macOS: the visual
highlight initially appeared about 500 ms early and was acceptable in a second
trial with a temporary 500 ms presentation-only delay. That delay remains in the
application, not in this library. This is a limited listening observation, not a
measured alignment-accuracy result or a universal playback-latency correction.
Word-event offsets remain estimates on the generated source-PCM timeline.
`.github/workflows/rust-timestamps.yml` prepares the dependency, runs Rust tests
and checks alignment parity on pull requests. The workflow has not yet run remotely; the results above were obtained locally.

Checkpoint calibration in the app bridge uses `calibrations.json`: the source
weights SHA-256, language and model dimensions must match. The September and
October Italian checkpoints deliberately use different heads, as in the Python
reference history. An unknown checkpoint keeps ordinary audio and sentence-level
presentation; the app must not guess a timing calibration from its language.
