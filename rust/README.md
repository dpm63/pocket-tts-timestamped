# pocket-tts, the standalone binary

A port of the `pocket-tts` CLI (`generate`, `serve`, `export-voice`) that runs without Python,
built on the [xn-ptts](https://github.com/gradium-ai/xn-ptts) implementation of the model,
through [a fork](https://github.com/gabrieldemarmiesse/xn-ptts/tree/pocket-tts-small-models)
that adds what pocket-tts models need. The release workflow (`.github/workflows/rust-release.yml`) attaches its builds to every GitHub
release.

```bash
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

## Stable baseline imported on 23 November 2026

Sources and configurations come from Pocket TTS 4.0.1, commit
`a922b912dfa366975294afb568d815b305acde5d`, without later main changes.
The legacy Rust package label at that tag is 3.3.0; the source release is4.0.1.
The build embeds the unmodified stable configs under `rust/config`, and reads
the web page from this fork’s renamed Python package. No Rust word-timestamp
implementation is included in this baseline commit.
