# Leading-silence vectors

Pocket TTS starts every generation with 0.5–1 s of silence. `skip_leading_silence=True` removes
it with a few learned vectors added to the FlowLM on the BOS step. This folder trains them, one
file per checkpoint, and the `leading-silence` workflow runs it on GitHub's runners.

The method comes from the bos_shortcut research (its handoff document, sections 1, 2 and 7):

- **Where:** one vector per residual-stream index (each layer's input and the transformer
  output), added only on the BOS step.
- **Target:** at BOS, the frozen model plus vectors should produce the hidden state (z) the
  model itself had at its speech onset, in free-running "teacher" generations. The onset is the
  first sustained voiced segment that isn't a breath. Rollouts already speaking at frame 0 are
  left out.
- **Losses:**
  - z-MSE at BOS;
  - z-MSE over the next 4 frames, teacher-forced;
  - z-MSE over 7 whole sequences per step (onset to past EOS, texts without a final period),
    so frames long after BOS stay on course;
  - an EOS loss on every teacher-forced frame.

  The last two stop the vectors from making EOS fire early on text without a final period.
- **Setting:** 256 rows, lr 0.3, 100 steps for 6- and 8-layer models; 128 rows, lr 0.1, 50
  steps for 24 layers. Adam with warm-up, then cosine decay.
- **Output:** `vectors` ([layers + 1, d_model]) and `warmup_latents` (8 frames of
  Mimi-encoded silence, which the decoder warms up on), in safetensors.

## Checks

Each run measures the fork's own generation on the dataset's test voices and texts, without and
with the exported file:

- speech onset (first frame above −35 dB with another within 3 frames);
- abrupt starts (the first 30 ms of the onset louder than −45 dB);
- how often the frames generated after EOS hold speech, with and without a final period.

Vectors ship only if:

- speech starts by frame 1 in at least 70% of trials, and 20 points more often than without;
- speech after EOS without a period stays under 30% (or baseline + 15 points);
- speech after EOS with a period stays under 10% (or baseline + 5 points).

## Data

The [dataset repo](https://huggingface.co/datasets/geli95us/pocket-tts-leading-silence) (`data.py`):

- voices: the built-in voices plus Kokoro voice variants;
- texts: LibriTTS sentences;
- splits by voice and by text.

`build_dataset.py` builds it from the research workspace. Set `LEADING_SILENCE_DATA` to a
local copy to skip the download.

## Running

```bash
python -m scripts.leading_silence plan --models english   # group configs by weights hash
python -m scripts.leading_silence train --config english --out work/english [--smoke]
python -m scripts.leading_silence publish --work work [--open-pr]
```

Training needs the voice-cloning weights (`HF_TOKEN`). On a 4-core CPU, a 6-layer model takes
about TODO min.

The workflow runs these steps on `workflow_dispatch` (choose `models`; `force` retrains). It
opens a pull request that adds `pocket_tts_timestamped/config/leading_silence/<name>.safetensors`
and points every config with the same weights at it.

**Not covered yet:**
- English 2026-01: it can't encode silence into its latent space, so it needs the research's
  generated-silence warm-up.
- Languages other than English have no data in the dataset yet; the research used MLS speakers.
