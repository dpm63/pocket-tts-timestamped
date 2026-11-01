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
  Mimi-encoded silence, which the decoder warms up on), in safetensors. English 2026-01 can't
  encode silence into its latent space, so it warms up on its own generated silence instead
  (frames before speech in plain generations, below −65 dB).
- **Weights:** each file records the SHA-256 of the weights it was trained on
  (`weights_sha256` metadata).

## Checks

Each run measures the fork's own generation on the dataset's test voices and texts, without and
with the exported file:

- speech onset by the VAD (the Silero model faster-whisper bundles: the first 3 windows of
  32 ms in a row with speech probability ≥ 0.5, in 80 ms frames). Loudness can't tell quiet
  speech from silence, so quiet voices looked late; the loudness onset (first frame above
  −35 dB with another within 3 frames) is still reported as `loud_*`;
- abrupt starts (the first 30 ms of the onset louder than −45 dB);
- after EOS, with and without a final period: how often the tail holds speech (a frame
  louder than −35 dB), and how often EOS fires on the last loud frame, shortening the ending.
  The rate of any sound above −50 dB is reported too; on `english` it is mostly the decay
  of the last word, even without vectors (13–28%), and inaudible;
- Whisper (large-v3-turbo, int8 on CPU) on 60 of the generations with a period, spread over
  voices and texts: how often the first word is right, and the mean WER. Onset metrics can't
  catch a clipped or garbled first word (on `english_drifting_26-09` a setting with the best
  onsets said "lumi" for "Jimmy").

Vectors ship only if, against the same model without them:

- speech starts by frame 1 in at least 70% of trials, and 20 points more often;
- speech after EOS and early EOS each rise by at most 5 points with a period, 10 without;
- Whisper's first word right falls by at most 10 points, and its WER rises by at most 4.

The slack covers the noise of 180 trials (9 voices × 20 texts). At 10 texts, early EOS
without a period swung by 5 points against 20 on the same vectors.

## Data

A private [dataset repo](https://huggingface.co/datasets/geli95us/pocket-tts-leading-silence), pinned in `data.py`; `HF_TOKEN` must be able to read it:

- English: the built-in voices plus Kokoro voice variants, and LibriTTS sentences;
- French, German, Spanish, Italian, Portuguese, Dutch: Multilingual LibriSpeech speakers (plus
  the 9-hour subset's for Dutch, Italian and Portuguese) and transcript prefixes, with the
  language's built-in voice as a test voice;
- splits by voice and by text.

`build_dataset.py` builds it from the research workspace. Set `LEADING_SILENCE_DATA` to a
local copy to skip the download.

## Running

```bash
python -m scripts.leading_silence plan --models english   # group configs by weights hash
python -m scripts.leading_silence train --config english_2026-10 --out work/english [--smoke | --timing]
python -m scripts.leading_silence publish --work work [--open-pr]
```

Training needs the voice-cloning weights (`HF_TOKEN`) and `requirements.txt` (the VAD and
Whisper). For a 6-layer model at 4 threads on a desktop
CPU, training takes about 9 min (rollouts 2, whole-sequence rollouts 3, training 4), plus the
checks.

Vectors that fail a check stay in the run's artifact (90 days) and are listed under "Not
published". To hear what a check flagged, `listen` replays the check's own trials (same voices,
texts and seeds) without and with the vectors, and saves the pairs it flags with the vectors
only (at most `--most` per check), with a `manifest.json`:

```bash
python -m scripts.leading_silence listen --config spanish --vectors RESULTS/vectors.safetensors     --out listening/spanish --checks tail_period,tail_no_period   # also first_word, abrupt
```

If they turn out to be false negatives, publish them without retraining:

```bash
python -m scripts.leading_silence publish --from-run RUN_ID --work work/RUN_ID --accept dutch
```

It downloads the run's results, copies in the vectors of the passing configs and of the
accepted ones (under their checkpoint name, with every alias pointed at them, weights checked),
and writes `work/RUN_ID/pr-body.md` with the accepted reasons. Commit the result yourself.

`--timing` runs every phase at reduced size with the real batch shapes (64 rows, 3 steps, one
whole-sequence batch, 2 texts per voice, 6 Whisper clips per side). Its report extrapolates the
full run's time per phase (`full_run_times_s`) and the memory the stored prefix keys/values take
(`full_run_stored_mb`), beside each phase's peak RSS. Use it before a first run on a model size.

The workflow runs these steps on `workflow_dispatch` (choose `models`; `force` retrains;
`timing`). The plan trains every group of same-weights configs whose vectors are missing or
were trained on other weights. It opens a pull request that adds
`pocket_tts_timestamped/config/leading_silence/<name>.safetensors` and points every config with
the same weights at it. `<name>` is the group's dated config, never a bare language name such
as `english`: upstream moves those aliases to new weights.
