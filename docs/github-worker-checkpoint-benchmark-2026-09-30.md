# Checkpoint evaluation on GitHub-hosted CPU workers

Measured on 30 September 2026 in [this successful Actions run](https://github.com/dpm63/pocket-tts-timestamped/actions/runs/36656357594). The run used the public repository's `ubuntu-24.04` standard runners, which GitHub lists as 4 CPU cores, 16 GB RAM, 14 GB SSD, and no GPU. The benchmark used 36 representative English prompts from `.timestamp-eval`: 4 warmups, followed by 32 timed samples. Their measured texts ranged from 19 to 92 characters and covered 16 voices. STT used the 6-layer model's generated audio (96.64 seconds across the 32 timed clips). The 24-layer TTS produced 105.36 seconds of audio for those prompts.

| Inference path | Batching | Timed 32 samples | Seconds/sample | Extrapolated 500 |
| --- | ---: | ---: | ---: | ---: |
| Pocket TTS English 6L, with timestamps | 1 | 36.39 s | 1.137 s | 9.48 min |
| Pocket TTS English 24L, with timestamps | 1 | 103.64 s | 3.239 s | 26.99 min |
| Parakeet TDT 0.6B v3, word timestamps | 16 | 19.75 s | 0.617 s | 5.14 min |
| CrisperWhisper 2.0 Small, CT2 int8 | 1 | 61.54 s | 1.923 s | 16.03 min |
| CrisperWhisper 2.0 Small, Transformers float32 | 4 | 137.80 s | 4.306 s | 35.88 min |
| CrisperWhisper 2.0 Large, CT2 int8 | 1 | 223.87 s | 6.996 s | 58.30 min |

CT2 int8 was **2.24 times faster** than batched Transformers float32 for Small, so the workflow selected CT2 int8 for Large. The CT2 Large model returned word timestamps for 29 of the 32 timed inputs; three returned no word list despite requesting timestamps. This matters for usable reference count but does not invalidate the runtime measurement.

The 500-sample figures are `500 × (timed inference seconds / 32)`. They exclude runner setup, dependency installation, model downloads/loading, voice embedding setup, artifact transfer, and the four warmups. They assume a similar prompt and audio-length mix. In this run, the job wall times for the 36-sample benchmark, including those startup steps, were 1:47 (6L), 3:15 (24L), 1:48 (Parakeet), 1:59 (Small CT2), 3:21 (Small Transformers), and 5:31 (Large CT2). The whole workflow completed in about 10:47 because jobs ran concurrently and Large waited for the Small comparison.

For one checkpoint using Parakeet, CT2 Small, and CT2 Large as references, the summed **inference-only runner time** is about **89 minutes with 6L TTS** or **106 minutes with 24L TTS**. If the three STT jobs start in parallel after generation, the inference-only **wall time** is about **68 minutes for 6L** or **85 minutes for 24L**, governed by Large. The latter estimate uses STT timing on 6L audio; 24L produced about 9% more audio on this input set, so its STT time may be longer. A recurring evaluation that uses fewer references should sum only its chosen paths. Running the Small Transformers comparison again adds about 36 inference-only runner minutes.

GitHub says standard hosted runners are free and unlimited for public repositories, so this repository's measured CPU workflow has no Actions compute charge. Private repository standard runners have different CPU/RAM specifications and consume included minutes before per-minute billing, so these timings should not be transferred to a private runner. GitHub's paid GPU larger runner uses a Tesla T4 at $0.052/minute and has different performance; this benchmark did not measure it. Each GitHub-hosted job has a 6-hour limit, comfortably above these inference-only estimates.

The checked-in [workflow](../.github/workflows/benchmark-checkpoint-workers.yml) runs on a benchmark branch push or manual dispatch. The [driver](../scripts/benchmark_github_workers.py) and [fixture](../scripts/benchmark-data/checkpoint-prompts.json) reproduce the measurements. Raw result JSON files are in `docs/benchmarks/checkpoint-workers-2026-09-30/`.

Sources: [GitHub-hosted runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners), [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions), [runner pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing), [larger GPU runner specifications](https://docs.github.com/en/actions/reference/runners/larger-runners), [Actions limits](https://docs.github.com/en/actions/reference/limits).
