# Automatic timestamp head selection

The `Evaluate checkpoint timestamp heads` workflow checks checkpoint references on pushes to `main`. New references are resolved to the weights actually available to the worker and SHA256-hashed. Identical content shares results, including config aliases. Changes to timestamp heads or generation context alone do not invalidate an evaluation. New language families need an entry in `common.py` before sampling.

Each new checkpoint uses a frozen order of Tatoeba prompts, interleaved across word-length bands and its published evaluation voices. German checkpoints exclude all prompts containing en or em dashes (`–`, `—`) before generation. Generation captures all heads once. The first 1,500 generated samples form the skip cohort. Pinned CrisperWhisper Medium runs on CPU with CT2 float32, explicit language, intended mode, and word timestamps. The first 500 strict normalized lexical matches form the MAE cohort. Extra samples are generated when the initial pool is insufficient, without changing the skip cohort.

All individual heads are ranked by raw MAE, without a skip prefilter. The search evaluates all 637 groups of one to five from the top ten, plus existing configured combinations. All individual heads remain candidates. Equal head weights are used. Every candidate with a measured MAE is scored without a skip-rate cutoff:

```text
score_ms = MAE_ms + 10 * skip_percentage + 0.5 * head_count
```

No separate validation samples are collected. MAE is explicitly a selection-cohort measurement. If no candidate has a measured MAE, the current configuration is preserved and the report flags it for review. The PR description includes current configured heads, five best candidates by MAE, five best candidates by score, and five zero-skip candidates by MAE. README rows separate skip and MAE sample/word counts and share the reference named above the table.

## Running

Use **Actions → Evaluate checkpoint timestamp heads → Run workflow**:

- **Initial/full rerun:** `force=true`, leave `models` blank. This evaluates every available config, deduplicating identical checkpoint content.
- **Force only selected checkpoints:** set `force_models=german_24l`; other selected checkpoints still reuse matching hashes. This can refresh a sampling policy without forcing every checkpoint.
- **Selected checkpoint:** set `models=english_2026-09` (or comma-separated config names).
- **Smoke test:** `models=english_2026-09`, `force=true`, `publish=false`, `skip_samples=60`, `mae_samples=20`. Reduced cohorts cannot publish configurations or enter the production reuse cache.
- **Rescore:** set `rescore_run` to a completed run ID and change `skip_penalty`, `head_penalty`. This reads saved metrics; generation, transcription, and alignment jobs are skipped. Set `publish=false` for a report-only comparison.
- **Resume:** reference jobs save accepted and rejected transcription progress and automatically dispatch another bounded run when needed. `resume_run` also allows manual continuation. Failed capture jobs can be rerun using their partial failure artifacts.

`HF_TOKEN` is optional. Without access to voice-cloning weights, the public weights are used when provided by the config; their actual content hash is recorded. GitHub Actions needs permission to create PRs, as in the repository's upstream-sync workflow. The implementation uses `GITHUB_TOKEN`; a separately configured App token is unnecessary for the initial version.

## Storage and concurrency

Three checkpoints can run concurrently, each with eight generation and eight alignment workers. Transcription is ordered within each checkpoint and parallel across checkpoints. Reference jobs checkpoint before the six-hour job limit. Multiple bounded runs continue collection until the match target is reached.

Large audio and all-head unit scores are temporary Actions caches, deleted after successful evaluation. Compact aggregate and per-utterance metrics, reference outputs, runtime provenance, reports, and frozen plans are Actions artifacts retained for 90 days. Failure/intermediate metric artifacts expire after seven days. No evaluation data is committed to the repository. After results expire, a changed/new checkpoint reference can require evaluation again; scoring an expired run requires fresh evaluation.

Publication starts from current `main`, checks that checkpoint references and configured heads still match the evaluated inputs, and preserves configs changed during the run. Unrelated main commits are incorporated. Each evaluation has its own results PR, so a later run cannot drop changes from an older unmerged PR. Rescoring updates the original evaluation's PR. A results PR with human edits is preserved rather than overwritten.

## Development

Evaluation-only dependencies live in `requirements.txt`; the inference package does not depend on CrisperWhisper or Numba. Worker environments install the repository, then these requirements. Run stages with `python -m scripts.timestamp_eval --help`. Scoring functions consume raw aggregate metrics independently of capture and alignment.

`tests/test_timestamp_evaluation.py` checks replay equivalence to production `WordAlignment`, skip/scoring units, candidate inclusion, raw-metric preservation, lexical acceptance, configuration/README updates, and artifact extraction. The dedicated CI job installs Numba so replay equivalence is exercised even though ordinary inference environments do not require it.
