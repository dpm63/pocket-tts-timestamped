# BOS-shortcut CPU benchmark

Times the BOS-shortcut recipe (a learned `bos_all` steer that removes Pocket-TTS's
leading silence) on a GitHub-hosted runner: voice encoding, 160 teacher rollouts
(128 kept), cached-prefix training (25 steps, lr 0.1, batch 64), then a small
onset check. `bos_shortcut/` is a copy of the research code; `texts_train.txt` is
2000 LibriTTS training texts. Workflow: `.github/workflows/bos-shortcut-bench.yml`.
