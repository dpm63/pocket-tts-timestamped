"""End-to-end CPU run of the locked BOS-shortcut recipe, timed per phase, as a CI job would run it.

    load model -> encode voice prompts -> teacher rollouts (keeping their prefix keys/values)
    -> cached bos_all training -> save the steer -> onset check (baseline vs steer, warm Mimi)

Voices are predefined pocket voices (downloaded from HF, nothing to commit);
texts come from a plain text file, one per line.

--timing-only runs without the gated voice-cloning weights: the voice projection
is random (same shape and scale), so costs are those of the real recipe but the
learned steer and the onset check mean nothing.

Run: python -m bos_shortcut.ci_bench --language english --texts texts_train.txt --out result.json
"""

import argparse
import json
import os
import platform
import random
import time

import torch

from .evaluate import measure_onsets, silence_latents
from .sequences import encode_voice
from .sites import all_sites, save_steer
from .train import Batcher, make_rollouts, subset_rollouts, summarize_rollouts, train_steer

TRAIN_VOICES = "alba,anna,azelma,bill_boerst,caro_davy,eve,george,jane,javert,jean,marius,mary,michael,paul,peter_yearsley"  # pocket built-ins of the train split
CHECK_VOICES = "charles,eponine"  # test split
ROLLOUT_FRAMES = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="english")
    parser.add_argument("--texts", required=True)
    parser.add_argument("--voices", default=TRAIN_VOICES)
    parser.add_argument("--check-voices", default=CHECK_VOICES)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument(
        "--pairs", type=int, default=160, help="rollouts generated; rows without onset are dropped"
    )
    parser.add_argument("--rollout-batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--check-texts", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = all CPUs)")
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="no voice-cloning weights: random voice projection",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from pocket_tts import TTSModel
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

    # After the import: pocket_tts pins torch to one thread (streaming latency).
    torch.set_num_threads(args.threads or os.cpu_count())

    times, info = {}, {"language": args.language, "torch_threads": torch.get_num_threads(), "cpu_count": os.cpu_count(),
                       "processor": platform.processor() or platform.machine(), "torch": torch.__version__, "args": vars(args)}  # fmt: skip
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu_model"] = next(
                line.split(":", 1)[1].strip() for line in f if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass

    def phase(name, fn):
        t0 = time.perf_counter()
        result = fn()
        times[name] = round(time.perf_counter() - t0, 2)
        print(f"[{name}] {times[name]:.1f}s", flush=True)
        return result

    total0 = time.perf_counter()
    model = phase("load_model", lambda: TTSModel.load_model(language=args.language))
    info["timing_only"] = not model.has_voice_cloning
    if model.has_voice_cloning:
        pass
    elif args.timing_only:
        fl = model.flow_lm
        # Voice embeddings at the text embeddings' scale.
        rms = float(fl.conditioner.embed.weight.pow(2).mean().sqrt())
        fl.speaker_proj_weight = torch.nn.Parameter(
            torch.randn(fl.dim, fl.ldim) * rms / fl.ldim**0.5, requires_grad=False
        )
    else:
        raise SystemExit(
            "voice-cloning weights missing (HF_TOKEN?); pass --timing-only for costs alone"
        )
    for p in model.parameters():
        p.requires_grad_(False)
    names = args.voices.split(",")
    latents = phase(
        "encode_voices",
        lambda: {
            i: encode_voice(model, _ORIGINS_OF_PREDEFINED_VOICES[n]) for i, n in enumerate(names)
        },
    )
    with open(args.texts, encoding="utf-8") as f:
        pool = [line.strip() for line in f if line.strip()]
    rng = random.Random(0)
    pairs = [(rng.randrange(len(names)), rng.choice(pool)) for _ in range(args.pairs)]
    # The rollouts keep their prefix keys/values, so cached training never recomputes the prefix.
    data = phase(
        "rollouts",
        lambda: make_rollouts(
            model, latents, pairs, args.rollout_batch, ROLLOUT_FRAMES, seed=1, keep_prefix_kv=True
        ),
    )
    info["rollouts"] = summarize_rollouts(data)
    n = min(args.n_train, len(data["k"]))
    data = subset_rollouts(data, n)
    train = phase("prefix_kv", lambda: Batcher(model, data, latents, store_prefix_kv=True))
    site = all_sites(len(model.flow_lm.transformer.layers))["bos_all"]
    steer, _ = phase(
        "train",
        lambda: train_steer(
            model,
            site,
            train,
            train,
            args.steps,
            args.batch,
            args.lr,
            seed=0,
            log_every=0,
            log=lambda *a: None,
        ),
    )
    phase(
        "save",
        lambda: save_steer(
            os.path.splitext(args.out)[0] + "_steer.pt",
            steer,
            steps=args.steps,
            lr=args.lr,
            n_train=n,
        ),
    )
    times["pipeline_total"] = round(time.perf_counter() - total0, 2)

    states = phase(
        "check_voice_states",
        lambda: {v: model.get_state_for_audio_prompt(v) for v in args.check_voices.split(",")},
    )
    warmup = silence_latents(model, 8)
    check_texts = pool[-args.check_texts :]
    info["check_baseline"] = phase(
        "check_baseline", lambda: measure_onsets(model, None, states, check_texts, warmup)
    )
    info["check_steer"] = phase(
        "check_steer", lambda: measure_onsets(model, steer, states, check_texts, warmup)
    )
    info["times_s"] = times
    with open(args.out, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
