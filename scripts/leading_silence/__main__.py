"""python -m scripts.leading_silence {plan,train,publish} — see README.md."""

from __future__ import annotations

import argparse

from scripts.leading_silence import pipeline, publish


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m scripts.leading_silence")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="group configs by weights; choose what to train")
    plan.add_argument("--models", default="", help="comma-separated configs; blank: all")
    plan.add_argument("--force", action="store_true", help="retrain checkpoints with vectors")
    plan.add_argument("--work", default="leading-silence-work")
    plan.set_defaults(run=publish.plan)

    train = commands.add_parser("train", help="train, export and check one checkpoint")
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--eval-texts", type=int, default=20)
    train.add_argument("--threads", type=int, default=0, help="torch threads (0: all CPUs)")
    train.add_argument("--smoke", action="store_true", help="tiny run, never published")
    train.add_argument(
        "--timing",
        action="store_true",
        help="every phase at reduced size; reports full-run times and memory (never published)",
    )
    train.add_argument(
        "--whisper-trials",
        type=int,
        default=60,
        help="generations per side Whisper checks (0: none)",
    )
    train.set_defaults(run=pipeline.run)

    publish_parser = commands.add_parser("publish", help="open a PR with passing vectors")
    publish_parser.add_argument("--work", default="leading-silence-work")
    publish_parser.add_argument("--open-pr", action="store_true")
    publish_parser.add_argument(
        "--from-run", default="", help="download this finished workflow run's results into --work"
    )
    publish_parser.add_argument(
        "--accept",
        default="",
        help="comma-separated configs to publish despite failed checks (false negatives)",
    )
    publish_parser.set_defaults(run=publish.publish)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
