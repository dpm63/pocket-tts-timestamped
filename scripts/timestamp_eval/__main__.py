"""Run stages with python -m scripts.timestamp_eval."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["plan", "capture", "reference", "groups", "collect", "publish"]
    )
    parser.add_argument("--work", default="timestamp-work")
    parser.add_argument("--key", default="")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--models", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--before", default="")
    parser.add_argument("--resume-run", type=int, default=0)
    parser.add_argument("--rescore-run", type=int, default=0)
    parser.add_argument("--skip-samples", type=int, default=1500)
    parser.add_argument("--mae-samples", type=int, default=500)
    parser.add_argument("--skip-penalty", type=float, default=10.0)
    parser.add_argument("--head-penalty", type=float, default=0.5)
    parser.add_argument("--skip-limit", type=float, default=0.5)
    parser.add_argument("--budget-seconds", type=int, default=18000)
    args = parser.parse_args()
    if args.stage == "publish":
        from scripts.timestamp_eval.publish import publish_run

        publish_run(args)
    else:
        from scripts.timestamp_eval import runner

        stages = {
            "plan": runner.plan_run,
            "capture": runner.capture_run,
            "reference": runner.reference_run,
            "groups": runner.groups_run,
            "collect": runner.collect_run,
        }
        stages[args.stage](args)


if __name__ == "__main__":
    main()
