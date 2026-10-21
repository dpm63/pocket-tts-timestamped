"""Plan which checkpoints to train, and publish trained vectors as a pull request.

Configs that load the same weights share vectors: the plan groups them by the weights file's
SHA-256, read from the Hub's file metadata (nothing is downloaded). Published vectors go to
`pocket_tts_timestamped/config/leading_silence/<name>.safetensors`, and every config of the
group gets `leading_silence_vectors` pointing at them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from pocket_tts_timestamped.utils.config import CONFIGS_DIR
from scripts.leading_silence.pipeline import REPORT_FILE, VECTORS_FILE

VECTORS_DIR = "leading_silence"


def weights_sha256(source: str) -> str:
    """SHA-256 of an `hf://owner/repo/path[@revision]` file, from the Hub's LFS metadata."""
    from huggingface_hub import HfApi

    match = re.fullmatch(r"hf://([^/]+/[^/]+)/([^@]+)(?:@(.+))?", source)
    if match is None:
        raise ValueError(f"Not an hf:// weights path: {source}")
    repo, path, revision = match.groups()
    (info,) = HfApi().get_paths_info(repo, [path], revision=revision)
    lfs = getattr(info, "lfs", None)
    if lfs is None:
        raise ValueError(f"No LFS metadata for {source}")
    return lfs.sha256


def _configs() -> dict[str, dict[str, Any]]:
    return {
        path.stem: yaml.safe_load(path.read_text()) for path in sorted(CONFIGS_DIR.glob("*.yaml"))
    }


def plan(args: argparse.Namespace) -> None:
    configs = _configs()
    wanted = [m.strip() for m in args.models.split(",") if m.strip()] or sorted(configs)
    unknown = set(wanted) - set(configs)
    if unknown:
        raise SystemExit(f"Unknown configs: {sorted(unknown)}")
    groups: dict[str, list[str]] = {}
    for name, config in configs.items():
        source = config.get("weights_path")
        if source:
            groups.setdefault(weights_sha256(source), []).append(name)
    matrix = []
    for digest, aliases in sorted(groups.items()):
        if not set(aliases) & set(wanted):
            continue
        if not args.force and all(configs[a].get("leading_silence_vectors") for a in aliases):
            print(f"{', '.join(aliases)}: already has vectors", flush=True)
            continue
        matrix.append({"key": digest[:12], "config": aliases[0], "aliases": aliases})
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "plan.json").write_text(json.dumps({"matrix": matrix}, indent=2) + "\n")
    print(json.dumps(matrix, indent=2), flush=True)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"matrix={json.dumps({'include': matrix})}\n")
            stream.write(f"active={'true' if matrix else 'false'}\n")


def set_vectors(text: str, relative: str) -> str:
    """The config text with `leading_silence_vectors` set (added before `flow_lm:`)."""
    line = f"leading_silence_vectors: {relative}\n"
    pattern = r"(?m)^leading_silence_vectors:[^\n]*\n"
    if re.search(pattern, text):
        return re.sub(pattern, line, text, count=1)
    position = text.index("flow_lm:")
    return text[:position] + line + "\n" + text[position:]


def _row(name: str, report: dict[str, Any]) -> str:
    before, after = report["baseline"], report["vectors"]

    def pair(section: str, key: str, percent: bool = True) -> str:
        a, b = before[section][key], after[section][key]
        return f"{a:.0%} → {b:.0%}" if percent else f"{a:g} → {b:g}"

    return (
        f"| {name} | {pair('onsets', 'speech_by_frame_1')} | "
        f"{pair('onsets', 'onset_p95', percent=False)} | {pair('onsets', 'abrupt_rate')} | "
        f"{pair('tail_without_period', 'tail_speech_rate')} | "
        f"{pair('tail_with_period', 'tail_speech_rate')} |"
    )


def publish(args: argparse.Namespace) -> None:
    work = Path(args.work)
    matrix = json.loads((work / "plan.json").read_text())["matrix"]
    run = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/" + str(
        os.environ.get("GITHUB_RUN_ID")
    )
    lines = [
        "Leading-silence vectors trained by the `leading-silence` workflow "
        f"([run and artifacts]({run})).",
        "",
        "Recipe: `scripts/leading_silence` (bos_shortcut handoff, section 2, with 7 "
        "whole-sequence rows per step and the EOS loss of section 7). Checked through the "
        "fork's own generation, without and with the exported file, on the dataset's test "
        "voices and texts.",
        "",
        "| Checkpoint | Speech by frame 1 | Onset p95 (frames) | Abrupt starts "
        "| Speech after EOS, no period | Speech after EOS, period |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    published, failed, smoke = [], [], False
    for item in matrix:
        folder = work / "results" / item["key"]
        if not (folder / REPORT_FILE).exists():
            failed.append(f"{item['config']}: no result (the training job failed)")
            continue
        report = json.loads((folder / REPORT_FILE).read_text())
        smoke = smoke or report["smoke"]
        lines.append(_row(", ".join(item["aliases"]), report))
        if report["problems"]:
            failed.append(f"{item['config']}: " + "; ".join(report["problems"]))
            continue
        name = item["config"]
        target = CONFIGS_DIR / VECTORS_DIR / f"{name}.safetensors"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(folder / VECTORS_FILE, target)
        for alias in item["aliases"]:
            path = CONFIGS_DIR / f"{alias}.yaml"
            path.write_text(set_vectors(path.read_text(), f"{VECTORS_DIR}/{name}.safetensors"))
        published.append(name)
    if failed:
        lines += ["", "Not published:", *(f"- {reason}" for reason in failed)]
    body = work / "pr-body.md"
    body.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(body.read_text(), flush=True)
    if not published or smoke or not args.open_pr:
        print("Nothing to publish" if not published else "Publishing disabled", flush=True)
        return
    base = os.environ["GITHUB_REF_NAME"]
    branch = f"automation/leading-silence-{os.environ['GITHUB_RUN_ID']}"

    def command(*arguments: str) -> None:
        subprocess.run(list(arguments), check=True)

    command("git", "config", "user.name", "github-actions[bot]")
    command("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    command("git", "switch", "-c", branch)
    command("git", "add", "--", str(CONFIGS_DIR))
    command("git", "commit", "-m", "Add leading-silence vectors for " + ", ".join(published))
    command("git", "push", "origin", f"HEAD:{branch}")
    command(
        "gh", "pr", "create", "--base", base, "--head", branch,
        "--title", "Leading-silence vectors for " + ", ".join(published),
        "--body-file", str(body),
    )  # fmt: skip
