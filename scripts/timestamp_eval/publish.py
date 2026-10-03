"""Publish only README/config changes; evaluation data stays in Actions artifacts."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from scripts.timestamp_eval.artifacts import Artifacts
from scripts.timestamp_eval.common import PROTOCOL, config_identity, git, head_label, read, write
from scripts.timestamp_eval.scoring import rank, report


def update_config(text: str, heads: list[list[int]]) -> str:
    block = "timestamp_heads:\n" + "".join(
        f"- layer: {layer}\n  head: {head}\n" for layer, head in heads
    )
    pattern = r"(?m)^timestamp_heads:[^\n]*\n(?:(?:[ \t]+[^\n]*|- [^\n]*)\n)*"
    if re.search(pattern, text):
        return re.sub(pattern, lambda _: block, text, count=1)
    position = text.index("flow_lm:")
    return text[:position] + block + "\n" + text[position:]


def display_name(model_id: str) -> str:
    return (
        model_id.replace("_24l", " 24L")
        .replace("_", " ")
        .replace("english", "English", 1)
        .replace("french", "French", 1)
        .replace("german", "German", 1)
        .replace("spanish", "Spanish", 1)
        .replace("italian", "Italian", 1)
        .replace("portuguese", "Portuguese", 1)
        .replace("dutch", "Dutch", 1)
        .replace("drifting", "Drifting")
    )


def update_readme(text: str, updates: dict[str, dict[str, Any]]) -> str:
    start = text.index("| Checkpoint")
    end = text.find("\n\n", start)
    if end < 0:
        raise ValueError("Cannot locate checkpoint table boundary")
    old_lines = text[start:end].splitlines()
    records: dict[str, list[str]] = {}
    for line in old_lines[2:]:
        if not line.startswith("|"):
            continue
        columns = [c.strip() for c in line.strip("|").split("|")]
        key = columns[0].casefold().replace(" ", "_")
        if len(columns) == 6:
            name, heads, samples, words, skip, mae = columns
            columns = [name, heads, samples, words, skip, samples, words, mae]
        records[key] = columns[:8]
    for model_id, row in updates.items():
        result = row["result"]
        candidate = row["candidate"]
        records[model_id] = [
            display_name(model_id),
            head_label(candidate["heads"]),
            str(result["skip_samples"]),
            f"{candidate['skip_words']:,}",
            f"{100 * candidate['skip_rate']:.4f}%",
            str(result["mae_samples"]),
            f"{candidate['mae_words']:,}",
            "n/a" if candidate["mae_ms"] is None else f"{candidate['mae_ms']:.2f} ms",
        ]
    header = [
        "| Checkpoint | Heads | Skip samples | Skip words | Skip rate | MAE samples | MAE words | Start/end MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    table = "\n".join(
        header + ["| " + " | ".join(columns) + " |" for _, columns in sorted(records.items())]
    )
    text = text[:start] + table + text[end:]
    replacement = "All official Pocket TTS checkpoints are supported, but accuracy differs between them. Selection results, MAE is measured against CrisperWhisper 2.0 medium"
    introduction = text.rfind("All official Pocket TTS checkpoints", 0, start)
    if introduction < 0:
        introduction = text.rfind("Evaluation results,", 0, start)
    if introduction < 0:
        raise ValueError("Cannot locate checkpoint table introduction")
    return text[:introduction] + replacement + "\n" + text[start:]


def command(*arguments: str) -> None:
    subprocess.run(list(arguments), check=True)


def publish_run(args: argparse.Namespace) -> None:
    work = Path(args.work)
    plan = read(work / "plan.json")
    transport = Artifacts()
    records = [record["result"] for record in plan["cached"]]
    pending = []
    for item in plan["active"]:
        destination = work / "finished" / item["key"]
        name = f"timestamp-v{PROTOCOL}-{item['key']}"
        smoke = f"timestamp-smoke-v{PROTOCOL}-{item['key']}"
        if transport.named(plan["run_id"], name, destination, False) or transport.named(
            plan["run_id"], smoke, destination, False
        ):
            records.append(read(destination / "result.json"))
        else:
            state = work / "unfinished" / item["key"]
            if (
                transport.named(plan["run_id"], f"timestamp-reference-{item['key']}", state, False)
                and (state / "status.json").exists()
                and not read(state / "status.json")["complete"]
            ):
                pending.append(item)
            else:
                raise RuntimeError(
                    f"Evaluation failed for {item['id']}; no configuration will be published"
                )
    if pending:
        # An explicit workflow_dispatch can start a continuation with GITHUB_TOKEN.
        command(
            "gh",
            "workflow",
            "run",
            "timestamp-heads.yml",
            "--repo",
            os.environ["GITHUB_REPOSITORY"],
            "--ref",
            os.environ["GITHUB_REF_NAME"],
            "-f",
            f"resume_run={plan['run_id']}",
            "-f",
            f"skip_penalty={plan['scoring']['skip_penalty']}",
            "-f",
            f"head_penalty={plan['scoring']['head_penalty']}",
        )
        print("Continuation dispatched for " + ", ".join(i["id"] for i in pending), flush=True)
        return
    if not records:
        print("No checkpoint changes", flush=True)
        return
    scoring = {record["checkpoint_sha256"]: rank(record, **plan["scoring"]) for record in records}
    root = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{plan['run_id']}"
    lines = [
        "Checkpoint head selection using pinned CrisperWhisper Medium, CT2 float32.",
        "",
        f"[Evaluation run and downloadable result artifacts]({root}). Results are retained for 90 days; large temporary captures are deleted after completion (failure artifacts expire after seven days). No evaluation data is committed to the repository.",
        "",
        f"Reference revision: `{plan['reference']['revision']}`.",
        "",
        f"Score: MAE_ms + {plan['scoring']['skip_penalty']} × skip_percentage + {plan['scoring']['head_penalty']} × head_count. No skip-rate eligibility cutoff.",
        "",
        "Individual heads were ranked by raw MAE without a skip-rate eligibility cutoff. The search tests all groups of 1–5 from the top ten, all individual heads, and existing configurations. Voices, length bands, and seeds were frozen before transcription; MAE uses the first matching clips in that order.",
        "",
        "MAE is measured on the selection cohort, without separately collected validation. Zero observed skips do not establish a zero population rate.",
        "",
        "Prompts are sourced from [Tatoeba](https://tatoeba.org/); sentence IDs and frozen sample metadata are retained in the plan artifact.",
    ]
    for record in records:
        lines += ["", report(record, scoring[record["checkpoint_sha256"]])]
    work.mkdir(parents=True, exist_ok=True)
    body = work / "pr-body.md"
    body.write_text("\n".join(lines), encoding="utf-8")
    write(
        work / "rescored.json",
        [{"result": r, "scoring": scoring[r["checkpoint_sha256"]]} for r in records],
    )
    if not plan["publish"]:
        print("Reduced-sample/test run complete; publishing disabled", flush=True)
        return
    if any(r["skip_samples"] != 1500 or r["mae_samples"] != 500 for r in records):
        raise ValueError("Cannot publish reduced-sample results")
    command("git", "fetch", "origin", "main")
    # A later checkpoint change must not replace an older, unmerged results PR.
    # Rescoring/continuations retain the capture run ID and update their own PR.
    branch = f"codex/timestamp-head-results-{plan['capture_run_id']}"
    command("git", "switch", "-C", branch, "origin/main")
    updates = {}
    stale = []
    for record in records:
        winner = scoring[record["checkpoint_sha256"]]["winner"]
        rows = {tuple(sorted(map(tuple, r["heads"]))): r for r in record["rows"]}
        for alias in record["aliases"]:
            path = Path(alias["config_path"])
            if not path.exists():
                stale.append(alias["id"])
                continue
            text = path.read_text()
            current = yaml.safe_load(text)
            baseline = [[h["layer"], h["head"]] for h in (current.get("timestamp_heads") or [])]
            if list(config_identity(current)) != alias["identity"] or baseline != alias["baseline"]:
                stale.append(alias["id"])
                continue
            if winner is not None:
                path.write_text(update_config(text, winner["heads"]), encoding="utf-8")
                updates[alias["id"]] = {"result": record, "candidate": winner}
            elif tuple(sorted(map(tuple, baseline))) in rows:
                updates[alias["id"]] = {
                    "result": record,
                    "candidate": rows[tuple(sorted(map(tuple, baseline)))],
                }
    if stale:
        body.write_text(
            body.read_text()
            + "\nConfigs changed or removed while evaluating; preserved for manual review: "
            + ", ".join(stale)
            + ".\n"
        )
    readme = Path("README.md")
    if updates:
        readme.write_text(update_readme(readme.read_text(), updates), encoding="utf-8")
    changed = git("diff", "--name-only")
    if not changed:
        print("No publishable configuration or table change", flush=True)
        return
    allowed = {
        "README.md",
        *(alias["config_path"] for record in records for alias in record["aliases"]),
    }
    if set(changed.splitlines()) - allowed:
        raise RuntimeError("Unexpected files in evaluation publication")
    command("git", "config", "user.name", "github-actions[bot]")
    command("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    command("git", "add", "--", *changed.splitlines())
    command("git", "commit", "-m", "Update checkpoint timestamp heads and evaluation table")
    # A human-edited results PR is preserved rather than force-overwritten.
    existing = subprocess.check_output(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            os.environ["GITHUB_REPOSITORY"],
            "--state",
            "open",
            "--head",
            branch,
            "--json",
            "number",
            "--jq",
            ".[0].number // empty",
        ],
        text=True,
    ).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin", branch], text=True
    ).strip()
    if remote:
        command("git", "fetch", "origin", f"{branch}:refs/remotes/origin/{branch}")
    if existing:
        author = git("log", "-1", "--format=%an", f"origin/{branch}")
        if author != "github-actions[bot]":
            raise RuntimeError("Existing results PR has human edits; preserved for manual review")
    command("git", "push", "--force-with-lease", "origin", f"HEAD:{branch}")
    if existing:
        command(
            "gh",
            "pr",
            "edit",
            existing,
            "--repo",
            os.environ["GITHUB_REPOSITORY"],
            "--title",
            "Update checkpoint timestamp heads",
            "--body-file",
            str(body.resolve()),
        )
    else:
        command(
            "gh",
            "pr",
            "create",
            "--repo",
            os.environ["GITHUB_REPOSITORY"],
            "--base",
            "main",
            "--head",
            branch,
            "--title",
            "Update checkpoint timestamp heads",
            "--body-file",
            str(body.resolve()),
        )
