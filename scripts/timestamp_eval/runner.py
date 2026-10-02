"""Bounded Actions stages: plan, capture, reference, groups, collect."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.timestamp_eval.alignment import metrics
from scripts.timestamp_eval.artifacts import Artifacts
from scripts.timestamp_eval.common import (
    PROTOCOL,
    REFERENCE,
    config_identity,
    discover,
    git,
    read,
    resolve_checkpoint,
    strict_match,
    write,
)
from scripts.timestamp_eval.sampling import extend
from scripts.timestamp_eval.scoring import candidates, mae_key, rank, report, summarize


def output(name: str, value: str) -> None:
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            stream.write(f"{name}={value}\n")


def item_for(plan: dict[str, Any], key: str) -> dict[str, Any]:
    return next(item for item in plan["active"] if item["key"] == key)


def plan_run(args: argparse.Namespace) -> None:
    import json

    work = Path(args.work)
    artifacts = Artifacts()
    plan: dict[str, Any]
    if args.resume_run:
        artifacts.named(args.resume_run, "timestamp-plan", work)
        plan = read(work / "plan.json")
        plan["previous_run_id"] = args.resume_run
        plan["run_id"] = int(os.environ["GITHUB_RUN_ID"])
        plan.setdefault("history_run_ids", [args.resume_run]).append(plan["run_id"])
        completed = []
        for item in plan["active"]:
            target = work / "resume" / item["key"]
            if artifacts.named(
                args.resume_run, f"timestamp-v{PROTOCOL}-{item['key']}", target, False
            ) or artifacts.named(
                args.resume_run, f"timestamp-smoke-v{PROTOCOL}-{item['key']}", target, False
            ):
                completed.append({"item": item, "result": read(target / "result.json")})
        keys = {c["item"]["key"] for c in completed}
        plan["active"] = [item for item in plan["active"] if item["key"] not in keys]
        plan["cached"].extend(completed)
    elif args.rescore_run:
        artifacts.named(args.rescore_run, "timestamp-plan", work)
        plan = read(work / "plan.json")
        cached = []
        for item in plan["active"]:
            target = work / "rescore" / item["key"]
            if not artifacts.named(
                args.rescore_run, f"timestamp-v{PROTOCOL}-{item['key']}", target, False
            ):
                artifacts.named(
                    args.rescore_run, f"timestamp-smoke-v{PROTOCOL}-{item['key']}", target
                )
            cached.append({"item": item, "result": read(target / "result.json")})
        plan["cached"].extend(cached)
        plan["active"] = []
        plan["run_id"] = int(os.environ["GITHUB_RUN_ID"])
        plan["mode"] = "rescore"
        plan["publish"] = args.publish
    else:
        if min(args.skip_samples, args.mae_samples, args.shards) < 1:
            raise ValueError("Cohort sizes and shard count must be positive")
        if args.publish and (args.skip_samples != 1500 or args.mae_samples != 500):
            raise ValueError("Reduced-sample test runs cannot publish configurations")
        all_items = discover(Path("pocket_tts_timestamped/config"))
        selected = set(args.models.split(",")) if args.models else set()
        if selected and selected - {item["id"] for item in all_items}:
            raise ValueError("Unknown checkpoint selector")
        before = args.before
        wanted = []
        for item in all_items:
            if selected and item["id"] not in selected:
                continue
            if args.force or selected or not before or set(before) == {"0"}:
                wanted.append(item)
                continue
            try:
                import yaml

                previous = yaml.safe_load(git("show", f"{before}:{item['config_path']}"))
            except Exception:
                previous = {}
            if config_identity(previous) != config_identity(item["config"]):
                wanted.append(item)
        available = artifacts.listing() if wanted and not args.force else []
        grouped: dict[str, dict[str, Any]] = {}
        resolved_by_source: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
        for item in wanted:
            identity = config_identity(item["config"])
            if identity in resolved_by_source:
                resolved = {
                    **item,
                    **{
                        k: resolved_by_source[identity][k]
                        for k in ("checkpoint_sha256", "effective_weights")
                    },
                }
            else:
                resolved = resolve_checkpoint(item)
                resolved_by_source[identity] = resolved
            digest = resolved["checkpoint_sha256"]
            alias = {
                "id": item["id"],
                "config_path": item["config_path"],
                "baseline": item["baseline"],
                "identity": list(identity),
            }
            if digest not in grouped:
                grouped[digest] = {**resolved, "key": digest[:16], "aliases": [], "baselines": []}
            grouped[digest]["aliases"].append(alias)
            if item["baseline"] and item["baseline"] not in grouped[digest]["baselines"]:
                grouped[digest]["baselines"].append(item["baseline"])
        active = []
        cached = []
        for item in grouped.values():
            reusable = next(
                (a for a in available if a["name"] == f"timestamp-v{PROTOCOL}-{item['key']}"), None
            )
            if reusable:
                target = work / "cache" / item["key"]
                artifacts.download(reusable, target)
                result = read(target / "result.json")
                if (
                    result["checkpoint_sha256"] == item["checkpoint_sha256"]
                    and result["skip_samples"] >= args.skip_samples
                    and result["mae_samples"] >= args.mae_samples
                    and result["reference"] == REFERENCE
                ):
                    result["aliases"] = item["aliases"]
                    cached.append({"item": item, "result": result})
                    continue
            from huggingface_hub import HfApi

            from pocket_tts_timestamped.utils.utils import get_predefined_voice

            voice_source = get_predefined_voice(item["id"], item["voices"][0])
            revision = voice_source.rsplit("@", 1)[1]
            catalog = HfApi().list_repo_files(
                "kyutai/pocket-tts-without-voice-cloning", revision=revision
            )
            prefix = f"languages/{item['id']}/embeddings/"
            item["voices"] = [
                voice for voice in item["voices"] if prefix + voice + ".safetensors" in catalog
            ]
            if not item["voices"]:
                raise ValueError(f"No predefined evaluation voices available for {item['id']}")
            item["samples"] = extend(
                item, work / "prompts" / (item["key"] + ".json"), args.skip_samples
            )
            active.append(item)
        plan = {
            "protocol": PROTOCOL,
            "reference": REFERENCE,
            "base_sha": git("rev-parse", "origin/main"),
            "run_id": int(os.environ["GITHUB_RUN_ID"]),
            "capture_run_id": int(os.environ["GITHUB_RUN_ID"]),
            "previous_run_id": None,
            "history_run_ids": [int(os.environ["GITHUB_RUN_ID"])],
            "skip_samples": args.skip_samples,
            "mae_samples": args.mae_samples,
            "shards": args.shards,
            "publish": args.publish,
            "mode": "evaluate",
            "active": active,
            "cached": cached,
        }
    plan["scoring"] = {
        "skip_penalty": args.skip_penalty,
        "head_penalty": args.head_penalty,
        "skip_limit": args.skip_limit,
    }
    write(work / "plan.json", plan)
    output(
        "matrix",
        json.dumps(
            {"include": [{"key": item["key"], "model": item["id"]} for item in plan["active"]]}
        ),
    )
    output("active", str(bool(plan["active"])).lower())
    output("resume", str(bool(args.resume_run)).lower())
    output("changed", str(bool(plan["active"] or plan["cached"])).lower())
    print(
        f"Plan: {len(plan['active'])} new checkpoints, {len(plan['cached'])} cached; {plan['skip_samples']} skip samples, {plan['mae_samples']} MAE matches",
        flush=True,
    )


def capture_run(args: argparse.Namespace) -> None:
    from scripts.timestamp_eval.capture import Generator

    work = Path(args.work)
    plan = read(work / "plan.json")
    item = item_for(plan, args.key)
    shard = args.shard
    raw = work / "units"
    audio = work / "audio"
    # A rerun of a failed job can recover its previously uploaded completion markers.
    transport = Artifacts()
    transport.named(plan["capture_run_id"], f"timestamp-units-{args.key}-{shard}", raw, False)
    transport.named(plan["capture_run_id"], f"timestamp-audio-{args.key}-{shard}", audio, False)
    generator = None
    for sample in item["samples"]:
        if sample["index"] % plan["shards"] != shard:
            continue
        directory = raw / "samples" / f"{sample['index']:06d}"
        target = audio / "samples" / f"{sample['index']:06d}"
        if (directory / "sample.json").exists() and (target / "audio.wav").exists():
            continue
        generator = generator or Generator(item)
        generator.generate(sample, directory)
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("audio.wav", "sample.json", "singles.npz"):
            shutil.copy2(directory / filename, target / filename)
        (directory / "audio.wav").unlink()
        print(f"{item['id']} capture {sample['index'] + 1}/{plan['skip_samples']}", flush=True)
    write(
        raw / "complete.json",
        {
            "shard": shard,
            "indices": [
                s["index"] for s in item["samples"] if s["index"] % plan["shards"] == shard
            ],
        },
    )


def reference_run(args: argparse.Namespace) -> None:
    import importlib.metadata
    from importlib import import_module

    CrisperWhisperModel = import_module("crisperwhisper").CrisperWhisperModel
    from huggingface_hub import snapshot_download

    from scripts.timestamp_eval.capture import Generator

    work = Path(args.work)
    plan = read(work / "plan.json")
    item = item_for(plan, args.key)
    artifacts = Artifacts()
    data = work / "audio"
    # Audio caches are restored by the workflow; failure artifacts support job retries.
    if not all(
        (data / "samples" / f"{i:06d}" / "sample.json").exists()
        for i in range(plan["skip_samples"])
    ):
        for shard in range(plan["shards"]):
            artifacts.named(
                plan["capture_run_id"], f"timestamp-audio-{args.key}-{shard}", data, False
            )
    if not all(
        (data / "samples" / f"{i:06d}" / "sample.json").exists()
        for i in range(plan["skip_samples"])
    ):
        raise RuntimeError("Missing transient audio cache; rerun capture jobs before reference")
    result_dir = work / "reference"
    extras = work / "extras"
    source_run = plan["previous_run_id"] or plan["run_id"]
    artifacts.named(
        source_run, f"timestamp-reference-{args.key}", result_dir, bool(plan["previous_run_id"])
    )
    artifacts.named(source_run, f"timestamp-extras-{args.key}", extras, False)
    cache = read(result_dir / "reference.json") if (result_dir / "reference.json").exists() else {}
    accepted = sorted(int(key) for key, record in cache.items() if record["usable"])
    # Checkpointed before job time limits; continuation resumes failed and accepted clips alike.
    budget = time.monotonic() + args.budget_seconds
    snapshot = snapshot_download(REFERENCE["repo"], revision=REFERENCE["revision"])
    model = CrisperWhisperModel(snapshot, backend="ct2", device="cpu", compute_type="float32")
    if model._engine.model.compute_type != "float32":
        raise RuntimeError("CT2 did not retain float32 precision")
    write(
        result_dir / "runtime.json",
        {
            **REFERENCE,
            "effective_compute_type": model._engine.model.compute_type,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("crisperwhisper", "ctranslate2-crisperwhisper", "torch", "transformers")
            },
        },
    )
    generator = None
    index = 0
    extra_prompts = extras / "prompts.json"
    while len(accepted) < plan["mae_samples"] and time.monotonic() < budget:
        key = str(index)
        if key in cache:
            index += 1
            continue
        directory = (
            data / "samples" / f"{index:06d}"
            if index < plan["skip_samples"]
            else extras / "samples" / f"{index:06d}"
        )
        if not (directory / "sample.json").exists():
            # Extra collection starts only after every initial clip has been tried.
            if not extra_prompts.exists():
                write(extra_prompts, item["samples"])
            samples = extend(item, extra_prompts, index + 100)
            sample = samples[index]
            generator = generator or Generator(item)
            generator.generate(sample, directory)
        metadata = read(directory / "sample.json")
        from scripts.timestamp_eval.common import file_hash

        if metadata["audio_sha256"] != file_hash(directory / "audio.wav"):
            raise RuntimeError("Audio content differs from generation record")
        output_result = model.transcribe(
            str(directory / "audio.wav"),
            language=item["asr_language"],
            mode="intended",
            word_timestamps=True,
        )
        words = []
        for word in output_result.words or []:
            if word.start is None or word.end is None:
                words = []
                break
            words.append(
                {"word": str(word.word).strip(), "start": float(word.start), "end": float(word.end)}
            )
        usable = strict_match(words, metadata["words"])
        cache[key] = {
            "words": words,
            "text": str(output_result.text),
            "usable": usable,
            "audio_sha256": metadata["audio_sha256"],
        }
        write(result_dir / "reference.json", cache)
        if usable:
            accepted.append(index)
        index += 1
        if index % 25 == 0:
            print(
                f"{item['id']}: {len(accepted)}/{plan['mae_samples']} matches, {index} attempts",
                flush=True,
            )
    complete = len(accepted) >= plan["mae_samples"]
    selected = sorted(accepted)[: plan["mae_samples"]]
    write(
        result_dir / "status.json",
        {
            "complete": complete,
            "attempts": len(cache),
            "matches": len(accepted),
            "selected_ids": selected,
        },
    )
    output("complete", str(complete).lower())
    output("has_extras", str((extras / "samples").exists()).lower())
    if not complete:
        print("Saved reference progress for an automatic continuation", flush=True)
        return
    first = read(data / "samples/000000/sample.json")
    count = len(first["heads"])
    skips = np.zeros((count, 4), dtype=np.float64)
    maes = np.zeros_like(skips)
    skip_cases = []
    mae_cases = []
    required = sorted(set(range(plan["skip_samples"])) | set(selected))
    for index in required:
        directory = (
            data / "samples" / f"{index:06d}"
            if index < plan["skip_samples"]
            else extras / "samples" / f"{index:06d}"
        )
        metadata = read(directory / "sample.json")
        with np.load(directory / "singles.npz") as saved:
            bounds = saved["bounds"][:count]
        if index < plan["skip_samples"]:
            row = metrics(bounds, len(metadata["words"]))
            skips += row
            skip_cases.append(row)
        if index in selected:
            reference = np.asarray([[w["start"], w["end"]] for w in cache[str(index)]["words"]])
            row = metrics(bounds, len(metadata["words"]), reference)
            maes += row
            mae_cases.append(row)
    single_rows = [
        summarize([head], skip, mae)
        for head, skip, mae in zip(first["heads"], skips, maes, strict=True)
    ]
    groups = candidates(single_rows, item["baselines"])
    write(
        result_dir / "selection.json",
        {
            "groups": groups,
            "single_rows": single_rows,
            "selected_ids": selected,
            "attempts": len(cache),
            "required_ids": required,
        },
    )
    np.savez_compressed(
        result_dir / "single-metrics.npz", skip=np.stack(skip_cases), mae=np.stack(mae_cases)
    )


def groups_run(args: argparse.Namespace) -> None:
    from scripts.timestamp_eval.capture import replay

    work = Path(args.work)
    plan = read(work / "plan.json")
    selection = read(work / "reference/selection.json")
    artifacts = Artifacts()
    units = work / "units"
    extras = work / "extras"
    if not (units / "complete.json").exists():
        artifacts.named(
            plan["capture_run_id"], f"timestamp-units-{args.key}-{args.shard}", units, False
        )
    if not (units / "complete.json").exists():
        raise RuntimeError("Missing transient unit cache; rerun capture jobs before replay")
    artifacts.named(plan["run_id"], f"timestamp-extras-{args.key}", extras, False)
    cache = read(work / "reference/reference.json")
    groups = selection["groups"]
    selected = set(selection["selected_ids"])
    ids = [i for i in selection["required_ids"] if i % plan["shards"] == args.shard]
    skip = np.zeros((len(groups), 4), dtype=np.float64)
    mae = np.zeros_like(skip)
    skip_cases = []
    mae_cases = []
    skip_ids = []
    mae_ids = []
    for index in ids:
        directory = (
            units / "samples" / f"{index:06d}"
            if index < plan["skip_samples"]
            else extras / "samples" / f"{index:06d}"
        )
        metadata = read(directory / "sample.json")
        bounds = replay(directory, groups)
        if index < plan["skip_samples"]:
            row = metrics(bounds, len(metadata["words"]))
            skip += row
            skip_cases.append(row)
            skip_ids.append(index)
        if index in selected:
            reference = np.asarray([[w["start"], w["end"]] for w in cache[str(index)]["words"]])
            row = metrics(bounds, len(metadata["words"]), reference)
            mae += row
            mae_cases.append(row)
            mae_ids.append(index)
    target = work / "group-metrics"
    target.mkdir(parents=True, exist_ok=True)
    write(
        target / "totals.json",
        {"skip": skip.tolist(), "mae": mae.tolist(), "skip_ids": skip_ids, "mae_ids": mae_ids},
    )
    np.savez_compressed(
        target / "per-sample.npz",
        skip=np.stack(skip_cases) if skip_cases else np.empty((0, len(groups), 4)),
        mae=np.stack(mae_cases) if mae_cases else np.empty((0, len(groups), 4)),
        skip_ids=np.asarray(skip_ids),
        mae_ids=np.asarray(mae_ids),
    )


def collect_run(args: argparse.Namespace) -> None:
    work = Path(args.work)
    plan = read(work / "plan.json")
    item = item_for(plan, args.key)
    selection = read(work / "reference/selection.json")
    artifacts = Artifacts()
    count = len(selection["groups"])
    skips = np.zeros((count, 4))
    maes = np.zeros_like(skips)
    skip_ids = []
    mae_ids = []
    for shard in range(plan["shards"]):
        destination = work / "result" / "per-sample" / str(shard)
        artifacts.named(plan["run_id"], f"timestamp-groups-{args.key}-{shard}", destination)
        totals = read(destination / "totals.json")
        skips += np.asarray(totals["skip"])
        maes += np.asarray(totals["mae"])
        skip_ids.extend(totals["skip_ids"])
        mae_ids.extend(totals["mae_ids"])
    if (
        sorted(skip_ids) != list(range(plan["skip_samples"]))
        or sorted(mae_ids) != selection["selected_ids"]
    ):
        raise RuntimeError("Missing or duplicate sample metrics")
    rows = {tuple(sorted(map(tuple, row["heads"]))): row for row in selection["single_rows"]}
    for heads, skip, mae in zip(selection["groups"], skips, maes, strict=True):
        row = summarize(heads, skip, mae)
        key = tuple(sorted(map(tuple, heads)))
        if key in rows:
            old = rows[key]
            if (
                old["skipped_words"] != row["skipped_words"]
                or old["mae_predicted_words"] != row["mae_predicted_words"]
                or not np.isclose(
                    old["start_error_sum_ms"], row["start_error_sum_ms"], atol=1e-5, rtol=0
                )
                or not np.isclose(
                    old["end_error_sum_ms"], row["end_error_sum_ms"], atol=1e-5, rtol=0
                )
            ):
                raise RuntimeError("Replayed singleton metrics differ from capture")
        rows[key] = row
    result = {
        "protocol": PROTOCOL,
        "reference": REFERENCE,
        "model_id": item["id"],
        "checkpoint_sha256": item["checkpoint_sha256"],
        "effective_weights": item["effective_weights"],
        "aliases": item["aliases"],
        "skip_samples": plan["skip_samples"],
        "mae_samples": plan["mae_samples"],
        "transcription_attempts": selection["attempts"],
        "voices": item["voices"],
        "rows": sorted(rows.values(), key=mae_key),
        "source_run_id": plan["run_id"],
    }
    scoring = rank(result, **plan["scoring"])
    target = work / "result"
    write(target / "result.json", result)
    write(target / "scoring.json", scoring)
    (target / "report.md").write_text(report(result, scoring), encoding="utf-8")
    for filename in (
        "single-metrics.npz",
        "runtime.json",
        "reference.json",
        "selection.json",
        "status.json",
    ):
        shutil.copy2(work / "reference" / filename, target / filename)
    print(f"Scored {len(result['rows'])} candidates for {item['id']}", flush=True)
    keys = [
        f"timestamp-{kind}-{plan['capture_run_id']}-{args.key}-{shard}"
        for kind in ("audio", "units")
        for shard in range(plan["shards"])
    ]
    keys += [
        f"timestamp-extras-{run_id}-{args.key}"
        for run_id in plan.get("history_run_ids", [plan["run_id"]])
    ]
    artifacts.delete_caches(keys)
