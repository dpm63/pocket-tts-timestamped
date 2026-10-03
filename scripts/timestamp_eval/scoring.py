"""Penalty-only rescoring and human-readable candidate reports."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.timestamp_eval.common import head_label


def summarize(
    heads: list[list[int]], skip: NDArray[np.float64], mae: NDArray[np.float64]
) -> dict[str, Any]:
    words, predicted = int(skip[0]), int(skip[1])
    mae_words, mae_predicted = int(mae[0]), int(mae[1])
    return {
        "heads": heads,
        "skip_words": words,
        "skipped_words": words - predicted,
        "skip_rate": (words - predicted) / words if words else 1.0,
        "mae_words": mae_words,
        "mae_predicted_words": mae_predicted,
        "start_error_sum_ms": float(mae[2]),
        "end_error_sum_ms": float(mae[3]),
        "start_mae_ms": float(mae[2] / mae_predicted) if mae_predicted else None,
        "end_mae_ms": float(mae[3] / mae_predicted) if mae_predicted else None,
        "mae_ms": float((mae[2] + mae[3]) / (2 * mae_predicted)) if mae_predicted else None,
    }


def mae_key(row: dict[str, Any]) -> tuple[float, int, str]:
    return (
        row["mae_ms"] if row["mae_ms"] is not None else math.inf,
        len(row["heads"]),
        head_label(row["heads"]),
    )


def candidates(
    single_rows: list[dict[str, Any]], baselines: list[list[list[int]]]
) -> list[list[list[int]]]:
    # No skip gate on partner heads: poor individual completeness can improve a mixture.
    top = [row["heads"][0] for row in sorted(single_rows, key=mae_key)[:10]]
    groups = [
        list(group)
        for size in range(1, min(5, len(top)) + 1)
        for group in itertools.combinations(top, size)
    ]
    signatures = {tuple(sorted(map(tuple, group))) for group in groups}
    for baseline in baselines:
        key = tuple(sorted(map(tuple, baseline)))
        if baseline and key not in signatures:
            groups.append(baseline)
            signatures.add(key)
    return groups


def rank(
    result: dict[str, Any], skip_penalty: float = 10.0, head_penalty: float = 0.5
) -> dict[str, Any]:
    if min(skip_penalty, head_penalty) < 0:
        raise ValueError("Penalties must be non-negative")
    rows = []
    for original in result["rows"]:
        row = {**original}
        row["score_ms"] = (
            None
            if row["mae_ms"] is None
            else row["mae_ms"]
            + skip_penalty * 100 * row["skip_rate"]
            + head_penalty * len(row["heads"])
        )
        row["eligible"] = row["mae_ms"] is not None
        rows.append(row)
    best_mae = sorted(rows, key=mae_key)
    eligible = sorted(
        (r for r in rows if r["eligible"]), key=lambda r: (r["score_ms"], *mae_key(r))
    )
    zero = sorted((r for r in rows if r["skipped_words"] == 0), key=mae_key)
    return {
        "winner": eligible[0] if eligible else None,
        "best_mae": best_mae[:5],
        "best_score": eligible[:5],
        "best_zero_skip": zero[:5],
        "parameters": {
            "skip_penalty_ms_per_percentage_point": skip_penalty,
            "head_penalty_ms": head_penalty,
        },
        "scored_candidates": len(eligible),
    }


def table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Heads | MAE | Skipped/total words | Skip rate | Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        mae = "n/a" if row["mae_ms"] is None else f"{row['mae_ms']:.2f} ms"
        score = "n/a" if row["score_ms"] is None else f"{row['score_ms']:.2f} ms"
        lines.append(
            f"| {head_label(row['heads'])} | {mae} | {row['skipped_words']}/{row['skip_words']} | {100 * row['skip_rate']:.4f}% | {score} |"
        )
    return lines


def report(result: dict[str, Any], scoring: dict[str, Any]) -> str:
    lines = [
        f"## {result['model_id']}",
        "",
        f"Checkpoint SHA256: `{result['checkpoint_sha256']}`.",
        "",
        f"Skip cohort: **{result['skip_samples']} samples**; MAE cohort: **{result['mae_samples']} strict matches** from {result['transcription_attempts']} attempts.",
        "",
        f"Aliases: {', '.join(a['id'] for a in result['aliases'])}.",
        "",
        f"{scoring['scored_candidates']} candidates have a measured MAE and are scored without a skip-rate cutoff.",
    ]
    if scoring["winner"] is None:
        lines += [
            "",
            "**No candidate has a measured MAE. Preserve the configured heads; manual review required.**",
        ]
    else:
        lines += ["", f"Selected: **{head_label(scoring['winner']['heads'])}**."]
    lookup = {tuple(sorted(map(tuple, row["heads"]))): row for row in result["rows"]}
    lines += [
        "",
        "### Current configured heads",
        "",
        "| Config | Heads | MAE | Skip rate |",
        "|---|---|---:|---:|",
    ]
    for alias in result["aliases"]:
        row = lookup.get(tuple(sorted(map(tuple, alias["baseline"]))))
        label = head_label(alias["baseline"]) or "not configured"
        mae = "not available" if row is None or row["mae_ms"] is None else f"{row['mae_ms']:.2f} ms"
        skip = "not available" if row is None else f"{100 * row['skip_rate']:.4f}%"
        lines.append(f"| {alias['id']} | {label} | {mae} | {skip} |")
    for title, key in [
        ("Five best by MAE", "best_mae"),
        ("Five best by score", "best_score"),
        ("Five best with zero skips, by MAE", "best_zero_skip"),
    ]:
        lines += ["", f"### {title}", "", *table(scoring[key])]
    return "\n".join(lines) + "\n"
