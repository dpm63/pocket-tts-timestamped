#!/usr/bin/env python3
"""Materialize the pinned XN dependency and selected-head patch, without changing app models."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = "7efa26973f81bba42cf2ace8da2c1eb9b605d54d"
START = time.monotonic()


def log(message: str) -> None:
    print(
        f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} {message} elapsedMs={int((time.monotonic() - START) * 1000)}",
        flush=True,
    )


def run(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            args, cwd=cwd, stderr=subprocess.STDOUT, timeout=180
        ).decode()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        output = error.output or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        raise RuntimeError(
            f"{type(error).__name__}: {args[0]} failed: {error}\n{output}"
        ) from error


def prepare(destination: Path, patch: Path) -> str:
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    manifest = json.loads(patch.with_suffix(".json").read_text())
    if manifest["baseCommit"] != BASE or manifest["patchSha256"] != digest:
        raise RuntimeError("Dependency manifest/patch identity mismatch")
    stamp = destination / ".pocket-timestamp-patch"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        log("PHASE fetching pinned xn-ptts timeoutSeconds=180")
        run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/gabrieldemarmiesse/xn-ptts.git",
                str(destination),
            ]
        )
        run(["git", "checkout", "--detach", BASE], destination)
    if run(["git", "rev-parse", "HEAD"], destination).strip() != BASE:
        raise RuntimeError("Dependency HEAD changed; refusing to overwrite it")
    if not stamp.exists():
        if run(["git", "status", "--porcelain"], destination).strip():
            raise RuntimeError("Dependency has local changes; refusing to overwrite them")
        run(["git", "apply", "--check", str(patch)], destination)
        run(["git", "apply", str(patch)], destination)
        stamp.write_text(digest)
    if stamp.read_text() != digest:
        raise RuntimeError("Cached dependency patch differs; use a fresh destination")
    run(["git", "apply", "--reverse", "--check", str(patch)], destination)
    for name, expected in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("Invalid dependency manifest path")
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Cached dependency content differs: " + name)
    allowed = set(manifest["files"]) | {".pocket-timestamp-patch"}
    changed = run(
        ["git", "status", "--porcelain", "--untracked-files=all"], destination
    ).splitlines()
    if any(line[3:] not in allowed for line in changed):
        raise RuntimeError("Unexpected tracked or untracked edits in cached dependency")
    # Include ignored files too: a global Git ignore must not hide an auto-detected build.rs.
    untracked = run(["git", "ls-files", "--others"], destination).splitlines()
    if any(name not in allowed for name in untracked):
        raise RuntimeError("Unexpected tracked or untracked edits in cached dependency")
    return digest


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=root / ".deps/xn-ptts")
    parser.add_argument("--patch", type=Path, default=root / "patches/xn-ptts-timestamps.patch")
    args = parser.parse_args()
    log("START pinned timestamp dependency timeoutSeconds=180 per phase")
    try:
        digest = prepare(args.destination.resolve(), args.patch.resolve())
        log("FINISH dependency verified patch=" + digest)
    except Exception:
        log("FAIL timestamp dependency")
        raise
