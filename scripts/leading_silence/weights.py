"""Which weights a vectors file was trained on.

Vectors only fit the weights they were trained on, and a config's weights can change under the
same name (upstream replaces checkpoints in place and moves aliases such as `english` to new
models). So every vectors file records the SHA-256 of its weights, and the plan retrains a
config whose file doesn't match.
"""

from __future__ import annotations

import re
from pathlib import Path

from safetensors import safe_open

METADATA_KEY = "weights_sha256"


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


def trained_on(path: Path) -> str | None:
    """The weights SHA-256 recorded in a vectors file; None if it records none."""
    with safe_open(str(path), framework="pt") as stream:
        return (stream.metadata() or {}).get(METADATA_KEY)
