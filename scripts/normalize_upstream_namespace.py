"""Keep upstream changes in the fork's collision-free public namespace.

The upstream-sync workflow runs this after merging upstream ``main``.  The
transformation is intentionally idempotent so it can also be run locally while
resolving an upstream merge.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORMALIZER_PATH = Path(__file__).resolve()
UPSTREAM_PACKAGE = "pocket_tts"
FORK_PACKAGE = "pocket_tts_timestamped"
UPSTREAM_CLI = "pocket-tts"
FORK_CLI = "pocket-tts-timestamped"

TEXT_SUFFIXES = {".ipynb", ".md", ".py", ".pyi", ".toml", ".yaml", ".yml"}
TEXT_ROOTS = (
    ROOT / FORK_PACKAGE,
    ROOT / "docs",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "training",
)
TEXT_FILES = (
    ROOT / "AGENTS.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "Dockerfile",
    ROOT / "README.md",
    ROOT / "mkdocs.yml",
    ROOT / "pyproject.toml",
)


def _move_upstream_package_files() -> None:
    old_root = ROOT / UPSTREAM_PACKAGE
    new_root = ROOT / FORK_PACKAGE
    if not old_root.exists():
        return

    for source in sorted(path for path in old_root.rglob("*") if path.is_file()):
        relative_path = source.relative_to(old_root)
        destination = new_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if source.read_bytes() != destination.read_bytes():
                raise RuntimeError(
                    f"Cannot normalize {source}: {destination} already exists with different content"
                )
            source.unlink()
        else:
            source.rename(destination)

    for directory in sorted(
        (path for path in old_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.rmdir()
    old_root.rmdir()


def _iter_text_files() -> list[Path]:
    files = [path for path in TEXT_FILES if path.is_file()]
    for root in TEXT_ROOTS:
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix in TEXT_SUFFIXES
                and path.resolve() != NORMALIZER_PATH
            )
    return sorted(set(files))


def _normalize_text(text: str) -> str:
    text = re.sub(rf"\b{UPSTREAM_PACKAGE}\b", FORK_PACKAGE, text)
    text = re.sub(
        rf"\b{UPSTREAM_CLI}\b(?=\s+(?:generate|serve|export-voice|--help)\b)", FORK_CLI, text
    )
    text = text.replace(
        f'{UPSTREAM_CLI} = "{FORK_PACKAGE}.main:cli_app"',
        f'{FORK_CLI} = "{FORK_PACKAGE}.main:cli_app"',
    )
    text = text.replace(
        f'ENTRYPOINT ["uv", "run", "{UPSTREAM_CLI}"]', f'ENTRYPOINT ["uv", "run", "{FORK_CLI}"]'
    )

    # These resources intentionally remain shared with upstream.
    text = re.sub(
        r"(raw\.githubusercontent\.com/kyutai-labs/pocket-tts/[^\s\"')]+)"
        rf"/{FORK_PACKAGE}/",
        rf"\1/{UPSTREAM_PACKAGE}/",
        text,
    )
    text = text.replace(
        f'Path.home() / ".cache" / "{FORK_PACKAGE}"',
        f'Path.home() / ".cache" / "{UPSTREAM_PACKAGE}"',
    )
    return text


def main() -> None:
    _move_upstream_package_files()
    changed_files: list[Path] = []
    for path in _iter_text_files():
        text = path.read_text(encoding="utf-8")
        normalized = _normalize_text(text)
        if normalized != text:
            path.write_text(normalized, encoding="utf-8", newline="")
            changed_files.append(path.relative_to(ROOT))

    if changed_files:
        print("Normalized upstream namespace in:")
        for path in changed_files:
            print(f"  {path}")
    else:
        print("Upstream namespace is already normalized.")


if __name__ == "__main__":
    main()
