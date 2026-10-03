"""Actions artifact transport, including reuse across workflow runs."""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import requests


class Artifacts:
    def __init__(self, repository: str | None = None, token: str | None = None):
        self.repository = repository or os.environ["GITHUB_REPOSITORY"]
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": "Bearer " + (token or os.environ["GH_TOKEN"]),
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        self.base = f"https://api.github.com/repos/{self.repository}"

    def get(self, route: str) -> dict[str, Any]:
        for attempt in range(4):
            response = self.session.get(self.base + route, timeout=120)
            if response.status_code < 500:
                response.raise_for_status()
                return response.json()
            time.sleep(2**attempt)
        response.raise_for_status()
        raise RuntimeError("Unreachable")

    def listing(self, run_id: int | None = None) -> list[dict[str, Any]]:
        route = f"/actions/runs/{run_id}/artifacts" if run_id else "/actions/artifacts"
        records = []
        for page in range(1, 101):
            payload = self.get(f"{route}?per_page=100&page={page}")
            batch = payload["artifacts"]
            records.extend(a for a in batch if not a["expired"])
            if len(batch) < 100:
                break
        return records

    def download(self, artifact: dict[str, Any], destination: Path) -> None:
        response = self.session.get(artifact["archive_download_url"], timeout=300)
        response.raise_for_status()
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for member in archive.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Artifact path escapes output directory")
            archive.extractall(root)

    def named(self, run_id: int, name: str, destination: Path, required: bool = True) -> bool:
        artifact = next((a for a in self.listing(run_id) if a["name"] == name), None)
        if artifact is None:
            if required:
                raise FileNotFoundError(f"Missing artifact {name} in run {run_id}")
            return False
        self.download(artifact, destination)
        return True

    def delete_caches(self, keys: list[str]) -> None:
        from urllib.parse import urlencode

        for key in keys:
            if not key.startswith("timestamp-"):
                raise ValueError("Refusing to delete an unrelated cache")
            query = urlencode({"key": key, "ref": os.environ["GITHUB_REF"]})
            response = self.session.delete(self.base + "/actions/caches?" + query, timeout=90)
            if response.status_code not in (200, 204, 404):
                print(f"Cache cleanup deferred: {key}, HTTP {response.status_code}", flush=True)
