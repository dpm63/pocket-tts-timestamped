"""Offline regression tests for the pinned, patched dependency materializer."""

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "timestamp_prepare", Path(__file__).with_name("prepare.py")
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def run(self, result: unittest.TestResult | None = None) -> unittest.TestResult | None:
        result = result if result is not None else unittest.TestResult()
        before = len(result.failures) + len(result.errors)
        started = time.monotonic()
        print(
            f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} START {self.id()} timeoutSeconds=180-per-Git-command",
            flush=True,
        )
        try:
            return super().run(result)
        finally:
            status = "FAIL" if len(result.failures) + len(result.errors) > before else "FINISH"
            print(
                f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} {status} {self.id()} durationMs={int((time.monotonic() - started) * 1000)}",
                flush=True,
            )

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="timestamp-prepare-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "cache"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "user.name", "Fixture")
        self.git("config", "core.autocrlf", "false")
        file = self.repo / "ptts/src/lib.rs"
        file.parent.mkdir(parents=True)
        file.write_text("unchanged\n" * 20 + "pub fn old() {}\n" + "tail\n" * 20, newline="\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture base")
        base = self.git("rev-parse", "HEAD").strip()
        file.write_text(file.read_text().replace("pub fn old()", "pub fn captured()"), newline="\n")
        patchfile = self.root / "feature.patch"
        patchfile.write_text(self.git("diff", "--binary"), newline="\n")
        digest = hashlib.sha256(patchfile.read_bytes()).hexdigest()
        patchfile.with_suffix(".json").write_text(
            json.dumps(
                {
                    "baseCommit": base,
                    "patchSha256": digest,
                    "files": {"ptts/src/lib.rs": hashlib.sha256(file.read_bytes()).hexdigest()},
                }
            )
        )
        self.git("checkout", "--", "ptts/src/lib.rs")
        self.feature = patchfile
        mocked = patch.object(module, "BASE", base)
        mocked.start()
        self.addCleanup(mocked.stop)

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, stderr=subprocess.STDOUT, timeout=30
        ).decode()

    def test_clean_cache_is_repeatable_and_outside_hunk_edit_is_rejected(self):
        digest = module.prepare(self.repo, self.feature)
        self.assertEqual(module.prepare(self.repo, self.feature), digest)
        file = self.repo / "ptts/src/lib.rs"
        file.write_text(file.read_text() + "\n// extra edit\n")
        with self.assertRaisesRegex(RuntimeError, "content differs"):
            module.prepare(self.repo, self.feature)

    def test_global_windows_line_endings_do_not_change_pinned_bytes(self):
        # Simulate a clean clone before its first checkout under autocrlf=true.
        destination = self.root / "windows-cache"
        base = self.git("rev-parse", "HEAD").strip()
        subprocess.check_output(
            ["git", "clone", "--no-checkout", str(self.repo), str(destination)],
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        subprocess.check_output(
            ["git", "config", "core.autocrlf", "true"],
            cwd=destination,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        module.run(["git", "checkout", "--detach", base], destination)
        self.assertNotIn(b"\r\n", (destination / "ptts/src/lib.rs").read_bytes())
        digest = module.prepare(destination, self.feature)
        self.assertEqual(module.prepare(destination, self.feature), digest)
        self.assertEqual(
            subprocess.check_output(
                ["git", "config", "--local", "core.autocrlf"],
                cwd=destination,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            .decode()
            .strip(),
            "true",
        )

    def test_untracked_build_script_is_rejected(self):
        module.prepare(self.repo, self.feature)
        # Even an ignored Cargo auto-build script must invalidate the source cache.
        self.git("config", "core.excludesFile", str(self.root / "global-ignore"))
        (self.root / "global-ignore").write_text("build.rs\n")
        (self.repo / "ptts/build.rs").write_text("fn main(){}")
        with self.assertRaisesRegex(RuntimeError, "Unexpected tracked or untracked"):
            module.prepare(self.repo, self.feature)

    def test_command_failure_and_timeout_preserve_original_output_and_cause(self):
        failure = subprocess.CalledProcessError(
            9, ["git", "apply"], output=b"original Git diagnostic"
        )
        with patch.object(module.subprocess, "check_output", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "original Git diagnostic") as captured:
                module.run(["git", "apply"])
            self.assertIs(captured.exception.__cause__, failure)
        failure = subprocess.TimeoutExpired(
            ["git", "fetch"], 180, output=b"partial timeout diagnostic"
        )
        with patch.object(module.subprocess, "check_output", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "partial timeout diagnostic") as captured:
                module.run(["git", "fetch"])
            self.assertIs(captured.exception.__cause__, failure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
