#!/usr/bin/env python3
"""Regression gates for release authorization, provenance and archive integrity.

Uses disposable Git repositories and synthetic archive bytes. It does not call
GitHub, invoke Cargo, create a release or claim to execute a native binary.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile


def module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


package = module("package_release_tested", "package-release.py")
prepare = module("prepare_release_tested", "prepare-release.py")
gate = module("verify_release_tested", "verify-release.py")
COMMIT = "a" * 40


class GitProvenance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="forge-release-git-")
        self.root = Path(self.temporary.name)
        self.patches = [mock.patch.object(package, "ROOT", self.root), mock.patch.object(gate, "ROOT", self.root)]
        for patch in self.patches:
            patch.start()
        self.git("init", "--initial-branch=main")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "release-test@example.invalid")
        (self.root / "Cargo.toml").write_text('[package]\nname = "fixture"\nversion = "0.9.1"\n')
        self.git("add", "Cargo.toml")
        self.git("commit", "-m", "Synthetic main commit")
        self.commit = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/remotes/origin/main", self.commit)
        self.git("tag", "v0.9.1")

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True,
                                       stderr=subprocess.DEVNULL).strip()

    def test_tag_must_match_version_and_main_ancestry(self):
        self.assertEqual(package.validate_tag("v0.9.1"), ("0.9.1", self.commit))
        self.assertEqual(gate.release_commit("v0.9.1"), self.commit)
        self.git("tag", "v0.9.2")
        with self.assertRaises(ValueError):
            gate.release_commit("v0.9.2")
        with self.assertRaises(ValueError):
            package.validate_tag("v0.9.1; echo unsafe")
        self.git("checkout", "-b", "unmerged")
        (self.root / "unmerged.txt").write_text("unmerged source")
        self.git("add", "unmerged.txt")
        self.git("commit", "-m", "Unmerged feature")
        self.git("tag", "-f", "v0.9.1")
        with self.assertRaises(subprocess.CalledProcessError):
            gate.release_commit("v0.9.1")
        with self.assertRaises(subprocess.CalledProcessError):
            package.validate_tag("v0.9.1")

    def test_packaging_refuses_a_different_checkout(self):
        (self.root / "extra.txt").write_text("new main source")
        self.git("add", "extra.txt")
        self.git("commit", "-m", "Later commit")
        with self.assertRaises(ValueError):
            package.validate_tag("v0.9.1")


class GitHubProvenance(unittest.TestCase):
    def setUp(self):
        self.check = {"id": 10, "name": "required", "head_sha": COMMIT,
                      "status": "completed", "conclusion": "success", "app": {"slug": "github-actions"},
                      "details_url": f"https://github.com/{gate.REPOSITORY}/actions/runs/50/job/60"}
        self.workflow = {"head_sha": COMMIT, "head_branch": "main", "event": "push",
                         "path": ".github/workflows/ci.yml", "status": "completed", "conclusion": "success",
                         "html_url": f"https://github.com/{gate.REPOSITORY}/actions/runs/50"}
        self.pull = {"number": 1, "merged": True, "merge_commit_sha": COMMIT,
                     "base": {"ref": "main", "repo": {"full_name": gate.REPOSITORY}},
                     "merged_by": {"id": gate.OWNER_ID},
                     "html_url": f"https://github.com/{gate.REPOSITORY}/pull/1"}

    def api(self, path):
        if "/check-runs?" in path:
            return {"check_runs": [self.check]}
        if "/actions/runs/" in path:
            return self.workflow
        if "/commits/" in path and "/pulls?" in path:
            return [{"number": 1}]
        if path.endswith("/pulls/1"):
            return self.pull
        self.fail(f"Unexpected API endpoint {path}")

    def test_owner_merge_and_exact_main_ci_are_required(self):
        self.assertEqual(gate.verify_ci(COMMIT, api=self.api), self.workflow["html_url"])
        self.assertEqual(gate.verify_owner_merge(COMMIT, api=self.api), self.pull["html_url"])

    def test_failing_or_wrong_provider_checks_are_rejected(self):
        for changes in [{"status": "in_progress"}, {"conclusion": "failure"},
                        {"head_sha": "b" * 40}, {"app": {"slug": "other-app"}},
                        {"details_url": "https://github.com/other/repo/actions/runs/50"}]:
            with self.subTest(changes=changes), mock.patch.dict(self.check, changes):
                with self.assertRaises(ValueError):
                    gate.verify_ci(COMMIT, api=self.api)

    def test_different_workflow_or_pull_request_ci_is_not_release_ci(self):
        for changes in [{"event": "pull_request"}, {"head_branch": "feature"},
                        {"path": ".github/workflows/unrelated.yml"}, {"head_sha": "b" * 40},
                        {"conclusion": "failure"}]:
            with self.subTest(changes=changes), mock.patch.dict(self.workflow, changes):
                with self.assertRaises(ValueError):
                    gate.verify_ci(COMMIT, api=self.api)

    def test_old_green_check_does_not_mask_latest_failure(self):
        failed = dict(self.check, id=11, conclusion="failure")
        def api(path):
            return {"check_runs": [self.check, failed]} if "/check-runs?" in path else self.api(path)
        with self.assertRaises(ValueError):
            gate.verify_ci(COMMIT, api=api)

    def test_owner_identity_and_exact_merged_result_are_required(self):
        for changes in [{"merged_by": {"id": 123}}, {"merged": False},
                        {"merge_commit_sha": "b" * 40},
                        {"base": {"ref": "develop", "repo": {"full_name": gate.REPOSITORY}}}]:
            with self.subTest(changes=changes), mock.patch.dict(self.pull, changes):
                with self.assertRaises(ValueError):
                    gate.verify_owner_merge(COMMIT, api=self.api)


class ArchiveIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="forge-release-archive-")
        self.root = Path(self.temporary.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        self.engine_version = package.version(package.ROOT / "crates/forge-engine/Cargo.toml")
        self.stages = {}
        for target in sorted(package.TARGETS):
            windows = target.endswith("windows-msvc")
            stem = f"forge9070-v0.9.1-{target}"
            stage = self.root / stem
            stage.mkdir()
            binary = stage / ("forge-engine.exe" if windows else "forge-engine")
            binary.write_bytes(b"Synthetic archive integrity fixture; not an executable.")
            manifest = {"project_version": "0.9.1", "engine_version": self.engine_version,
                        "tag": "v0.9.1", "commit": COMMIT, "target": target,
                        "cpu_baseline": "x86-64", "binary": binary.name,
                        "rustflags": "-C target-cpu=x86-64" + (" -C target-feature=+crt-static" if windows else ""),
                        "binary_sha256": package.digest(binary), "dependency_notice_count": 1,
                        "model_weights_included": False, "target_gpu_verified": False}
            raw = json.dumps(manifest, indent=2) + "\n"
            (stage / "build-manifest.json").write_text(raw)
            (self.dist / (stem + ".build.json")).write_text(raw)
            for notice in ["LICENSE-MIT", "GGML-NOTICE.md", "THIRD-PARTY-NOTICES.md", "README.md"]:
                (stage / notice).write_text("Synthetic required notice\n")
            self.stages[target] = stage
            self.write_archive(target)

    def tearDown(self):
        self.temporary.cleanup()

    def write_archive(self, target):
        result = package.archive(self.stages[target], self.dist, target.endswith("windows-msvc"), epoch=1794955706)
        (self.dist / (result.name + ".sha256")).write_text(f"{package.digest(result)}  {result.name}\n")
        return result

    def verify(self):
        with mock.patch.object(prepare.package, "validate_tag", return_value=("0.9.1", COMMIT)), \
             mock.patch.object(sys, "argv", ["prepare-release.py", "--tag", "v0.9.1", "--directory", str(self.dist)]), \
             contextlib.redirect_stdout(io.StringIO()):
            prepare.main()

    def test_both_archives_verify_and_include_exact_checksums(self):
        self.verify()
        self.assertEqual(len((self.dist / "SHA256SUMS").read_text().splitlines()), 2)

    def test_tampered_executable_is_rejected_even_with_fresh_archive_checksum(self):
        target = "x86_64-unknown-linux-gnu"
        (self.stages[target] / "forge-engine").write_bytes(b"modified executable")
        self.write_archive(target)
        with self.assertRaisesRegex(ValueError, "Executable checksum mismatch"):
            self.verify()

    def test_archive_and_sidecar_manifest_must_match(self):
        sidecar = next(self.dist.glob("*.build.json"))
        value = json.loads(sidecar.read_text())
        value["commit"] = "b" * 40
        sidecar.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "manifests disagree"):
            self.verify()

    def test_extra_artifact_and_missing_platform_are_rejected(self):
        extra = self.dist / "unreviewed.bin"
        extra.write_bytes(b"unreviewed")
        with self.assertRaisesRegex(ValueError, "Unexpected files"):
            self.verify()
        extra.unlink()
        next(self.dist.glob("*.zip")).unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_archives_are_stable_and_linux_binary_is_executable(self):
        for target in package.TARGETS:
            archive = self.write_archive(target)
            before = archive.read_bytes()
            self.write_archive(target)
            self.assertEqual(before, archive.read_bytes())
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as data:
                    self.assertTrue(all("\\" not in name for name in data.namelist()))
            else:
                with tarfile.open(archive) as data:
                    binary = data.getmember(self.stages[target].name + "/forge-engine")
                    self.assertEqual(binary.mode, 0o755)
                    self.assertEqual((binary.uid, binary.gid), (0, 0))

    def test_portable_cpu_baseline_and_windows_runtime_are_enforced(self):
        linux = "x86_64-unknown-linux-gnu"
        windows = "x86_64-pc-windows-msvc"
        package.validate_rustflags("-C target-cpu=x86-64", linux)
        package.validate_rustflags("-Ctarget-cpu=x86-64 -Ctarget-feature=+crt-static", windows)
        for flags, target in [("", linux), ("-C target-cpu=native", linux),
                              ("-C target-cpu=x86-64 -C target-cpu=native", linux),
                              ("-C target-cpu=x86-64", windows)]:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                package.validate_rustflags(flags, target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
