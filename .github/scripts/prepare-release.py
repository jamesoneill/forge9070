#!/usr/bin/env python3
"""Verify both platform archives before a draft GitHub release is created."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import zipfile

spec = importlib.util.spec_from_file_location("forge_package_release", Path(__file__).with_name("package-release.py"))
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def read_member(path: Path, name: str) -> bytes:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            return archive.read(name)
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(name)
        if handle is None:
            raise ValueError(f"Missing archive member: {name}")
        return handle.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    project_version, commit = package.validate_tag(args.tag)
    engine_version = package.version(package.ROOT / "crates/forge-engine/Cargo.toml")
    expected_files, checksums, rows = set(), [], []
    for target in sorted(package.TARGETS):
        stem = f"forge9070-{args.tag}-{target}"
        extension = ".zip" if target.endswith("windows-msvc") else ".tar.gz"
        archive = args.directory / (stem + extension)
        manifest_file = args.directory / (stem + ".build.json")
        checksum_file = args.directory / (archive.name + ".sha256")
        expected_files.update((archive.name, manifest_file.name, checksum_file.name))
        external = json.loads(manifest_file.read_text(encoding="utf-8"))
        internal = json.loads(read_member(archive, stem + "/build-manifest.json"))
        if external != internal:
            raise ValueError(f"Archive/sidecar build manifests disagree: {target}")
        for key, expected in {"tag": args.tag, "commit": commit, "project_version": project_version,
                              "engine_version": engine_version, "target": target,
                              "cpu_baseline": "x86-64", "model_weights_included": False,
                              "target_gpu_verified": False}.items():
            if external.get(key) != expected:
                raise ValueError(f"Unexpected {key} in {target} manifest")
        executable = "forge-engine.exe" if target.endswith("windows-msvc") else "forge-engine"
        if external.get("binary") != executable:
            raise ValueError(f"Unexpected executable name: {target}")
        package.validate_rustflags(external.get("rustflags", ""), target)
        if not isinstance(external.get("dependency_notice_count"), int) or external["dependency_notice_count"] <= 0:
            raise ValueError(f"Missing resolved dependency notices: {target}")
        actual_binary = hashlib.sha256(read_member(archive, stem + "/" + executable)).hexdigest()
        if actual_binary != external["binary_sha256"]:
            raise ValueError(f"Executable checksum mismatch: {target}")
        for notice in ("LICENSE-MIT", "GGML-NOTICE.md", "THIRD-PARTY-NOTICES.md", "README.md"):
            if not read_member(archive, stem + "/" + notice).strip():
                raise ValueError(f"Empty required notice: {notice}")
        actual_checksum = package.digest(archive)
        expected_line = f"{actual_checksum}  {archive.name}"
        if checksum_file.read_text(encoding="ascii").strip() != expected_line:
            raise ValueError(f"Archive checksum mismatch: {target}")
        checksums.append(expected_line)
        rows.append(f"| {target} | `{archive.name}` | {archive.stat().st_size / (1024 ** 2):.1f} MiB |")
    # Re-verification may replace only this helper's own generated outputs.
    unexpected = {p.name for p in args.directory.iterdir()} - expected_files - {"SHA256SUMS", "RELEASE-NOTES.md"}
    if unexpected:
        raise ValueError(f"Unexpected files in release artifact set: {sorted(unexpected)}")
    (args.directory / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="ascii")
    notes = [f"# Forge9070 {args.tag}", "", f"Native engine **{engine_version}** · source commit `{commit}`", "",
             "## Downloads", "", "| Platform | Archive | Size |", "| --- | --- | --- |", *rows, "",
             "Extract the archive and follow its README to start the embedded configuration UI.",
             "The archives contain the independent native engine, licenses and attribution,",
             "a build manifest, and portable x86-64 CPU builds. Model weights are separate downloads.", "",
             "## Validation scope", "",
             "The release workflow runs native unit tests and synthetic CPU CLI/API/configuration",
             "checks on Linux and Windows before packaging. This does not establish full pretrained-model",
             "loading, real RX 9070 XT execution, model quality, or token throughput.", "",
             "Linux builds use Ubuntu 24.04/glibc; Windows builds use MSVC with a static C runtime.",
             "GPU operation requires a suitable Vulkan driver. A WSL software adapter is insufficient.", "",
             "## Integrity and licensing", "",
             "Verify the selected archive against `SHA256SUMS` before use. Per-platform build manifests",
             "identify the exact source commit, compiler, target and executable checksum.",
             "The standalone engine is MIT-licensed; included third-party notices retain their own terms.",
             "The wider repository's compatibility implementation has a separate license scope.", "",
             "This release starts as a draft for the repository owner's review.", ""]
    (args.directory / "RELEASE-NOTES.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Verified both archives for {args.tag} at {commit}")


if __name__ == "__main__":
    main()
