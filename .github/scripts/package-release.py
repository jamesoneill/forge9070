#!/usr/bin/env python3
"""Package only the independent native engine; never download model weights.

Uses the checked-out tag, the actual executable version and Cargo's resolved
dependency graph. No publishing or repository mutation is performed here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {"x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"}


def run(*command: str) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True, encoding="utf-8").strip()


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def version(path: Path) -> str:
    with path.open("rb") as handle:
        return tomllib.load(handle)["package"]["version"]


def validate_tag(tag: str) -> tuple[str, str]:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", tag):
        raise ValueError("Expected a release tag such as v0.9.1")
    project_version = version(ROOT / "Cargo.toml")
    if tag != "v" + project_version:
        raise ValueError(f"Tag {tag!r} does not match Cargo.toml version {project_version!r}")
    commit = run("git", "rev-parse", "HEAD")
    tagged_commit = run("git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    if commit != tagged_commit:
        raise ValueError("Checkout is not the requested tag's commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   cwd=ROOT, check=True)
    return project_version, commit


def validate_rustflags(rustflags: str, target: str) -> None:
    options = [item.removeprefix("-C") for item in shlex.split(rustflags)]
    if [item for item in options if item.startswith("target-cpu=")] != ["target-cpu=x86-64"]:
        raise ValueError("Release build must explicitly use RUSTFLAGS=-C target-cpu=x86-64")
    if target.endswith("windows-msvc") and [item for item in options if item.startswith("target-feature=")] != ["target-feature=+crt-static"]:
        raise ValueError("Windows release build must explicitly enable +crt-static")


def dependency_notices(destination: Path, target: str) -> int:
    metadata = json.loads(run("cargo", "metadata", "--locked", "--format-version", "1",
                              "--filter-platform", target))
    packages = {p["id"]: p for p in metadata["packages"]}
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    engine = next(p["id"] for p in packages.values() if p["name"] == "nextora-engine")
    pending, selected = [engine], set()
    while pending:
        package_id = pending.pop()
        if package_id in selected:
            continue
        selected.add(package_id)
        pending.extend(nodes[package_id]["dependencies"])
    lines = ["# Third-party notices", "",
             "The independent native binary includes third-party components under their own licenses.",
             "This inventory conservatively includes resolved build-time dependencies as well.",
             "Full upstream notices are under `licenses/`; the Forge MIT license and GGML notice",
             "are provided separately. No RustingStrata runtime is included in this archive.", "",
             "| Package | Version | Declared license | Notices |",
             "| --- | --- | --- | --- |"]
    for package_id in sorted(selected - {engine}):
        package = packages[package_id]
        base = Path(package["manifest_path"]).parent.resolve()
        slug = f'{package["name"]}-{package["version"]}'
        legal = {p for p in base.rglob("*") if p.is_file() and
                 p.name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "UNLICENSE", "AUTHORS", "COPYRIGHT"))}
        if package.get("license_file"):
            legal.add(base / package["license_file"])
        if not legal:
            raise ValueError(f"No packaged license text found for {slug}; review before releasing")
        for source in sorted(legal):
            resolved = source.resolve()
            if not resolved.is_relative_to(base):
                raise ValueError(f"License file escapes crate directory: {slug}")
            relative = source.relative_to(base)
            output = destination / "licenses" / slug / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
        license_name = (package.get("license") or "See upstream notice").replace("|", "\\|")
        lines.append(f'| {package["name"]} | {package["version"]} | {license_name} | [Texts](licenses/{slug}/) |')

    # Rust's standard library is linked into the executable too. The minimal
    # toolchain ships its attribution document and referenced license texts.
    rust_docs = Path(run("rustc", "--print", "sysroot")) / "share" / "doc" / "rust"
    copyright_file = rust_docs / "COPYRIGHT-library.html"
    if not copyright_file.is_file() or not (rust_docs / "licenses").is_dir():
        raise ValueError("Rust standard-library notices are missing from the installed toolchain")
    rust_output = destination / "licenses" / "rust-standard-library"
    rust_output.mkdir(parents=True)
    shutil.copyfile(copyright_file, rust_output / copyright_file.name)
    shutil.copytree(rust_docs / "licenses", rust_output / "licenses")
    lines += ["", "## Rust standard library", "",
              "The toolchain's standard-library attribution and license texts are retained in",
              "[`licenses/rust-standard-library/`](licenses/rust-standard-library/).", ""]
    (destination / "THIRD-PARTY-NOTICES.md").write_text("\n".join(lines), encoding="utf-8")
    return len(selected) - 1


def usage(tag: str, engine_version: str, target: str) -> str:
    windows = target.endswith("windows-msvc")
    command = (".\\nextora-engine.exe serve --open-ui --backend auto --models-dir 'F:\\Models' "
               "--settings-file '.\\forge-service.json'" if windows else
               "./nextora-engine serve --open-ui --backend auto --models-dir \"$HOME/Models\" "
               "--settings-file ./forge-service.json")
    doctor = ".\\nextora-engine.exe doctor" if windows else "./nextora-engine doctor"
    return f"""# Nextora {tag} — independent native engine

Engine version: **{engine_version}**. Target: **{target}**.

## Start the configuration UI

Extract the complete archive, open a terminal in its directory, and run:

```{'powershell' if windows else 'bash'}
{command}
```

The UI is served at http://127.0.0.1:9070/ by default. If a browser does not
open automatically, open that address yourself. Choose your model directory,
the first shard of a complete compatible GGUF, and the runtime settings in the UI.
Keep every shard of one checkpoint in the same directory.

Automatic mode chooses compatible Vulkan hardware, or CPU when no suitable
adapter is available, and reports the choice. Existing saved profiles retain
their settings. Explicit GPU choices and model-loading errors remain strict.

To inspect CPU availability and Vulkan adapter requirements:

```{'powershell' if windows else 'bash'}
{doctor}
```

## Build and platform scope

This is an experimental native Rust CPU/Vulkan executable. It was built with
the x86-64 CPU baseline, rather than the build runner's native instruction set.
The Windows package uses a statically linked C runtime. The Linux package is
built on Ubuntu 24.04 for glibc systems; it is not a musl or universal Linux build.
A compatible Vulkan driver must be installed separately for GPU execution.

WSL must expose a real compatible Vulkan adapter. A CPU/software adapter or
the presence of `/dev/dxg` alone is not sufficient. Windows GPU execution can
be used separately from a WSL coding client; configure its reachable API URL.

CI exercises CPU execution using tiny untrained fixtures. These checks do not
establish RX 9070 XT execution, full pretrained-model loading, model quality,
or hardware token rates. See the source repository's validation documentation.

## Contents and attribution

This archive contains `nextora-engine`, its embedded configuration UI and shaders,
the engine MIT license, GGML attribution, resolved dependency notices, and a
build manifest. It does not contain model weights, the compatibility frontend,
or the RustingStrata runtime. The complete source repository has separate
licensing scopes; this native engine's license does not relicense other code.

Do not expose the service publicly without appropriate authentication and
network access controls. The default bind address is loopback. See `serve --help`
for API authentication and transport settings.
"""


def archive(source: Path, output: Path, windows: bool, *, epoch: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    if windows:
        path = output / f"{source.name}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
            for item in sorted(source.rglob("*")):
                if item.is_file():
                    name = (Path(source.name) / item.relative_to(source)).as_posix()
                    info = zipfile.ZipInfo(name, date_time=time.gmtime(max(epoch, 315532800))[:6])
                    info.create_system = 3
                    info.external_attr = (0o100644 << 16)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    handle.writestr(info, item.read_bytes(), compresslevel=9)
    else:
        path = output / f"{source.name}.tar.gz"
        # Normalize timestamps, ownership and file ordering. Identical staged
        # content has identical archive bytes, independent of runner metadata.
        with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as handle:
                for item in [source, *sorted(source.rglob("*"))]:
                    info = handle.gettarinfo(item, arcname=(Path(source.name) / item.relative_to(source)).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = epoch
                    info.mode = 0o755 if item.is_dir() or item.name == "nextora-engine" else 0o644
                    if item.is_file():
                        with item.open("rb") as data:
                            handle.addfile(info, data)
                    else:
                        handle.addfile(info)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    project_version, commit = validate_tag(args.tag)
    engine_version = version(ROOT / "crates/nextora-engine/Cargo.toml")
    binary = args.binary.resolve(strict=True)
    actual_version = run(str(binary), "--version")
    if actual_version != f"nextora-engine {engine_version}":
        raise ValueError(f"Unexpected executable version: {actual_version!r}")
    rustflags = os.environ.get("RUSTFLAGS", "")
    if os.environ.get("CARGO_ENCODED_RUSTFLAGS"):
        raise ValueError("CARGO_ENCODED_RUSTFLAGS would override the documented release CPU baseline")
    validate_rustflags(rustflags, args.target)
    windows = args.target.endswith("windows-msvc")
    epoch = int(run("git", "show", "-s", "--format=%ct", commit))
    stem = f"nextora-cli-{args.tag}-{args.target}"
    with tempfile.TemporaryDirectory(prefix="nextora-release-") as temporary:
        stage = Path(temporary) / stem
        stage.mkdir()
        executable = stage / ("nextora-engine.exe" if windows else "nextora-engine")
        shutil.copy2(binary, executable)
        executable.chmod(0o755)
        shutil.copyfile(ROOT / "crates/nextora-engine/LICENSE-MIT", stage / "LICENSE-MIT")
        shutil.copyfile(ROOT / "crates/nextora-engine/GGML-NOTICE.md", stage / "GGML-NOTICE.md")
        count = dependency_notices(stage, args.target)
        (stage / "README.md").write_text(usage(args.tag, engine_version, args.target), encoding="utf-8")
        manifest = {"project_version": project_version, "engine_version": engine_version,
                    "tag": args.tag, "commit": commit, "target": args.target,
                    "cpu_baseline": "x86-64", "rustflags": rustflags,
                    "rustc": run("rustc", "--version"), "dependency_notice_count": count,
                    "binary": executable.name, "binary_sha256": digest(executable),
                    "model_weights_included": False, "target_gpu_verified": False}
        manifest_text = json.dumps(manifest, indent=2) + "\n"
        (stage / "build-manifest.json").write_text(manifest_text, encoding="utf-8")
        result = archive(stage, args.output, windows, epoch=epoch)
        (args.output / f"{stem}.build.json").write_text(manifest_text, encoding="utf-8")
        (args.output / f"{result.name}.sha256").write_text(f"{digest(result)}  {result.name}\n", encoding="ascii")
        print(f"Packaged {result.name}; {count} dependency notice sets; SHA-256 {digest(result)}")


if __name__ == "__main__":
    main()
