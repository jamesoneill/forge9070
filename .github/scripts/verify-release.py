#!/usr/bin/env python3
"""Read-only release gate: a tagged, CI-checked commit merged by the owner.

Run this helper from main before executing the requested tag's build scripts.
It never creates tags, merges code or writes to GitHub. GH_TOKEN needs only
Contents, Checks, Actions and Pull requests read access.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "jamesoneill/forge9070"
OWNER_ID = 3801829


def command(*arguments: str) -> str:
    return subprocess.check_output(arguments, cwd=ROOT, text=True, encoding="utf-8").strip()


def github(path: str):
    return json.loads(command("gh", "api", "--method", "GET", "-H",
                              "Accept: application/vnd.github+json", path))


def release_commit(tag: str) -> str:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", tag):
        raise ValueError("Expected an existing release tag such as v0.9.1")
    commit = command("git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The tag did not resolve to a full commit SHA")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   cwd=ROOT, check=True)
    project = tomllib.loads(command("git", "show", f"{commit}:Cargo.toml"))
    if tag != "v" + project["package"]["version"]:
        raise ValueError("Release tag does not match the tagged Cargo.toml version")
    return commit


def verify_ci(commit: str, *, api=github) -> str:
    endpoint = f"repos/{REPOSITORY}/commits/{commit}/check-runs"
    checks = []
    page = 1
    while True:
        response = api(f"{endpoint}?check_name=required&filter=latest&per_page=100&page={page}")
        batch = response["check_runs"]
        checks.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    # Accept only the latest real Actions gate. A similarly named third-party
    # status or an older green rerun must not mask a current failed check.
    candidates = [check for check in checks if check.get("name") == "required"
                  and check.get("head_sha") == commit
                  and check.get("app", {}).get("slug") == "github-actions"]
    if not candidates:
        raise ValueError("No GitHub Actions required check exists on the tagged commit")
    check = max(candidates, key=lambda item: item["id"])
    if check.get("status") != "completed" or check.get("conclusion") != "success":
        raise ValueError("The latest required check on the tagged commit has not succeeded")
    match = re.fullmatch(rf"https://github\.com/{re.escape(REPOSITORY)}/actions/runs/([0-9]+)(?:/job/[0-9]+)?",
                         check.get("details_url", ""))
    if not match:
        raise ValueError("The required check does not identify a workflow run in this repository")
    workflow = api(f"repos/{REPOSITORY}/actions/runs/{match[1]}")
    expected = {"head_sha": commit, "head_branch": "main", "event": "push",
                "path": ".github/workflows/ci.yml", "status": "completed", "conclusion": "success"}
    if any(workflow.get(key) != value for key, value in expected.items()):
        raise ValueError("Release requires successful main-branch CI for the exact tagged commit")
    return workflow["html_url"]


def verify_owner_merge(commit: str, *, api=github) -> str:
    page = 1
    while True:
        pulls = api(f"repos/{REPOSITORY}/commits/{commit}/pulls?per_page=100&page={page}")
        for associated in pulls:
            pull = api(f"repos/{REPOSITORY}/pulls/{associated['number']}")
            if (pull.get("merged") is True and pull.get("merge_commit_sha") == commit
                    and pull.get("base", {}).get("ref") == "main"
                    and pull.get("base", {}).get("repo", {}).get("full_name") == REPOSITORY
                    and pull.get("merged_by", {}).get("id") == OWNER_ID):
                return pull["html_url"]
        if len(pulls) < 100:
            break
        page += 1
    raise ValueError("The tagged commit must be the result of a pull request merged into main by @jamesoneill")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise ValueError(f"Release publishing is configured only for {REPOSITORY}")
    if os.environ.get("GITHUB_ACTOR") != "jamesoneill" or os.environ.get("GITHUB_TRIGGERING_ACTOR") != "jamesoneill":
        raise ValueError("Only @jamesoneill may start or rerun the release workflow")
    commit = release_commit(args.tag)
    ci = verify_ci(commit)
    pull = verify_owner_merge(commit)
    output = {"tag": args.tag, "commit": commit, "ci": ci, "owner_merged_pull_request": pull}
    if destination := os.environ.get("GITHUB_OUTPUT"):
        with open(destination, "a", encoding="utf-8") as handle:
            handle.write(f"commit={commit}\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
