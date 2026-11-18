#!/usr/bin/env python3
"""Verify, or explicitly apply, Forge9070's owner-only GitHub governance.

Requires Python 3.10+ and an authenticated GitHub CLI. No third-party modules.
The default mode is read-only. This script never pushes code or merges a PR.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlencode


OWNER = "jamesoneill"
OWNER_ID = 3801829
REPOSITORY = f"{OWNER}/forge9070"
API_BASE = f"repos/{REPOSITORY}"
CHECK_NAME = "required"
ROOT = Path(__file__).resolve().parents[1]
RULESET_FILES = ("main-owner-only.json", "main-quality.json")


class ConfigurationError(RuntimeError):
    """A precondition, API operation or verification failed."""


class GitHub:
    def __init__(self, allow_writes: bool = False) -> None:
        self.allow_writes = allow_writes
        self.gh = shutil.which("gh")
        if not self.gh:
            raise ConfigurationError("GitHub CLI (gh) is required: https://cli.github.com/")

    def api(self, path: str, method: str = "GET", body: Any = None) -> Any:
        if method != "GET" and not self.allow_writes:
            raise ConfigurationError("Writes require the explicit --apply option.")
        command = [
            self.gh, "api", "--hostname", "github.com", "--method", method,
            "-H", "Accept: application/vnd.github+json",
            "-H", "X-GitHub-Api-Version: 2026-03-10", path,
        ]
        payload = None
        if body is not None:
            command.extend(["--input", "-"])
            payload = json.dumps(body)
        environment = os.environ.copy()
        environment.pop("GH_DEBUG", None)
        environment["GH_PROMPT_DISABLED"] = "1"
        try:
            result = subprocess.run(
                command, input=payload, capture_output=True, text=True,
                encoding="utf-8", env=environment, timeout=60, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ConfigurationError(f"GitHub CLI failed during {method} {path}: {exc}") from exc
        if result.returncode:
            # Keep credentials and potentially verbose gh diagnostics out of output.
            match = re.search(r"HTTP (\d{3})", result.stderr)
            status = f" (HTTP {match.group(1)})" if match else ""
            raise ConfigurationError(
                f"GitHub rejected {method} {path}{status}. Check gh auth status and "
                "repository administration/checks permissions. No retry or permission "
                "weakening was attempted."
            )
        if not result.stdout.strip():
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Unexpected non-JSON response to {method} {path}.") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def identity_and_repository(client: GitHub) -> dict[str, Any]:
    profile = client.api("user")
    require(
        profile.get("login", "").lower() == OWNER and profile.get("id") == OWNER_ID,
        f"Authenticate gh as {OWNER} (user {OWNER_ID}); another account cannot run this script.",
    )
    repo = client.api(API_BASE)
    owner = repo.get("owner", {})
    require(
        repo.get("full_name", "").lower() == REPOSITORY
        and owner.get("login", "").lower() == OWNER
        and owner.get("id") == OWNER_ID and owner.get("type") == "User",
        f"The target must be the personal repository {REPOSITORY}.",
    )
    require(repo.get("private") is False and repo.get("visibility") == "public",
            "The repository must already be public; this script never changes visibility.")
    require(not repo.get("archived") and not repo.get("disabled") and not repo.get("fork"),
            "The target must be an active, non-fork repository.")
    require(repo.get("permissions", {}).get("admin") is True,
            "The authenticated account needs repository administration access.")
    require(repo.get("default_branch") == "main", "The default branch must already be main.")
    return repo


def main_sha(client: GitHub) -> str:
    sha = client.api(f"{API_BASE}/branches/main").get("commit", {}).get("sha", "")
    require(bool(re.fullmatch(r"[0-9a-f]{40,64}", sha)),
            "Push the initial source to main before configuring protections.")
    return sha


def ci_check(client: GitHub, sha: str, require_success: bool) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for page in range(1, 51):
        query = urlencode({"check_name": CHECK_NAME, "filter": "latest", "per_page": 100, "page": page})
        response = client.api(f"{API_BASE}/commits/{sha}/check-runs?{query}")
        batch = response.get("check_runs", [])
        runs.extend(batch)
        if len(batch) < 100:
            break
    else:
        raise ConfigurationError("Too many check runs to inspect safely; no settings changed.")
    candidates = [run for run in runs if run.get("name") == CHECK_NAME
                  and run.get("head_sha") == sha
                  and run.get("app", {}).get("slug") == "github-actions"]
    require(bool(candidates),
            "No GitHub Actions check named 'required' exists on the current main commit. "
            "Run CI first; its displayed label is CI / required. Do not substitute that label as the context.")
    check = max(candidates, key=lambda run: run.get("id", 0))
    require(isinstance(check.get("app", {}).get("id"), int) and check["app"]["id"] > 0,
            "Could not determine the GitHub Actions integration ID from the observed check.")
    if require_success:
        require(check.get("status") == "completed" and check.get("conclusion") == "success",
                "The latest CI / required check on main must succeed before --apply. "
                "Fix or finish that CI run first; no settings changed.")
    return check


def desired_rulesets(integration_id: int) -> list[dict[str, Any]]:
    rulesets = [json.loads((ROOT / ".github" / "rulesets" / name).read_text(encoding="utf-8"))
                for name in RULESET_FILES]
    for item in rulesets:
        require(item.get("target") == "branch" and item.get("enforcement") == "active"
                and item.get("conditions") == {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
                "A ruleset template no longer targets only main with active enforcement.")
    owner, quality = rulesets
    require(owner.get("bypass_actors") == [{"actor_id": OWNER_ID, "actor_type": "User", "bypass_mode": "pull_request"}],
            "The owner template must have exactly the verified owner with PR-only bypass.")
    require(owner.get("rules") == [{"type": "update", "parameters": {"update_allows_fetch_and_merge": False}}],
            "The owner template must contain only the restrict-updates rule.")
    require(quality.get("bypass_actors") == [], "The quality template must have no bypass actors.")
    rules = {rule["type"]: rule for rule in quality.get("rules", [])}
    require(set(rules) == {"pull_request", "required_status_checks", "deletion", "non_fast_forward"}
            and len(quality["rules"]) == 4, "Unexpected quality ruleset structure.")
    parameters = rules["required_status_checks"]["parameters"]
    require(parameters["required_status_checks"] == [{"context": CHECK_NAME}]
            and parameters["strict_required_status_checks_policy"] is True
            and parameters["do_not_enforce_on_create"] is False,
            "The quality template must require the stable CI check with strict enforcement.")
    require(rules["pull_request"]["parameters"] == {
        "dismiss_stale_reviews_on_push": True, "require_code_owner_review": False,
        "require_last_push_approval": False, "required_approving_review_count": 0,
        "required_review_thread_resolution": True,
    }, "The PR template changed; review governance requirements before editing this script.")
    parameters["required_status_checks"][0]["integration_id"] = integration_id
    return rulesets


def existing_rulesets(client: GitHub) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for page in range(1, 51):
        batch = client.api(f"{API_BASE}/rulesets?includes_parents=false&per_page=100&page={page}")
        require(isinstance(batch, list), "Unexpected repository ruleset listing.")
        items.extend(batch)
        if len(batch) < 100:
            return items
    raise ConfigurationError("Too many rulesets to inspect safely; no settings changed.")


def find_existing(items: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matches = [item for item in items if item.get("name") == name]
    require(len(matches) <= 1, f"Duplicate rulesets named {name!r}; resolve those manually first.")
    if not matches:
        return None
    item = matches[0]
    require(item.get("source_type") == "Repository" and item.get("source", "").lower() == REPOSITORY,
            f"The matching ruleset {name!r} is not owned by the target repository.")
    require(isinstance(item.get("id"), int), "Ruleset has no usable ID.")
    return item


def matches(actual: Any, expected: Any) -> bool:
    """Compare required fields while tolerating server-added read-only defaults."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and matches(actual[key], value)
                                               for key, value in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        if expected and isinstance(expected[0], dict) and "type" in expected[0]:
            actual = sorted(actual, key=lambda value: value.get("type", ""))
            expected = sorted(expected, key=lambda value: value["type"])
        return all(matches(left, right) for left, right in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def inspect_rulesets(client: GitHub, desired: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    existing = existing_rulesets(client)
    result = []
    for item in desired:
        match = find_existing(existing, item["name"])
        current = client.api(f"{API_BASE}/rulesets/{match['id']}") if match else None
        result.append((item, current))
    return result


def configure(client: GitHub, apply: bool) -> bool:
    repo = identity_and_repository(client)
    sha = main_sha(client)
    check = ci_check(client, sha, require_success=apply)
    desired = desired_rulesets(check["app"]["id"])
    plans = inspect_rulesets(client, desired)
    reporting = client.api(f"{API_BASE}/private-vulnerability-reporting")
    require(isinstance(reporting, dict) and isinstance(reporting.get("enabled"), bool),
            "Could not inspect private vulnerability reporting; no settings changed.")
    require(main_sha(client) == sha, "main changed during preflight; rerun against its new CI result.")
    print(f"Verified owner and public personal repository: {REPOSITORY}")
    print(f"main: {sha}")
    print(f"CI / {CHECK_NAME}: {check.get('status')} / {check.get('conclusion')}; "
          f"GitHub Actions integration {check['app']['id']}")
    if apply:
        if repo.get("allow_auto_merge") is not False:
            client.api(API_BASE, "PATCH", {"allow_auto_merge": False})
        if reporting["enabled"] is not True:
            client.api(f"{API_BASE}/private-vulnerability-reporting", "PUT")
        for item, current in plans:
            if current is not None and matches(current, item):
                print(f"Already configured: {item['name']}")
                continue
            if current:
                client.api(f"{API_BASE}/rulesets/{current['id']}", "PUT", item)
            else:
                client.api(f"{API_BASE}/rulesets", "POST", item)
            print(f"Configured: {item['name']}")
        repo = identity_and_repository(client)
        reporting = client.api(f"{API_BASE}/private-vulnerability-reporting")
        plans = inspect_rulesets(client, desired)
    problems = []
    if repo.get("allow_auto_merge") is not False:
        problems.append("Repository auto-merge is not disabled.")
    if reporting.get("enabled") is not True:
        problems.append("Private vulnerability reporting is not enabled.")
    for item, current in plans:
        if current is None or not matches(current, item):
            problems.append(f"Ruleset is missing or differs: {item['name']}")
    if problems:
        for problem in problems:
            print(f"MISMATCH: {problem}", file=sys.stderr)
        if not apply:
            print("Read-only check finished; use --apply after successful main CI to configure.")
        return False
    print("Verified: owner-only PR merges, required CI, no force push/deletion, auto-merge off, private reports on.")
    print(f"Review live rules: https://github.com/{REPOSITORY}/rules")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--check", action="store_true", help="verify remote settings without writes (default)")
    options.add_argument("--apply", action="store_true", help="configure and read back the exact target repository")
    args = parser.parse_args()
    try:
        return 0 if configure(GitHub(allow_writes=args.apply), apply=args.apply) else 1
    except (ConfigurationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.apply:
            print("Earlier successful changes may remain. Nothing was rolled back or weakened; "
                  "inspect settings and rerun after resolving the error.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
