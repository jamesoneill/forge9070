# Published validation evidence

These are reviewed publication copies of validation reports from the
`Forge9070-0.9.1-rx9070xt-source.zip` source snapshot. They preserve what was
recorded during development, including failures, skipped checks and unverified
capabilities. They are **historical evidence**, not results from the current
GitHub Actions run.

Start with the [current validation record](../../VALIDATION.md). The most recent
native UI/API checks cover Forge9070 0.9.1 and `forge-engine` 0.2.1:

| Evidence | Published report |
|---|---|
| Optimized Linux build | [Build result](build-release-0.9.1.json) |
| Native Rust unit tests | [Native core result](native-core-tests-0.9.1.json) |
| CLI and native dispatch | [CLI checks](forge-engine-cli-0.9.1.json) |
| HTTP/SSE, admission and model switching | [API checks](native-api-0.9.1.json) |
| Configuration and model discovery | [UI/API checks](native-ui-0.9.1.json) |
| Launcher argument checks | [Launcher report](native-ui-launchers-0.9.1.json) |
| Embedded UI source checks | [Source report](native-ui-source-checks-0.9.1.json) |
| Browser attempts and limitations | [Chromium attempt](native-ui-browser-attempt-0.9.1.json), [separate browser result](native-ui-browser-preview-0.9.1.json) |
| Prior GPU/numerical validation | [Version 0.9.0 record](v0.9.0/VALIDATION.md) |
| Source/published file inventory and hashes | [Manifest](manifest.json) |

Earlier snapshots are retained under `v0.1.0/` through `v0.9.0/`, with related
reports in this directory and `qwen38-header-checks/`. Historical documents
describe their own release, not the current feature set. Header acceptance is
not successful full-checkpoint inference.

## Publication edits

The selected material consists of JSON and Markdown reports present in the
source ZIP. Raw `.log`/`.err` output, debug-named files, dependency listings and
binary payloads are not published here. Structured reports may include relevant
compiler/test output. A report can mention an original log path even though that
raw log is not included in this selection.

The following text substitutions remove transient machine paths:

| Placeholder | Meaning |
|---|---|
| `<workspace>/<project>` | Original transient Forge9070 checkout |
| `<workspace>` | Parent scratch workspace |
| `<build-home>` | Build account home directory |
| `<temporary-directory>` | Temporary filesystem directory |

Hardware-owner wording is generalized to the target machine. Historical document
links are rebased to their published location and a snapshot note is added.
Recorded test outcomes, timings, numerical values, source receipts and limitations
are retained. The [manifest](manifest.json) records each original archive-member
hash and its published-file hash so these edits are explicit. Paths in a report
are historical descriptions, not executable instructions for your machine.

## Interpretation and future checks

Synthetic fixtures and software Vulkan establish only the behavior they exercise.
They do not establish Windows operation, native WSL hardware access, full
pretrained-model quality, RX 9070 XT throughput or long-context capacity.
The recorded browser attempts did not verify rendering or interaction.

New local checks should continue writing to ignored `artifacts/`; GitHub Actions
stores its own results as workflow-run artifacts. Do not overwrite these historical
reports with a new run. Publish new evidence deliberately, identifying the commit,
environment, model/fixture, commands and limits, and redact sensitive information
before adding it to source control.
