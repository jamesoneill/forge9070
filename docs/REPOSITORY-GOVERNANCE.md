# Repository governance

**Only [@jamesoneill](https://github.com/jamesoneill) merges pull requests into `main`.**
Contributions arrive through pull requests, run CI, and wait for the maintainer's
decision. Automatic merging is disabled. Bots and assistants must not merge PRs
on the owner's behalf without a subsequent explicit instruction.

## Required protection model

The intended live configuration has two active branch rulesets targeting
`refs/heads/main`. Verify the live settings; this document does not apply them:

| Ruleset | Enforcement |
| --- | --- |
| `main - owner merges only` | Restricts updates. Only GitHub user `3801829` (`jamesoneill`) has bypass authority, and only for pull requests. |
| `main - pull requests and checks` | Requires a PR, the successful `required` GitHub Actions check, an up-to-date branch, and resolved review threads. Blocks force pushes and deletion. No bypass actors. |

Keeping the rulesets separate lets the owner merge while retaining the quality
requirements. The CI workflow is displayed as **CI / required**; its actual
status-check context is **`required`**. The setup script discovers and binds the
GitHub Actions integration ID from a real check run instead of trusting any
provider that submits the same check name.

The required approval count is zero because GitHub does not allow authors to
approve their own PRs. The owner's exclusive merge authority remains the human
decision point. [CODEOWNERS](../.github/CODEOWNERS) requests the maintainer's
attention; that file alone does not restrict who can merge.

The configuration also enables private vulnerability reporting.
See [SECURITY.md](../SECURITY.md) for reporting guidance.

## Initial publication and subsequent verification

The JSON files in [.github/rulesets](../.github/rulesets/) describe the intended
policy. **Checking them into Git does not enable repository protection.** The
live GitHub settings must be applied and verified separately.

Publish the initial source on a separate branch and open a PR against `main`.
Let **CI / required** finish on that PR, then configure both rulesets in
GitHub's rule editor before the owner merges it. Select the observed GitHub
Actions provider for the check. A GitHub identity-verification prompt must be
completed through GitHub; it is not evidence that the settings were saved.
Never bypass the owner's merge checkpoint to bootstrap the repository.

After the first owner merge, the repeatable administration script verifies
the current main commit and can apply or repair the same configuration:

1. Let **CI / required** finish successfully on the current `main` commit.
2. With Python 3.10+ and [GitHub CLI](https://cli.github.com/) installed,
   authenticate `gh` locally as `jamesoneill`. Use credentials with repository
   Administration write access and Checks read access. Do not put credentials
   in files, issues, or chat.
3. From the project root, run:

   ```console
   python scripts/configure-github-repository.py --check
   python scripts/configure-github-repository.py --apply
   python scripts/configure-github-repository.py --check
   ```

On Linux, use `python3` if that is your Python command. Without an option, the
script performs the read-only check. A nonzero check result means configuration
is missing, differs, or could not be verified; it is not a successful deployment.

The script verifies the exact account ID, public personal repository, admin
access, default branch and real CI check before writing. It updates matching
rulesets by name, refuses duplicate matches, disables auto-merge, enables private
reports, and reads all settings back. It never pushes source, merges a PR,
changes visibility, invites collaborators, or deletes existing rulesets.
Unrelated protections remain in place and may impose additional requirements.

There is no bypass fallback if GitHub rejects the `User` actor or another
setting. Stop and inspect that failure. A partially completed run retains its
successful changes; rerunning is idempotent. Do not loosen protections to make
the script pass.

For manual setup, use the two templates and select the observed GitHub Actions
provider for the `required` check. Some GitHub interfaces expose only role-based
bypass selectors: in a personal repository, **Repository admin / For pull
requests only** corresponds to the sole owner. The script deliberately uses the
specific user ID and does not silently switch to a broader role.

## Maintenance boundaries

- Keep the `required` job name stable. Update its ruleset together with any CI
  rename, after observing the new check on GitHub.
- Do not add Actions, Dependabot, apps, or collaborators to the bypass list.
- Keep workflow permissions minimal. No workflow should auto-approve or merge
  pull requests. Release jobs can publish release assets without merging code.
- Repository administrators can edit protection settings. GitHub also treats
  tokens acting as the owner as that owner; these settings cannot distinguish
  a human action from an owner-authenticated tool.

## GitHub references

- [Rulesets and how multiple rulesets combine](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
- [Creating rulesets and PR-only bypass](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
- [Ruleset API: actors, updates, PRs and required checks](https://docs.github.com/en/rest/repos/rules?apiVersion=2026-03-10)
- [Check runs API](https://docs.github.com/en/rest/checks/runs)
- [Repository settings and private vulnerability reporting API](https://docs.github.com/en/rest/repos/repos)
- [Personal repository permission levels](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository)
- [Authors cannot approve their own pull requests](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/approving-a-pull-request-with-required-reviews)
