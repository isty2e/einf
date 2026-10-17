# CI dependencies

## Action updates

External Actions in the CI workflow use full commit SHAs. Version comments
identify the corresponding upstream releases; they do not select the code
that runs.

Dependabot checks for Action updates weekly and opens pull requests against
the repository's default branch. These PRs run the same CI checks as code
changes and require maintainer review before merging. CI has read-only
repository permissions; no update-publishing workflow or automatic merge
is configured.

## Python snapshots

Each CI role uses a hashed requirements snapshot in
`.github/requirements/`. A separate build snapshot pins the tools used
when a dependency has no wheel for the selected Python version. CI
assembles wheels from hash-verified distributions in an isolated build
environment, derives exact runtime and build constraints from their
metadata, and limits subsequent pip installs to that wheelhouse.
Missing packages, incompatible ranges,
and hash mismatches fail the job instead of resolving newer dependencies.
The package dependency ranges in `pyproject.toml` are unchanged.

The snapshots describe available dependencies, not bundles to install.
Each job still installs its own extras. For example, the core snapshot
includes LSP dependencies for isolated wheel/sdist checks without adding
them to the core test environment. The Torch role uses CPU-only wheels.
The same wheelhouse and constraints apply to nested artifact-test and
isolated build environments.

Dependabot checks the Python snapshots weekly and groups updates into a
separate PR. These PRs run the full CI suite and require review before
merging, just like Action updates. An incompatible update fails its own PR;
ordinary PRs keep using the accepted snapshots.

If an update introduces new transitive dependencies, regenerate the
snapshots in that PR before merging. The offline checks reject an
incomplete dependency set. New source-build requirements must also be
added to the build snapshot's inputs in `tests/ci/dependencies.py`.

After changing package requirements or CI roles, refresh the snapshots in
the same PR. Run from the repository root:

```sh
python -m pip install uv
python tests/ci/dependencies.py refresh
```

Review the generated changes before committing. Refreshing resolves
available versions within the package's ranges; CI never runs this
command during an ordinary install.

These snapshots fix project, build, and test dependency versions, not the
entire runner environment. Runner images, Python patch versions, and
bundled environment-bootstrap tooling can still change. A green run is
evidence for the tested revision and environment, not every later release.

## Before a release

Run **Release dependency check** against the branch or tag you intend to
release. It resolves the newest compatible dependency combination within
the package's ranges, then calls the existing CI workflow. It runs the
same 13 checks, including the supported Python matrix, optional LSP and
Torch environments, actual checker executables, and installed artifacts.
It does not update the committed snapshots or publish a package.

The workflow must be present on the default branch before GitHub allows
manual dispatch. With the GitHub CLI, for example:

```sh
gh workflow run release-check.yml --ref develop
gh run list --workflow release-check.yml --event workflow_dispatch --limit 5
```

Use the release candidate's branch or tag instead of `develop` when
appropriate. Check the run's commit SHA against the intended candidate;
a moving branch name is not sufficient evidence. Only a completed,
successful run of the whole workflow establishes compatibility for that
candidate. Successful resolution alone is not a passing release check.

The run retains a `latest-compatible-dependencies` artifact, suffixed with
the resolution attempt number. It contains all seven hashed snapshots and
`resolution.json`, which records the source commit, resolution time,
resolver version, resolution environment, and input/snapshot hashes. Each
job downloads the exact artifact ID produced by resolution and verifies
its source and hashes before installing. Missing or mismatched evidence
fails rather than falling back to the fixed snapshots.

Retrying failed checks uses the completed resolution job's artifact. Start
a new workflow run when you need a fresh dependency resolution; the result
only covers the versions selected for that run. Artifact availability is
subject to GitHub's retention policy. Ordinary PR and push checks continue
to use the committed snapshots.
