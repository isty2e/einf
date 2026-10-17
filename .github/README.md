# CI dependencies

External Actions in the CI workflow use full commit SHAs. Version comments
identify the corresponding upstream releases; they do not select the code
that runs.

Dependabot checks for Action updates weekly and opens pull requests against
the repository's default branch. These PRs run the same CI checks as code
changes and require maintainer review before merging. CI has read-only
repository permissions; no update-publishing workflow or automatic merge
is configured.

This pins Action implementations, not the whole CI environment. Python
packages still resolve from the ranges in `pyproject.toml`, and the runner
images and Python patch versions can change. A green run is evidence for
the environment used by that run, not every later dependency release.
