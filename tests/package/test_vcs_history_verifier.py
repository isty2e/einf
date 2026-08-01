import subprocess
from pathlib import Path

import pytest

import tests.package.verify_vcs_history as verifier


def _run_git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", repository, *arguments],
        check=True,
    )


def test_vcs_history_verifier_accepts_reachable_release_tags() -> None:
    assert (
        verifier._validate_history(
            shallow_state="false\n",
            describe_returncode=0,
            described_tag="v0.2.0\n",
        )
        == "v0.2.0"
    )


@pytest.mark.parametrize(
    "tag",
    (
        "ci-marker",
        "v1garbage",
        "v1.2",
        "v1.2.3.4",
        "v1.2.3garbage",
        "v01.2.3",
    ),
)
def test_vcs_history_verifier_rejects_nonrelease_tags(tag: str) -> None:
    with pytest.raises(RuntimeError, match="no reachable vX.Y.Z release tag"):
        verifier._validate_history(
            shallow_state="false",
            describe_returncode=0,
            described_tag=tag,
        )


def test_vcs_history_verifier_rejects_failed_describe_with_valid_stdout() -> None:
    with pytest.raises(RuntimeError, match="no reachable vX.Y.Z release tag"):
        verifier._validate_history(
            shallow_state="false",
            describe_returncode=1,
            described_tag="v1.2.3",
        )


def test_vcs_history_verifier_rejects_shallow_checkout() -> None:
    with pytest.raises(RuntimeError, match="complete Git history"):
        verifier._validate_history(
            shallow_state="true",
            describe_returncode=0,
            described_tag="v0.2.0",
        )


def test_hatch_vcs_release_tag_policy_matches_ci_verifier() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert (
        "tag-pattern = '^v(?P<version>"
        f"{verifier._RELEASE_VERSION_PATTERN}"
        ")$'" in pyproject
    )
    assert (
        'git_describe_command = "git describe --dirty --tags --long --match '
        f'{verifier._RELEASE_TAG_GLOB}"' in pyproject
    )


def test_vcs_history_verifier_rejects_repository_with_only_nonrelease_tag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _run_git(tmp_path, "init", "--quiet")
    (tmp_path / "file.txt").write_text("probe\n", encoding="utf-8")
    _run_git(tmp_path, "add", "file.txt")
    _run_git(
        tmp_path,
        "-c",
        "user.name=probe",
        "-c",
        "user.email=probe@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )
    _run_git(tmp_path, "tag", "ci-marker")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="no reachable vX.Y.Z release tag"):
        verifier.main()
