import argparse
import re
import subprocess

_RELEASE_VERSION_PATTERN = (
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
)
_RELEASE_TAG_PATTERN = re.compile(rf"v{_RELEASE_VERSION_PATTERN}")
_RELEASE_TAG_GLOB = "v[0-9]*.[0-9]*.[0-9]*"


def _validate_history(
    *,
    shallow_state: str,
    describe_returncode: int,
    described_tag: str,
) -> str:
    if shallow_state.strip() != "false":
        raise RuntimeError("artifact checkout must contain complete Git history")

    release_tag = described_tag.strip()
    if describe_returncode != 0 or not _RELEASE_TAG_PATTERN.fullmatch(release_tag):
        raise RuntimeError("artifact checkout has no reachable vX.Y.Z release tag")
    return release_tag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check VCS history required for package artifact versioning."
    )
    parser.parse_args()

    shallow_state = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    describe_result = subprocess.run(
        [
            "git",
            "describe",
            "--tags",
            "--abbrev=0",
            "--match",
            _RELEASE_TAG_GLOB,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    release_tag = _validate_history(
        shallow_state=shallow_state,
        describe_returncode=describe_result.returncode,
        described_tag=describe_result.stdout,
    )
    print(f"reachable release tag: {release_tag}")


if __name__ == "__main__":
    main()
