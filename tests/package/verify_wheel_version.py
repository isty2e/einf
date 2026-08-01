import argparse
import ast
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from zipfile import ZipFile


def _generated_version(source: str) -> str:
    module = ast.parse(source)
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            continue
        value = ast.literal_eval(statement.value)
        if isinstance(value, str):
            return value
    raise ValueError("generated version file has no string __version__ assignment")


def _wheel_version(metadata: bytes) -> str:
    message = BytesParser(policy=default).parsebytes(metadata)
    if message.get("Name") != "einf":
        raise ValueError("wheel metadata does not describe einf")
    version = message.get("Version")
    if not isinstance(version, str) or not version:
        raise ValueError("wheel metadata has no version")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check wheel metadata against the build-generated VCS version."
    )
    parser.add_argument("wheel", type=Path)
    parser.add_argument("generated_version", type=Path)
    arguments = parser.parse_args()

    wheel = arguments.wheel
    if wheel.is_dir():
        wheels = tuple(wheel.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"expected one wheel file: {wheel}")
        wheel = wheels[0]
    elif not (wheel.is_file() and wheel.suffix == ".whl"):
        raise ValueError(f"expected one wheel file: {wheel}")

    expected_version = _generated_version(
        arguments.generated_version.read_text(encoding="utf-8")
    )
    with ZipFile(wheel) as archive:
        metadata_names = tuple(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_names) != 1:
            raise ValueError(f"expected one wheel metadata file: {wheel}")
        actual_version = _wheel_version(archive.read(metadata_names[0]))

    if actual_version != expected_version:
        raise RuntimeError(
            "wheel version does not match the VCS-derived build version: "
            f"{actual_version!r} != {expected_version!r}"
        )


if __name__ == "__main__":
    main()
