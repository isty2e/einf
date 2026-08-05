import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class BoundaryRule:
    name: str
    source_prefixes: tuple[str, ...]
    target_prefix: str
    ticket: str
    rationale: str


@dataclass(frozen=True, slots=True)
class ImportReference:
    module: str
    line: int


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    path: str
    line: int
    imported_module: str
    rule_name: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.path, self.imported_module, self.rule_name)


RUNTIME_PACKAGE_PREFIXES = (
    "einf.axis",
    "einf.backend",
    "einf.diagnostics",
    "einf.ir",
    "einf.lowering",
    "einf.operations",
    "einf.output_normalization",
    "einf.plans",
    "einf.reduction",
    "einf.shape",
    "einf.signature",
    "einf.solver",
    "einf.steps",
    "einf.tensor_types",
)


BOUNDARY_RULES = (
    BoundaryRule(
        name="steps-must-not-import-plans",
        source_prefixes=("einf.steps",),
        target_prefix="einf.plans",
        ticket="td-e8uw",
        rationale="step execution code should depend on step-owned context/scoring contracts, not plan internals",
    ),
    BoundaryRule(
        name="steps-must-not-import-lowering",
        source_prefixes=("einf.steps",),
        target_prefix="einf.lowering",
        ticket="td-mfui",
        rationale="expand step artifacts should be owned by steps.expand rather than lowering",
    ),
    BoundaryRule(
        name="plans-must-not-import-operations",
        source_prefixes=("einf.plans",),
        target_prefix="einf.operations",
        ticket="td-jy33",
        rationale="TensorOp call execution glue belongs with operations, not plans",
    ),
    BoundaryRule(
        name="plans-must-not-import-concrete-lowering",
        source_prefixes=("einf.plans",),
        target_prefix="einf.lowering",
        ticket="td-fgax",
        rationale="plans should depend on a lowering protocol seam, not concrete lowering implementation",
    ),
    BoundaryRule(
        name="operations-must-not-import-concrete-steps",
        source_prefixes=("einf.operations",),
        target_prefix="einf.steps",
        ticket="td-esj6",
        rationale="operation construction validation should not be owned by concrete step modules",
    ),
    BoundaryRule(
        name="ir-must-not-import-plans",
        source_prefixes=("einf.ir",),
        target_prefix="einf.plans",
        ticket="td-e8uw",
        rationale="IR routing should not depend on plan-layer runtime context helpers",
    ),
    BoundaryRule(
        name="ir-must-not-import-backend",
        source_prefixes=("einf.ir",),
        target_prefix="einf.backend",
        ticket="td-osrc",
        rationale="IR should model route intent and static solving; call-time backend namespace validation belongs to plans",
    ),
    BoundaryRule(
        name="ir-must-not-import-steps",
        source_prefixes=("einf.ir",),
        target_prefix="einf.steps",
        ticket="td-osrc",
        rationale="IR should not depend on step-owned runtime context normalization",
    ),
    BoundaryRule(
        name="analysis-must-not-import-concrete-steps",
        source_prefixes=("einf.analysis",),
        target_prefix="einf.steps",
        ticket="td-nw7p",
        rationale="static analysis should not reach into concrete runtime step packages",
    ),
    BoundaryRule(
        name="analysis-must-not-import-lowering",
        source_prefixes=("einf.analysis",),
        target_prefix="einf.lowering",
        ticket="60a1",
        rationale="static analysis should consume canonical operation semantics rather than runtime lowering",
    ),
    BoundaryRule(
        name="analysis-must-not-import-plans",
        source_prefixes=("einf.analysis",),
        target_prefix="einf.plans",
        ticket="60a1",
        rationale="static analysis should not construct or inspect runtime plans",
    ),
    BoundaryRule(
        name="analysis-must-not-import-runtime-tensor-op",
        source_prefixes=("einf.analysis",),
        target_prefix="einf.operations.tensor_op",
        ticket="60a1",
        rationale="static analysis should use canonical definitions rather than executable TensorOp objects",
    ),
    BoundaryRule(
        name="lsp-must-not-import-validator",
        source_prefixes=("einf.analysis.lsp",),
        target_prefix="einf.analysis.validator",
        ticket="fv6c",
        rationale="editor semantics must not couple to batch discovery and CLI policy; shared per-file models live in neutral analysis ownership",
    ),
    BoundaryRule(
        name="runtime-packages-must-not-import-analysis",
        source_prefixes=RUNTIME_PACKAGE_PREFIXES,
        target_prefix="einf.analysis",
        ticket="td-rh9d",
        rationale="analysis is a read-only sidecar and must not become a runtime dependency",
    ),
)


KNOWN_BOUNDARY_DEBT: Mapping[tuple[str, str, str], str] = {}


def test_package_import_boundaries_follow_taxonomy_guardrails() -> None:
    violations = _find_boundary_violations()
    violation_keys = {violation.key for violation in violations}
    allowed_keys = set(KNOWN_BOUNDARY_DEBT)

    unexpected = tuple(
        violation
        for violation in sorted(violations, key=_violation_sort_key)
        if violation.key not in allowed_keys
    )
    stale_allowlist = tuple(sorted(allowed_keys - violation_keys))

    assert not unexpected and not stale_allowlist, _format_failure(
        unexpected, stale_allowlist
    )


def test_import_boundary_rules_cover_taxonomy_audit_targets() -> None:
    assert {rule.name for rule in BOUNDARY_RULES} == {
        "analysis-must-not-import-concrete-steps",
        "analysis-must-not-import-lowering",
        "analysis-must-not-import-plans",
        "analysis-must-not-import-runtime-tensor-op",
        "ir-must-not-import-backend",
        "ir-must-not-import-plans",
        "ir-must-not-import-steps",
        "lsp-must-not-import-validator",
        "operations-must-not-import-concrete-steps",
        "plans-must-not-import-concrete-lowering",
        "plans-must-not-import-operations",
        "runtime-packages-must-not-import-analysis",
        "steps-must-not-import-lowering",
        "steps-must-not-import-plans",
    }


def test_import_boundary_scanner_normalizes_relative_imports() -> None:
    tree = ast.parse(
        "from .lowering_protocol import LoweringProgram\n"
        "from ..steps.context import PlanSelectionContext\n"
    )
    references = {
        reference.module
        for reference in _import_references(
            tree=tree, module_name="einf.plans.abstract", path=Path("abstract.py")
        )
    }

    assert "einf.plans.lowering_protocol" in references
    assert "einf.steps.context" in references


def test_import_boundary_scanner_detects_forbidden_import(tmp_path: Path) -> None:
    path = tmp_path / "src" / "einf" / "steps" / "sample.py"
    path.parent.mkdir(parents=True)
    path.write_text("from einf.plans import AbstractPlan\n", encoding="utf-8")

    assert _find_boundary_violations(tmp_path) == (
        BoundaryViolation(
            path="src/einf/steps/sample.py",
            line=1,
            imported_module="einf.plans",
            rule_name="steps-must-not-import-plans",
        ),
    )


def test_import_boundary_scanner_rejects_missing_package_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError, match="package source root does not exist"):
        _find_boundary_violations(tmp_path)


def test_import_boundary_scanner_rejects_empty_package_root(tmp_path: Path) -> None:
    (tmp_path / "src" / "einf").mkdir(parents=True)

    with pytest.raises(AssertionError, match="found no Python files"):
        _find_boundary_violations(tmp_path)


def _find_boundary_violations(
    project_root: Path = PROJECT_ROOT, /
) -> tuple[BoundaryViolation, ...]:
    source_root = project_root / "src"
    package_root = source_root / "einf"
    if not package_root.is_dir():
        raise AssertionError(f"package source root does not exist: {package_root}")

    paths = tuple(sorted(package_root.rglob("*.py")))
    if not paths:
        raise AssertionError(
            f"package boundary scan found no Python files: {package_root}"
        )

    violations: set[BoundaryViolation] = set()
    for path in paths:
        module_name = _module_name_for_path(path, source_root=source_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for reference in _import_references(
            tree=tree, module_name=module_name, path=path
        ):
            for rule in BOUNDARY_RULES:
                if _matches_any_prefix(
                    module_name, rule.source_prefixes
                ) and _matches_prefix(reference.module, rule.target_prefix):
                    violations.add(
                        BoundaryViolation(
                            path=_relative_posix(path, project_root=project_root),
                            line=reference.line,
                            imported_module=reference.module,
                            rule_name=rule.name,
                        )
                    )
    return tuple(violations)


def _import_references(
    *, tree: ast.AST, module_name: str, path: Path
) -> Iterable[ImportReference]:
    package_name = _package_name_for_imports(module_name=module_name, path=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield ImportReference(module=alias.name, line=node.lineno)
        elif isinstance(node, ast.ImportFrom):
            base_module = _resolve_import_from_module(
                node=node, package_name=package_name
            )
            if base_module is None:
                continue
            yield ImportReference(module=base_module, line=node.lineno)
            if base_module == "einf" or node.module is None:
                for alias in node.names:
                    if alias.name != "*":
                        yield ImportReference(
                            module=f"{base_module}.{alias.name}", line=node.lineno
                        )


def _resolve_import_from_module(
    *, node: ast.ImportFrom, package_name: str
) -> str | None:
    if node.level == 0:
        return node.module
    relative_module = "." * node.level + (node.module or "")
    return resolve_name(relative_module, package_name)


def _module_name_for_path(path: Path, *, source_root: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def _package_name_for_imports(*, module_name: str, path: Path) -> str:
    if path.name == "__init__.py":
        return module_name
    return module_name.rpartition(".")[0]


def _relative_posix(path: Path, *, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _matches_any_prefix(module_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(_matches_prefix(module_name, prefix) for prefix in prefixes)


def _matches_prefix(module_name: str, prefix: str) -> bool:
    return module_name == prefix or module_name.startswith(f"{prefix}.")


def _violation_sort_key(violation: BoundaryViolation) -> tuple[str, int, str, str]:
    return (
        violation.path,
        violation.line,
        violation.imported_module,
        violation.rule_name,
    )


def _format_failure(
    unexpected: tuple[BoundaryViolation, ...],
    stale_allowlist: tuple[tuple[str, str, str], ...],
) -> str:
    lines = ["package import-boundary guardrail drift detected"]
    if unexpected:
        lines.append("")
        lines.append("Unexpected forbidden imports:")
        lines.extend(
            f"- {violation.path}:{violation.line}: imports {violation.imported_module} "
            f"({violation.rule_name})"
            for violation in unexpected
        )
    if stale_allowlist:
        lines.append("")
        lines.append("Stale known-debt allowlist entries:")
        lines.extend(
            f"- {path}: no longer imports {module_name} ({rule_name}); remove "
            f"{KNOWN_BOUNDARY_DEBT[(path, module_name, rule_name)]}"
            for path, module_name, rule_name in stale_allowlist
        )
    return "\n".join(lines)
