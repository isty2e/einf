import ast
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path
from typing import Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "einf"


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
        name="analysis-must-not-import-concrete-steps",
        source_prefixes=("einf.analysis",),
        target_prefix="einf.steps",
        ticket="td-nw7p",
        rationale="static analysis should not reach into concrete runtime step packages",
    ),
    BoundaryRule(
        name="runtime-packages-must-not-import-analysis",
        source_prefixes=RUNTIME_PACKAGE_PREFIXES,
        target_prefix="einf.analysis",
        ticket="td-rh9d",
        rationale="analysis is a read-only sidecar and must not become a runtime dependency",
    ),
)


KNOWN_BOUNDARY_DEBT: Mapping[tuple[str, str, str], str] = {
    (
        "src/einf/plans/abstract.py",
        "einf.lowering",
        "plans-must-not-import-concrete-lowering",
    ): "td-fgax: separate lowering protocol seam from concrete lowering",
    (
        "src/einf/steps/expand/__init__.py",
        "einf.lowering.expand",
        "steps-must-not-import-lowering",
    ): "td-mfui: move expand program artifacts to steps.expand",
    (
        "src/einf/steps/expand/runtime.py",
        "einf.lowering.expand",
        "steps-must-not-import-lowering",
    ): "td-mfui: move expand program artifacts to steps.expand",
    (
        "src/einf/steps/expand/solve.py",
        "einf.lowering.expand",
        "steps-must-not-import-lowering",
    ): "td-mfui: move expand program artifacts to steps.expand",
    (
        "src/einf/steps/expand/step.py",
        "einf.lowering.expand",
        "steps-must-not-import-lowering",
    ): "td-mfui: move expand program artifacts to steps.expand",
}


def test_package_import_boundaries_follow_taxonomy_guardrails() -> None:
    violations = _find_boundary_violations()
    violation_keys = {violation.key for violation in violations}
    allowed_keys = set(KNOWN_BOUNDARY_DEBT)

    unexpected = tuple(
        violation for violation in sorted(violations, key=_violation_sort_key) if violation.key not in allowed_keys
    )
    stale_allowlist = tuple(sorted(allowed_keys - violation_keys))

    assert not unexpected and not stale_allowlist, _format_failure(unexpected, stale_allowlist)


def test_import_boundary_rules_cover_taxonomy_audit_targets() -> None:
    assert {rule.name for rule in BOUNDARY_RULES} == {
        "analysis-must-not-import-concrete-steps",
        "ir-must-not-import-plans",
        "operations-must-not-import-concrete-steps",
        "plans-must-not-import-concrete-lowering",
        "plans-must-not-import-operations",
        "runtime-packages-must-not-import-analysis",
        "steps-must-not-import-lowering",
        "steps-must-not-import-plans",
    }


def test_import_boundary_scanner_normalizes_relative_imports() -> None:
    tree = ast.parse(
        "from ..lowering import LoweringProgram\n"
        "from ..steps.context import PlanSelectionContext\n"
    )
    references = {
        reference.module
        for reference in _import_references(tree=tree, module_name="einf.plans.abstract", path=Path("abstract.py"))
    }

    assert "einf.lowering" in references
    assert "einf.steps.context" in references


def _find_boundary_violations() -> tuple[BoundaryViolation, ...]:
    violations: set[BoundaryViolation] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module_name = _module_name_for_path(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for reference in _import_references(tree=tree, module_name=module_name, path=path):
            for rule in BOUNDARY_RULES:
                if _matches_any_prefix(module_name, rule.source_prefixes) and _matches_prefix(
                    reference.module, rule.target_prefix
                ):
                    violations.add(
                        BoundaryViolation(
                            path=_relative_posix(path),
                            line=reference.line,
                            imported_module=reference.module,
                            rule_name=rule.name,
                        )
                    )
    return tuple(violations)


def _import_references(*, tree: ast.AST, module_name: str, path: Path) -> Iterable[ImportReference]:
    package_name = _package_name_for_imports(module_name=module_name, path=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield ImportReference(module=alias.name, line=node.lineno)
        elif isinstance(node, ast.ImportFrom):
            base_module = _resolve_import_from_module(node=node, package_name=package_name)
            if base_module is None:
                continue
            yield ImportReference(module=base_module, line=node.lineno)
            if base_module == "einf" or node.module is None:
                for alias in node.names:
                    if alias.name != "*":
                        yield ImportReference(module=f"{base_module}.{alias.name}", line=node.lineno)


def _resolve_import_from_module(*, node: ast.ImportFrom, package_name: str) -> str | None:
    if node.level == 0:
        return node.module
    relative_module = "." * node.level + (node.module or "")
    return resolve_name(relative_module, package_name)


def _module_name_for_path(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def _package_name_for_imports(*, module_name: str, path: Path) -> str:
    if path.name == "__init__.py":
        return module_name
    return module_name.rpartition(".")[0]


def _relative_posix(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _matches_any_prefix(module_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(_matches_prefix(module_name, prefix) for prefix in prefixes)


def _matches_prefix(module_name: str, prefix: str) -> bool:
    return module_name == prefix or module_name.startswith(f"{prefix}.")


def _violation_sort_key(violation: BoundaryViolation) -> tuple[str, int, str, str]:
    return (violation.path, violation.line, violation.imported_module, violation.rule_name)


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
