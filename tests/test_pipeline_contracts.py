import ast
from pathlib import Path

from einf.steps.axis_slice import AxisSliceRuntimeStep, AxisSliceSymbolicStep
from einf.steps.base import RuntimeStep, SymbolicStep
from einf.steps.concat import ConcatRuntimeStep, ConcatSymbolicStep
from einf.steps.einsum import EinsumRuntimeStep, EinsumSymbolicStep
from einf.steps.expand import ExpandRuntimeStep, ExpandSymbolicStep
from einf.steps.permute import (
    AxisPermuteSymbolicStep,
    PermuteRuntimeStep,
    PermuteSymbolicStep,
)
from einf.steps.reduce import ReduceRuntimeStep, ReduceSymbolicStep
from einf.steps.reshape import ReshapeRuntimeStep, ReshapeSymbolicStep


def _field_names(cls: type[object]) -> set[str]:
    dataclass_fields = getattr(cls, "__dataclass_fields__", None)
    if not isinstance(dataclass_fields, dict):
        raise TypeError(f"{cls.__name__} is not a dataclass type")
    return set(dataclass_fields)


def _import_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = "" if node.module is None else node.module
            modules.append("." * node.level + module)
    return tuple(modules)


def test_pipeline_contract_symbolic_steps_embed_program_field() -> None:
    symbolic_step_classes = (
        AxisSliceSymbolicStep,
        ConcatSymbolicStep,
        EinsumSymbolicStep,
        ExpandSymbolicStep,
        PermuteSymbolicStep,
        AxisPermuteSymbolicStep,
        ReduceSymbolicStep,
        ReshapeSymbolicStep,
    )
    for symbolic_step_class in symbolic_step_classes:
        assert "program" in _field_names(symbolic_step_class)


def test_pipeline_contract_runtime_steps_embed_program_field() -> None:
    runtime_step_classes = (
        AxisSliceRuntimeStep,
        ConcatRuntimeStep,
        EinsumRuntimeStep,
        ExpandRuntimeStep,
        PermuteRuntimeStep,
        ReduceRuntimeStep,
        ReshapeRuntimeStep,
    )
    for runtime_step_class in runtime_step_classes:
        assert "program" in _field_names(runtime_step_class)


def test_pipeline_contract_step_interfaces_require_program_attribute() -> None:
    assert "program" in RuntimeStep.__annotations__
    assert "program" in SymbolicStep.__annotations__


def test_pipeline_contract_tensor_op_does_not_import_steps() -> None:
    tensor_op_path = Path("src/einf/operations/tensor_op.py")
    modules = _import_modules(tensor_op_path)
    assert all(
        not module.endswith("steps") and ".steps" not in module for module in modules
    )


def test_pipeline_contract_abstract_plan_imports_only_step_contracts() -> None:
    abstract_path = Path("src/einf/plans/abstract.py")
    modules = _import_modules(abstract_path)
    step_modules = {module for module in modules if ".steps" in module}
    assert step_modules <= {"einf.steps.base"}
    assert all("lowering.builders" not in module for module in modules)


def test_pipeline_contract_concat_step_has_no_reindex_dependency() -> None:
    concat_path = Path("src/einf/steps/concat.py")
    modules = _import_modules(concat_path)
    assert all("reindex" not in module for module in modules)
