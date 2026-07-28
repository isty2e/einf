import ast
from dataclasses import dataclass
from pathlib import Path

from einf.analysis.model import TextSpan
from einf.analysis.source import SourceText

EINF_OP_NAMES = frozenset(
    {
        "rearrange",
        "reduce",
        "repeat",
        "contract",
        "einop",
        "view",
    }
)


@dataclass(frozen=True, slots=True)
class _OpBinding:
    """Binding for one imported einf operation symbol."""

    op_name: str


@dataclass(frozen=True, slots=True)
class _ModuleBinding:
    """Binding for one recognized einf module path."""

    module_name: str


@dataclass(frozen=True, slots=True)
class _OtherBinding:
    """Binding that shadows einf names but is not an einf symbol."""


_Binding = _OpBinding | _ModuleBinding | _OtherBinding
_OTHER_BINDING = _OtherBinding()


@dataclass(frozen=True, slots=True)
class CallBindings:
    """Visible binding snapshot for one call expression."""

    bindings: dict[str, _Binding]

    def resolve_einf_op(self, expr: ast.expr) -> str | None:
        """Resolve one callee expression to an einf operation name."""
        if isinstance(expr, ast.Name):
            binding = self.bindings.get(expr.id)
            if isinstance(binding, _OpBinding):
                return binding.op_name
            return None

        if not isinstance(expr, ast.Attribute):
            return None
        if expr.attr not in EINF_OP_NAMES:
            return None

        module_name = self._resolve_module_name(expr.value)
        if module_name in {"einf", "einf.operations"}:
            return expr.attr
        return None

    def binding_for_value(self, expr: ast.expr) -> _Binding | None:
        """Resolve one assignment rhs to a binding that can be aliased."""
        op_name = self.resolve_einf_op(expr)
        if op_name is not None:
            return _OpBinding(op_name)

        module_name = self._resolve_module_name(expr)
        if module_name is not None:
            return _ModuleBinding(module_name)
        return None

    def _resolve_module_name(self, expr: ast.expr) -> str | None:
        """Resolve one expression to a recognized einf module path."""
        if isinstance(expr, ast.Name):
            binding = self.bindings.get(expr.id)
            if isinstance(binding, _ModuleBinding):
                return binding.module_name
            return None

        if not isinstance(expr, ast.Attribute):
            return None

        base_module_name = self._resolve_module_name(expr.value)
        if base_module_name == "einf" and expr.attr == "operations":
            return "einf.operations"
        return None


class _BindingCollector:
    """Sequential collector for import/alias bindings at each call site."""

    def __init__(self, source_text: SourceText) -> None:
        self._source_text = source_text
        self._scope_stack: list[dict[str, _Binding]] = [{}]
        self._call_bindings: dict[TextSpan, CallBindings] = {}

    def collect(self, module_node: ast.Module) -> dict[TextSpan, CallBindings]:
        """Collect call-site binding snapshots for one parsed module."""
        for statement in module_node.body:
            self._visit_statement(statement)
        return self._call_bindings

    def _visit_statement(self, statement: ast.stmt) -> None:
        match statement:
            case ast.Import():
                self._bind_import(statement)
            case ast.ImportFrom():
                self._bind_import_from(statement)
            case ast.Assign():
                self._visit_expression(statement.value)
                binding = self._current_bindings().binding_for_value(statement.value)
                for target in statement.targets:
                    self._bind_target(target, binding)
            case ast.AnnAssign():
                if statement.value is not None:
                    self._visit_expression(statement.value)
                    binding = self._current_bindings().binding_for_value(
                        statement.value
                    )
                else:
                    binding = None
                self._bind_target(statement.target, binding)
            case ast.AugAssign():
                self._visit_expression(statement.value)
                self._bind_target(statement.target, None)
            case ast.FunctionDef():
                for decorator in statement.decorator_list:
                    self._visit_expression(decorator)
                self._bind_name(statement.name, None)
                self._visit_function_scope(statement.args, statement.body)
            case ast.AsyncFunctionDef():
                for decorator in statement.decorator_list:
                    self._visit_expression(decorator)
                self._bind_name(statement.name, None)
                self._visit_function_scope(statement.args, statement.body)
            case ast.ClassDef():
                for decorator in statement.decorator_list:
                    self._visit_expression(decorator)
                self._bind_name(statement.name, None)
                self._push_scope()
                for class_statement in statement.body:
                    self._visit_statement(class_statement)
                self._pop_scope()
            case ast.For():
                self._visit_expression(statement.iter)
                self._bind_target(statement.target, None)
                self._visit_statement_list(statement.body)
                self._visit_statement_list(statement.orelse)
            case ast.AsyncFor():
                self._visit_expression(statement.iter)
                self._bind_target(statement.target, None)
                self._visit_statement_list(statement.body)
                self._visit_statement_list(statement.orelse)
            case ast.With():
                for item in statement.items:
                    self._visit_expression(item.context_expr)
                    if item.optional_vars is not None:
                        self._bind_target(item.optional_vars, None)
                self._visit_statement_list(statement.body)
            case ast.AsyncWith():
                for item in statement.items:
                    self._visit_expression(item.context_expr)
                    if item.optional_vars is not None:
                        self._bind_target(item.optional_vars, None)
                self._visit_statement_list(statement.body)
            case ast.Try():
                self._visit_statement_list(statement.body)
                for handler in statement.handlers:
                    if handler.type is not None:
                        self._visit_expression(handler.type)
                    self._push_scope()
                    if handler.name is not None:
                        self._bind_name(handler.name, None)
                    self._visit_statement_list(handler.body)
                    self._pop_scope()
                self._visit_statement_list(statement.orelse)
                self._visit_statement_list(statement.finalbody)
            case ast.If():
                self._visit_expression(statement.test)
                self._visit_statement_list(statement.body)
                self._visit_statement_list(statement.orelse)
            case ast.While():
                self._visit_expression(statement.test)
                self._visit_statement_list(statement.body)
                self._visit_statement_list(statement.orelse)
            case ast.Match():
                self._visit_expression(statement.subject)
                for case in statement.cases:
                    if case.guard is not None:
                        self._visit_expression(case.guard)
                    self._visit_statement_list(case.body)
            case ast.Expr():
                self._visit_expression(statement.value)
            case ast.Return():
                if statement.value is not None:
                    self._visit_expression(statement.value)
            case ast.Raise():
                if statement.exc is not None:
                    self._visit_expression(statement.exc)
                if statement.cause is not None:
                    self._visit_expression(statement.cause)
            case ast.Assert():
                self._visit_expression(statement.test)
                if statement.msg is not None:
                    self._visit_expression(statement.msg)
            case ast.Delete():
                for target in statement.targets:
                    self._bind_target(target, None)
            case _:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        self._visit_expression(child)
                    elif isinstance(child, ast.stmt):
                        self._visit_statement(child)

    def _visit_statement_list(self, statements: list[ast.stmt]) -> None:
        """Visit one ordered statement list."""
        for statement in statements:
            self._visit_statement(statement)

    def _visit_function_scope(
        self,
        arguments: ast.arguments,
        body: list[ast.stmt],
    ) -> None:
        """Visit one nested function scope with argument shadowing."""
        self._push_scope()
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            self._bind_name(argument.arg, None)
        if arguments.vararg is not None:
            self._bind_name(arguments.vararg.arg, None)
        if arguments.kwarg is not None:
            self._bind_name(arguments.kwarg.arg, None)
        self._visit_statement_list(body)
        self._pop_scope()

    def _visit_expression(self, expression: ast.expr) -> None:
        """Visit one expression and record call-site bindings."""
        if isinstance(expression, ast.Call):
            call_span = self._source_text.span_from_ast_node(expression)
            if call_span is not None:
                self._call_bindings[call_span] = self._current_bindings()

        for child in ast.iter_child_nodes(expression):
            if isinstance(child, ast.expr):
                self._visit_expression(child)

    def _bind_import(self, statement: ast.Import) -> None:
        """Bind names introduced by one import statement."""
        for alias in statement.names:
            if alias.asname is not None:
                if alias.name in {"einf", "einf.operations"}:
                    self._bind_name(alias.asname, _ModuleBinding(alias.name))
                else:
                    self._bind_name(alias.asname, None)
                continue

            head = alias.name.split(".", maxsplit=1)[0]
            if head == "einf":
                self._bind_name(head, _ModuleBinding("einf"))
            else:
                self._bind_name(head, None)

    def _bind_import_from(self, statement: ast.ImportFrom) -> None:
        """Bind names introduced by one from-import statement."""
        module_name = statement.module
        for alias in statement.names:
            bound_name = alias.asname or alias.name
            if module_name == "einf" and alias.name in EINF_OP_NAMES:
                self._bind_name(bound_name, _OpBinding(alias.name))
                continue
            if module_name == "einf" and alias.name == "operations":
                self._bind_name(bound_name, _ModuleBinding("einf.operations"))
                continue
            if module_name == "einf.operations" and alias.name in EINF_OP_NAMES:
                self._bind_name(bound_name, _OpBinding(alias.name))
                continue
            self._bind_name(bound_name, None)

    def _current_bindings(self) -> CallBindings:
        """Materialize one visible binding snapshot from the current scope stack."""
        bindings: dict[str, _Binding] = {}
        for scope in self._scope_stack:
            bindings.update(scope)
        return CallBindings(bindings)

    def _bind_target(self, target: ast.expr, binding: _Binding | None) -> None:
        """Bind names introduced or shadowed by one assignment target."""
        if isinstance(target, ast.Name):
            self._bind_name(target.id, binding)
            return

        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(element, binding)

    def _bind_name(self, name: str, binding: _Binding | None) -> None:
        """Bind one visible name in the current scope."""
        self._scope_stack[-1][name] = binding or _OTHER_BINDING

    def _push_scope(self) -> None:
        """Enter one nested lexical scope."""
        self._scope_stack.append({})

    def _pop_scope(self) -> None:
        """Leave one nested lexical scope."""
        self._scope_stack.pop()


def build_call_bindings(
    *,
    source: str,
    path: Path,
    source_text: SourceText,
) -> dict[TextSpan, CallBindings]:
    """Build import/alias-aware binding snapshots for each call in one module."""
    try:
        module_node = ast.parse(source, filename=str(path))
    except SyntaxError:
        return {}
    return _BindingCollector(source_text).collect(module_node)
