import einf.analysis as analysis
import einf.backend as backend
from einf.analysis.parser import AstParserBackend, ParserBackend
from einf.backend.memory_alias import tensor_numel
from einf.backend.namespace import derive_namespace_id
from einf.backend.runtime import ArrayNamespaceBinding, load_backend_module


def test_analysis_top_level_exports_high_level_api_only() -> None:
    assert hasattr(analysis, "AnalysisOutput")
    assert hasattr(analysis, "AnalysisDiagnostic")
    assert hasattr(analysis, "analyze_module")

    assert not hasattr(analysis, "AstParserBackend")
    assert not hasattr(analysis, "LibCstParserBackend")
    assert not hasattr(analysis, "ParsedModule")
    assert not hasattr(analysis, "ParsedNode")
    assert not hasattr(analysis, "ParserBackend")
    assert not hasattr(analysis, "TextEdit")
    assert not hasattr(analysis, "analyze_einf_calls")


def test_analysis_parser_surface_remains_available() -> None:
    assert AstParserBackend.__name__ == "AstParserBackend"
    assert ParserBackend.__name__ == "ParserBackend"


def test_backend_top_level_exports_dispatch_runtime_contract_only() -> None:
    assert hasattr(backend, "BACKEND_RESOLVER")
    assert hasattr(backend, "BackendProfile")
    assert hasattr(backend, "BackendArrayOps")
    assert hasattr(backend, "bind_array_namespace")
    assert hasattr(backend, "get_backend_array_ops")

    assert not hasattr(backend, "ArrayNamespaceBinding")
    assert not hasattr(backend, "derive_namespace_id")
    assert not hasattr(backend, "infer_backend_family")
    assert not hasattr(backend, "derive_family_key")
    assert not hasattr(backend, "is_namespace_family")
    assert not hasattr(backend, "load_backend_module")
    assert not hasattr(backend, "resolve_backend_array_ops")
    assert not hasattr(backend, "numpy_shares_memory")
    assert not hasattr(backend, "tensor_numel")
    assert not hasattr(backend, "torch_storage_ptr")


def test_backend_submodule_surface_remains_available() -> None:
    assert callable(derive_namespace_id)
    assert callable(tensor_numel)
    assert ArrayNamespaceBinding.__name__ == "ArrayNamespaceBinding"
    assert callable(load_backend_module)
