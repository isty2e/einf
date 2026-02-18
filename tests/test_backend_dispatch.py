from dataclasses import dataclass

import numpy as np
import pytest

from einf.backend import BACKEND_RESOLVER
from einf.diagnostics import ErrorCode, ValidationError


@dataclass(frozen=True, slots=True)
class UnsupportedTensor:
    shape: tuple[int, ...]


class _FakeNamespaceA:
    __name__ = "fake.a"


class _FakeNamespaceB:
    __name__ = "fake.b"


@dataclass(frozen=True, slots=True)
class FakeArrayA:
    shape: tuple[int, ...]

    def __array_namespace__(
        self, api_version: str | None = None
    ) -> type[_FakeNamespaceA]:
        _ = api_version
        return _FakeNamespaceA


@dataclass(frozen=True, slots=True)
class FakeArrayB:
    shape: tuple[int, ...]

    def __array_namespace__(
        self, api_version: str | None = None
    ) -> type[_FakeNamespaceB]:
        _ = api_version
        return _FakeNamespaceB


@dataclass(frozen=True, slots=True)
class BrokenArrayNamespace:
    shape: tuple[int, ...]

    def __array_namespace__(
        self, api_version: str | None = None
    ) -> type[_FakeNamespaceB]:
        _ = api_version
        raise RuntimeError("broken namespace")


@dataclass(frozen=True, slots=True)
class NamespaceWithoutNameArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version
        return object()


@dataclass(frozen=True, slots=True)
class NamespaceWithInvalidNameArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __module__ = "fake"

            def __init__(self) -> None:
                self.__name__ = "   "

        return Namespace()


@dataclass(frozen=True, slots=True)
class NestedCompatNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.numpy.linalg"

        return Namespace()


@dataclass(frozen=True, slots=True)
class RootNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "numpy.random"

        return Namespace()


@dataclass(frozen=True, slots=True)
class NamespaceWithNonStringNameArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __module__ = "fake"

            def __init__(self) -> None:
                self.__name__ = 123

        return Namespace()


@dataclass(frozen=True, slots=True)
class NamespaceWithNonStringModuleArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            def __init__(self) -> None:
                self.__name__ = "fake"
                setattr(self, "__module__", 123)

        return Namespace()


@dataclass(frozen=True, slots=True)
class NamespaceWithNoneModuleArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "fake.none_module"

            def __init__(self) -> None:
                setattr(self, "__module__", None)

        return Namespace()


@dataclass(frozen=True, slots=True)
class NamespaceWithWhitespaceModuleArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "fake.whitespace_module"
            __module__ = "   "

        return Namespace()


@dataclass(frozen=True, slots=True)
class NamespaceWithEmptyModuleArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "fake.empty_module"
            __module__ = ""

        return Namespace()


@dataclass(frozen=True, slots=True)
class NamespaceWithoutModuleAttributeArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "fake.no_module_attr"

            def __getattribute__(self, name: str) -> object:
                if name == "__module__":
                    raise AttributeError
                return super().__getattribute__(name)

        return Namespace()


@dataclass(frozen=True, slots=True)
class UnknownCompatNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.numpylike.linalg"

        return Namespace()


@dataclass(frozen=True, slots=True)
class MixedCaseCompatNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.Numpy.linalg"

        return Namespace()


@dataclass(frozen=True, slots=True)
class CupyCompatNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.cupy.linalg"

        return Namespace()


@dataclass(frozen=True, slots=True)
class JaxCompatNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.jax.numpy"

        return Namespace()


@dataclass(frozen=True, slots=True)
class MlxCoreNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "mlx.core"

        return Namespace()


@dataclass(frozen=True, slots=True)
class MlxCompatNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.mlx.core.linalg"

        return Namespace()


@dataclass(frozen=True, slots=True)
class NamespaceWithExplodingModuleArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            def __init__(self) -> None:
                self.__name__ = "fake_exploding_module"
                self.__module__ = "fake"

            def __getattribute__(self, name: str) -> object:
                if name == "__module__":
                    raise RuntimeError("module lookup exploded")
                return super().__getattribute__(name)

        return Namespace()


@dataclass(frozen=True, slots=True)
class NumpyLinalgCompatNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.numpy.linalg"

        return Namespace()


@dataclass(frozen=True, slots=True)
class NumpyRandomCompatNamespaceArray:
    shape: tuple[int, ...]

    def __array_namespace__(self, api_version: str | None = None) -> object:
        _ = api_version

        class Namespace:
            __name__ = "array_api_compat.numpy.random"

        return Namespace()


@pytest.fixture
def numpy_tensor() -> np.ndarray:
    """Return one simple NumPy tensor fixture."""
    return np.zeros((2, 3), dtype=np.float32)


def test_backend_dispatch_resolves_numpy_profile(numpy_tensor: np.ndarray) -> None:
    profile = BACKEND_RESOLVER.resolve(numpy_tensor, op_name="rearrange")
    assert profile.namespace_id == "array_api_compat.numpy"
    assert profile.backend_family == "numpy"
    assert profile.supports_contract_einsum
    assert profile.supports_strict_view


def test_backend_dispatch_accepts_multiple_same_family_inputs(
    numpy_tensor: np.ndarray,
) -> None:
    profile = BACKEND_RESOLVER.resolve(numpy_tensor, numpy_tensor, op_name="rearrange")
    assert profile.namespace_id == "array_api_compat.numpy"
    assert profile.backend_family == "numpy"


def test_backend_dispatch_rejects_unsupported_input() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            UnsupportedTensor(shape=(2, 3)), op_name="rearrange"
        )

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value
    assert error.value.external_code == "BACKEND_DISPATCH_UNSUPPORTED_INPUT"


def test_backend_dispatch_rejects_empty_input_list() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(op_name="rearrange")

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value
    assert error.value.data["operation"] == "rearrange"


def test_backend_dispatch_rejects_mixed_tensor_families(
    numpy_tensor: np.ndarray,
) -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            numpy_tensor, FakeArrayA(shape=(2, 3)), op_name="rearrange"
        )

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_MIXED_FAMILY.value
    assert error.value.external_code == "BACKEND_DISPATCH_MIXED_FAMILY"
    assert "namespace_ids" in error.value.data
    namespace_ids = error.value.data["namespace_ids"]
    assert isinstance(namespace_ids, str)
    assert "array_api_compat.numpy" in namespace_ids


def test_backend_dispatch_rejects_three_way_mixed_tensor_families(
    numpy_tensor: np.ndarray,
) -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            numpy_tensor,
            FakeArrayA(shape=(2, 3)),
            FakeArrayB(shape=(2, 3)),
            op_name="rearrange",
        )

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_MIXED_FAMILY.value
    assert error.value.data["families"] == 3


def test_backend_dispatch_requires_einsum_extension_for_contract() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(FakeArrayA(shape=(2, 3)), op_name="contract")

    assert error.value.code == ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING.value
    assert error.value.external_code == "BACKEND_REQUIRED_EXTENSION_MISSING"


def test_backend_dispatch_requires_einsum_extension_for_einop() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(FakeArrayA(shape=(2, 3)), op_name="einop")

    assert error.value.code == ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING.value
    assert error.value.data["operation"] == "einop"


def test_backend_dispatch_allows_non_contract_ops_without_einsum_extension() -> None:
    profile = BACKEND_RESOLVER.resolve(FakeArrayB(shape=(2, 3)), op_name="rearrange")
    assert profile.namespace_id.endswith("._FakeNamespaceB")
    assert profile.backend_family is None
    assert not profile.supports_contract_einsum
    assert not profile.supports_strict_view


def test_backend_dispatch_normalizes_non_typeerror_namespace_failures() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            BrokenArrayNamespace(shape=(2, 3)), op_name="rearrange"
        )

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value


def test_backend_dispatch_operation_name_is_normalized() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(FakeArrayA(shape=(2, 3)), op_name="CONTRACT")

    assert error.value.code == ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING.value
    assert error.value.data["operation"] == "contract"


def test_backend_dispatch_operation_name_is_normalized_with_whitespace() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(FakeArrayA(shape=(2, 3)), op_name="  CONTRACT  ")

    assert error.value.code == ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING.value
    assert error.value.data["operation"] == "contract"


def test_backend_dispatch_operation_name_is_normalized_with_whitespace_for_valid_backend(
    numpy_tensor: np.ndarray,
) -> None:
    profile = BACKEND_RESOLVER.resolve(numpy_tensor, op_name="  ConTract  ")
    assert profile.backend_family == "numpy"
    assert profile.supports_contract_einsum


def test_backend_dispatch_rejects_non_string_operation_name() -> None:
    with pytest.raises(TypeError, match="op_name must be a non-empty string"):
        _ = BACKEND_RESOLVER.resolve(FakeArrayA(shape=(2, 3)), op_name=123)  # type: ignore[arg-type]


def test_backend_dispatch_rejects_empty_operation_name() -> None:
    with pytest.raises(TypeError, match="op_name must be a non-empty string"):
        _ = BACKEND_RESOLVER.resolve(FakeArrayA(shape=(2, 3)), op_name="   ")


def test_backend_dispatch_rejects_namespace_without_name() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            NamespaceWithoutNameArray(shape=(2, 3)), op_name="rearrange"
        )

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value


def test_backend_dispatch_rejects_namespace_with_invalid_name() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            NamespaceWithInvalidNameArray(shape=(2, 3)), op_name="rearrange"
        )

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value


def test_backend_dispatch_infers_opt_backend_from_nested_array_api_compat_namespace() -> (
    None
):
    profile = BACKEND_RESOLVER.resolve(
        NestedCompatNamespaceArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.backend_family == "numpy"


def test_backend_dispatch_infers_opt_backend_from_root_namespace() -> None:
    profile = BACKEND_RESOLVER.resolve(
        RootNamespaceArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.backend_family == "numpy"
    assert profile.supports_contract_einsum
    assert profile.supports_strict_view


def test_backend_dispatch_rejects_namespace_with_non_string_name() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            NamespaceWithNonStringNameArray(shape=(2, 3)), op_name="rearrange"
        )

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value


def test_backend_dispatch_allows_unknown_operation_without_einsum_requirement() -> None:
    profile = BACKEND_RESOLVER.resolve(FakeArrayA(shape=(2, 3)), op_name="mystery_op")
    assert not profile.supports_contract_einsum


def test_backend_dispatch_rejects_namespace_with_non_string_module() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            NamespaceWithNonStringModuleArray(shape=(2, 3)), op_name="rearrange"
        )

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value


def test_backend_dispatch_einsum_probe_runtime_error_maps_to_missing_extension(
    monkeypatch: pytest.MonkeyPatch,
    numpy_tensor: np.ndarray,
) -> None:
    def broken_probe(_backend_name: str) -> bool:
        raise RuntimeError("backend probe exploded")

    monkeypatch.setattr("einf.backend.dispatch.oe_backends.has_einsum", broken_probe)

    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(numpy_tensor, op_name="contract")

    assert error.value.code == ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING.value


def test_backend_dispatch_einsum_probe_runtime_error_does_not_break_non_einsum_ops(
    monkeypatch: pytest.MonkeyPatch,
    numpy_tensor: np.ndarray,
) -> None:
    def broken_probe(_backend_name: str) -> bool:
        raise RuntimeError("backend probe exploded")

    monkeypatch.setattr("einf.backend.dispatch.oe_backends.has_einsum", broken_probe)

    profile = BACKEND_RESOLVER.resolve(numpy_tensor, op_name="rearrange")
    assert profile.backend_family == "numpy"
    assert not profile.supports_contract_einsum


def test_backend_dispatch_allows_namespace_with_none_module() -> None:
    profile = BACKEND_RESOLVER.resolve(
        NamespaceWithNoneModuleArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.namespace_id == "fake.none_module"


def test_backend_dispatch_allows_namespace_with_whitespace_module() -> None:
    profile = BACKEND_RESOLVER.resolve(
        NamespaceWithWhitespaceModuleArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.namespace_id == "fake.whitespace_module"


def test_backend_dispatch_allows_namespace_with_empty_module() -> None:
    profile = BACKEND_RESOLVER.resolve(
        NamespaceWithEmptyModuleArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.namespace_id == "fake.empty_module"


def test_backend_dispatch_allows_namespace_without_module_attribute() -> None:
    profile = BACKEND_RESOLVER.resolve(
        NamespaceWithoutModuleAttributeArray(shape=(2, 3)),
        op_name="rearrange",
    )
    assert profile.namespace_id == "fake.no_module_attr"


def test_backend_dispatch_treats_mixed_case_compat_backend_as_unknown() -> None:
    profile = BACKEND_RESOLVER.resolve(
        MixedCaseCompatNamespaceArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.backend_family is None
    assert not profile.supports_contract_einsum
    assert not profile.supports_strict_view


def test_backend_dispatch_unknown_compat_backend_skips_einsum_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_probe(_backend_name: str) -> bool:
        raise RuntimeError("probe should not be called")

    monkeypatch.setattr("einf.backend.dispatch.oe_backends.has_einsum", broken_probe)

    profile = BACKEND_RESOLVER.resolve(
        UnknownCompatNamespaceArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.backend_family is None
    assert not profile.supports_contract_einsum


def test_backend_dispatch_unknown_compat_backend_contract_reports_missing_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_probe(_backend_name: str) -> bool:
        raise RuntimeError("probe should not be called")

    monkeypatch.setattr("einf.backend.dispatch.oe_backends.has_einsum", broken_probe)

    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            UnknownCompatNamespaceArray(shape=(2, 3)), op_name="contract"
        )

    assert error.value.code == ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING.value


def test_backend_dispatch_known_non_numpy_backend_uses_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_backends: list[str] = []

    def fake_probe(backend_name: str) -> bool:
        seen_backends.append(backend_name)
        return backend_name == "cupy"

    monkeypatch.setattr("einf.backend.dispatch.oe_backends.has_einsum", fake_probe)

    profile = BACKEND_RESOLVER.resolve(
        CupyCompatNamespaceArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.backend_family == "cupy"
    assert profile.supports_contract_einsum
    assert not profile.supports_strict_view
    assert seen_backends == ["cupy"]


def test_backend_dispatch_known_jax_backend_uses_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_backends: list[str] = []

    def fake_probe(backend_name: str) -> bool:
        seen_backends.append(backend_name)
        return backend_name == "jax"

    monkeypatch.setattr("einf.backend.dispatch.oe_backends.has_einsum", fake_probe)

    profile = BACKEND_RESOLVER.resolve(
        JaxCompatNamespaceArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.backend_family == "jax"
    assert profile.supports_contract_einsum
    assert not profile.supports_strict_view
    assert seen_backends == ["jax"]


def test_backend_dispatch_known_mlx_backend_uses_mlx_core_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_backends: list[str] = []

    def fake_probe(backend_name: str) -> bool:
        seen_backends.append(backend_name)
        return backend_name == "mlx.core"

    monkeypatch.setattr("einf.backend.dispatch.oe_backends.has_einsum", fake_probe)

    profile = BACKEND_RESOLVER.resolve(
        MlxCoreNamespaceArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.backend_family == "mlx.core"
    assert profile.supports_contract_einsum
    assert not profile.supports_strict_view
    assert seen_backends == ["mlx.core"]


def test_backend_dispatch_known_mlx_compat_backend_uses_mlx_core_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_backends: list[str] = []

    def fake_probe(backend_name: str) -> bool:
        seen_backends.append(backend_name)
        return backend_name == "mlx.core"

    monkeypatch.setattr("einf.backend.dispatch.oe_backends.has_einsum", fake_probe)

    profile = BACKEND_RESOLVER.resolve(
        MlxCompatNamespaceArray(shape=(2, 3)), op_name="rearrange"
    )
    assert profile.backend_family == "mlx.core"
    assert profile.supports_contract_einsum
    assert not profile.supports_strict_view
    assert seen_backends == ["mlx.core"]


def test_backend_dispatch_coerces_probe_result_to_bool(
    monkeypatch: pytest.MonkeyPatch,
    numpy_tensor: np.ndarray,
) -> None:
    def weird_probe(_backend_name: str) -> str:
        return "definitely"

    monkeypatch.setattr("einf.backend.dispatch.oe_backends.has_einsum", weird_probe)

    profile = BACKEND_RESOLVER.resolve(numpy_tensor, op_name="rearrange")
    assert isinstance(profile.supports_contract_einsum, bool)
    assert profile.supports_contract_einsum is True


def test_backend_dispatch_accepts_compat_subnamespaces_from_same_family() -> None:
    profile = BACKEND_RESOLVER.resolve(
        NumpyLinalgCompatNamespaceArray(shape=(2, 3)),
        NumpyRandomCompatNamespaceArray(shape=(2, 3)),
        op_name="rearrange",
    )
    assert profile.backend_family == "numpy"


def test_backend_dispatch_accepts_mixed_root_and_compat_namespaces_for_same_family(
    numpy_tensor: np.ndarray,
) -> None:
    profile = BACKEND_RESOLVER.resolve(
        numpy_tensor,
        RootNamespaceArray(shape=(2, 3)),
        op_name="rearrange",
    )
    assert profile.backend_family == "numpy"


def test_backend_dispatch_normalizes_namespace_module_lookup_failures() -> None:
    with pytest.raises(ValidationError) as error:
        _ = BACKEND_RESOLVER.resolve(
            NamespaceWithExplodingModuleArray(shape=(2, 3)), op_name="rearrange"
        )

    assert error.value.code == ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT.value
