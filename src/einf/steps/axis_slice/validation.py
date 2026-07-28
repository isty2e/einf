from math import gcd
from typing import Protocol, TypeGuard

from einf.backend import BackendProfile
from einf.backend.memory_alias import (
    numpy_shares_memory,
    tensor_numel,
    torch_storage_ptr,
)
from einf.backend.namespace import is_namespace_family
from einf.diagnostics import ErrorCode, ValidationError
from einf.tensor_types import TensorLike


class _TorchTensorLike(Protocol):
    """Subset of torch tensor API needed for view overlap checks."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    def numel(self) -> int: ...

    def storage_offset(self) -> int: ...

    def stride(self) -> tuple[int, ...]: ...

    def element_size(self) -> int: ...


def _supports_torch_storage_interface(value: TensorLike) -> TypeGuard[_TorchTensorLike]:
    """Return whether one value exposes torch-like storage/stride methods."""
    if not hasattr(value, "shape"):
        return False
    if not callable(getattr(value, "numel", None)):
        return False
    if not callable(getattr(value, "storage_offset", None)):
        return False
    if not callable(getattr(value, "stride", None)):
        return False
    if not callable(getattr(value, "element_size", None)):
        return False
    return True


def validate_view_outputs(
    *,
    input_tensor: TensorLike,
    outputs: tuple[TensorLike, ...],
    profile: BackendProfile,
) -> None:
    """Validate zero-copy and non-overlap constraints for strict `view` outputs."""
    for output_index, output_tensor in enumerate(outputs):
        shares_with_input = _shares_memory(
            lhs=input_tensor,
            rhs=output_tensor,
            profile=profile,
        )
        if shares_with_input is not True:
            raise ValidationError(
                code=ErrorCode.NOT_A_VIEW,
                message=(
                    "not a view: view requires affine zero-copy mapping "
                    "for every output tensor"
                ),
                help=(
                    "use view mappings that preserve aliasing to the same "
                    "base buffer without materialization"
                ),
                related=("view affine mapping", "zero-copy check"),
                data={
                    "operation": "view",
                    "backend": profile.namespace_id,
                    "output_index": output_index,
                },
            )

    for lhs_index in range(len(outputs)):
        for rhs_index in range(lhs_index + 1, len(outputs)):
            overlaps = _outputs_overlap(
                lhs=outputs[lhs_index],
                rhs=outputs[rhs_index],
                profile=profile,
            )
            if overlaps is not False:
                raise ValidationError(
                    code=ErrorCode.NOT_A_VIEW,
                    message=(
                        "not a view: view split outputs must be affine "
                        "zero-copy non-overlapping regions"
                    ),
                    help="ensure split outputs do not alias overlapping segments",
                    related=("view affine mapping", "split non-overlap"),
                    data={
                        "operation": "view",
                        "backend": profile.namespace_id,
                        "lhs_output_index": lhs_index,
                        "rhs_output_index": rhs_index,
                    },
                )


def _shares_memory(
    *, lhs: TensorLike, rhs: TensorLike, profile: BackendProfile
) -> bool | None:
    """Return whether tensors share memory under one backend profile."""
    if tensor_numel(rhs) == 0:
        return True

    if is_namespace_family(profile.namespace_id, "numpy"):
        numpy_shares = numpy_shares_memory(lhs=lhs, rhs=rhs)
        if numpy_shares is not None:
            return numpy_shares

    if is_namespace_family(profile.namespace_id, "torch"):
        lhs_ptr = torch_storage_ptr(lhs)
        rhs_ptr = torch_storage_ptr(rhs)
        if lhs_ptr is None or rhs_ptr is None:
            return None
        return lhs_ptr == rhs_ptr

    return None


def _outputs_overlap(
    *, lhs: TensorLike, rhs: TensorLike, profile: BackendProfile
) -> bool | None:
    """Return true for overlap, false for disjointness, or none when unknown."""
    if tensor_numel(lhs) == 0 or tensor_numel(rhs) == 0:
        return False

    if is_namespace_family(profile.namespace_id, "numpy"):
        numpy_shares = numpy_shares_memory(lhs=lhs, rhs=rhs)
        if numpy_shares is not None:
            return numpy_shares

    if is_namespace_family(profile.namespace_id, "torch"):
        lhs_ptr = torch_storage_ptr(lhs)
        rhs_ptr = torch_storage_ptr(rhs)
        if lhs_ptr is None or rhs_ptr is None:
            return None
        if lhs_ptr != rhs_ptr:
            return False

        lhs_offsets = _torch_storage_offsets(lhs)
        rhs_offsets = _torch_storage_offsets(rhs)
        if lhs_offsets is not None and rhs_offsets is not None:
            return bool(lhs_offsets & rhs_offsets)

        lhs_span = _torch_byte_span(lhs)
        rhs_span = _torch_byte_span(rhs)
        if lhs_span is None or rhs_span is None:
            return None
        lhs_start, lhs_stop = lhs_span
        rhs_start, rhs_stop = rhs_span
        if lhs_stop <= rhs_start or rhs_stop <= lhs_start:
            return False
        if _torch_storage_congruence_proves_disjoint(lhs=lhs, rhs=rhs):
            return False
        return None

    return None


def _torch_storage_offsets(
    tensor: TensorLike, *, max_elements: int = 100_000
) -> set[int] | None:
    """Return exact storage element offsets when feasible."""
    if not _supports_torch_storage_interface(tensor):
        return None

    if tensor.numel() == 0:
        return set()

    try:
        storage_offset = int(tensor.storage_offset())
        shape = tuple(int(dim) for dim in tensor.shape)
        strides = tuple(int(stride) for stride in tensor.stride())
    except Exception:
        return None

    if len(shape) != len(strides):
        return None

    unique_offset_count = 1
    for dim, stride in zip(shape, strides):
        if stride == 0:
            continue
        unique_offset_count *= dim
        if unique_offset_count > max_elements:
            return None

    offsets: set[int] = {storage_offset}
    for dim, stride in zip(shape, strides):
        if stride == 0:
            continue
        next_offsets: set[int] = set()
        for offset in offsets:
            for index in range(dim):
                next_offsets.add(offset + (index * stride))
        offsets = next_offsets

    return offsets


def _torch_storage_congruence_proves_disjoint(
    *, lhs: TensorLike, rhs: TensorLike
) -> bool:
    """Prove disjointness when affine storage offsets occupy distinct residues."""
    if not _supports_torch_storage_interface(lhs):
        return False
    if not _supports_torch_storage_interface(rhs):
        return False

    try:
        lhs_offset = int(lhs.storage_offset())
        rhs_offset = int(rhs.storage_offset())
        lhs_shape = tuple(int(dim) for dim in lhs.shape)
        rhs_shape = tuple(int(dim) for dim in rhs.shape)
        lhs_strides = tuple(int(stride) for stride in lhs.stride())
        rhs_strides = tuple(int(stride) for stride in rhs.stride())
        lhs_element_size = int(lhs.element_size())
        rhs_element_size = int(rhs.element_size())
    except Exception:
        return False

    if lhs_element_size <= 0 or lhs_element_size != rhs_element_size:
        return False
    if len(lhs_shape) != len(lhs_strides):
        return False
    if len(rhs_shape) != len(rhs_strides):
        return False

    active_strides = [
        abs(stride)
        for shape, strides in (
            (lhs_shape, lhs_strides),
            (rhs_shape, rhs_strides),
        )
        for dim, stride in zip(shape, strides)
        if dim > 1 and stride != 0
    ]
    offset_delta = abs(lhs_offset - rhs_offset)
    if not active_strides:
        return offset_delta != 0

    stride_gcd = gcd(*active_strides)
    return offset_delta % stride_gcd != 0


def _torch_byte_span(tensor: TensorLike) -> tuple[int, int] | None:
    """Return tensor byte span relative to storage base pointer."""
    if not _supports_torch_storage_interface(tensor):
        return None

    try:
        storage_offset = int(tensor.storage_offset())
        strides = tuple(int(stride) for stride in tensor.stride())
        shape = tuple(int(dim) for dim in tensor.shape)
        element_size = int(tensor.element_size())
    except Exception:
        return None

    if len(strides) != len(shape):
        return None

    min_offset = storage_offset
    max_offset = storage_offset
    for dim, stride in zip(shape, strides):
        delta = stride * (dim - 1)
        if delta < 0:
            min_offset += delta
            continue
        max_offset += delta

    start = min_offset * element_size
    stop = (max_offset + 1) * element_size
    if start <= stop:
        return start, stop
    return stop, start


__all__ = ["validate_view_outputs"]
