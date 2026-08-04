from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from inspect import signature

import numpy as np

from .types import Array, BackendName, NumpyArray, Output, TorchDevice, torch


def _synchronize_numpy() -> None:
    """Synchronize NumPy execution, which is complete on return."""


def _bind_synchronizer(
    synchronize: Callable[..., None],
    device: TorchDevice,
) -> Callable[[], None]:
    """Bind a Torch synchronizer that may accept a device or no arguments."""
    try:
        synchronize_signature = signature(synchronize)
    except (TypeError, ValueError):
        return partial(synchronize, device)

    try:
        synchronize_signature.bind(device)
    except TypeError:
        try:
            synchronize_signature.bind()
        except TypeError as error:
            raise TypeError(
                f"torch synchronization callable has unsupported signature: "
                f"{synchronize_signature}"
            ) from error
        return synchronize
    return partial(synchronize, device)


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Resolved benchmark backend and execution target."""

    name: BackendName
    requested_device: str = "cpu"
    _torch_device: TorchDevice | None = field(init=False, repr=False)
    _synchronize: Callable[[], None] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize and validate the execution target once at ingress."""
        if not self.requested_device:
            raise ValueError("benchmark device must not be empty")

        if self.name == "numpy":
            if self.requested_device != "cpu":
                raise ValueError(
                    "numpy benchmark backend only supports the cpu device; "
                    f"got {self.requested_device!r}"
                )
            object.__setattr__(self, "_torch_device", None)
            object.__setattr__(self, "_synchronize", _synchronize_numpy)
            return

        if torch is None:
            raise RuntimeError("torch backend requested but torch is not installed")

        try:
            requested_device = torch.device(self.requested_device)
            probe = torch.empty(0, device=requested_device)
            resolved_device = probe.device
        except Exception as error:
            raise RuntimeError(
                f"torch benchmark device {self.requested_device!r} is unavailable"
            ) from error

        if resolved_device.type == "cpu":
            synchronize = getattr(getattr(torch, "cpu", None), "synchronize", None)
            resolved_synchronize = (
                _bind_synchronizer(synchronize, resolved_device)
                if callable(synchronize)
                else _synchronize_numpy
            )
        else:
            synchronize = getattr(
                getattr(torch, "accelerator", None),
                "synchronize",
                None,
            )
            if not callable(synchronize):
                device_module = getattr(torch, resolved_device.type, None)
                if device_module is None:
                    module_name = f"torch.{resolved_device.type}"
                    try:
                        device_module = import_module(module_name)
                    except ModuleNotFoundError as error:
                        if error.name != module_name:
                            raise
                synchronize = getattr(device_module, "synchronize", None)
            if not callable(synchronize):
                raise TypeError(
                    f"torch benchmark device {resolved_device} has no "
                    "synchronization capability"
                )
            resolved_synchronize = _bind_synchronizer(
                synchronize,
                resolved_device,
            )

        try:
            resolved_synchronize()
        except Exception as error:
            raise RuntimeError(
                f"torch benchmark device {resolved_device} failed to synchronize"
            ) from error

        object.__setattr__(self, "_torch_device", resolved_device)
        object.__setattr__(self, "_synchronize", resolved_synchronize)

    @property
    def resolved_device(self) -> str:
        """Return the canonical execution device label."""
        if self._torch_device is None:
            return "cpu"
        return str(self._torch_device)

    def synchronize(self) -> None:
        """Wait until work submitted to the execution target is complete."""
        self._synchronize()

    @contextmanager
    def preserve_input_batch(self, batch: tuple[Array, ...]) -> Iterator[None]:
        """Prevent or detect runner mutation without copying input tensors."""
        if self.name == "numpy":
            numpy_batch: list[NumpyArray] = []
            protected_ids: set[int] = set()
            restoration_state: dict[int, tuple[NumpyArray, bool]] = {}
            try:
                for index, value in enumerate(batch):
                    if not isinstance(value, np.ndarray):
                        raise TypeError(
                            "numpy benchmark input has an unsupported type at "
                            f"index {index}: {type(value)!r}"
                        )
                    if id(value) in protected_ids:
                        continue
                    protected_ids.add(id(value))
                    numpy_batch.append(value)

                    current: object | None = value
                    while isinstance(current, np.ndarray):
                        restoration_state.setdefault(
                            id(current),
                            (current, bool(current.flags.writeable)),
                        )
                        current = current.base
                    value.setflags(write=False)

                yield
                for index, value in enumerate(numpy_batch):
                    if value.flags.writeable:
                        raise RuntimeError(
                            "numpy benchmark runner removed input write protection "
                            f"at index {index}"
                        )
            finally:
                restoration = list(restoration_state.values())

                def base_depth(item: tuple[NumpyArray, bool]) -> int:
                    depth = 0
                    base = item[0].base
                    while isinstance(base, np.ndarray):
                        depth += 1
                        base = base.base
                    return depth

                restoration.sort(key=base_depth)
                for value, writeable in restoration:
                    try:
                        value.setflags(write=True)
                    except ValueError:
                        if writeable:
                            raise
                for value, writeable in reversed(restoration):
                    if not writeable:
                        value.setflags(write=False)
            return

        if torch is None:
            raise RuntimeError("torch benchmark target is not resolved")

        torch_batch = []
        versions: list[int] = []
        for index, value in enumerate(batch):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "torch benchmark input has an unsupported type at "
                    f"index {index}: {type(value)!r}"
                )
            try:
                version = value._version
            except RuntimeError as error:
                raise RuntimeError(
                    "torch benchmark input does not expose a mutation version "
                    f"at index {index}"
                ) from error
            torch_batch.append(value)
            versions.append(version)

        yield
        for index, (value, version) in enumerate(
            zip(torch_batch, versions, strict=True)
        ):
            if value._version != version:
                raise RuntimeError(
                    f"torch benchmark runner mutated input at index {index}"
                )

    def to_backend_batch(
        self,
        batch: tuple[NumpyArray, ...],
    ) -> tuple[Array, ...]:
        """Take ownership of a fresh NumPy batch and materialize it on the backend."""
        if self.name == "numpy":
            return batch
        if torch is None or self._torch_device is None:
            raise RuntimeError("torch benchmark target is not resolved")
        return tuple(
            torch.from_numpy(array).to(device=self._torch_device) for array in batch
        )

    def validate_output_target(self, output: Output) -> None:
        """Validate that output tensors remain on the configured target."""
        values = output if isinstance(output, tuple) else (output,)
        for value in values:
            if self.name == "numpy":
                if not isinstance(value, np.ndarray):
                    raise TypeError(
                        "numpy benchmark runner returned an unsupported output type: "
                        f"{type(value)!r}"
                    )
                continue
            if torch is None or not isinstance(value, torch.Tensor):
                raise TypeError(
                    "torch benchmark runner returned an unsupported output type: "
                    f"{type(value)!r}"
                )
            if value.device != self._torch_device:
                raise RuntimeError(
                    "torch benchmark runner moved output away from the configured device: "
                    f"expected {self.resolved_device}, got {value.device}"
                )

    def to_numpy_array(self, value: Array) -> NumpyArray:
        """Convert one backend tensor to NumPy float32."""
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float32)
        if torch is not None and isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            return tensor.numpy().astype(np.float32, copy=False)
        raise TypeError(f"unsupported output type: {type(value)!r}")

    def to_numpy_output(self, output: Output) -> tuple[NumpyArray, ...]:
        """Convert one backend output to tuple of NumPy float32 arrays."""
        if isinstance(output, tuple):
            return tuple(self.to_numpy_array(item) for item in output)
        return (self.to_numpy_array(output),)
