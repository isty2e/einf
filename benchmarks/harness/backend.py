from dataclasses import dataclass

import numpy as np

from .types import Array, BackendName, NumpyArray, Output, torch


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Backend declaration and tensor conversion helpers."""

    name: BackendName

    def validate_available(self) -> None:
        """Validate that the configured backend is available."""
        if self.name == "torch" and torch is None:
            raise RuntimeError("torch backend requested but torch is not installed")

    def is_torch_tensor(self, value: Array | Output) -> bool:
        """Return whether value is a torch tensor."""
        return torch is not None and isinstance(value, torch.Tensor)

    def to_backend_batch(
        self,
        batch: tuple[NumpyArray, ...],
    ) -> tuple[Array, ...]:
        """Convert NumPy batch arrays to backend tensor batch."""
        if self.name == "numpy":
            return batch
        if torch is None:
            raise RuntimeError("torch backend requested but torch is not installed")
        return tuple(torch.from_numpy(array.copy()) for array in batch)

    def clone_array(self, value: Array) -> Array:
        """Clone one backend array to independent physical storage."""
        if isinstance(value, np.ndarray):
            return value.copy()
        if torch is not None and isinstance(value, torch.Tensor):
            return value.clone()
        raise TypeError(f"unsupported output type: {type(value)!r}")

    def clone_batch(self, batch: tuple[Array, ...]) -> tuple[Array, ...]:
        """Clone one backend batch to independent physical storage."""
        return tuple(self.clone_array(item) for item in batch)

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

    def touch_array(self, value: Array) -> None:
        """Touch one synchronous CPU output to make it observable."""
        _ = tuple(value.shape)
        if isinstance(value, np.ndarray):
            if value.size > 0:
                if value.ndim == 0:
                    _ = float(value.item())
                else:
                    _ = float(value[(0,) * value.ndim])
            return
        if torch is not None and isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise RuntimeError(
                    "eager benchmark timing requires synchronous CPU outputs; "
                    f"got torch device {value.device}"
                )
            if value.numel() > 0:
                if value.ndim == 0:
                    _ = float(value.item())
                else:
                    _ = float(value[(0,) * value.ndim].item())
            return
        raise TypeError(f"unsupported output type: {type(value)!r}")

    def touch_output(self, output: Output) -> None:
        """Touch one synchronous CPU output tuple or tensor."""
        if isinstance(output, tuple):
            for item in output:
                self.touch_array(item)
            return
        self.touch_array(output)
