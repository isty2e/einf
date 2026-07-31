from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
    from torch import device as TorchDevice
else:

    class TorchDevice:
        """Runtime placeholder type when torch is unavailable."""

    class TorchTensor:
        """Runtime placeholder type when torch is unavailable."""


BackendName: TypeAlias = Literal["numpy", "torch"]
LibraryName: TypeAlias = Literal["einf", "einops", "einx"]
NumpyArray: TypeAlias = NDArray[np.float32]
Array: TypeAlias = NumpyArray | TorchTensor
Output: TypeAlias = Array | tuple[Array, ...]
NumpyOutput: TypeAlias = NumpyArray | tuple[NumpyArray, ...]
Runner: TypeAlias = Callable[[tuple[Array, ...]], Output]
Reference: TypeAlias = Callable[[tuple[NumpyArray, ...]], NumpyOutput]
