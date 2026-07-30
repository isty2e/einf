from dataclasses import dataclass

import numpy as np

from .backend import BackendSpec
from .types import Array, NumpyArray


@dataclass(slots=True)
class TensorGenerator:
    """RNG-controlled tensor generator for fixed/dynamic benchmark inputs."""

    backend: BackendSpec
    random_state: np.random.RandomState

    @classmethod
    def from_seed(cls, *, backend: BackendSpec, seed: int) -> "TensorGenerator":
        """Construct one tensor generator from seed."""
        return cls(
            backend=backend,
            random_state=np.random.RandomState(seed),
        )

    @staticmethod
    def dimension_bounds(*, base: int, floor: int = 1) -> tuple[int, int]:
        """Return the inclusive integer bounds used for one dynamic dimension."""
        low = max(floor, int(base * 0.6))
        high = max(low, int(base * 1.4))
        return low, high

    def draw_dimension(self, *, base: int, floor: int = 1) -> int:
        """Draw one positive dynamic dimension around base size."""
        low, high = self.dimension_bounds(base=base, floor=floor)
        return int(self.random_state.randint(low, high + 1))

    def randn_numpy(self, shape: tuple[int, ...]) -> NumpyArray:
        """Draw one float32 NumPy tensor with Gaussian entries."""
        return np.asarray(
            self.random_state.standard_normal(shape),
            dtype=np.float32,
        )

    def randn_backend(self, shape: tuple[int, ...]) -> Array:
        """Draw one backend tensor with Gaussian entries."""
        return self.backend.to_backend_batch((self.randn_numpy(shape),))[0]

    def backend_batch(self, batch: tuple[NumpyArray, ...]) -> tuple[Array, ...]:
        """Convert one NumPy batch to configured backend batch."""
        return self.backend.to_backend_batch(batch)
