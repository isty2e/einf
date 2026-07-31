from dataclasses import dataclass

import numpy as np

from .backend import BackendSpec
from .types import Array, NumpyArray


def derive_coordinate_seed(
    *,
    seed: int,
    case_index: int,
    round_index: int,
    stream_index: int,
) -> int:
    """Derive one stable RandomState seed from a dynamic input coordinate."""
    components = (seed, case_index, round_index, stream_index)
    if any(component < 0 for component in components):
        raise ValueError("coordinate seed components must be non-negative")
    return int(
        np.random.SeedSequence(components).generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


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
        """Transfer ownership of a fresh NumPy batch to the configured backend."""
        return self.backend.to_backend_batch(batch)
