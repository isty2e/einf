from hashlib import sha256

import numpy as np

from .types import NumpyArray


def numpy_batch_fingerprint(batch: tuple[NumpyArray, ...]) -> bytes:
    """Fingerprint ordered numeric inputs without retaining their storage.

    Parameters
    ----------
    batch : tuple[NumpyArray, ...]
        Host arrays whose dtype, shape, strides and exact contents identify one
        generated workload. Numeric and boolean dtypes are supported.

    Returns
    -------
    bytes
        SHA-256 digest of the input representations and logical C-order bytes.
        Non-contiguous arrays use at most 16,384 buffered elements at a time.

    Raises
    ------
    TypeError
        If an input is not a numeric or boolean NumPy array.
    """
    digest = sha256()
    digest.update(len(batch).to_bytes(8, "big"))
    for index, array in enumerate(batch):
        if not isinstance(array, np.ndarray) or array.dtype.kind not in "buifc":
            raise TypeError(
                f"dynamic input[{index}] must be a numeric or boolean NumPy array"
            )

        metadata = repr((array.dtype.str, array.shape, array.strides)).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        if not array.size:
            continue

        if array.flags.c_contiguous:
            digest.update(array.data.cast("B"))
        else:
            # Copy only bounded chunks, not the whole strided input.
            with np.nditer(
                array,
                flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=["readonly"],
                order="C",
                buffersize=16_384,
            ) as chunks:
                for _ in chunks:
                    digest.update(chunks[0].tobytes(order="C"))
    return digest.digest()
