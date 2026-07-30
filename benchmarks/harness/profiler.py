import statistics
import time

import numpy as np

from .backend import BackendSpec
from .result import TimingSummary
from .types import Array, Runner


class Profiler:
    """Profiler for eager CPU benchmark call latency distributions."""

    def __init__(self, *, backend: BackendSpec) -> None:
        self.backend = backend

    def summarize(self, samples_ms: list[float]) -> TimingSummary:
        """Summarize timing samples."""
        if not samples_ms:
            raise ValueError("cannot summarize empty sample list")
        sample_array = np.asarray(samples_ms, dtype=np.float64)
        p25_ms, p75_ms, p95_ms = np.percentile(sample_array, [25.0, 75.0, 95.0])
        return TimingSummary(
            count=len(samples_ms),
            p25_ms=float(p25_ms),
            median_ms=statistics.median(samples_ms),
            p75_ms=float(p75_ms),
            iqr_ms=float(p75_ms - p25_ms),
            p95_ms=float(p95_ms),
            mean_ms=statistics.fmean(samples_ms),
            min_ms=min(samples_ms),
            max_ms=max(samples_ms),
        )

    def measure_dynamic(
        self,
        *,
        runner: Runner,
        batches: list[tuple[Array, ...]],
        warmup_batches: int,
        repeats: int,
    ) -> list[float]:
        """Measure per-batch dynamic latency across repeated batch cycles."""
        if warmup_batches > len(batches):
            raise ValueError(
                f"warmup_batches={warmup_batches} exceeds batch count={len(batches)}"
            )
        warmup_slice = batches[:warmup_batches]
        measure_slice = batches[warmup_batches:]
        if not measure_slice:
            raise ValueError("no batches left for measurement after warmup")

        for batch in warmup_slice:
            self.backend.touch_output(runner(batch))

        samples_ms: list[float] = []
        for _ in range(repeats):
            for batch in measure_slice:
                started = time.perf_counter()
                output = runner(batch)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.backend.touch_output(output)
                samples_ms.append(elapsed_ms)
        return samples_ms
