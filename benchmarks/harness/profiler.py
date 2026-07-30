import statistics
import time

import numpy as np

from .backend import BackendSpec
from .result import TimingSummary
from .types import Array, Runner


class Profiler:
    """Profiler for synchronized benchmark call latency distributions."""

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

    def measure_call(
        self,
        *,
        runner: Runner,
        batch: tuple[Array, ...],
    ) -> float:
        """Measure one call through completion on the configured target."""
        self.backend.synchronize()
        started = time.perf_counter()
        try:
            output = runner(batch)
        except Exception:
            self.backend.synchronize()
            raise
        self.backend.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.backend.validate_output_target(output)
        return elapsed_ms
