from dataclasses import dataclass
from typing import Literal, Self

AnalysisResource = Literal["documents", "document_bytes", "source_bytes"]


class LspAnalysisLimitExceeded(ValueError):
    """A configured analysis resource cannot admit another source capture.

    Parameters
    ----------
    resource : AnalysisResource
        Resource whose configured ceiling was exceeded.
    limit : int
        Configured maximum.
    observed : int
        Usage that the attempted admission would require.
    """

    def __init__(self, resource: AnalysisResource, limit: int, observed: int) -> None:
        self.resource = resource
        self.limit = limit
        self.observed = observed
        super().__init__(f"analysis {resource} limit exceeded: {observed} > {limit}")


@dataclass(frozen=True, slots=True)
class LspAnalysisLimits:
    """Independent opt-in limits on logical analysis source captures.

    Parameters
    ----------
    max_documents : int or None, optional
        Maximum distinct URIs with live captures, including retiring work.
    max_document_bytes : int or None, optional
        Maximum UTF-8 bytes in an individual captured source.
    max_source_bytes : int or None, optional
        Maximum sum of live capture sizes; shared holders count a capture once.

    Raises
    ------
    ValueError
        A configured limit is not a positive built-in integer.
    """

    max_documents: int | None = None
    max_document_bytes: int | None = None
    max_source_bytes: int | None = None

    def __post_init__(self) -> None:
        for name, maximum in (
            ("max_documents", self.max_documents),
            ("max_document_bytes", self.max_document_bytes),
            ("max_source_bytes", self.max_source_bytes),
        ):
            if maximum is not None and (type(maximum) is not int or maximum < 1):
                raise ValueError(f"{name} must be a positive integer or None")

    @property
    def enabled(self) -> bool:
        """Return whether any analysis resource is limited.

        Returns
        -------
        bool
            At least one ceiling is configured.
        """
        return any(
            maximum is not None
            for maximum in (
                self.max_documents,
                self.max_document_bytes,
                self.max_source_bytes,
            )
        )

    @property
    def measures_bytes(self) -> bool:
        """Return whether admission needs a source-byte count.

        Returns
        -------
        bool
            A per-document or aggregate byte ceiling is configured.
        """
        return self.max_document_bytes is not None or self.max_source_bytes is not None

    def require(self, resource: AnalysisResource, observed: int) -> None:
        """Check one resource against its independent ceiling.

        Parameters
        ----------
        resource : AnalysisResource
            Resource being checked.
        observed : int
            Required usage after the proposed admission.

        Raises
        ------
        LspAnalysisLimitExceeded
            Required usage exceeds the configured ceiling.
        """
        maximum = {
            "documents": self.max_documents,
            "document_bytes": self.max_document_bytes,
            "source_bytes": self.max_source_bytes,
        }[resource]
        if maximum is not None and observed > maximum:
            raise LspAnalysisLimitExceeded(resource, maximum, observed)


class LspSourceBudget:
    """Own parent-side accounting without storing source text or workflow tasks.

    Parameters
    ----------
    limits : LspAnalysisLimits
        Canonical limits for this session. Calls are serialized by the server's
        event loop; this is not a thread-safe or process-memory budget.
    """

    def __init__(self, limits: LspAnalysisLimits) -> None:
        self._limits = limits
        self._documents: dict[str, int] = {}
        self._source_bytes = 0

    @property
    def limits(self) -> LspAnalysisLimits:
        """Return this budget's immutable session policy.

        Returns
        -------
        LspAnalysisLimits
            Policy used for every reservation from this budget.
        """
        return self._limits

    def reserve(self, *, uri: str, source: str) -> "LspSourceReservation":
        """Admit one logical capture with one initial owner.

        Parameters
        ----------
        uri : str
            Document identity, independent of filesystem or workspace membership.
        source : str
            Captured text; this budget does not retain it.

        Returns
        -------
        LspSourceReservation
            Source-free accounting token; every holder must eventually release.

        Raises
        ------
        LspAnalysisLimitExceeded
            Admission would exceed a configured ceiling; counters are unchanged.
        UnicodeEncodeError
            A byte-limited source cannot be represented as UTF-8.
        """
        return LspSourceReservation(self, uri=uri, source=source)

    def _acquire(self, *, uri: str, source: str) -> int:
        self.limits.require(
            "documents", len(self._documents) + (uri not in self._documents)
        )
        source_bytes = 0
        if self.limits.measures_bytes:
            if source.isascii():
                source_bytes = len(source)
            else:
                for start in range(0, len(source), 16_384):
                    source_bytes += len(source[start : start + 16_384].encode("utf-8"))
        self.limits.require("document_bytes", source_bytes)
        self.limits.require("source_bytes", self._source_bytes + source_bytes)

        self._documents[uri] = self._documents.get(uri, 0) + 1
        self._source_bytes += source_bytes
        return source_bytes

    def _release(self, *, uri: str, source_bytes: int) -> None:
        count = self._documents[uri] - 1
        if count:
            self._documents[uri] = count
        else:
            del self._documents[uri]
        self._source_bytes -= source_bytes


class LspSourceReservation:
    """Account for one capture until its last workflow owner releases it.

    Parameters
    ----------
    budget : LspSourceBudget
        Session budget charged by construction, not by a later registration step.
    uri : str
        Document identity of the captured source.
    source : str
        Text used for admission only; no source reference is stored.

    Raises
    ------
    LspAnalysisLimitExceeded
        Admission would exceed a configured ceiling.
    UnicodeEncodeError
        A byte-limited source cannot be represented as UTF-8.
    """

    def __init__(self, budget: LspSourceBudget, *, uri: str, source: str) -> None:
        self._source_bytes = budget._acquire(uri=uri, source=source)
        self._budget = budget
        self._uri = uri
        self._holders = 1

    def retain(self) -> Self:
        """Add an owner of this already charged capture.

        Returns
        -------
        Self
            The same reservation, not a second capture.

        Raises
        ------
        RuntimeError
            The last owner already released this reservation.
        """
        if not self._holders:
            raise RuntimeError("source reservation has already been released")
        self._holders += 1
        return self

    def release(self) -> None:
        """Release one owner and return capacity after the last owner.

        Raises
        ------
        RuntimeError
            The reservation has no remaining owner.
        """
        if not self._holders:
            raise RuntimeError("source reservation has already been released")
        self._holders -= 1
        if not self._holders:
            self._budget._release(uri=self._uri, source_bytes=self._source_bytes)
