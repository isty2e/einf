from .base import AxisTermBase
from .collections import AxisTerms


class AxisTermsFactory:
    """Factory object enabling concise `ax[...]` axis-term syntax."""

    def __getitem__(
        self,
        spec: AxisTerms | AxisTermBase | int | tuple[AxisTermBase | int, ...],
    ) -> AxisTerms:
        """Create one normalized axis-term tuple from bracket syntax.

        Parameters
        ----------
        spec
            One axis term or a tuple of axis terms.

        Returns
        -------
        AxisTerms
            Normalized axis-term tuple used to build a `Signature`.
        """
        return AxisTerms.from_spec(spec)


ax = AxisTermsFactory()

__all__ = ["AxisTermsFactory", "ax"]
