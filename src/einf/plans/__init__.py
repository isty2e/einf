"""Plan package namespace.

This package intentionally avoids eager re-exports to prevent import cycles.
Import concrete symbols from submodules (for example, `einf.plans.abstract`).
"""

__all__: list[str] = []
