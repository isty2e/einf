def normalize_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Validate one concrete tensor shape."""
    if not isinstance(shape, tuple):
        raise TypeError("each input shape must be a tuple[int, ...]")
    normalized: list[int] = []
    for value in shape:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("shape entries must be integers")
        if value < 0:
            raise ValueError("shape entries must be non-negative")
        normalized.append(value)
    return tuple(normalized)


def normalize_explicit_sizes(
    *,
    explicit_sizes: dict[str, int] | None,
    axis_names: set[str],
    pack_names: set[str],
) -> dict[str, int]:
    """Validate scalar `.with_sizes(...)` bindings."""
    if explicit_sizes is None:
        return {}
    if not isinstance(explicit_sizes, dict):
        raise TypeError("explicit_sizes must be dict[str, int]")

    normalized: dict[str, int] = {}
    for key, value in explicit_sizes.items():
        if not isinstance(key, str):
            raise TypeError("explicit_sizes keys must be strings")
        if key in pack_names:
            raise ValueError(
                f"explicit size {key!r} refers to an axis pack; only scalar axis bindings are supported"
            )
        if key not in axis_names:
            raise ValueError(f"explicit size {key!r} is not used by the signature")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"explicit size for {key!r} must be an int")
        if value < 0:
            raise ValueError(f"explicit size for {key!r} must be non-negative")
        normalized[key] = value
    return normalized


__all__ = [
    "normalize_explicit_sizes",
    "normalize_shape",
]
