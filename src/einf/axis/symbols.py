from .terms import Axis, AxisPack, _validate_identifier


def _find_duplicates(names: tuple[str, ...]) -> list[str]:
    """Return duplicate names preserving first duplicate encounter order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
            continue
        seen.add(name)
    return duplicates


def _parse_variadic_names(names: tuple[str, ...], *, kind: str) -> tuple[str, ...]:
    """Parse explicit one-name-per-argument declarations."""
    if not names:
        raise ValueError(f"{kind} declaration requires at least one name")

    normalized: list[str] = []
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{kind} names must be strings")
        if any(char.isspace() for char in name):
            raise ValueError(
                f"{kind} names must be passed one per argument, not whitespace-delimited strings"
            )
        _validate_identifier(name, kind=kind)
        normalized.append(name)

    parsed_names = tuple(normalized)
    duplicates = _find_duplicates(parsed_names)
    if duplicates:
        raise ValueError(f"duplicate {kind} names: {duplicates}")

    return parsed_names


def _normalize_symbol_group(
    names: tuple[str, ...], *, kind: str, allow_empty: bool
) -> tuple[str, ...]:
    """Validate one tuple-based symbol declaration group."""
    if not isinstance(names, tuple):
        raise TypeError(f"{kind} declarations must be tuple[str, ...]")
    if not names and not allow_empty:
        raise ValueError(f"{kind} declaration requires at least one name")

    normalized: list[str] = []
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{kind} names must be strings")
        if any(char.isspace() for char in name):
            raise ValueError(
                f"{kind} names must be provided as single tokens, without embedded whitespace"
            )
        _validate_identifier(name, kind=kind)
        normalized.append(name)

    parsed_names = tuple(normalized)
    duplicates = _find_duplicates(parsed_names)
    if duplicates:
        raise ValueError(f"duplicate {kind} names: {duplicates}")

    return parsed_names


def axes(*names: str) -> tuple[Axis, ...]:
    """Create named axis symbols.

    Parameters
    ----------
    *names
        Axis names, one name per argument.

    Returns
    -------
    tuple[Axis, ...]
        Axes in declaration order.

    Raises
    ------
    ValueError
        If no names are provided, if names are duplicated, or if a name is invalid.
    """
    parsed_names = _parse_variadic_names(names, kind="axis")
    return tuple(Axis(name) for name in parsed_names)


def packs(*names: str) -> tuple[AxisPack, ...]:
    """Create named axis-pack symbols.

    Parameters
    ----------
    *names
        Axis-pack names, one name per argument.

    Returns
    -------
    tuple[AxisPack, ...]
        Axis packs in declaration order.

    Raises
    ------
    ValueError
        If no names are provided, if names are duplicated, or if a name is invalid.
    """
    parsed_names = _parse_variadic_names(names, kind="axis pack")
    return tuple(AxisPack(name) for name in parsed_names)


def symbols(
    *, axes: tuple[str, ...], packs: tuple[str, ...] = ()
) -> tuple[tuple[Axis, ...], tuple[AxisPack, ...]]:
    """Create axis and axis-pack symbols from tuple-based declarations.

    Parameters
    ----------
    axes
        Tuple of axis names.
    packs
        Tuple of axis-pack names.

    Returns
    -------
    tuple[tuple[Axis, ...], tuple[AxisPack, ...]]
        Pair of `(axes, packs)` declaration tuples.

    Raises
    ------
    ValueError
        If declarations are empty, duplicated, invalid, or overlap across groups.
    """
    axis_names = _normalize_symbol_group(axes, kind="axis", allow_empty=False)
    pack_names = _normalize_symbol_group(packs, kind="axis pack", allow_empty=True)

    overlapping_names = sorted(set(axis_names) & set(pack_names))
    if overlapping_names:
        raise ValueError(
            f"axis and axis-pack names must be distinct: {overlapping_names}"
        )

    axis_symbols = tuple(Axis(name) for name in axis_names)
    pack_symbols = tuple(AxisPack(name) for name in pack_names)
    return axis_symbols, pack_symbols
