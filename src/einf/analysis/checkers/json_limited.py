"""Bounded incremental JSON object parsing for checker adapters."""

import json

_MAX_JSON_CONTAINER_DEPTH = 64


class _MalformedJson(Exception):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"checker JSON output is not valid JSON: {value}")


def load_list_field_limited(
    text: str,
    *,
    field: str,
    max_entries: int,
) -> tuple[list[object], bool, str | None]:
    """Parse a bounded list field from one checker JSON object.

    Parameters
    ----------
    text : str
        Checker output containing one JSON object.
    field : str
        Name of the list field to extract.
    max_entries : int
        Maximum number of list entries to materialize.

    Returns
    -------
    tuple[list[object], bool, str or None]
        Parsed entries, whether the list exceeded ``max_entries``, and a
        parse error message. Invalid syntax and excessive container nesting
        return no entries.
    """
    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    try:
        _validate_container_depth(text)
        entries, truncated = _parse_object(
            decoder=decoder,
            text=text,
            field=field,
            max_entries=max_entries,
        )
    except _MalformedJson as error:
        return [], False, str(error)
    except (RecursionError, ValueError):
        return [], False, "checker JSON output is not valid JSON"
    return entries, truncated, None


def _validate_container_depth(text: str) -> None:
    expected_closings: list[str] = []
    inside_string = False
    escaped = False

    for char in text:
        if inside_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                inside_string = False
            continue

        if char == '"':
            inside_string = True
            continue
        if char == "{":
            expected_closings.append("}")
        elif char == "[":
            expected_closings.append("]")
        elif char in "}]" and expected_closings and expected_closings[-1] == char:
            expected_closings.pop()

        if len(expected_closings) > _MAX_JSON_CONTAINER_DEPTH:
            raise _MalformedJson("checker JSON output is not valid JSON")


def _parse_object(
    *,
    decoder: json.JSONDecoder,
    text: str,
    field: str,
    max_entries: int,
) -> tuple[list[object], bool]:
    index = _skip_whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        raise _MalformedJson("checker output must be a JSON object")
    index += 1
    entries: list[object] = []
    truncated = False
    seen_target = False
    expect_key = True
    while True:
        index = _skip_whitespace(text, index)
        if index >= len(text):
            raise _MalformedJson("unexpected end of checker JSON output")
        if text[index] == "}":
            if expect_key and seen_target:
                raise _MalformedJson("checker JSON output is not valid JSON")
            if not seen_target:
                raise _MalformedJson(f"checker JSON output missing {field!r} list")
            index += 1
            if _skip_whitespace(text, index) != len(text):
                raise _MalformedJson("checker JSON output has trailing content")
            return entries, truncated
        if not expect_key:
            if text[index] != ",":
                raise _MalformedJson("checker JSON output is not valid JSON")
            index += 1
            index = _skip_whitespace(text, index)
            if index >= len(text):
                raise _MalformedJson("unexpected end of checker JSON output")
            if text[index] == "}":
                raise _MalformedJson("checker JSON output is not valid JSON")
        key, index = _parse_key(decoder, text, index)
        index = _skip_whitespace(text, index)
        if index >= len(text) or text[index] != ":":
            raise _MalformedJson("checker JSON output is not valid JSON")
        index = _skip_whitespace(text, index + 1)
        if key == field:
            if seen_target:
                raise _MalformedJson(f"checker JSON output repeats {field!r} field")
            seen_target = True
            entries, truncated, index = _parse_list(
                decoder=decoder,
                text=text,
                index=index,
                max_entries=max_entries,
            )
        else:
            index = _skip_value(decoder, text, index)
        expect_key = False


def _parse_key(
    decoder: json.JSONDecoder,
    text: str,
    index: int,
) -> tuple[str, int]:
    try:
        key, new_index = decoder.raw_decode(text, index)
    except json.JSONDecodeError:
        raise _MalformedJson("checker JSON output is not valid JSON") from None
    if not isinstance(key, str):
        raise _MalformedJson("checker JSON output keys must be strings")
    return key, new_index


def _parse_list(
    *,
    decoder: json.JSONDecoder,
    text: str,
    index: int,
    max_entries: int,
) -> tuple[list[object], bool, int]:
    if index >= len(text) or text[index] != "[":
        raise _MalformedJson("checker JSON output list field is malformed")
    index += 1
    entries: list[object] = []
    truncated = False
    expect_value = True
    while True:
        index = _skip_whitespace(text, index)
        if index >= len(text):
            raise _MalformedJson("unexpected end of checker JSON output")
        if text[index] == "]":
            if expect_value and entries:
                raise _MalformedJson("checker JSON output is not valid JSON")
            return entries, truncated, index + 1
        if not expect_value:
            if text[index] != ",":
                raise _MalformedJson("checker JSON output is not valid JSON")
            index += 1
            index = _skip_whitespace(text, index)
            if index >= len(text):
                raise _MalformedJson("unexpected end of checker JSON output")
            if text[index] == "]":
                raise _MalformedJson("checker JSON output is not valid JSON")
        if len(entries) >= max_entries:
            truncated = True
            index = _skip_list_remainder(decoder, text, index)
            return entries, truncated, index
        try:
            entry, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            raise _MalformedJson("checker JSON output is not valid JSON") from None
        entries.append(entry)
        expect_value = False


def _skip_list_remainder(
    decoder: json.JSONDecoder,
    text: str,
    index: int,
) -> int:
    """Validate and skip the rest of a list after the entry cap."""
    while True:
        index = _skip_value(decoder, text, index)
        index = _skip_whitespace(text, index)
        if index >= len(text):
            raise _MalformedJson("unexpected end of checker JSON output")
        if text[index] == "]":
            return index + 1
        if text[index] != ",":
            raise _MalformedJson("checker JSON output is not valid JSON")
        index += 1


def _skip_value(decoder: json.JSONDecoder, text: str, index: int) -> int:
    """Skip one JSON value without materializing it.

    Strings, arrays, and objects are scanned structurally; scalars are
    validated and materialized transiently via ``raw_decode``.
    """
    index = _skip_whitespace(text, index)
    if index >= len(text):
        raise _MalformedJson("unexpected end of checker JSON output")
    char = text[index]
    if char == '"':
        return _skip_string(text, index)
    if char in "{[":  # pragma: no cover - exercised via recursion branches
        return _skip_container(decoder, text, index)
    try:
        _, index = decoder.raw_decode(text, index)
    except json.JSONDecodeError:
        raise _MalformedJson("checker JSON output is not valid JSON") from None
    return index


def _skip_string(text: str, index: int) -> int:
    index += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            if index + 1 >= len(text):
                raise _MalformedJson("unexpected end of checker JSON output")
            escaped = text[index + 1]
            if escaped == "u":
                if index + 6 > len(text):
                    raise _MalformedJson("unexpected end of checker JSON output")
                hex_digits = text[index + 2 : index + 6]
                if any(digit not in "0123456789abcdefABCDEF" for digit in hex_digits):
                    raise _MalformedJson("checker JSON output is not valid JSON")
                index += 6
            elif escaped in '"\\/bfnrt':
                index += 2
            else:
                raise _MalformedJson("checker JSON output is not valid JSON")
            continue
        if char == '"':
            return index + 1
        if ord(char) < 0x20:
            raise _MalformedJson("checker JSON output is not valid JSON")
        index += 1
    raise _MalformedJson("unexpected end of checker JSON output")


def _skip_container(decoder: json.JSONDecoder, text: str, index: int) -> int:
    opening = text[index]
    closing = "]" if opening == "[" else "}"
    index += 1
    expect_value = True
    first = True
    while True:
        index = _skip_whitespace(text, index)
        if index >= len(text):
            raise _MalformedJson("unexpected end of checker JSON output")
        if text[index] == closing:
            if expect_value and not first:
                raise _MalformedJson("checker JSON output is not valid JSON")
            return index + 1
        if not expect_value:
            if text[index] != ",":
                raise _MalformedJson("checker JSON output is not valid JSON")
            index += 1
            index = _skip_whitespace(text, index)
            if index >= len(text):
                raise _MalformedJson("unexpected end of checker JSON output")
            if text[index] == closing:
                raise _MalformedJson("checker JSON output is not valid JSON")
        if opening == "{":
            if text[index] != '"':
                raise _MalformedJson("checker JSON output is not valid JSON")
            index = _skip_string(text, index)
            index = _skip_whitespace(text, index)
            if index >= len(text) or text[index] != ":":
                raise _MalformedJson("checker JSON output is not valid JSON")
            index = _skip_value(decoder, text, index + 1)
        else:
            index = _skip_value(decoder, text, index)
        expect_value = False
        first = False


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\n\r":
        index += 1
    return index
