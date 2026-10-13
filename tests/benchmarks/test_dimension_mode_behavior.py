import json

import pytest

from benchmarks.harness import DimensionMode


@pytest.mark.parametrize("mode", tuple(DimensionMode))
def test_dimension_mode_preserves_string_value_and_display(mode: DimensionMode) -> None:
    qualified_name = f"DimensionMode.{mode.name}"

    assert DimensionMode(mode.value) is mode
    assert mode == mode.value
    assert json.loads(json.dumps(mode)) == mode.value
    assert str(mode) == qualified_name
    assert f"{mode}" == qualified_name
    assert format(mode, ">30") == qualified_name.rjust(30)
