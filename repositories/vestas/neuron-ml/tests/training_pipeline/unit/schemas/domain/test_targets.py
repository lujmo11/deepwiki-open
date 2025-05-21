from enum import Enum

import pytest

from neuron.schemas.domain import Target


@pytest.mark.parametrize("enum_element", Target)
def test_target_enum_name_and_value_match(enum_element: Enum) -> None:
    """Test that the name and value of the enum match.

    For example Targets.MxBld_0000_max should have the value "MxBld_0000_max"
    """
    assert enum_element.name == enum_element.value


def test_target_enum_values_are_unique() -> None:
    """Test that the values of the enum are unique."""
    assert len(Target) == len(set(Target))
