from enum import Enum

import pytest

from neuron.evaluation.evaluation_utils import CONDITION_NAMES, CONDITION_OPERATIONS, Condition


@pytest.mark.parametrize("enum_element", Condition)
def test_condition_enum_name_and_value_match(enum_element: Enum) -> None:
    """Test that the name and value of the enum match.

    For example Condition.eq should have the name value "eq".
    """
    assert enum_element.name == enum_element.value


def test_condition_enum_values_are_unique() -> None:
    """Test that the values of the enum are unique."""
    assert len(Condition) == len(set(Condition))


@pytest.mark.parametrize("enum_element", Condition)
def test_condition_names_and_operators(enum_element: Enum) -> None:
    """Test that all conditions include a name and an operator."""
    assert enum_element in CONDITION_OPERATIONS
    assert enum_element in CONDITION_NAMES
