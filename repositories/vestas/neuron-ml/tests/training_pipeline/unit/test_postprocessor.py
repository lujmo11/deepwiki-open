import math
from typing import Dict

import numpy as np
import pytest

from neuron.postprocessing.directional_twr import DirectionalTwr
from neuron.schemas.domain import Target


@pytest.fixture
def directional_twr() -> DirectionalTwr:
    return DirectionalTwr()


@pytest.fixture
def target_values_dict() -> Dict[Target, np.array]:
    return {
        Target.MxTwrBot_Twr_m400: np.array([4, 5, 10]),
        Target.MyTwrBot_Twr_m400: np.array([3, 6, 7]),
    }


def test_scenario_1(
    directional_twr: DirectionalTwr, target_values_dict: Dict[Target, np.array]
) -> None:
    mr_values = directional_twr.get_directional_twr_values_scenario_1(
        target_values_dict[Target.MxTwrBot_Twr_m400],
        target_values_dict[Target.MyTwrBot_Twr_m400],
        4,
    )
    mr_expected = (
        target_values_dict[Target.MxTwrBot_Twr_m400] ** 2
        + target_values_dict[Target.MyTwrBot_Twr_m400] ** 2
    ) ** 0.5

    assert np.allclose(mr_values, mr_expected, rtol=1e-2)


def test_scenario_2(
    directional_twr: DirectionalTwr, target_values_dict: Dict[Target, np.array]
) -> None:
    m = 4
    mr_values = directional_twr.get_directional_twr_values_scenario_2(
        target_values_dict[Target.MxTwrBot_Twr_m400],
        target_values_dict[Target.MyTwrBot_Twr_m400],
        m,
    )
    mr_expected = (
        0.5 * target_values_dict[Target.MxTwrBot_Twr_m400] ** m
        + 0.5 * target_values_dict[Target.MyTwrBot_Twr_m400] ** m
    ) ** (1 / m)

    assert np.allclose(mr_values, mr_expected, rtol=1e-2)


def test_scenario_3(
    directional_twr: DirectionalTwr, target_values_dict: Dict[Target, np.array]
) -> None:
    m = 4

    mx = target_values_dict[Target.MxTwrBot_Twr_m400]
    my = target_values_dict[Target.MyTwrBot_Twr_m400]

    mr_values = directional_twr.get_directional_twr_values_scenario_3(
        mx,
        my,
        m,
    )

    load = np.zeros((180, len(mx)))
    corr_fac = -0.3
    for i in range(180):
        dir_rad = math.radians(i)
        load[i, :] = (
            (mx * math.cos(dir_rad)) ** 2
            + (my * math.sin(dir_rad)) ** 2
            - 2 * corr_fac * (mx * math.cos(dir_rad)) * (my * math.sin(dir_rad))
        ) ** 0.5
    dam = np.zeros(len(mx))
    for i in range(180):
        dam += 1 / 180 * load[i, :] ** m
    mr_expected = dam ** (1 / m)

    assert np.allclose(mr_values, mr_expected, rtol=1e-2)
