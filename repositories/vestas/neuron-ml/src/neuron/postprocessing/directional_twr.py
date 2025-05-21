import logging
import math
from typing import Dict

import numpy as np

from neuron.postprocessing.base import Postprocessor
from neuron.schemas.domain import POSTPROCESSOR_REQUIRED_TARGETS_MAP, Target, TargetValues

logger = logging.getLogger(__name__)


class DirectionalTwr(Postprocessor):

    """
    Postprocessor to calculate the directional tower fatigue load based on the method used in
    Vestas Site Check (VSC). The VSC calculation method is documented in DMS: 0004-0533 -
    section: "Calculation of Directional Tower Loads".

    The postprocessor calculates the maximum directional tower fatigue load for a given set of
    Mx and My values with associated correlation factor and wind distribution.

    Motivation: The tower bottom side-side moment (MyTwrBot_Twr_m*) is not directly used as a
    stand alone target for site suitability assessment in VSC, but rather used as an input for
    calculating the directional tower fatigue moment MrTwrBot_Twr_m* given a site wind distribution
    and correlation coefficient between the MyTwrBot_Twr (side-side) and MxTwrBot_Twr (fore-aft)
    moments. This postprocessor calculates the MrTwrBot_Twr_m* values for some predefined scenarios
    of wind distribution and correlation factor.

    Three different scenarios are considered. The motivation for choosing these scenarios is
    documented in the Neuron tech doc.: DMS: 0142-4355.

    1. corr_fac=-1, wind_dist: wind coming from one direction
        -> Highest possible Mr

    2. corr_fac=0, wind_dist: wind coming from 0 and 90 deg sectors 50/50 probability
        -> Lowest possible Mr

    3. corr_fac=-0.3, wind_dist: omnidirectional wind distribution -> Average case

    NB:
    - Mr inherits the groupby and weightby from Mx.
    - value_list_std is set to 0 for all calculated Mr values.

    The postprocessor requires the following targets:
    - MyTwrTop_Twr_m400
    - MxTwrTop_Twr_m400
    - MyTwrBot_Twr_m400
    - MxTwrBot_Twr_m400

    """

    name = "directional_twr"

    DIRECTIONS = range(0, 360, 10)
    N_DIRECTIONS = len(DIRECTIONS)
    SAFETY_FACTOR = 1.1

    def postprocess(self, target_values_dict: Dict[str, TargetValues]) -> Dict[str, TargetValues]:
        logger.info("Postprocessing data to produce directional tower fatigue load.")

        required_targets = POSTPROCESSOR_REQUIRED_TARGETS_MAP[self.name]
        tvl = target_values_dict

        if not all(target in target_values_dict for target in required_targets):
            missing_targets = [
                target for target in required_targets if target not in target_values_dict
            ]
            raise ValueError(
                f"Missing required targets {missing_targets} "
                + "to calculate directional tower fatigue load."
            )

        SCENARIO_REGISTRY = {
            # "target_name_prefix": DirectionalTwr function for the calculation of the target
            "Mr1": self.get_directional_twr_values_scenario_1,
            "Mr2": self.get_directional_twr_values_scenario_2,
            "Mr3": self.get_directional_twr_values_scenario_3,
        }

        tower_section_list = ["Top", "Bot"]
        for scenario_prefix, scenario in SCENARIO_REGISTRY.items():
            for tower_section in tower_section_list:
                target_name_mx = "MxTwr" + tower_section + "_Twr_m400"
                target_name_my = "MyTwr" + tower_section + "_Twr_m400"
                mr_400_values = scenario(
                    tvl[Target[target_name_mx]].values_as_np,
                    tvl[Target[target_name_my]].values_as_np * self.SAFETY_FACTOR,
                    4,
                )
                target_values_dict[
                    Target[scenario_prefix + "Twr" + tower_section + "_Twr_m400"]
                ] = TargetValues(
                    target_name=Target[scenario_prefix + "TwrTop_Twr_m400"],
                    value_list=mr_400_values.tolist(),
                    value_list_std=np.zeros_like(mr_400_values).tolist(),
                    groupby=target_values_dict[Target[target_name_mx]].groupby,
                    weightby=target_values_dict[Target[target_name_mx]].weightby,
                )

        return target_values_dict

    def calculate_directional_damage_to_global_sectors(
        self,
        sector_index: int,
        mx: np.array,
        my: np.array,
        corr_fac: float,
        sector_probability: float,
        wohler: float,
    ) -> Dict[int, np.array]:
        """
        Calculates the directional damage and maps it to the global sectors
        for a given sector index

        Implementation based on the corresponding VSC implementation documented in
        the VSC documentation (dms: 0004-0533)
        """

        global_sector_damage = {}
        twr_direction_damage = {}

        for i, direction in enumerate(self.DIRECTIONS):
            dir_rad = math.radians(direction)
            load = (
                (mx * math.cos(dir_rad)) ** 2
                + (my * math.sin(dir_rad)) ** 2
                - 2 * corr_fac * (mx * math.cos(dir_rad)) * (my * math.sin(dir_rad))
            ) ** 0.5
            damage = load**wohler * sector_probability

            twr_direction_damage[i] = damage

        for twr_direction_index, _ in enumerate(self.DIRECTIONS):
            if sector_index >= self.N_DIRECTIONS:
                sector_index = 0

            global_sector_damage[sector_index] = twr_direction_damage[twr_direction_index]

            sector_index += 1

        return global_sector_damage

    def calculate_mr(
        self, mx: np.array, my: np.array, corr_fac: float, wind_dist: np.array, wohler: float
    ) -> float:
        """
        Calculates the maximum directional tower fatigue load for a given set of
        Mx and My values with associated correlation factor and wind distribution.

        Implementation based on the corresponding VSC implementation documented in
        the VSC documentation (dms: 0004-0533)
        """

        dam_res = np.zeros((self.N_DIRECTIONS, len(mx)))

        for sector_index, _ in enumerate(self.DIRECTIONS):
            global_sector_damage = self.calculate_directional_damage_to_global_sectors(
                sector_index, mx, my, corr_fac, wind_dist[sector_index], wohler
            )
            for twr_direction, _ in enumerate(self.DIRECTIONS):
                dam_res[twr_direction, :] += global_sector_damage[twr_direction]

        # Find the maximum damage across directions
        max_dam_res = np.max(dam_res, axis=0)

        return max_dam_res ** (1 / wohler)

    def get_directional_twr_values_scenario_1(
        self, mx_values: np.array, my_values: np.array, wohler: float
    ) -> np.array:
        # Scenario 1: corr_fac=-1, wind_dist: wind coming from one direction
        # -> Highest possible Mr
        corr_fac = -1
        wind_dist = np.zeros(len(self.DIRECTIONS))
        wind_dist[0] = 1  # 0 degrees
        mr_values = self.calculate_mr(
            mx_values.flatten(), my_values.flatten(), corr_fac, wind_dist, wohler
        )
        return mr_values

    def get_directional_twr_values_scenario_2(
        self, mx_values: np.array, my_values: np.array, wohler: float
    ) -> np.array:
        # Scenario 2: corr_fac=0, wind_dist: wind coming from 0 and 90 deg sectors 50/50 prob
        # -> Lowest possible Mr
        corr_fac = 0
        wind_dist = np.zeros(len(self.DIRECTIONS))
        wind_dist[0] = 0.5  # 0 degrees
        wind_dist[self.DIRECTIONS.index(90)] = 0.5  # 90 degrees
        mr_values = self.calculate_mr(
            mx_values.flatten(), my_values.flatten(), corr_fac, wind_dist, wohler
        )
        return mr_values

    def get_directional_twr_values_scenario_3(
        self, mx_values: np.array, my_values: np.array, wohler: float
    ) -> np.array:
        # Scenario 3: corr_fac=-0.3, wind_dist: omnidirectional wind distribution -> Average case
        corr_fac = -0.3
        wind_dist = np.ones(len(self.DIRECTIONS)) / len(self.DIRECTIONS)

        mr_values = self.calculate_mr(
            mx_values.flatten(), my_values.flatten(), corr_fac, wind_dist, wohler
        )
        return mr_values
