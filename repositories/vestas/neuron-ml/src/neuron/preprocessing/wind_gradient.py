import logging

import pandas as pd
from scipy.optimize import root_scalar

from neuron.data_validation.exceptions import DataValidationError
from neuron.preprocessing.base import Preprocessor
from neuron.schemas.domain import REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION, TurbineVariant

logger = logging.getLogger(__name__)


class WindGradientPreprocessor(Preprocessor):
    name = "wind_gradient"
    feature_name = "wnd_grad"

    def __init__(self, turbine_variant: TurbineVariant):
        self.turbine = turbine_variant
        self.diameter = float(turbine_variant.rotor_diameter)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Bypass preprocessing if wnd_grad is provided as input
        if self.feature_name in df.columns:
            logger.info(
                f"Feature {self.feature_name} is provided as input - "
                "Wind gradient preprocessor bypassed and vexp is calculated "
                "based on the provided wind gradient."
            )

            required_columns = REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION["vexp"]
            if not all(item in df.columns for item in required_columns):
                raise DataValidationError(
                    f"Missing required columns {required_columns} in dataframe to "
                    f"calculate wind shear exponent."
                )

            # Calculate vexp based on the provided wind gradient
            if "vexp" in df.columns:
                df.drop(columns=["vexp"])
            df["vexp"] = df.apply(
                lambda row: self._calculate_vexp(
                    twr_hh=float(row["twr_hh"]),
                    ws=float(row["ws"]),
                    wnd_grad=float(row[self.feature_name]),
                ),
                axis=1,
            )
        else:
            # Calculate wind gradient based on the provided vexp
            required_columns = REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION[self.feature_name]
            if not all(item in df.columns for item in required_columns):
                raise DataValidationError(
                    f"Missing required columns {required_columns} in dataframe to "
                    f"calculate wind gradient."
                )

            df[self.feature_name] = df.apply(
                lambda row: self._calculate_wnd_grad(
                    twr_hh=float(row["twr_hh"]),
                    ws=float(row["ws"]),
                    vexp=float(row["vexp"]),
                ),
                axis=1,
            )

        return df

    def _calculate_vexp(self, twr_hh: float, ws: float, wnd_grad: float) -> float:
        """
        Calculate the wind shear exponent given the tower hub height,
        wind speed at hub height, rotor diameter, and wind gradient over the rotor.

        """
        if ws <= 0:
            raise DataValidationError("Wind speed must have a positive value.")

        if (self.diameter / 2) >= twr_hh:
            raise DataValidationError("Rotor radius is larger than tower hub height.")

        # Approximated vexp to sanity check the input params
        vexp_estimate = (wnd_grad * twr_hh) / ws
        if abs(vexp_estimate) > 5:
            raise DataValidationError(
                "Invalid input parameters: An estimate of the wind shear exponent "
                f"({vexp_estimate:.2f}) is outside a sensible range (+/- 5) for the provided "
                "input parameters: "
                f"wind gradient: {wnd_grad}, ws: {ws}, twr_hh: {twr_hh}, diameter: {self.diameter}."
                "The wind shear exponent cannot be calculated with the given parameters."
            )

        z_top = twr_hh + (self.diameter / 2)
        z_bottom = twr_hh - (self.diameter / 2)

        """
        The math:
        
        The wind speeds at the top and bottom of the rotor:
        ws_top = ws * (z_top / twr_hh) ** vexp
        ws_bottom = ws * (z_bottom / twr_hh) ** vexp

        The wind gradient across the rotor:
        wnd_grad = (ws_top - ws_bottom) / diameter
        wnd_grad = [ws * ((z_top / twr_hh) ** vexp - (z_bottom / twr_hh) ** vexp)] / diameter

        Rearranging:
        (wnd_grad * diameter) / ws = (z_top / twr_hh) ** vexp - (z_bottom / twr_hh) ** vexp
        (z_top / twr_hh) ** vexp - (z_bottom / twr_hh) ** vexp - (wnd_grad * diameter) / ws = 0
        
        """

        def _func(vexp_candidate: float) -> float:
            return (
                (z_top / twr_hh) ** vexp_candidate
                - (z_bottom / twr_hh) ** vexp_candidate
                - (wnd_grad * self.diameter) / ws
            )

        # Solving for vexp
        sol = root_scalar(_func, bracket=[-10, 10], method="brentq")
        if sol.converged:
            vexp = sol.root
        else:
            raise DataValidationError(
                f"Could not calculate wind shear exponent for the given parameters:"
                f"twr_hh: {twr_hh}, ws: {ws}, diameter: "
                f"{self.diameter}, wnd_grad: {wnd_grad}"
            )

        return vexp

    def _calculate_wnd_grad(self, twr_hh: float, ws: float, vexp: float) -> float:
        """
        Calculate the wind gradient given the tower hub height, wind speed at hub height,
        rotor diameter, and wind shear exponent.
        """
        if ws <= 0:
            raise DataValidationError("Wind speed must have a positive value.")
        if (self.diameter / 2) >= twr_hh:
            raise DataValidationError("Rotor radius is larger than tower hub height.")

        z_top = twr_hh + (self.diameter / 2)
        z_bottom = twr_hh - (self.diameter / 2)

        # Calculate wind speeds at the top and bottom of the rotor
        ws_top = ws * (z_top / twr_hh) ** vexp
        ws_bottom = ws * (z_bottom / twr_hh) ** vexp

        # Calculate the wind gradient
        wnd_grad = (ws_top - ws_bottom) / self.diameter

        return wnd_grad
