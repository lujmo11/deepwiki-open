from typing import Dict, List, Self, Tuple

import pandas as pd

from neuron.safe_domain.validators.base import Validator
from neuron.schemas.domain import Feature


class RangeValidator(Validator):
    """To ensure backwards compatibility, we should avoid refactoring this class.

    If the functionality of this validator is insufficient, a new validator should be created.
    """

    def __init__(self):
        self.feature_range_dict_interp = {}
        self.feature_range_dict_extrap = {}

    def fit(self, df: pd.DataFrame, features: List[Feature]) -> None:
        for feature in features:
            min = df[feature.name].min()
            max = df[feature.name].max()
            self.feature_range_dict_interp[feature.name] = (min, max)

            if feature.extrapolation_domain_offset is not None:
                min_extrap = min - feature.extrapolation_domain_offset
                max_extrap = max + feature.extrapolation_domain_offset
                self.feature_range_dict_extrap[feature.name] = (min_extrap, max_extrap)
            else:
                self.feature_range_dict_extrap[feature.name] = (min, max)

    def validate_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if values are within the range for each feature.

        1: In range, 0: Out of range
        """
        df_out_interp = pd.DataFrame()
        for feature in df:
            df_out_interp[feature] = (
                (df[feature] >= self.feature_range_dict_interp[feature][0])
                & (df[feature] <= self.feature_range_dict_interp[feature][1])
            ).astype(int)

        return df_out_interp

    def validate_extrapolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if values are within the range for each feature.

        1: In range, 0: Out of range
        """
        df_out_extrap = pd.DataFrame()
        for feature in df:
            df_out_extrap[feature] = (
                (df[feature] >= self.feature_range_dict_extrap[feature][0])
                & (df[feature] <= self.feature_range_dict_extrap[feature][1])
            ).astype(int)

        return df_out_extrap

    def get_model_data(self) -> Dict:
        return {
            "feature_range_dict_interp": self.feature_range_dict_interp,
            "feature_range_dict_extrap": self.feature_range_dict_extrap,
        }

    def _get_ranges(self, feature: str, method: str = "interp") -> Tuple:
        if method == "interp":
            range_values = self.feature_range_dict_interp[feature]
        elif method == "extrap":
            range_values = self.feature_range_dict_extrap[feature]

        return range_values

    @classmethod
    def load(cls, model_data: Dict) -> Self:
        validator = cls()
        validator.feature_range_dict_interp = model_data["feature_range_dict_interp"]
        validator.feature_range_dict_extrap = model_data["feature_range_dict_extrap"]
        return validator
