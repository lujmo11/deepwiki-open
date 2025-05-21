from typing import Dict, List, Self

import numpy as np
import pandas as pd

from neuron.safe_domain.validators.base import Validator
from neuron.schemas.domain import Feature


class DiscreteValidator(Validator):
    """To ensure backwards compatibility, we should avoid refactoring this class.

    If the functionality of this validator is insufficient, a new validator should be created.
    """

    discrete_values_dict: Dict[str, np.ndarray]

    def fit(self, df: pd.DataFrame, features: List[Feature]) -> None:
        """Get unique categories from dataframe"""
        self.discrete_values_dict = {
            feature.name: df[feature.name].unique() for feature in features
        }

    def validate_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if values are in the set of values for the each feature .

        1: In set, 0: Not in set
        """
        df_out = pd.DataFrame()
        for feature in df:
            df_out[feature] = df[feature].isin(self.discrete_values_dict[feature]).astype(int)

        return df_out

    def validate_extrapolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if values are in the set of values for the each feature .

        1: In set, 0: Not in set

        Extrapolation is not applicable for discrete features in the context of Neuron,
        therefore the extrapolation validation output is set to be identical to the
        interpolaiton validation output. In practice a feature with a discrete validator
        specified should not have an extrapolation offset defined.
        """
        return self.validate_interpolation(df)

    def get_model_data(self) -> Dict:
        model_data = {
            feature: values.tolist() for feature, values in self.discrete_values_dict.items()
        }
        return {"discrete_values_dict": model_data}

    def _get_ranges(self, feature_name: str, method: str = "interp") -> None:
        """
        Discrete validator does not have ranges, the allowed values
        correspond to the exact feature values.
        """

    @classmethod
    def load(cls, model_data: Dict) -> Self:
        validator = cls()
        validator.discrete_values_dict = {
            column: np.array(values)
            for column, values in model_data["discrete_values_dict"].items()
        }
        return validator
