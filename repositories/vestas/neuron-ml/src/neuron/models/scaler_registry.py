"""Supporting module that acts as a registry for data scalers."""


from enum import StrEnum
from typing import Dict

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class DataScaler(StrEnum):
    """List of allowed data scalers"""

    MIN_MAX_SCALER = "min_max_scaler"
    ROBUST_SCALER = "robust_scaler"
    STANDARD_SCALER = "standard_scaler"


DATA_SCALER_REGISTRY: Dict[DataScaler, TransformerMixin] = {
    DataScaler.MIN_MAX_SCALER: MinMaxScaler,
    DataScaler.ROBUST_SCALER: RobustScaler,
    DataScaler.STANDARD_SCALER: StandardScaler,
}


class ScalerNotRegisteredError(Exception):
    """Exception raised when the model name is not in the registry"""

    pass


def get_registered_scaler_from_name(name: DataScaler) -> TransformerMixin:
    try:
        scaler_class = DATA_SCALER_REGISTRY[name]
        return scaler_class()
    except KeyError as e:
        raise ScalerNotRegisteredError(
            f"The data scaler {name} is not the scaler registry. "
            f"Available scalers are: {list(DATA_SCALER_REGISTRY.keys())}"
        ) from e
