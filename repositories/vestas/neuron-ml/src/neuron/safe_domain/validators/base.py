from abc import ABC, abstractmethod
from typing import Dict, List, Self

import pandas as pd

from neuron.schemas.domain import Feature


class Validator(ABC):
    """Protocol for safe domain validation methods"""

    @abstractmethod
    def fit(self, df: pd.DataFrame, features: List[Feature]) -> None:
        pass

    @abstractmethod
    def validate_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def validate_extrapolation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_model_data(self) -> Dict:
        pass

    @abstractmethod
    def _get_ranges(self, feature_name: str, method: str = "interp"):
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_data: Dict) -> Self:
        pass
