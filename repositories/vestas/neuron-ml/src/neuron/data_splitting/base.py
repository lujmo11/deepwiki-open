from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd


class DataSplitter(ABC):
    """Protocol for data splitter class"""

    @abstractmethod
    def train_test_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train and test dataframes"""
        pass

    @abstractmethod
    def validate_params(self) -> bool:
        """Validate the parameters for the data splitter"""
        pass
