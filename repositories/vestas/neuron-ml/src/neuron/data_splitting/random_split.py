from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from neuron.data_splitting.base import DataSplitter
from neuron.utils import set_seed


class RandomTestTrain(DataSplitter):
    name = "random_test_train_split"

    def __init__(self, test_size: float = 0.2):
        self.test_size = test_size

    def train_test_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return sklearn_train_test_split(data, test_size=self.test_size, random_state=set_seed())

    def validate_params(self) -> bool:
        return 0 < self.test_size < 1
