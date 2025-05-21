from typing import Any, Dict, Type

from neuron.data_splitting.base import DataSplitter
from neuron.data_splitting.feature_based import FeatureBasedSplitter
from neuron.data_splitting.random_split import RandomTestTrain

DATA_SPLITTING_REGISTRY: Dict[str, Type[DataSplitter]] = {
    "random_test_train_split": RandomTestTrain,
    "feature_group_split": FeatureBasedSplitter,
}


class DataSplitterNotRegisteredError(Exception):
    """Exception raised when the method to split the data is not registered."""

    pass


def get_data_splitter(name: str, params: Dict[str, Any]) -> DataSplitter:
    """Gets the registered data splitting method class from the method name."""

    try:
        DATA_SPLITTING_REGISTRY[name]
    except KeyError as e:
        raise DataSplitterNotRegisteredError(
            f"Data splitting method {name} not in method registry. "
            f"Available models are: {list(DATA_SPLITTING_REGISTRY.keys())}"
        ) from e

    data_split_class = DATA_SPLITTING_REGISTRY.get(name)
    return data_split_class(**params)  # type: ignore
