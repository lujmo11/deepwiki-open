import shutil
from typing import Any, Dict, List, Type

import pandas as pd
import pytest

from neuron.models.target_models.base import Model
from neuron.models.target_models.constant import ConstantRegressor
from neuron.models.target_models.deep_gpr import DeepGPRPytorch
from neuron.models.target_models.feature_range_gpr import FeatureRangeGPRPytorch
from neuron.models.target_models.gpr import GPRPytorch
from neuron.models.target_models.gpr_sklearn import GPR, KERNEL_CLASS_REGISTRY
from neuron.models.target_models.linear_regressor import NeuronLinearRegressor

TRAINING_DATA_FILE_PATH_FAT = (
    "tests/data/training_pipeline/sample_training_data_performance_steps_fat.parquet"
)
TRAINING_DATA_FILE_PATH_EXT = (
    "tests/data/training_pipeline/sample_training_data_performance_steps_ext.parquet"
)
TRAINING_DATA_FIXED_FILE_PATH = "tests/data/training_pipeline/fixed_training_data.parquet"


@pytest.fixture(scope="function")
def mlflow_artifacts_folder(tmp_path_factory):  # noqa: ANN201, ANN001
    """Set up MLFlow to log to a temporary folder and return the folder path."""
    mlflow_artifacts_folder = tmp_path_factory.mktemp("mlflow_artifacts")
    yield str(mlflow_artifacts_folder)
    shutil.rmtree(mlflow_artifacts_folder)


@pytest.fixture(scope="function")
def training_df_fat() -> pd.DataFrame:
    return pd.read_parquet(TRAINING_DATA_FILE_PATH_FAT)


@pytest.fixture(scope="function")
def training_df_ext() -> pd.DataFrame:
    return pd.read_parquet(TRAINING_DATA_FILE_PATH_EXT)


@pytest.fixture(scope="function")
def training_df_sample_fat() -> pd.DataFrame:
    return pd.read_parquet(TRAINING_DATA_FILE_PATH_FAT).sample(10)


@pytest.fixture(scope="function")
def training_df_sample_ext() -> pd.DataFrame:
    return pd.read_parquet(TRAINING_DATA_FILE_PATH_EXT).sample(10)


@pytest.fixture(scope="function")
def training_data_file_path_fat() -> str:
    return TRAINING_DATA_FILE_PATH_FAT


@pytest.fixture(scope="function")
def training_data_file_path_ext() -> str:
    return TRAINING_DATA_FILE_PATH_EXT


@pytest.fixture(scope="function")
def training_df_fixed_data() -> pd.DataFrame:
    return pd.read_parquet(TRAINING_DATA_FIXED_FILE_PATH)


model_class_sample_configs = {
    NeuronLinearRegressor: [{}],
    GPRPytorch: [{"n_inducing_points": 20}],
    GPR: [{"kernel_name": name} for name in KERNEL_CLASS_REGISTRY.keys()],
    FeatureRangeGPRPytorch: [
        {"range_feature": "feature_0", "range_upper_bounds": [0.0], "n_inducing_points": 20}
    ],
    DeepGPRPytorch: [{"n_inducing_points": 20}],
    ConstantRegressor: [{}],
}


@pytest.fixture(scope="function")
def model_class_sample_configs_fixture() -> Dict[Type[Model], List[Dict[str, Any]]]:
    """Mapping between model class and a list of sample configurations for each model class."""
    return model_class_sample_configs
