"""Module that acts as a registry for target models.

The module is used to
- Initialize a target model from a model config.
This is used in the training pipeline to get a new untrained model.
- Load a target model from a folder path.
This is used to load a trained model from a folder path, without knowing the model class upfront.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Type

from pydantic import BaseModel, field_validator

from neuron.models.target_models.base import Model, ModelMetadata
from neuron.models.target_models.constant import ConstantRegressor
from neuron.models.target_models.deep_gpr import DeepGPRPytorch
from neuron.models.target_models.feature_range_gpr import FeatureRangeGPRPytorch
from neuron.models.target_models.gpr import GPRPytorch
from neuron.models.target_models.gpr_sklearn import GPR
from neuron.models.target_models.linear_regressor import NeuronLinearRegressor

TARGET_MODEL_REGISTER: Dict[str, Type[Model]] = {
    NeuronLinearRegressor.name: NeuronLinearRegressor,
    GPRPytorch.name: GPRPytorch,
    GPR.name: GPR,
    FeatureRangeGPRPytorch.name: FeatureRangeGPRPytorch,
    DeepGPRPytorch.name: DeepGPRPytorch,
    ConstantRegressor.name: ConstantRegressor,
}
TARGET_MODEL_METADATA_FILE_NAME = "target_model_metadata.json"


class ModelNotRegisteredError(Exception):
    """Exception raised when the model name is not in the registry"""

    pass


def get_registered_model_class_from_name(name: str) -> Type[Model]:
    """Gets the registered model class from the model name.

    Raises:
        ModelNotRegisteredError: If the model name is not in the registry
    """
    try:
        return TARGET_MODEL_REGISTER[name]
    except KeyError as e:
        raise ModelNotRegisteredError(
            f"Model name {name} not in target model registry. "
            f"Available models are: {list(TARGET_MODEL_REGISTER.keys())}"
        ) from e


class TargetModelInitializationConfig(BaseModel):
    name: str
    params: Dict[str, Any]
    features: List[str]
    target_col: str

    @field_validator("name")
    @classmethod
    def model_name_must_be_registered(cls, name: str) -> str:
        if name not in TARGET_MODEL_REGISTER.keys():
            raise ValueError(
                f"Model name in model {name=} "
                f"not in list of registered models: {TARGET_MODEL_REGISTER.keys()}"
            )
        return name


def initialize_target_model(model_config: TargetModelInitializationConfig) -> Model:
    """Initializes a target model class instance from a model config"""
    model_class = get_registered_model_class_from_name(model_config.name)
    model_init_vars = (
        model_config.params
        | {"features": model_config.features}
        | {"target_col": model_config.target_col}
    )
    return model_class(**model_init_vars)  # type: ignore


def load_target_model(target_model_folder_path: str) -> Model:
    """Load a target model from a folder path.

    The folder path must contain a model metadata file. This contains the information on which model
    class to load and the model parameters. The model class must be registered in the target model.
    """
    if not Path(target_model_folder_path).exists():
        raise FileNotFoundError(
            f"Target model folder path {target_model_folder_path} does not exist"
        )
    if not Path(target_model_folder_path).is_dir():
        raise FileNotFoundError(
            f"Target model folder path {target_model_folder_path} is not a directory"
        )
    model_metadata_path = Path(target_model_folder_path) / TARGET_MODEL_METADATA_FILE_NAME
    if not model_metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata file not found for model in {target_model_folder_path}"
        )
    with open(model_metadata_path, "r") as fh:
        model_metadata_dict = json.load(fh)
        try:
            model_metadata = ModelMetadata(**model_metadata_dict)
        except Exception as e:
            raise ValueError(f"Error loading model metadata from {model_metadata_path}") from e
    model_class = get_registered_model_class_from_name(name=model_metadata.model_class_name)
    return model_class.load_model(str(Path(target_model_folder_path)))
