"""The module contains a load case model pipeline class
that combines multiple target models into one model.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Self, Union

import numpy as np
import pandas as pd

from neuron.data_validation.exceptions import DataValidationError
from neuron.data_validation.validation import (
    check_required_input_features_for_load_case,
    check_required_safe_domain_features_for_load_case,
)
from neuron.models.target_models import registry
from neuron.models.target_models.base import Model
from neuron.postprocessing.base import Postprocessor
from neuron.postprocessing.registry import get_postprocessor
from neuron.preprocessing.base import Preprocessor
from neuron.preprocessing.registry import get_preprocessor
from neuron.safe_domain.safe_domain_validator import SafeDomainValidator
from neuron.schemas.domain import (
    FeatureType,
    LoadCase,
    TargetValues,
    TurbineVariant,
)
from neuron.schemas.training_run_config import LoadCaseModelConfig

logger = logging.getLogger(__name__)


LOAD_CASE_MODEL_METADATA_FILE_NAME = "load_case.json"
TURBINE_VARIANT_METADATA_FILE_NAME = "turbine_variant.json"
LOAD_CASE_PREPROCESSING_FILE_NAME = "preprocessor.pkl"
LOAD_CASE_SAFE_DOMAIN_VALIDATOR_FILE_NAME = "safe_domain_validator.json"

LoadCasePredictions = Dict[str, TargetValues]


class LoadCaseModelPipeline:
    """Class that combines multiple target models into one model for a load case"""

    def __init__(
        self,
        target_models: Dict[str, Model],
        load_case: LoadCase,
        turbine_variant: TurbineVariant,
        postprocessor: Postprocessor,
        preprocessors: List[Preprocessor],
        safe_domain_validator: Union[SafeDomainValidator, None] = None,
    ) -> None:
        self.models = target_models
        self.load_case = load_case
        self.turbine_variant = turbine_variant
        self.postprocessor = postprocessor
        self.preprocessors = preprocessors
        self.safe_domain_validator = safe_domain_validator

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data using all the preprocessors"""
        preprocessed_df = df.copy()
        for preprocessor in self.preprocessors:
            preprocessed_df = preprocessor.preprocess(preprocessed_df)
        return preprocessed_df

    def fit(self, df: pd.DataFrame) -> Self:
        """
        - Preprocess data
        - Fit safe domain validator
        - Fit target models"""
        feature_df = self.preprocess(df)
        if self.safe_domain_validator:
            self.safe_domain_validator.fit(feature_df)

        for target_name, model in self.models.items():
            logger.info(f"Fitting model for target {target_name}")
            model.fit(feature_df)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        return_std: bool,
        targets: Union[List[str], None] = None,
        grad_features: Union[List[str], None] = None,
    ) -> LoadCasePredictions:
        """Preprocess data and predict for all the target models specified in the targets list.
        If targets list is None or empty, then predict for all target models"""
        feature_df = self.preprocess(df)
        self._validate_model_input(feature_df)

        target_predictions = {}

        if targets is None or len(targets) == 0:
            target_list = self.models.keys()
        else:
            if not all(target in self.models.keys() for target in targets):
                raise DataValidationError("One or more target names are not present in the model")
            target_list = targets

        if grad_features is None or len(grad_features) == 0:
            return_grads = False
        else:
            features_not_in_model = set(grad_features) - set(self.load_case.get_feature_name_list())
            if len(features_not_in_model) > 0:
                raise DataValidationError(
                    "The following feature names in grad_features are not present in the model: "
                    + f"{features_not_in_model}"
                )
            return_grads = True

        for target_name in target_list:
            model = self.models[target_name]
            predictions = model.predict(feature_df, return_std, return_grads=return_grads)

            # only return gradients for specified features
            if return_grads:
                gradients_dict = {}
                for feature in grad_features:
                    gradients_dict[feature] = predictions.gradients_dict[feature]

                target_predictions[target_name] = TargetValues(
                    target_name=target_name,
                    value_list=predictions.value_list,
                    value_list_std=predictions.value_list_std,
                    gradients_dict=gradients_dict,
                )
            else:
                target_predictions[target_name] = predictions

        return target_predictions

    def interpolation_domain_validation(self, df: pd.DataFrame) -> List[int]:
        """Validate data against safe domain"""
        self._validate_safe_domain_features(df)
        if self.safe_domain_validator:
            df_interp = self.safe_domain_validator.validate_interpolation(df)
            # Create envelope column that contains 1 if all features are within safe domain, else 0
            envelope_interp = df_interp.apply(
                lambda row: 0 if (row == 0).any() else 1, axis=1
            ).tolist()
        else:
            raise ValueError("No safe domain validator found.")
        return envelope_interp

    def extrapolation_domain_validation(self, df: pd.DataFrame) -> List[int]:
        """Validate data against safe domain"""
        self._validate_safe_domain_features(df)
        if self.safe_domain_validator:
            df_extrap = self.safe_domain_validator.validate_extrapolation(df)
            # Create envelope column that contains 1 if all features are within safe domain, else 0
            envelope_extrap = df_extrap.apply(
                lambda row: 0 if (row == 0).any() else 1, axis=1
            ).tolist()
        else:
            raise ValueError("No safe domain validator found.")
        return envelope_extrap

    def save_model(self, folder_path: str) -> None:
        for target_col, model in self.models.items():
            target_model_folder_path = Path(folder_path) / target_col
            model.save_model(folder_path=str(target_model_folder_path))

        with open(Path(folder_path) / LOAD_CASE_MODEL_METADATA_FILE_NAME, "w") as fh:
            json.dump(self.load_case.model_dump(), fh)

        with open(Path(folder_path) / TURBINE_VARIANT_METADATA_FILE_NAME, "w") as fh:
            json.dump(self.turbine_variant.model_dump(), fh)

        if self.safe_domain_validator:
            model_path = Path(folder_path) / LOAD_CASE_SAFE_DOMAIN_VALIDATOR_FILE_NAME
            self.safe_domain_validator.save(path=str(model_path))

    @classmethod
    def load_model(cls, folder_path: str) -> Self:
        if not Path(folder_path).exists():
            raise FileNotFoundError(f"Folder path {folder_path} does not exist.")

        with open(Path(folder_path) / LOAD_CASE_MODEL_METADATA_FILE_NAME, "r") as fh:
            load_case = LoadCase(**json.load(fh))

        with open(Path(folder_path) / TURBINE_VARIANT_METADATA_FILE_NAME, "r") as fh:
            turbine_variant = TurbineVariant(**json.load(fh))

        target_models = {}
        target_model_folder_paths = [p for p in Path(folder_path).glob("*") if p.is_dir()]
        for target_model_folder_path in target_model_folder_paths:
            target_model = registry.load_target_model(
                target_model_folder_path=str(target_model_folder_path),
            )
            target_models[target_model.target_col] = target_model

        postprocessor = get_postprocessor(
            postprocessor_name=load_case.postprocessor,
        )
        preprocessors = [
            get_preprocessor(engineered_feature_name=feat.name, turbine_variant=turbine_variant)
            for feat in load_case.feature_list.features
            if feat.feature_type == FeatureType.ENGINEERED
        ]

        safe_domain_validator_path = Path(folder_path) / LOAD_CASE_SAFE_DOMAIN_VALIDATOR_FILE_NAME
        if safe_domain_validator_path.exists():
            safe_domain_validator = SafeDomainValidator.load(path=str(safe_domain_validator_path))
        else:
            safe_domain_validator = None

        return LoadCaseModelPipeline(
            target_models=target_models,
            load_case=load_case,
            turbine_variant=turbine_variant,
            postprocessor=postprocessor,
            preprocessors=preprocessors,
            safe_domain_validator=safe_domain_validator,
        )

    def _validate_model_input(self, df: pd.DataFrame) -> None:
        """Validate input data against the load case model"""
        check_required_input_features_for_load_case(df=df, load_case=self.load_case)

    def _validate_safe_domain_features(self, df: pd.DataFrame) -> None:
        """Validate input data against the load case model"""
        check_required_safe_domain_features_for_load_case(df=df, load_case=self.load_case)

    def postprocess(
        self, loadcase_target_values: Dict[str, TargetValues]
    ) -> Dict[str, TargetValues]:
        """Postprocess target values"""

        return self.postprocessor.postprocess(loadcase_target_values)

    def run_speed_test(
        self, test_df: pd.DataFrame, num_rows: int = 1000, num_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Runs a prediction speed test by creating an artificial dataset with the model features
        and a fixed number of rows.
        """

        feature_names = self.load_case.get_feature_name_list()
        n_targets = len(self.models)

        df_num_rows = pd.concat([test_df.iloc[[0]]] * num_rows, ignore_index=True)

        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            self.predict(df_num_rows, return_std=True)
            end_time = time.time()
            times.append(end_time - start_time)

        mean_time = np.mean(times)
        std_time = np.std(times)

        logger.info(
            f"Prediction time over {num_iterations} runs with {num_rows} rows: "
            f"mean = {mean_time:.6f}s, std = {std_time:.6f}s"
        )

        results_dict = {
            "n_rows": num_rows,
            "n_iterations": num_iterations,
            "mean_time": mean_time,
            "std_time": std_time,
            "mean_time_per_target": mean_time / n_targets,
            "n_model_features": len(feature_names),
            "n_targets": n_targets,
        }

        return results_dict


def initialize_load_case_model_from_config(
    turbine_variant: TurbineVariant,
    load_case: LoadCase,
    load_case_model_config: LoadCaseModelConfig,
) -> LoadCaseModelPipeline:
    """Initialize a load case model from a training run config"""
    target_models = {}
    for target in load_case.target_list.targets:
        target_model_config = load_case_model_config.get_target_model_config(target)

        target_model_initialization_config = registry.TargetModelInitializationConfig(
            name=target_model_config.name,
            params=target_model_config.params,
            features=[
                feature.name
                for feature in load_case.feature_list.features
                if feature.is_model_input
            ],
            target_col=target,
        )
        logger.info(
            f"Initializing load_case_model of type {target_model_initialization_config.name} "
            f"for target {target_model_initialization_config.target_col}."
        )
        target_models[
            target_model_initialization_config.target_col
        ] = registry.initialize_target_model(model_config=target_model_initialization_config)

    postprocessor = get_postprocessor(postprocessor_name=load_case.postprocessor)
    preprocessors = [
        get_preprocessor(engineered_feature_name=feat.name, turbine_variant=turbine_variant)
        for feat in load_case.feature_list.features
        if feat.feature_type == FeatureType.ENGINEERED
    ]

    safe_domain_validator = SafeDomainValidator(
        features=load_case.feature_list.features,
    )

    logger.info("Initializing load case load_case_model from target models.")
    return LoadCaseModelPipeline(
        target_models=target_models,
        load_case=load_case,
        turbine_variant=turbine_variant,
        postprocessor=postprocessor,
        preprocessors=preprocessors,
        safe_domain_validator=safe_domain_validator,
    )
