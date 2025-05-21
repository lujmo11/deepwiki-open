import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from neuron.models.scaler_registry import DataScaler
from neuron.models.target_models.base import Model
from neuron.models.target_models.gpr import GPRPytorch
from neuron.schemas.domain import TargetValues
from neuron.utils import check_columns_in_dataframe

logger = logging.getLogger(__name__)


class Range(BaseModel):
    lower_bound: float | None = None
    upper_bound: float | None = None

    @model_validator(mode="after")
    def check_for_increasing_values(self) -> Self:
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound >= self.upper_bound:
                raise ValueError(
                    "'ranges' used for feature range splitting should be defined in ascending order"
                )
        return self

    def __repr__(self) -> str:
        return f"Range(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"

    def __hash__(self) -> int:
        return hash((self.lower_bound, self.upper_bound))

    def series_in_range(self, series: pd.Series) -> List[bool]:
        """Get list of booleans indicating whether the row of the series is in the range.

        The interval is inclusive on the right side.
        """
        return series.between(
            left=self.lower_bound or -np.inf, right=self.upper_bound or np.inf, inclusive="right"
        ).to_list()


class FeatureRangeGPRPytorch(Model):
    name = "feature_range_gpr"

    def __init__(
        self,
        features: List[str],
        target_col: str,
        range_feature: str,
        range_upper_bounds: List[float],
        minimum_training_size: int = 10,
        n_inducing_points: int = 300,
        nu: float = 2.5,
        training_iter: int = 500,
        es_patience: int = 40,
        learning_rate: float = 0.02,
        validation_size: float = 0.1,
        feature_scaler_name: DataScaler = DataScaler.ROBUST_SCALER,
        target_scaler_name: DataScaler = DataScaler.ROBUST_SCALER,
    ):
        self._features = features
        self._target_col = target_col

        # GPRPytorch parameters
        self.n_inducing_points = n_inducing_points
        self.nu = nu
        self.training_iter = training_iter
        self.es_patience = es_patience
        self.learning_rate = learning_rate
        self.validation_size = validation_size
        self.feature_scaler_name = feature_scaler_name
        self.target_scaler_name = target_scaler_name

        # Feature range parameters
        self.range_feature = range_feature
        self.minimum_training_size = minimum_training_size
        self.range_upper_bounds = range_upper_bounds

        models: Dict[Range, GPRPytorch] = {
            Range(lower_bound=None, upper_bound=range_upper_bounds[0]): GPRPytorch(
                **self.get_gpr_model_parameters()
            ),
            Range(lower_bound=range_upper_bounds[-1], upper_bound=None): GPRPytorch(
                **self.get_gpr_model_parameters()
            ),
        }

        for lower_bound, upper_bound in zip(
            range_upper_bounds[0:-1], range_upper_bounds[1:], strict=True
        ):
            models[Range(lower_bound=lower_bound, upper_bound=upper_bound)] = GPRPytorch(
                **self.get_gpr_model_parameters()
            )
        self.models = models

    @property
    def target_col(self) -> str:
        return self._target_col

    @property
    def features(self) -> List[str]:
        return self._features

    def fit(self, df: pd.DataFrame) -> Self:
        # Validating data before initiating model training
        check_columns_in_dataframe(df, self.features + [self.target_col])

        for _range, _model in self.models.items():
            feature_in_range = _range.series_in_range(df[self.range_feature])
            df_in_feature_range = df[feature_in_range]

            if len(df_in_feature_range) < self.minimum_training_size:
                logger.error(
                    f"The number of training data is below the minimum "
                    f"requirement of {self.minimum_training_size} in the range {_range}."
                )
                raise ValueError(
                    f"The number of training data is below the minimum "
                    f"requirement of {self.minimum_training_size} in the range {_range}."
                )

        # Model training
        for _range, model in self.models.items():
            feature_in_range = _range.series_in_range(df[self.range_feature])
            df_in_feature_range = df[feature_in_range]
            model.fit(df_in_feature_range)

        return self

    def predict(
        self, df: pd.DataFrame, return_std: bool = False, return_grads: bool = False
    ) -> TargetValues:
        check_columns_in_dataframe(df, self.features)
        dataf = df.copy()

        if return_grads:
            # Initialize gradient columns
            gradient_columns = ["gradients_" + feature for feature in self.features]
            for grad_col in gradient_columns:
                dataf[grad_col] = np.nan  # Initialize with NaN

        for _range, model in self.models.items():
            feature_in_range = _range.series_in_range(dataf[self.range_feature])
            df_in_feature_range = dataf.loc[feature_in_range]

            if df_in_feature_range.empty:
                continue

            target_predictions_for_range = model.predict(
                df_in_feature_range, return_std, return_grads
            )

            # Assign predictions using the indices of df_in_feature_range
            dataf.loc[
                df_in_feature_range.index, "predictions"
            ] = target_predictions_for_range.value_list

            if return_std:
                dataf.loc[
                    df_in_feature_range.index, "predictions_std"
                ] = target_predictions_for_range.value_list_std

            if return_grads:
                # Stack gradients for each feature into a 2D array
                gradients_array = np.column_stack(
                    [
                        target_predictions_for_range.gradients_dict[feature]
                        for feature in self.features
                    ]
                )
                gradient_columns = ["gradients_" + feature for feature in self.features]
                dataf.loc[df_in_feature_range.index, gradient_columns] = gradients_array

        if return_grads:
            gradients_dict = {}
            for feature in self.features:
                # Ensure the gradients are floats
                gradients_dict[feature] = dataf["gradients_" + feature].astype(float).tolist()

        return TargetValues(
            target_name=self.target_col,
            value_list=dataf["predictions"].tolist(),
            value_list_std=dataf["predictions_std"].tolist() if return_std else None,
            gradients_dict=gradients_dict if return_grads else None,
        )

    def get_gpr_model_parameters(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "target_col": self.target_col,
            "n_inducing_points": self.n_inducing_points,
            "nu": self.nu,
            "training_iter": self.training_iter,
            "es_patience": self.es_patience,
            "learning_rate": self.learning_rate,
            "validation_size": self.validation_size,
            "feature_scaler_name": self.feature_scaler_name,
            "target_scaler_name": self.target_scaler_name,
        }

    def get_params(self) -> Dict[str, Any]:
        return self.get_gpr_model_parameters() | {
            "range_feature": self.range_feature,
            "range_upper_bounds": self.range_upper_bounds,
        }

    def _save_model(self, folder_path: str) -> None:
        folder_path = Path(folder_path)
        model_artifact_path = folder_path / "model.pkl"

        if not folder_path.exists():
            folder_path.mkdir(parents=True)

        model_data_dict: Dict[str, Any] = {"range_model_params": self.get_params()}
        range_models_data_dict: Dict[Range, Any] = {}
        for _range in self.models:
            range_models_data_dict[_range] = self.models[_range].construct_model_data()
        model_data_dict["range_models_data_dict"] = range_models_data_dict

        with open(model_artifact_path, "wb") as f:
            pickle.dump(model_data_dict, f)

    @classmethod
    def load_model(cls, folder_path: str) -> Self:
        folder_path = Path(folder_path)
        model_artifact_path = folder_path / "model.pkl"
        with open(model_artifact_path, "rb") as f:
            models_data_dict = pickle.load(f)

        feature_range_model = FeatureRangeGPRPytorch(**models_data_dict["range_model_params"])
        for _range in models_data_dict["range_models_data_dict"]:
            feature_range_model.models[_range] = GPRPytorch.load_model_from_dict(
                models_data_dict["range_models_data_dict"][_range]
            )
        return feature_range_model
