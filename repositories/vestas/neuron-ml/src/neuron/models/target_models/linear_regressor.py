import pickle
from pathlib import Path
from typing import Any, Dict, List, Self

import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor
from sklearn.linear_model import LinearRegression

from neuron.models.target_models.base import Model
from neuron.schemas.domain import TargetValues
from neuron.utils import check_columns_in_dataframe


class NeuronLinearRegressor(Model):
    name = "linear_regression"

    def __init__(self, features: List[str], target_col: str, fit_intercept: bool = True):
        """Linear regression model wrapper.

        We keep the features as an attribute so that the predict method
        works directly on a dataframe.

        Parameters
        ----------
        features : List[str]
            List of features to use for training
        target_col : str
            Name of the target column
        fit_intercept : bool, optional
            Whether to fit an intercept, by default True
        """
        lr = LinearRegression(fit_intercept=fit_intercept)
        self.regressor = MapieRegressor(lr, method="plus", cv=5)
        self._features = features
        self._target_col = target_col

    @property
    def target_col(self) -> str:
        return self._target_col

    @property
    def features(self) -> List[str]:
        return self._features

    def fit(self, df: pd.DataFrame) -> Self:
        check_columns_in_dataframe(df, self.features + [self.target_col])
        self.regressor.fit(df[self.features], df[self.target_col])
        return self

    def predict(
        self, df: pd.DataFrame, return_std: bool = False, return_grads: bool = False
    ) -> TargetValues:
        check_columns_in_dataframe(df, self.features)
        if return_std:
            alpha_conformal_prediction = 0.32
            predictions, y_pis = self.regressor.predict(
                df[self.features], alpha=alpha_conformal_prediction
            )
            average_uncertainty = np.mean(
                [predictions - y_pis[:, 0, 0], y_pis[:, 1, 0] - predictions], axis=0, dtype=float
            )
        else:
            predictions = self.regressor.predict(df[self.features])

        if return_grads:
            gradients_dict = self._calculate_gradients_finite_diff(df)

        return TargetValues(
            target_name=self.target_col,
            value_list=list(predictions),
            value_list_std=list(average_uncertainty) if return_std else None,
            gradients_dict=gradients_dict if return_grads else None,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "target_col": self.target_col,
            "fit_intercept": self.regressor.estimator.fit_intercept,
        }

    def construct_model_data_dict(self) -> Dict[str, Any]:
        model_data_dict = self.get_params()
        return model_data_dict

    def _save_model(self, folder_path: str) -> None:
        folder_path = Path(folder_path)
        model_artifact_path = folder_path / "model.pkl"
        if not folder_path.exists():
            folder_path.mkdir(parents=True)

        model_data = self.construct_model_data_dict()

        # Save the MapieRegressor object to the pickle file
        model_data["regressor"] = self.regressor

        with open(model_artifact_path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, folder_path: str) -> Self:
        model_artifact_path = Path(folder_path) / "model.pkl"
        with open(model_artifact_path, "rb") as f:
            model_dict = pickle.load(f)

        model_instance = cls(
            features=model_dict["features"],
            target_col=model_dict["target_col"],
            fit_intercept=model_dict["fit_intercept"],
        )

        # Load the MapieRegressor from the pickle file
        model_instance.regressor = model_dict["regressor"]

        return model_instance
