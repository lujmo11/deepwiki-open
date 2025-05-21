import pickle
from pathlib import Path
from typing import Any, Dict, List, Self

import pandas as pd
from sklearn.dummy import DummyRegressor

from neuron.models.target_models.base import Model
from neuron.schemas.domain import TargetValues
from neuron.utils import check_columns_in_dataframe


class ConstantRegressor(Model):
    name = "constant_regression"

    def __init__(self, features: List[str], target_col: str, constant: float = 0.0) -> None:
        """Regressor that will always output a constant value

        Arguments:
            features: Only here for compatability.
            target_col: Target column name.
            constant: The value that will be outputted from `predict`.
        """
        super().__init__()
        self.regressor = DummyRegressor(strategy="constant", constant=constant)
        self._features = features
        self._target_col = target_col
        self._constant = constant

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
        predictions, predictions_std = self.regressor.predict(df[self.features], return_std=True)

        if return_grads:
            gradients_dict = {}
            for feature in self.features:
                gradients_dict[feature] = [0.0] * len(predictions)

        return TargetValues(
            target_name=self.target_col,
            value_list=list(predictions),
            value_list_std=list(predictions_std) if return_std else None,
            gradients_dict=gradients_dict if return_grads else None,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "target_col": self.target_col,
            "constant": self._constant,
        }

    def _save_model(self, folder_path: str) -> None:
        folder_path = Path(folder_path)
        model_artifact_path = folder_path / "model.pkl"
        if not folder_path.exists():
            folder_path.mkdir(parents=True)

        model_data = self.get_params()
        # We need to save the shape of the output
        model_data["n_outputs"] = self.regressor.n_outputs_

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
            constant=model_dict["constant"],
        )
        # Loading a value into `constant_` will tell scikit-learn that
        # this is a fitted model
        model_instance.regressor.constant_ = model_dict["constant"]
        model_instance.regressor.n_outputs_ = model_dict["n_outputs"]
        return model_instance
