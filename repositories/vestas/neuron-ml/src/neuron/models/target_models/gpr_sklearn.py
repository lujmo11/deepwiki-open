import pickle
from pathlib import Path
from typing import Any, Dict, List, Self

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, Matern, WhiteKernel

from neuron.models.scaler_registry import DataScaler, get_registered_scaler_from_name
from neuron.models.target_models.base import Model
from neuron.schemas.domain import TargetValues
from neuron.utils import check_columns_in_dataframe

KERNEL_CLASS_REGISTRY = {"Matern": Matern, "RBF": RBF}


def get_registered_kernel_class_from_name(name: str) -> Kernel:
    """Gets the registered kernel class from the class name."""
    try:
        kernel_class = KERNEL_CLASS_REGISTRY[name]
        return kernel_class()
    except KeyError as e:
        raise (
            f"Kernel name {name} not in registry. "
            f"Available kernels are: {list(KERNEL_CLASS_REGISTRY.keys())}"
        ) from e


class GPR(Model):
    name = "gpr_sklearn"

    def __init__(
        self,
        features: List[str],
        target_col: str,
        kernel_name: str = "Matern",
        kernel_params: Dict[str, Any] = None,
        gaussian_noise: bool = True,
        feature_scaler_name: DataScaler = DataScaler.ROBUST_SCALER,
        target_scaler_name: DataScaler = DataScaler.ROBUST_SCALER,
    ):
        self._features = features
        self._target_col = target_col
        self.kernel_name = kernel_name
        self.gaussian_noise = gaussian_noise
        self.feature_scaler_name = feature_scaler_name
        self.target_scaler_name = target_scaler_name

        self.target_scaler = get_registered_scaler_from_name(name=feature_scaler_name)
        self.feature_scaler = get_registered_scaler_from_name(name=target_scaler_name)
        kernel = get_registered_kernel_class_from_name(name=kernel_name)

        if kernel_params:
            kernel_params.update({"length_scale": np.ones(len(features))})
        else:
            kernel_params = {"length_scale": np.ones(len(features))}

        if any(var not in list(kernel.get_params().keys()) for var in list(kernel_params.keys())):
            raise ValueError(
                "The selected GPR kernel cannot be configured with the"
                "specified kernel parameters. "
                f"The allowed parameters are: {list(kernel_params.keys())}."
            )

        kernel.set_params(**kernel_params)

        if gaussian_noise:
            kernel = kernel + WhiteKernel()

        self.kernel = kernel
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    @property
    def target_col(self) -> str:
        return self._target_col

    @property
    def features(self) -> List[str]:
        return self._features

    def fit(self, df: pd.DataFrame) -> Self:
        check_columns_in_dataframe(df, self.features + [self.target_col])
        X = df[self._features]
        y = df[self._target_col]

        X = self.feature_scaler.fit_transform(X)

        self.model.fit(X, y)

        return self

    def predict(
        self, df: pd.DataFrame, return_std: bool = False, return_grads: bool = False
    ) -> TargetValues:
        check_columns_in_dataframe(df, self.features)

        X = df[self.features]

        X = self.feature_scaler.transform(X)

        pred = self.model.predict(X, return_std=return_std)
        pred_std = None

        if return_std:
            pred, pred_std = pred

        if return_grads:
            gradients_dict = self._calculate_gradients_finite_diff(df)

        return TargetValues(
            target_name=self.target_col,
            value_list=list(pred),
            value_list_std=list(pred_std) if pred_std is not None else None,
            gradients_dict=gradients_dict if return_grads else None,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "target_col": self.target_col,
            "kernel": self.kernel_name,
            "gaussian_noise": self.gaussian_noise,
            "feature_scaler_name": self.feature_scaler_name,
            "target_scaler_name": self.target_scaler_name,
        }

    def _save_model(self, folder_path: str) -> None:
        folder_path = Path(folder_path)
        model_artifact_path = folder_path / "model.pkl"
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        with open(model_artifact_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, folder_path: str) -> Self:
        model_artifact_path = Path(folder_path) / "model.pkl"
        with open(model_artifact_path, "rb") as f:
            return pickle.load(f)
