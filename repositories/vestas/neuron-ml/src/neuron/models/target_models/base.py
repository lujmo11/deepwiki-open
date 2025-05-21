import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from neuron.schemas.domain import TargetValues

TARGET_MODEL_METADATA_FILE_NAME = "target_model_metadata.json"


class ModelMetadata(BaseModel):
    model_class_name: str
    features: List[str]
    target_col: str

    # To avoid warning of clash between `model_class_name`and Pydantic protected namespace.
    model_config = ConfigDict(protected_namespaces=())


class Model(ABC):
    """Abstract base class for target model classes

    Any target model class should have this interface.
    """

    name: str

    @property
    @abstractmethod
    def target_col(self) -> str:
        ...

    @property
    @abstractmethod
    def features(self) -> List[str]:
        ...

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> Self:
        ...

    @abstractmethod
    def predict(
        self, df: pd.DataFrame, return_std: bool = False, return_grads: bool = False
    ) -> TargetValues:
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def _save_model(self, folder_path: str) -> None:
        ...

    def save_model(self, folder_path: str) -> None:
        self._save_model(folder_path)
        self._save_model_metadata(folder_path)

    @classmethod
    @abstractmethod
    def load_model(cls, folder_path: str) -> Self:
        ...

    @property
    def model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class_name=self.name,
            features=self.features,
            target_col=self.target_col,
        )

    def _save_model_metadata(self, folder: str) -> None:
        if not Path(folder).exists():
            Path(folder).mkdir(parents=True)
        with open(f"{folder}/{TARGET_MODEL_METADATA_FILE_NAME}", "w") as fh:
            json.dump(self.model_metadata.model_dump(), fh)

    def _calculate_gradients_finite_diff(
        self,
        df: pd.DataFrame,
        epsilon: float = 1e-6,
    ) -> Dict[str, List[float]]:
        """
        Gradient calculation using finite difference.
        """

        predictions = self.predict(df).values_as_np

        gradients_dict = {}
        for feature in self.features:
            epsilon_feature = epsilon * np.max(abs(df[feature]))

            X_perturbed = df.copy()
            X_perturbed[feature] += epsilon_feature

            predictions_perturbed = self.predict(X_perturbed).values_as_np

            gradients = (predictions_perturbed - predictions) / epsilon_feature

            gradients_dict[feature] = gradients.flatten().tolist()

        return gradients_dict
