import json
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from pydantic import BaseModel

from neuron.safe_domain.safe_domain_utils import get_hull_domain, get_range_domain
from neuron.safe_domain.validators.base import Validator
from neuron.safe_domain.validators.discrete_validator import DiscreteValidator
from neuron.safe_domain.validators.hull_validator import HullValidator
from neuron.safe_domain.validators.registry import get_validator
from neuron.schemas.domain import Feature, FeatureType, SafeDomainGroup


class SafeDomainValidatorData(BaseModel, frozen=True):
    """This class is used to store the data of a SafeDomainValidator object.

    To ensure backwards compatibility, we should avoid changing the class definition.
    """

    feature_name_list: List[str]
    safe_domain_method: SafeDomainGroup
    validator_model_data: Dict

    def to_dict(self) -> Dict:
        return {
            "safe_domain_method": self.safe_domain_method.value,
            "feature_name_list": self.feature_name_list,
            "validator_model_data": self.validator_model_data,
        }


class SafeDomainValidator:
    def __init__(self, features: List[Feature]):
        self.validator_dict: Dict[SafeDomainGroup, Validator] = {}

        # Fitting the validator exclusively on raw features
        features = [feature for feature in features if feature.feature_type == FeatureType.RAW]
        self.features = features

        # Group features by safe_domain_method
        self.feature_groups = defaultdict(list)
        for feature in features:
            self.feature_groups[feature.safe_domain_method].append(feature.name)

    def fit(self, df: pd.DataFrame) -> None:
        """Fit a validator for each feature group"""

        for safe_domain_method, feature_names in self.feature_groups.items():
            validator = get_validator(safe_domain_method)
            validator_features = [
                feature for feature in self.features if feature.name in feature_names
            ]
            validator.fit(df[feature_names], validator_features)

            self.validator_dict[safe_domain_method] = validator

    def validate_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate features in each feature group"""

        df_out_interp = pd.DataFrame()

        for safe_domain_method, validator in self.validator_dict.items():
            features = self.feature_groups[safe_domain_method]
            df_val_interp = validator.validate_interpolation(df[features])
            df_out_interp = pd.concat([df_out_interp, df_val_interp], axis=1)

        return df_out_interp

    def validate_extrapolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate features in each feature group"""

        df_out_extrap = pd.DataFrame()

        for safe_domain_method, validator in self.validator_dict.items():
            features = self.feature_groups[safe_domain_method]
            df_val_extrap = validator.validate_extrapolation(df[features])
            df_out_extrap = pd.concat([df_out_extrap, df_val_extrap], axis=1)

        return df_out_extrap

    def save(self, path: str) -> None:
        data = []

        for safe_domain_method, feature_names in self.feature_groups.items():
            data.append(
                SafeDomainValidatorData(
                    feature_name_list=feature_names,
                    safe_domain_method=safe_domain_method,
                    validator_model_data=self.validator_dict[safe_domain_method].get_model_data(),
                )
            )

        with open(path, "w") as f:
            json.dump([d.to_dict() for d in data], f)

    def _get_2d_feature_vertices(
        self, feature_x: str, feature_y: str, method: str = "interp"
    ) -> Tuple[List, List]:
        # A feature only exists in one safe domain group, i.e. we take the first one
        safe_domain_x = next(
            (method for method, features in self.feature_groups.items() if feature_x in features),
            None,
        )
        safe_domain_y = next(
            (method for method, features in self.feature_groups.items() if feature_y in features),
            None,
        )

        validator_x = self.validator_dict[safe_domain_x]
        validator_y = self.validator_dict[safe_domain_y]

        is_discrete_validator = isinstance(validator_x, DiscreteValidator) or isinstance(
            validator_y, DiscreteValidator
        )

        if is_discrete_validator:
            return [], []

        if not is_discrete_validator:
            if (safe_domain_x == safe_domain_y) and isinstance(validator_x, HullValidator):
                x, y = get_hull_domain(feature_x, feature_y, validator_x, method)
            else:
                x, y = get_range_domain(feature_x, feature_y, validator_x, validator_y, method)
        return x, y

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            model_data = json.load(f)

        safe_domain_validator = cls(features=[])

        for model in model_data:
            validator_data = SafeDomainValidatorData(**model)
            safe_domain_method = validator_data.safe_domain_method
            safe_domain_validator.feature_groups[
                safe_domain_method
            ] = validator_data.feature_name_list
            safe_domain_validator.validator_dict[safe_domain_method] = get_validator(
                safe_domain_method
            ).load(validator_data.validator_model_data)

        return safe_domain_validator
