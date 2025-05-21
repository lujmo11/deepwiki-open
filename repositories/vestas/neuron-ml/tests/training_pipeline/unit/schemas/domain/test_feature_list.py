from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.neuron.schemas.domain import FeatureList, FeatureType, FeatureValueType

valid_feature_list_data = {
    "name": "feature_list",
    "features": [
        {
            "name": "wnd_grad",
            "feature_type": FeatureType.ENGINEERED,
            "feature_value_type": FeatureValueType.CONTINUOUS,
            "is_model_input": True,
        },
        {
            "name": "twr_hh",
            "feature_type": FeatureType.RAW,
            "feature_value_type": FeatureValueType.CONTINUOUS,
            "is_model_input": True,
            "safe_domain_method": "range_validation",
        },
        {
            "name": "ws",
            "feature_type": FeatureType.RAW,
            "feature_value_type": FeatureValueType.CONTINUOUS,
            "is_model_input": True,
            "safe_domain_method": "range_validation",
        },
        {
            "name": "vexp",
            "feature_type": FeatureType.RAW,
            "feature_value_type": FeatureValueType.CONTINUOUS,
            "is_model_input": True,
            "safe_domain_method": "range_validation",
        },
    ],
}


def test_valid_required_features_for_preprocessing() -> None:
    """Test the case where all required raw features are present for the
    engineered wnd_grad feature.
    """
    _ = FeatureList(**valid_feature_list_data)


def test_invalid_required_features_for_preprocessing() -> None:
    """Test the case where not all required raw features are present for the
    engineered wnd_grad feature.
    """
    # Create a feature list where vexp is missing. required for wnd_grad
    feature_list_data_missing_non_model_input = deepcopy(valid_feature_list_data)
    feature_list_data_missing_non_model_input[
        "features"
    ] = feature_list_data_missing_non_model_input["features"][:-1]
    with pytest.raises(
        ValidationError,
        match=(
            "Engineered feature wnd_grad requires the following raw features: "
            "\['twr_hh', 'vexp', 'ws'\]."
        ),
    ):
        FeatureList(**feature_list_data_missing_non_model_input)


def test_to_many_non_model_input_features() -> None:
    """Test the case where there input features with is_model_input False
    that are not needed for any engineered feature.
    """
    # Create a feature list with an extra feature that is not needed for any
    # engineered feature.
    feature_list_data_extra_non_model_input = deepcopy(valid_feature_list_data)
    feature_list_data_extra_non_model_input["features"].append(
        {
            "name": "extra_feature",
            "feature_type": FeatureType.RAW,
            "feature_value_type": FeatureValueType.CONTINUOUS,
            "is_model_input": False,
            "safe_domain_method": "range_validation",
        }
    )
    with pytest.raises(
        ValidationError,
        match=(
            "Raw feature extra_feature is not used as model input "
            "and is not required for any engineered feature."
        ),
    ):
        FeatureList(**feature_list_data_extra_non_model_input)
