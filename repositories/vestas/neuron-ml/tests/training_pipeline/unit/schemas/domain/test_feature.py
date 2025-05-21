import pytest
from pydantic import ValidationError

from src.neuron.schemas.domain import Feature, FeatureType, FeatureValueType, SafeDomainGroup


def test_validate_engineered_feature_name() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "Feature name invalid_engineered_feature is not allowed "
            "for feature type `engineered`."
        ),
    ):
        Feature(
            name="invalid_engineered_feature",
            feature_type=FeatureType.ENGINEERED,
            feature_value_type=FeatureValueType.CONTINUOUS,
            is_model_input=True,
        )


def test_validate_model_input_and_feature_type() -> None:
    with pytest.raises(
        ValidationError, match="A feature of type engineered can only be model input"
    ):
        Feature(
            name="wnd_grad",
            feature_type=FeatureType.ENGINEERED,
            feature_value_type=FeatureValueType.CONTINUOUS,
            is_model_input=False,
        )


def test_validate_safe_domain_method() -> None:
    with pytest.raises(
        ValidationError, match="A feature of type engineered can not have a safe_domain_method."
    ):
        Feature(
            name="wnd_grad",
            feature_type=FeatureType.ENGINEERED,
            feature_value_type=FeatureValueType.CONTINUOUS,
            is_model_input=True,
            safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
        )

    with pytest.raises(
        ValidationError, match="A feature of type raw needs to have a safe_domain_method."
    ):
        Feature(
            name="ws",
            feature_type=FeatureType.RAW,
            feature_value_type=FeatureValueType.CONTINUOUS,
            is_model_input=True,
        )


def test_check_consistency_feature_type_safe_domain() -> None:
    with pytest.raises(
        ValidationError,
        match="Discrete feature value types require the discrete_validation method.",
    ):
        Feature(
            name="discrete_feature",
            feature_type=FeatureType.RAW,
            feature_value_type=FeatureValueType.DISCRETE,
            is_model_input=True,
            safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
        )

    with pytest.raises(
        ValidationError, match="features cannot have the discrete_validation method assigned."
    ):
        Feature(
            name="continuous_feature",
            feature_type=FeatureType.RAW,
            feature_value_type=FeatureValueType.CONTINUOUS,
            is_model_input=True,
            safe_domain_method=SafeDomainGroup.DISCRETE_VALIDATION,
        )

    with pytest.raises(
        ValidationError,
        match="An extrapolation_domain_offset is not allowed for discrete features.",
    ):
        Feature(
            name="discrete_feature",
            feature_type=FeatureType.RAW,
            feature_value_type=FeatureValueType.DISCRETE,
            is_model_input=True,
            safe_domain_method=SafeDomainGroup.DISCRETE_VALIDATION,
            extrapolation_domain_offset=0.1,
        )
