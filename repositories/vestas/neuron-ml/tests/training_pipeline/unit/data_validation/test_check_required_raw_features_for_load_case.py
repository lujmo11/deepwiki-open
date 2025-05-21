import pandas as pd
import pytest

from neuron.data_validation.exceptions import DataValidationError
from neuron.data_validation.validation import (
    MAX_UNIQUE_VALUES_FOR_DISCRETE_FEATURE,
    check_required_raw_features_for_load_case,
)
from neuron.schemas.domain import (
    Feature,
    FeatureList,
    FeatureType,
    FeatureValueType,
    LoadCase,
    PostprocessorName,
    SafeDomainGroup,
    Target,
    TargetList,
)


def test_check_required_raw_features_for_load_case_happy_path() -> None:
    """Test that the function does not raise an exception when all required columns are present."""
    test_load_case = LoadCase(
        name="test",
        feature_list=FeatureList(
            name="test_fl",
            features=[
                Feature(
                    name="continuous_feature",
                    feature_value_type=FeatureValueType.CONTINUOUS,
                    safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
                    feature_type=FeatureType.RAW,
                    is_model_input=True,
                ),
                Feature(
                    name="discrete_feature",
                    feature_value_type=FeatureValueType.DISCRETE,
                    safe_domain_method=SafeDomainGroup.DISCRETE_VALIDATION,
                    feature_type=FeatureType.RAW,
                    is_model_input=True,
                ),
            ],
        ),
        target_list=TargetList(name="test_tl", targets=[Target.Power_mean]),
        postprocessor=PostprocessorName.NO_POSTPROCESSOR,
    )
    test_df = pd.DataFrame.from_dict(
        {
            "continuous_feature": 100 * [0.0, 15.0, 100.0, 19.0],
            "discrete_feature": 100 * [1, 2, 3, 4],
        }
    )
    check_required_raw_features_for_load_case(df=test_df, load_case=test_load_case)


def test_check_required_raw_features_for_load_case_missing_feature_error() -> None:
    """Test that an exception is raised when a required column is missing."""
    test_load_case = LoadCase(
        name="test",
        feature_list=FeatureList(
            name="test_fl",
            features=[
                Feature(
                    name="feature_a",
                    feature_value_type=FeatureValueType.CONTINUOUS,
                    safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
                    feature_type=FeatureType.RAW,
                    is_model_input=True,
                )
            ],
        ),
        target_list=TargetList(name="test_tl", targets=[Target.Power_mean]),
        postprocessor=PostprocessorName.NO_POSTPROCESSOR,
    )
    test_df = pd.DataFrame.from_dict({"not_feature_a": [0.0, 15.0, 100.0, 19.0]})
    with pytest.raises(DataValidationError) as exc_info:
        check_required_raw_features_for_load_case(df=test_df, load_case=test_load_case)
    assert (
        "Feature feature_a is required for load case test but not present in the dataset."
        == str(exc_info.value)
    )


def test_check_required_raw_features_for_load_case_discrete_feature_error() -> None:
    """Test that an exception is raised when a required a discrete feature
    has to many unique values."""
    test_load_case = LoadCase(
        name="test_load_case",
        feature_list=FeatureList(
            name="test_fl",
            features=[
                Feature(
                    name="too_many_values",
                    feature_value_type=FeatureValueType.DISCRETE,
                    safe_domain_method=SafeDomainGroup.DISCRETE_VALIDATION,
                    feature_type=FeatureType.RAW,
                    is_model_input=True,
                )
            ],
        ),
        target_list=TargetList(name="test_tl", targets=[Target.Power_mean]),
        postprocessor=PostprocessorName.NO_POSTPROCESSOR,
    )
    test_df = pd.DataFrame.from_dict(
        {"too_many_values": range(MAX_UNIQUE_VALUES_FOR_DISCRETE_FEATURE + 1)}
    )
    expected_error_msg = (
        "Feature too_many_values is specified to be a discrete feature for load case "
        f"test_load_case. The data for a discrete feature type should have less than "
        f"{MAX_UNIQUE_VALUES_FOR_DISCRETE_FEATURE} unique values, "
        f"but found {MAX_UNIQUE_VALUES_FOR_DISCRETE_FEATURE + 1} unique values."
    )
    with pytest.raises(DataValidationError) as exc_info:
        check_required_raw_features_for_load_case(df=test_df, load_case=test_load_case)
    assert expected_error_msg == str(exc_info.value)
