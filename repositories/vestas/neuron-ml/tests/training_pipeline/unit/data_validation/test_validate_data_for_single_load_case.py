import pandas as pd
import pytest

from neuron.data_validation.exceptions import DataValidationError
from neuron.data_validation.validation import validate_data_for_single_load_case
from neuron.schemas.domain import (
    AggregationMethod,
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
from tests.test_doubles import TrainingDataRepositoryTestDouble


def test_validate_data_for_single_load_case_happy_path() -> None:
    test_load_case = LoadCase(
        name="test",
        feature_list=FeatureList(
            name="test_fl",
            features=[
                Feature(
                    name="ws",
                    feature_value_type=FeatureValueType.CONTINUOUS,
                    safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
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
            "ws": 100 * [0.0, 15.0, 100.0, 19.0],
            "Power_mean": 100 * [0.0, 1.0, 2.0, 3.0],
            "Power_mean_std": 100 * [0.0, 0.01, 0.02, 0.03],
        }
    )
    data_repo = TrainingDataRepositoryTestDouble(train_df=test_df, test_df=test_df, agg_df=test_df)
    validate_data_for_single_load_case(
        load_case=test_load_case,
        data_repo=data_repo,  # type: ignore
        calculate_aggregated_metrics=None,
    )


def test_validate_data_for_single_missing_std_raises_error() -> None:
    test_load_case = LoadCase(
        name="test",
        feature_list=FeatureList(
            name="test_fl",
            features=[
                Feature(
                    name="ws",
                    feature_value_type=FeatureValueType.CONTINUOUS,
                    safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
                    feature_type=FeatureType.RAW,
                    is_model_input=True,
                ),
            ],
        ),
        target_list=TargetList(name="test_tl", targets=[Target.Power_mean]),
        postprocessor=PostprocessorName.NO_POSTPROCESSOR,
    )

    train_df = pd.DataFrame.from_dict(
        {
            "ws": 100 * [0.0, 15.0, 100.0, 19.0],
            "Power_mean": 100 * [0.0, 1.0, 2.0, 3.0],
        }
    )

    data_repo_wrong = TrainingDataRepositoryTestDouble(
        train_df=train_df, test_df=train_df, agg_df=train_df
    )
    with pytest.raises(DataValidationError):
        validate_data_for_single_load_case(
            load_case=test_load_case,
            data_repo=data_repo_wrong,  # type: ignore
            calculate_aggregated_metrics=None,
        )


def test_validate_data_for_single_load_case_required_raw_feature_error() -> None:
    """Test that a DataValidationError is raised when a required feature is missing.

    This shows that we are hitting `check_required_raw_features_for_load_case`.
    `check_required_raw_features_for_load_case` is more thoroughly tested in a separate test file.
    """
    test_load_case = LoadCase(
        name="test",
        feature_list=FeatureList(
            name="test_fl",
            features=[
                Feature(
                    name="ws",
                    feature_value_type=FeatureValueType.CONTINUOUS,
                    safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
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
            "ws": 100 * [0.0, 15.0, 100.0, 19.0],
        }
    )
    data_repo = TrainingDataRepositoryTestDouble(train_df=test_df, test_df=test_df, agg_df=test_df)
    with pytest.raises(DataValidationError):
        validate_data_for_single_load_case(
            load_case=test_load_case,
            data_repo=data_repo,  # type: ignore
            calculate_aggregated_metrics=None,
        )


def test_validate_data_for_single_load_case_required_features_for_preprocessor() -> None:
    """Test that a DataValidationError is raised when a required feature
    for preprocessing is missing.

    This shows that we are hitting `check_required_features_for_preprocessor`.
    `check_required_features_for_preprocessor` is more thoroughly tested in a separate test file.
    """
    test_load_case = LoadCase(
        name="test",
        feature_list=FeatureList(
            name="test_fl",
            features=[
                Feature(
                    name="ws",
                    feature_value_type=FeatureValueType.CONTINUOUS,
                    safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
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
            "ws": 100 * [0.0, 15.0, 100.0, 19.0],
        }
    )
    data_repo = TrainingDataRepositoryTestDouble(train_df=test_df, test_df=test_df, agg_df=test_df)
    with pytest.raises(DataValidationError):
        validate_data_for_single_load_case(
            load_case=test_load_case,
            data_repo=data_repo,  # type: ignore
            calculate_aggregated_metrics=None,
        )


def test_validate_data_for_single_load_case_aggregation_method_error() -> None:
    """Test that a DataValidationError is raised when a required aggregation method is missing.

    This shows that we are hitting `check_required_aggregated_metrics_data`.
    `check_required_aggregated_metrics_data` is more thoroughly tested in a separate test file.
    """
    test_load_case = LoadCase(
        name="test",
        feature_list=FeatureList(
            name="test_fl",
            features=[
                Feature(
                    name="ws",
                    feature_value_type=FeatureValueType.CONTINUOUS,
                    safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
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
            "ws": 100 * [0.0, 15.0, 100.0, 19.0],
            "Power_mean": 100 * [0.0, 1.0, 2.0, 3.0],
            "Power_mean_std": 100 * [0.0, 0.01, 0.02, 0.03],
        }
    )
    data_repo = TrainingDataRepositoryTestDouble(train_df=test_df, test_df=test_df, agg_df=test_df)
    with pytest.raises(DataValidationError):
        validate_data_for_single_load_case(
            load_case=test_load_case,
            data_repo=data_repo,  # type: ignore
            calculate_aggregated_metrics=AggregationMethod(
                groupby="some_group_feature", weightby="some_weight_feature"
            ),
        )
