import pandas as pd
import pytest

from neuron.data_validation.exceptions import DataValidationError
from neuron.data_validation.validation import check_required_aggregated_metrics_data
from neuron.schemas.domain import AggregationMethod


def test_check_required_aggregated_metrics_data_happy_path() -> None:
    test_df = pd.DataFrame.from_dict(
        {
            "some_feature": 100 * [0.0, 15.0, 100.0, 19.0],
            "some_group_feature": 100 * [1, 2, 3, 4],
            "some_weight_feature": 100 * [5, 6, 7, 8],
            "a_target": 100 * [0.0, 1.0, 2.0, 3.0],
        }
    )
    check_required_aggregated_metrics_data(
        df=test_df,
        aggregation_method=AggregationMethod(
            groupby="some_group_feature", weightby="some_weight_feature"
        ),
    )


def test_check_required_aggregated_metrics_data_missing_weightby_feature_error() -> None:
    test_df = pd.DataFrame.from_dict(
        {
            "some_feature": 100 * [0.0, 15.0, 100.0, 19.0],
            "a_target": 100 * [0.0, 1.0, 2.0, 3.0],
            "some_group_feature": 100 * [1, 2, 3, 4],
        }
    )
    with pytest.raises(DataValidationError) as exc_info:
        check_required_aggregated_metrics_data(
            df=test_df,
            aggregation_method=AggregationMethod(
                groupby="some_group_feature", weightby="some_weight_feature"
            ),
        )
    expected_error_message = (
        f"Test data frame is missing the required weightby aggregation "
        f"method column some_weight_feature."
        f"The dataframe contains these columns: {list(test_df.columns)}"
    )
    assert str(exc_info.value) == expected_error_message


def test_check_required_aggregated_metrics_data_missing_groupby_feature_error() -> None:
    test_df = pd.DataFrame.from_dict(
        {
            "some_feature": 100 * [0.0, 15.0, 100.0, 19.0],
            "a_target": 100 * [0.0, 1.0, 2.0, 3.0],
            "some_weight_feature": 100 * [5, 6, 7, 8],
        }
    )
    with pytest.raises(DataValidationError) as exc_info:
        check_required_aggregated_metrics_data(
            df=test_df,
            aggregation_method=AggregationMethod(
                groupby="some_group_feature", weightby="some_weight_feature"
            ),
        )
    expected_error_message = (
        f"Test data frame is missing the required groupby aggregation "
        f"method column some_group_feature."
        f"The dataframe contains these columns: {list(test_df.columns)}"
    )
    assert str(exc_info.value) == expected_error_message
