"""Test of feature range GPRPytorch model"""

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from neuron.models.target_models.feature_range_gpr import FeatureRangeGPRPytorch


@pytest.mark.slow
def test_gpr_feature_range_model_end2end(
    training_df_fat: pd.DataFrame, mlflow_artifacts_folder: str
) -> None:
    """Test the end to end functionality of the Feature Range GPRPytorch model"""
    # Setup test training run
    features = [
        "ws",
        "ti",
        "vexp",
        "rho",
        "twr_frq1",
        "twr_frq2",
        "twr_hh",
        "epb",
        "vtl",
        "eco",
        "power_rating",
    ]
    target = "MxBldRoot_m1000"

    train, test = train_test_split(training_df_fat, test_size=0.2, random_state=42)

    # Train the Feature Range GPR Model
    n_inducing_points = int(len(train) / 4)

    range_feature = "ws"
    range_upper_bounds = [8, 14, 21]

    feature_gpr = FeatureRangeGPRPytorch(
        features=features,
        target_col=target,
        n_inducing_points=n_inducing_points,
        range_feature=range_feature,
        range_upper_bounds=range_upper_bounds,
    )

    feature_gpr.fit(train)

    # Check the number of individual trained models
    assert len(feature_gpr.models) == 4, "the number of trained range models is not correct"

    # Check the trained data size for each model range
    base_list_in_feature_range = [
        59,
        51,
        105,
        105,
    ]  # data count determined from an experimental run
    for item, (_range, _model) in enumerate(feature_gpr.models.items()):
        feature_in_range = _range.series_in_range(train[feature_gpr.range_feature])
        df_in_feature_range = train[feature_in_range]
        assert (
            len(df_in_feature_range) == base_list_in_feature_range[item]
        ), "the data splitting in ranges is not correct for the range"

    # Save the model
    feature_gpr.save_model(folder_path=mlflow_artifacts_folder)

    # Load the saved model
    loaded_feature_gpr = FeatureRangeGPRPytorch.load_model(folder_path=mlflow_artifacts_folder)

    # Make predictions using the loaded model
    preds = loaded_feature_gpr.predict(test, return_std=False)
    assert len(preds.value_list) == len(test), "the number of predictions is not correct"


def test_gpr_feature_bad_intervals(training_df_fat: pd.DataFrame) -> None:
    """Test that the Feature Range GPRPytorch model raises an error
    when the intervals are not correct

    """

    # Setup test training run
    features = [
        "ws",
        "ti",
        "vexp",
        "rho",
        "twr_frq1",
        "twr_frq2",
        "twr_hh",
        "epb",
        "vtl",
        "eco",
        "power_rating",
    ]
    target = "MxBldRoot_m1000"

    train, test = train_test_split(training_df_fat, test_size=0.2, random_state=42)

    # Train the Feature Range GPR Model
    n_inducing_points = int(len(train) / 4)

    range_feature = "ws"
    range_upper_bounds = [
        -20,
        14,
        21,
    ]  # The lower range should cause an error because of too few samples

    feature_gpr = FeatureRangeGPRPytorch(
        features=features,
        target_col=target,
        n_inducing_points=n_inducing_points,
        range_feature=range_feature,
        range_upper_bounds=range_upper_bounds,
    )
    with pytest.raises(ValueError):
        feature_gpr.fit(train)


def test_gpr_input_data(training_df_fat: pd.DataFrame) -> None:
    # Setup test training run

    # Check the missing features case
    features = [
        "these",
        "features",
        "do",
        "not",
        "exist",
        "twr_frq1",
        "twr_frq2",
        "twr_hh",
        "epb",
        "vtl",
        "eco",
        "power_rating",
    ]
    target = "MxBldRoot_m1000"

    feature_gpr = FeatureRangeGPRPytorch(
        features=features,
        target_col=target,
        range_feature="ws",
        range_upper_bounds=[8, 14, 21],
    )
    with pytest.raises(ValueError, match="are not available in the input data"):
        feature_gpr.fit(training_df_fat)

    # Check the missing target case
    features = ["twr_frq1", "twr_frq2", "twr_hh"]
    target = "missing_target"

    feature_gpr = FeatureRangeGPRPytorch(
        features=features,
        target_col=target,
        range_feature="ws",
        range_upper_bounds=[8, 14, 21],
    )
    with pytest.raises(ValueError, match="are not available in the input data"):
        feature_gpr.fit(training_df_fat)
