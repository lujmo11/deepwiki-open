import random
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch

from neuron.schemas.domain import (
    AggregationMethod,
    Feature,
    FeatureType,
    LoadCaseActuals,
    LoadCaseFeatureValues,
    SafeDomainGroup,
    TargetValues,
)


def set_seed(seed: int = 42) -> int:
    """Set seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed


def check_columns_in_dataframe(df: pd.DataFrame, columns: list[str]) -> None:
    columns_set = set(columns)
    if not columns_set.issubset(df.columns):
        raise ValueError(
            f"The columns {columns_set.difference(df.columns)} are not available in the input data"
        )


def get_load_case_actuals_from_df(
    df: pd.DataFrame,
    target_names: List[str],
    calculate_aggregated_metrics: Union[AggregationMethod, None],
) -> LoadCaseActuals:
    """Get load case actuals from a dataframe."""
    load_case_actuals: LoadCaseActuals = {}
    for target in target_names:
        actuals_std = df[target + "_std"].to_list()

        groupby = None
        weightby = None
        if calculate_aggregated_metrics is not None:
            groupby = df[calculate_aggregated_metrics.groupby].tolist()
            weightby = df[calculate_aggregated_metrics.weightby].tolist()

        load_case_actuals[target] = TargetValues(
            target_name=target,
            value_list=df[target].to_list(),
            value_list_std=actuals_std,
            groupby=groupby,
            weightby=weightby,
        )
    return load_case_actuals


def get_load_case_feature_values_df(
    df: pd.DataFrame, features: List[Feature]
) -> LoadCaseFeatureValues:
    """Get load case feature values from a dataframe."""
    load_case_feature_values: LoadCaseFeatureValues = {}
    for feature in features:
        try:
            load_case_feature_values[feature] = df[feature.name].to_numpy()
        except KeyError as e:
            raise ValueError(
                f"Could not find feature {feature.name} feature dataframe. "
                f"Available features: {df.columns}."
            ) from e
    return load_case_feature_values


def zip_folder(folder_path: Union[Path, str], zip_artifact_path: Union[Path, str]) -> None:
    """Zip a folder with all its contents into a zip file."""
    folder_path = Path(folder_path)
    zip_artifact_path = Path(zip_artifact_path)
    with zipfile.ZipFile(zip_artifact_path, "w") as zip_ref:
        for file in folder_path.rglob("*"):
            zip_ref.write(file, file.relative_to(folder_path.parent))


def plot_train_test_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    load_case_features: List[Feature],
    safe_domain: SafeDomainGroup = None,
) -> BytesIO:
    """
    Generating a plot with a representation of training and test data.
    As an optional feature, this function also plots the safe domain (interp and extrap)
    representation.
    """
    feature_names = [
        feature.name for feature in load_case_features if feature.feature_type == FeatureType.RAW
    ]
    feature_train_df = train_df[feature_names].copy(deep=True)
    feature_test_df = test_df[feature_names].copy(deep=True)
    feature_train_df["data_splitting"] = "train"
    feature_test_df["data_splitting"] = "test"
    full_data_df = pd.concat([feature_train_df, feature_test_df], axis=0)

    gnp = sns.pairplot(full_data_df, hue="data_splitting", plot_kws={"s": 10, "alpha": 0.5})

    if safe_domain:
        for irow, plot_row in enumerate(gnp.axes):
            for iax, ax in enumerate(plot_row):
                if irow != iax:
                    feature_x = ax.get_xlabel()
                    feature_y = ax.get_ylabel()
                    x, y = safe_domain._get_2d_feature_vertices(
                        feature_x=feature_x, feature_y=feature_y, method="extrap"
                    )
                    ax.plot(x, y, "r", linewidth=0.5)

                    x, y = safe_domain._get_2d_feature_vertices(
                        feature_x=feature_x, feature_y=feature_y, method="interp"
                    )
                    ax.plot(x, y, "k", linewidth=0.5)

    pair_plot_bytes = BytesIO()
    gnp.savefig(pair_plot_bytes, format="png")

    return pair_plot_bytes
