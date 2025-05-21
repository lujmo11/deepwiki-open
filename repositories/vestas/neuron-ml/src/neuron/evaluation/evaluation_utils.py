"""Module containing classes for evaluating models.

All concrete evaluators should implement the Evaluator interface/protocol.
"""
import operator
import re
from enum import StrEnum
from typing import Any, Dict, List, Self, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, model_validator
from pydantic.dataclasses import dataclass
from scipy import optimize, stats
from scipy.stats import iqr, norm, qmc
from sklearn.metrics import mean_absolute_error, r2_score

from neuron.schemas.domain import (
    Feature,
    FeatureValueType,
    LoadCaseActuals,
    LoadCasePredictions,
    Metric,
    TargetValues,
)
from neuron.utils import set_seed


class TargetMetrics(BaseModel):
    e_mean_norm: float
    e_std_norm: float
    e_max_norm: float
    mae: float
    mae_norm: float
    mane: float
    r2: float
    pred_coverage_error_1std: Union[float, None] = None
    pred_coverage_error_2std: Union[float, None] = None
    chi_squared: Union[float, None] = None
    critical_value: Union[float, None] = None
    sigma_model_form: Union[float, None] = None
    cov_model_form: Union[float, None] = None
    chi_squared_passed: Union[bool, None] = None

    @model_validator(mode="before")
    def _validate_consistency_of_statistical_tests(cls, values):  # noqa: ANN001
        chi_square_stats_vars = [
            values.get(Metric.CHI_SQUARED),
            values.get(Metric.CRITICAL_VALUE),
            values.get(Metric.SIGMA_MODEL_FORM),
            values.get(Metric.COV_MODEL_FORM),
            values.get(Metric.CHI_SQUARED_PASSED),
        ]

        if any(var is None for var in chi_square_stats_vars) and not all(
            var is None for var in chi_square_stats_vars
        ):
            raise ValueError(
                "If the chi square statistical test is being used for model evaluation, all the"
                "required variables should be defined"
            )

        return values

    def as_dict(self) -> Dict[str, Union[int, float, None]]:
        feature_dict = self.__dict__
        return {
            key: value for key, value in feature_dict.items() if key != "__pydantic_initialised__"
        }


class Condition(StrEnum):
    """List of allowed conditions to be applied for model evaluation criteria."""

    eq = "eq"
    lt = "lt"
    le = "le"
    gt = "gt"
    ge = "ge"


CONDITION_OPERATIONS = {
    Condition.eq: operator.eq,
    Condition.lt: operator.lt,
    Condition.le: operator.le,
    Condition.gt: operator.gt,
    Condition.ge: operator.ge,
}

CONDITION_NAMES = {
    Condition.eq: "equal to",
    Condition.lt: "less than",
    Condition.le: "less than or equal to",
    Condition.gt: "greater than",
    Condition.ge: "greater than or equal to",
}


class ModelAcceptanceCriteria(BaseModel):
    """Acceptance criteria for a trained model.

    Attributes:
    -----------
    metric: Metric
        The primary evaluation metric used to evaluate a trained model.
    value: Union[float, bool]
        The value of the primary metric that determines if the model meets the acceptance
        criteria.
    condition: Condition
        The condition applied to the metric value.
        E.g lower than, higher than, equal to, ... The allowed operations are
        defined in CONDITION_OPERATIONS.
    or_metric: Union[Metrics, None] = None
        An optional alternative metric that is used to evaluate a trained model. The
        model is accepted if either the metric OR the either metric meet the acceptace
        criteria.
    or_metric_value: Union[float, None] = None
        The alternative metric value. Similar to "value".
    or_metric_condition: Union[Condition, None] = None
        The alternative metric operation. Similar to "condition".
    """

    metric: Metric
    value: Union[float, bool]
    condition: Condition
    or_metric: Union[Metric, None] = None
    or_metric_value: Union[float, None] = None
    or_metric_condition: Union[Condition, None] = None

    @model_validator(mode="after")
    def check_consistency_in_or_metrics(self) -> Self:
        if self.metric == self.or_metric:
            raise ValueError("metric and or_metric cannot be the same for model_evaluation")
        return self


class FeatureBin(BaseModel):
    feature: Feature
    bin_value: float
    data_count: int
    lower_bound: float
    upper_bound: float

    @model_validator(mode="after")
    def check_for_increasing_values(self) -> Self:
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError(
                    "'ranges' used for feature range splitting should be defined in ascending order"
                )
        return self

    def filter_arrays_based_on_feature_bin(
        self, feature_values: np.array, arrays_to_filter: List[np.array]
    ) -> List[np.array]:
        """Filter arrays based on whether the absolute of target values (actuals)
        are within the max_load_evaluation_limit
        """
        if self.feature.feature_value_type == FeatureValueType.DISCRETE:
            filtering_condition_satisfied = feature_values == self.bin_value

        elif self.feature.feature_value_type == FeatureValueType.CONTINUOUS:
            filtering_condition_satisfied = (feature_values >= self.lower_bound) & (
                feature_values < self.upper_bound
            )

        return [
            array[filtering_condition_satisfied] if array is not None else None
            for array in arrays_to_filter
        ]

    def __hash__(self) -> int:
        return hash((self.bin_value, self.lower_bound, self.upper_bound))

    def as_dict(self) -> Dict[str, Union[int, float]]:
        """Convert FeatureBin to a dictionary that can be logged during the experiment logging.
        The feature type is reported as the feature type name.
        """
        return {
            "Feature": self.feature.name,
            "bin_value": self.bin_value,
            "data_count": self.data_count,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


BinnedTargetMetrics = Dict[FeatureBin, TargetMetrics]
BinnedLoadCaseMetrics = Dict[str, BinnedTargetMetrics]
StandardLoadCaseMetrics = Dict[str, TargetMetrics]


class LoadBin(BaseModel):
    bin_value: float
    data_count: int
    lower_bound: float
    upper_bound: float

    def __hash__(self) -> int:
        return hash((self.bin_value, self.lower_bound, self.upper_bound))


def get_target_limit_filtering_conditions(
    actuals: np.ndarray,
    max_load_evaluation_limit: float,
) -> np.ndarray:
    """Get filtering conditions for whether the absolute of target values (actuals)
    are within the max_load_evaluation_limit
    """
    max_fractile = 95
    max_abs = np.percentile(np.abs(actuals), max_fractile)

    filtering_condition_satisfied = (actuals <= -max_load_evaluation_limit * max_abs) | (
        actuals >= max_load_evaluation_limit * max_abs
    )
    return filtering_condition_satisfied


def filter_arrays_based_on_target_limit(
    actuals: np.ndarray,
    max_load_evaluation_limit: float,
    arrays_to_filter: List[np.array],
) -> List[np.array]:
    """Filter arrays based on whether the absolute of target values (actuals)
    are within the max_load_evaluation_limit
    """
    filtering_condition_satisfied = get_target_limit_filtering_conditions(
        actuals, max_load_evaluation_limit
    )
    return [
        array[filtering_condition_satisfied] if array is not None else None
        for array in arrays_to_filter
    ]


@dataclass
class EvaluationGroup:
    group_name: Union[str, int]
    data_count: int
    weight_vector: List[float]
    data_indices: List[int]


def plot_predictions_versus_actual(
    actuals: np.ndarray,
    predictions: np.ndarray,
    metrics: TargetMetrics,
    max_load_evaluation_limit: float,
    name_prefix: str,
    show: bool = False,
) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 4), ncols=2)

    filtering_condition = get_target_limit_filtering_conditions(
        actuals=actuals, max_load_evaluation_limit=max_load_evaluation_limit
    )

    # Prediction vs actuals scatter plot - data considered for evaluation
    ax[0].scatter(
        x=actuals[filtering_condition],
        y=predictions[filtering_condition],
        facecolors="none",
        edgecolors="k",
        label="Evaluation Data",
    )

    # Prediction vs actuals scatter plot - data filtered for evaluation
    reverse_filtering_condition = np.invert(filtering_condition)
    ax[0].scatter(
        x=actuals[reverse_filtering_condition],
        y=predictions[reverse_filtering_condition],
        facecolors="none",
        edgecolors="b",
        label="Filtered Data",
    )

    # Regression Line
    max_x_y = int(actuals.max())
    min_x_y = int(actuals.min())

    if min_x_y > 0:
        ax[0].plot([0, max_x_y], [0, max_x_y], c="red")
    else:
        ax[0].plot([min_x_y, max_x_y], [min_x_y, max_x_y], c="red")

    ax[0].set_xlabel("Actual values")
    ax[0].set_ylabel("Predicted values")
    ax[0].legend(loc=2)
    ax[0].text(
        0.75,
        0.05,
        "r2={:.3f}\nNMAE={:.3f}".format(metrics.r2, metrics.mae_norm),
        transform=ax[0].transAxes,
    )
    errors_all_norm = (predictions[filtering_condition] - actuals[filtering_condition]) / actuals[
        filtering_condition
    ]

    # Error distribution
    ax[1].hist(
        errors_all_norm[(errors_all_norm > -1e6) & (errors_all_norm < 1e6)],
        histtype="step",
        lw=3,
        color="k",
        bins=20,
    )
    ax[1].axvline(
        metrics.e_mean_norm,
        c="r",
        lw=2,
        label="mean error: {:.3f}".format(metrics.e_mean_norm),
    )
    ax[1].axvline(
        metrics.e_mean_norm - metrics.e_std_norm,
        c="b",
        lw=2,
        label="std error: {:.3f}".format(metrics.e_std_norm),
    )
    ax[1].axvline(metrics.e_mean_norm + metrics.e_std_norm, c="b", lw=2)
    ax[1].set_xlabel("Prediction error = pred - actual / actual")
    ax[1].set_ylabel("Count")
    ax[1].legend()
    plt.suptitle(name_prefix)

    if not show:
        plt.close()

    return fig


def plot_normalized_residual(
    actuals: np.ndarray,
    predictions: np.ndarray,
    predictions_std: np.ndarray,
    max_load_evaluation_limit: float,
    name_prefix: str,
    show: bool = False,
) -> Figure:
    filtering_condition = get_target_limit_filtering_conditions(
        actuals=actuals, max_load_evaluation_limit=max_load_evaluation_limit
    )

    actuals_filt = actuals[filtering_condition]
    predictions_filt = predictions[filtering_condition]
    predictions_std_filt = predictions_std[filtering_condition]

    err_filt = predictions_filt - actuals_filt

    err_div_std = err_filt / predictions_std_filt

    weights = 10 * np.ones_like(err_div_std) / len(err_div_std)

    fig = plt.figure(figsize=(8, 6), dpi=100)
    plt.hist(
        err_div_std, weights=weights, bins=np.arange(-8, 8, 0.1), density=True, label="err/pred_std"
    )
    plt.xlabel("Normalized Residual")

    x = np.linspace(-5, 5, 100)

    plt.plot(x, stats.norm.pdf(x, 0, 1), label="normal dist")
    plt.plot(x, stats.norm.pdf(x, np.mean(err_div_std), np.std(err_div_std)), label="fit")
    plt.title(name_prefix)
    plt.legend()
    if not show:
        plt.close()
    return fig


def calculate_chi2(
    actuals: np.array,
    actuals_std: np.array,
    predictions: np.array,
    alpha_significance_level: float,
) -> Tuple[float, float]:
    # TODO: use the predictions_std to calculate the model form error if necessary
    dof = len(actuals) - 1
    chi2 = (((predictions - actuals) / actuals_std) ** 2).sum() / dof
    critical_value = stats.chi2.ppf(1 - alpha_significance_level, dof) / dof
    return chi2, critical_value


def calculate_general_metrics(
    actuals: np.array,
    predictions: np.array,
) -> Dict[str, float]:
    errors_all = predictions - actuals
    errors_all_norm = errors_all / actuals

    # Actuals can be 0 - convert inf to nan
    errors_all_norm[np.isinf(errors_all_norm)] = np.nan

    e_mean_norm = np.nanmean(errors_all_norm)
    e_std_norm = np.nanstd(errors_all_norm)
    mae = mean_absolute_error(predictions, actuals)
    mae_norm = np.abs(mae / np.nanmean(actuals))
    mane = np.nanmean(np.abs(errors_all_norm))
    r2 = r2_score(actuals, predictions)
    e_max_norm = np.max(np.abs(errors_all_norm))

    metrics = {
        "e_mean_norm": e_mean_norm,
        "e_std_norm": e_std_norm,
        "e_max_norm": e_max_norm,
        "mae": mae,
        "mae_norm": mae_norm,
        "mane": mane,
        "r2": r2,
    }

    return metrics


def compute_aggregates(
    arrays_to_aggregate: List[np.array],
    evaluation_groups: List[EvaluationGroup],
    exponent_coefficient: float,
    arrays_std_to_aggregate: Union[List[np.array], None] = None,
    sample_size: int = 4000,
) -> List[np.array]:
    """Get target values as a function considering the binning and weighting defined
    by the grouping."""

    aggregated_values = []
    aggregated_std_values = []

    seed = set_seed()

    # Generate Halton sequence samples in range 0 to 1
    halton_sampler = qmc.Halton(d=1, seed=seed)
    uniform_samples = halton_sampler.random(n=sample_size).flatten()

    for i_array, array in enumerate(arrays_to_aggregate):
        array_aggregated_values = []
        array_aggregated_std_values = []
        for group_bin in evaluation_groups:
            # Get indices and weights for the current group
            group_indices = group_bin.data_indices
            group_weights = group_bin.weight_vector

            # Calculate the aggregated targets
            dataset_1hz = array[group_indices].flatten()

            # TODO: Should be ensured by the predict method
            dataset_1hz[dataset_1hz < 0] = 0

            eq_1Hz_full = sum(dataset_1hz**exponent_coefficient * group_weights) ** (
                1 / exponent_coefficient
            )

            array_aggregated_values.append(eq_1Hz_full)
            dataset_std_1hz = arrays_std_to_aggregate[i_array][group_indices].flatten()

            samples_matrix = np.zeros((len(dataset_std_1hz), sample_size))
            set_seed()
            for i_1Hzload, (mean_value, std_value) in enumerate(
                zip(dataset_1hz, dataset_std_1hz, strict=False)
            ):
                # # the normal dist approximates dirac step function when std = 0
                if std_value == 0:
                    normal_samples = np.ones((1, sample_size)) * mean_value
                else:
                    normal_samples = norm.ppf(uniform_samples, loc=mean_value, scale=std_value)
                    # We need to shuffle the samples to avoid using identical fractiles across the
                    # loads in the aggregation group
                    np.random.shuffle(normal_samples)

                # fatigue loads cannot be negative
                normal_samples[normal_samples < 0] = 0
                samples_matrix[i_1Hzload] = normal_samples

            damage_samples_matrix = samples_matrix**exponent_coefficient
            eq_load_list = (
                np.sum(damage_samples_matrix * np.array(group_weights).reshape(-1, 1), axis=0)
            ) ** (1 / exponent_coefficient)
            eq_std_full = np.std(eq_load_list)

            array_aggregated_std_values.append(eq_std_full)

        aggregated_values.append(array_aggregated_values)
        aggregated_std_values.append(array_aggregated_std_values)

    return aggregated_values, aggregated_std_values


def get_aggregated_actuals_and_predictions(
    load_case_actuals: LoadCaseActuals, load_case_predictions: LoadCasePredictions
) -> Tuple[LoadCaseActuals, LoadCasePredictions]:
    load_case_aggregated_actuals = {}
    load_case_aggregated_predictions = {}

    for target_name, target_actuals in load_case_actuals.items():
        evaluation_groups = get_evaluation_groups(target_actuals)
        exponent_coefficient = find_wohler_from_target_name(target_name=target_name)

        aggregated_targets, aggregated_targets_std = compute_aggregates(
            arrays_to_aggregate=[
                load_case_actuals[target_name].values_as_np,
                load_case_predictions[target_name].values_as_np,
            ],
            arrays_std_to_aggregate=[
                load_case_actuals[target_name].values_std_as_np,
                load_case_predictions[target_name].values_std_as_np,
            ],
            evaluation_groups=evaluation_groups,
            exponent_coefficient=exponent_coefficient,
        )

        for x in aggregated_targets_std:
            check_nans = np.isnan(x).any()

            if check_nans:
                raise ValueError("Error: NaNs in aggregated targets std.")

        load_case_aggregated_actuals[target_name] = TargetValues(
            target_name=target_name,
            value_list=aggregated_targets[0],
            value_list_std=aggregated_targets_std[0],
        )

        load_case_aggregated_predictions[target_name] = TargetValues(
            target_name=target_name,
            value_list=aggregated_targets[1],
            value_list_std=aggregated_targets_std[1],
        )

    return load_case_aggregated_actuals, load_case_aggregated_predictions


def calculate_coverage_metrics(
    actuals: np.array,
    predictions: np.array,
    predictions_std: np.array,
) -> Dict[str, float]:
    # Calculate average coverage error for one and two standard deviation prediction intervals

    cov_1_std = np.abs(actuals - predictions) <= predictions_std
    cov_2_std = np.abs(actuals - predictions) <= 2 * predictions_std

    pred_coverage_error_1std = cov_1_std.mean() - 0.6827
    pred_coverage_error_2std = cov_2_std.mean() - 0.9545

    metrics = {
        "pred_coverage_error_1std": pred_coverage_error_1std,
        "pred_coverage_error_2std": pred_coverage_error_2std,
    }

    return metrics


def chi_squared_function(
    actuals: np.ndarray, actuals_std: np.ndarray, predictions: np.ndarray, dof: int, sigma_mf: float
) -> float:
    """Calculate chi-squared function for the model form uncertainty"""
    total_uncertainty = np.sqrt((actuals_std**2 + sigma_mf**2).astype(float))
    chi_squared = ((actuals - predictions) / total_uncertainty) ** 2
    chi_squared = chi_squared.sum() / dof

    return chi_squared


def calculate_chi_squared_test(
    actuals: np.array,
    actuals_std: np.array,
    predictions: np.array,
    alpha_significance_level: float,
) -> Dict[str, float]:
    """Calculate chi-squared, critical value and model form uncertainty"""
    dof = len(actuals) - 1
    sigma_model_form = 0.0
    cov_model_form = 0.0

    chi_squared = chi_squared_function(actuals, actuals_std, predictions, dof, sigma_model_form)
    critical_value = stats.chi2.ppf(1 - alpha_significance_level, dof) / dof
    passed = "True"

    if chi_squared > critical_value:
        passed = "False"

        def _chi_squared_scalar(x: float):
            return abs(
                chi_squared_function(actuals, actuals_std, predictions, dof, x) - critical_value
            )

        sigma_mf = optimize.minimize_scalar(_chi_squared_scalar)
        sigma_model_form = sigma_mf.x
        cov_model_form = sigma_model_form / np.nanmean(predictions)
    chi_squared_results = {
        Metric.CHI_SQUARED: chi_squared,
        Metric.CRITICAL_VALUE: critical_value,
        Metric.SIGMA_MODEL_FORM: sigma_model_form,
        Metric.COV_MODEL_FORM: cov_model_form,
        Metric.CHI_SQUARED_PASSED: passed,
    }

    return chi_squared_results


# TODO: Should take a Series
def freedman_diaconis_number_of_bins_compute(feature_values: pd.Series, min_bin: int = 6) -> int:
    """Defining optimal number of bins using Freedman-Diaconis Rule"""

    if feature_values.empty or len(feature_values) == 1:
        return 1

    IQR = iqr(np.array(feature_values), rng=(25, 75), scale=1.0)

    if IQR == 0:
        number_of_bins = 1
    else:
        N = len(feature_values)
        bin_width = 2 * IQR / (N ** (1 / 3))

        min_value, max_value = feature_values.min(), feature_values.max()
        value_range = max_value - min_value
        number_of_bins = int((value_range / bin_width) + 1)

    return min(number_of_bins, min_bin)


def get_feature_bins(feature: Feature, feature_values: np.array) -> List[FeatureBin]:
    feature_bins: List[FeatureBin] = []
    if feature.feature_value_type == FeatureValueType.DISCRETE:
        category_values = np.unique(feature_values)
        for category_value in category_values:
            feature_bins.append(
                FeatureBin(
                    feature=feature,
                    bin_value=category_value,
                    data_count=np.sum(feature_values == category_value),
                    lower_bound=category_value,
                    upper_bound=category_value,
                )
            )
    elif feature.feature_value_type == FeatureValueType.CONTINUOUS:
        feature_series = pd.Series(feature_values)
        number_of_bins = freedman_diaconis_number_of_bins_compute(feature_series)
        bins_df = pd.cut(feature_values, bins=number_of_bins)
        for group_indx, group in feature_series.groupby(bins_df, observed=False):
            feature_bins.append(
                FeatureBin(
                    feature=feature,
                    bin_value=group_indx.mid,
                    data_count=len(group),
                    lower_bound=group_indx.left,
                    upper_bound=group_indx.right,
                )
            )
    else:
        raise ValueError(f"Feature type {feature.feature_value_type} not supported.")
    return feature_bins


def extract_binned_metrics_df_per_feature(
    binned_metrics: BinnedLoadCaseMetrics, evaluation_metric: str, binning_feature: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    targets = list(binned_metrics.keys())

    metrics_df = pd.DataFrame()
    data_count_df = pd.DataFrame()
    for target in targets:
        metrics_list = []
        range_list = []
        data_count_list = []
        for feature, feature_bins in binned_metrics[target].items():
            if feature.name == binning_feature:
                for bin_range, bin_metrics in feature_bins.items():
                    range_list += [bin_range.bin_value]

                    if bin_metrics is not None:
                        metrics_list += [bin_metrics.__getattribute__(evaluation_metric)]
                        data_count_list += [bin_range.data_count]
                    else:
                        metrics_list += [np.nan]
                        data_count_list += [bin_range.data_count]
        metrics_df[target] = metrics_list
        data_count_df[target] = data_count_list

    metrics_df[binning_feature] = range_list
    metrics_df[binning_feature] = metrics_df[binning_feature].round(3)
    data_count_df[binning_feature] = range_list
    data_count_df[binning_feature] = data_count_df[binning_feature].round(3)

    metrics_df = metrics_df.set_index(binning_feature)
    data_count_df = data_count_df.set_index(binning_feature)

    return metrics_df, data_count_df


def get_evaluation_groups(target_actuals: TargetValues) -> List[EvaluationGroup]:
    evaluation_groups: List[EvaluationGroup] = []
    groupby_values = np.array(target_actuals.groupby)
    weightby_values = np.array(target_actuals.weightby)
    unique_groups = np.unique(groupby_values)
    for group in unique_groups:
        group_indices = np.flatnonzero(groupby_values == group)
        group_weights = weightby_values[group_indices]
        evaluation_groups.append(
            EvaluationGroup(
                group_name=group,
                data_count=len(group_indices),
                weight_vector=group_weights.tolist(),
                data_indices=group_indices.tolist(),
            )
        )
    return evaluation_groups


def find_wohler_from_target_name(target_name: str) -> float:
    """
    Determining the wohler slope from the target name.
    The expected format for the target name is "basename_m{XXXX}".
    If the target name deviates from the above structure, the function returns 1.

    Example 1 - Explicitly stated in the target name:
        target_name: MxBldRoot_m1000
        value: 10

    Example 2 - Not stated in the target name
        target_name: Power_mean
        value: 1
    """
    token = re.findall(".+_m(\d+)$", target_name)

    if token:
        value = float(token[0]) / 100
    else:
        value = 1

    return value


def get_heatmap_config(heatmap_config: Metric) -> Dict[str, Any]:
    if heatmap_config == Metric.CHI_SQUARED_PASSED:
        return {"cmap": sns.color_palette(["red", "green"]), "vmin": 0, "vmax": 1, "cbar": False}
    elif heatmap_config == Metric.E_MEAN_NORM:
        return {"cmap": "coolwarm", "vmin": -0.06, "vmax": 0.06, "cbar": True}
    elif heatmap_config == Metric.MAE_NORM:
        return {"cmap": "YlOrBr", "vmin": 0, "vmax": 0.06, "cbar": True}
    else:
        raise ValueError(
            f"Invalid heatmap config string: {heatmap_config}"
            + f" Available options are: {Metric.CHI_SQUARED},"
            + f"{Metric.E_MEAN_NORM}, {Metric.MAE_NORM}"
        )


def plot_heatmap_from_df(
    plot_df: pd.DataFrame,
    plot_title: str,
    heatmap_config: Metric,
    annot_df: Union[pd.DataFrame, None] = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(16, 8))

    # Convert floats to strings for y-ticks if the indexes are floats
    if all(isinstance(i, float) for i in plot_df.index.to_list()):
        y_ticks_labels = ["{:.3f}".format(i) for i in plot_df.index.to_list()]
    else:
        y_ticks_labels = plot_df.index.to_list()

    config = get_heatmap_config(heatmap_config)
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

    sns.heatmap(
        plot_df,
        vmin=config["vmin"],
        vmax=config["vmax"],
        annot=annot_df,
        annot_kws={"fontsize": "x-small"},
        fmt="",
        cmap=config["cmap"],
        cbar=config["cbar"],
        ax=ax,
        yticklabels=y_ticks_labels,
    )

    plt.title(plot_title)
    plt.ylabel("Bins: " + plot_df.index.name)
    plt.subplots_adjust(bottom=0.3, top=0.8)
    plt.yticks(rotation=0)

    plt.close()
    return fig


def get_target_metrics(
    target_actuals: np.array,
    target_actuals_std: np.array,
    target_predictions: np.array,
    target_predictions_std: np.array,
    alpha_significance_level: float = 0.05,
) -> TargetMetrics:
    """Get metrics for a single target.

    If a max_load_evaluation_limit is set, the data is filtered based on this limit.
    """
    general_metrics = calculate_general_metrics(target_actuals, target_predictions)

    coverage_metrics = calculate_coverage_metrics(
        target_actuals, target_predictions, target_predictions_std
    )

    # Filter out zeros from actuals_std and corresponding entries in actuals and predictions
    mask = target_actuals_std != 0.0
    target_actuals = target_actuals[mask]
    target_actuals_std = target_actuals_std[mask]
    target_predictions = target_predictions[mask]
    chi_squared_metrics = calculate_chi_squared_test(
        target_actuals,
        target_actuals_std,
        target_predictions,
        alpha_significance_level,
    )

    return TargetMetrics(**{**general_metrics, **coverage_metrics, **chi_squared_metrics})


def get_table_html(table_df: pd.DataFrame) -> str:
    """Returns an html representation of a df in a similar style
    as tables displayed in a jupyter notebook."""
    styles = [
        dict(
            selector=" ",
            props=[
                ("margin", "0"),
                ("font-family", '"Helvetica", "Arial", sans-serif'),
                ("border-collapse", "collapse"),
                ("border", "none"),
            ],
        ),
        dict(selector="thead", props=[("background-color", "#cc8484")]),
        dict(selector="tbody tr:nth-child(even)", props=[("background-color", "#fff")]),
        dict(selector="tbody tr:nth-child(odd)", props=[("background-color", "#eee")]),
        dict(
            selector="td",
            props=[
                ("padding", "0.3em .6em"),
            ],
        ),
        dict(
            selector="th",
            props=[
                ("font-size", "100%"),
                ("text-align", "left"),
                ("padding", "0.3em 0.6em"),
            ],
        ),
    ]
    return (table_df.style.set_table_styles(styles))._repr_html_()


def metric_to_title(metric: Metric) -> str:
    dic = {
        Metric.E_MEAN_NORM: "Mean relative prediction error",
        Metric.MAE_NORM: "Normalized MAE",
    }

    if metric not in dic:
        raise ValueError(f"Title for {metric} is not defined.")

    return dic[metric]
