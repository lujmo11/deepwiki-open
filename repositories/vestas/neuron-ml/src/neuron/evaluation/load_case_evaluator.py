import logging
from io import BytesIO
from typing import Dict, List, Union

import numpy as np

from neuron.evaluation.evaluation_utils import (
    BinnedLoadCaseMetrics,
    BinnedTargetMetrics,
    FeatureBin,
    StandardLoadCaseMetrics,
    extract_binned_metrics_df_per_feature,
    filter_arrays_based_on_target_limit,
    get_feature_bins,
    get_target_metrics,
    metric_to_title,
    plot_heatmap_from_df,
    plot_normalized_residual,
    plot_predictions_versus_actual,
)
from neuron.schemas.domain import (
    LoadCaseActuals,
    LoadCaseFeatureValues,
    LoadCasePredictions,
    Metric,
)

logger = logging.getLogger(__name__)


class LoadCaseEvaluator:
    """Evaluates a load case regression model on a full and binned dataset dataset"""

    def __init__(
        self,
        feature_names: List[str],
        max_load_evaluation_limit: Union[float, None] = None,
        alpha_significance_level: Union[float, None] = None,
        generate_coverage_plots: bool = False,
        calculate_aggregated_metrics: bool = True,
    ):
        self.feature_names = feature_names
        self.max_load_evaluation_limit = max_load_evaluation_limit
        self.alpha_significance_level = alpha_significance_level
        self.generate_coverage_plots = generate_coverage_plots
        self.calculate_aggregated_metrics = calculate_aggregated_metrics

    def get_standard_metrics(
        self,
        load_case_actuals: LoadCaseActuals,
        load_case_predictions: LoadCasePredictions,
    ) -> StandardLoadCaseMetrics:
        """Get metrics for actual vs predicted values for a load case."""
        standard_metrics: StandardLoadCaseMetrics = {}
        for target_name in load_case_predictions:
            target_actuals = load_case_actuals[target_name].values_as_np
            target_actuals_std = load_case_actuals[target_name].values_std_as_np
            target_predictions = load_case_predictions[target_name].values_as_np
            (
                target_actuals_filt,
                target_actuals_std_filt,
                target_predictions_filt,
            ) = filter_arrays_based_on_target_limit(
                actuals=target_actuals,
                max_load_evaluation_limit=self.max_load_evaluation_limit,
                arrays_to_filter=[
                    target_actuals,
                    target_actuals_std,
                    target_predictions,
                ],
            )

            target_predictions_std = load_case_predictions[target_name].values_std_as_np
            target_predictions_std_filt = filter_arrays_based_on_target_limit(
                actuals=target_actuals,
                max_load_evaluation_limit=self.max_load_evaluation_limit,
                arrays_to_filter=[target_predictions_std],
            )[0]

            # Check if any entry in actuals_std is zero
            if np.any(target_actuals_std_filt == 0.0):
                zero_count = np.sum(target_actuals_std_filt == 0.0)
                logger.debug(
                    f"Column {target_name}_std contains {zero_count} zeros out of "
                    f"{len(target_actuals_std_filt)}."
                    + " Entries are omitted from the chi squared test."
                )

            standard_metrics[target_name] = get_target_metrics(
                target_actuals=target_actuals_filt,
                target_actuals_std=target_actuals_std_filt,
                target_predictions=target_predictions_filt,
                target_predictions_std=target_predictions_std_filt,
                alpha_significance_level=self.alpha_significance_level,
            )
        return standard_metrics

    def _get_feature_binned_target_metrics(
        self,
        target_actuals: np.array,
        target_actuals_std: np.array,
        target_predictions: np.array,
        target_predictions_std: np.array,
        feature_values: np.array,
        feature_bins: List[FeatureBin],
    ) -> BinnedTargetMetrics:
        """Get binned metrics for a single target."""
        binned_target_metrics = {}
        for feature_bin in feature_bins:
            (
                bin_target_actuals_filt,
                bin_target_actuals_std_filt,
                bin_target_predictions_filt,
            ) = feature_bin.filter_arrays_based_on_feature_bin(
                feature_values=feature_values.reshape(-1, 1),
                arrays_to_filter=[
                    target_actuals,
                    target_actuals_std,
                    target_predictions,
                ],
            )

            bin_target_predictions_std_filt = feature_bin.filter_arrays_based_on_feature_bin(
                feature_values=feature_values.reshape(-1, 1),
                arrays_to_filter=[
                    target_predictions_std,
                ],
            )[0]

            if len(bin_target_actuals_filt) < 3:
                binned_target_metrics[feature_bin] = None
            else:
                # Check if any entry in bin_target_actuals_std is zero
                if np.any(bin_target_actuals_std_filt == 0.0):
                    zero_count = np.sum(bin_target_actuals_std_filt == 0.0)
                    logger.debug(
                        f"The feature {feature_bin.feature.name} contains {zero_count} "
                        f"datasets out of {len(bin_target_actuals_std_filt)} that include a "
                        f"target std of zero for the bin {feature_bin.bin_value:.2f}. "
                        "These entries are omitted from the chi squared test."
                    )
                binned_target_metrics[feature_bin] = get_target_metrics(
                    target_actuals=bin_target_actuals_filt,
                    target_actuals_std=bin_target_actuals_std_filt,
                    target_predictions=bin_target_predictions_filt,
                    target_predictions_std=bin_target_predictions_std_filt,
                    alpha_significance_level=self.alpha_significance_level,
                )
        return binned_target_metrics

    def get_binned_metrics(
        self,
        load_case_actuals: LoadCaseActuals,
        load_case_predictions: LoadCasePredictions,
        load_case_feature_values: LoadCaseFeatureValues,
    ) -> BinnedLoadCaseMetrics:
        """Get binned metrics for a load case.

        This involves getting the feature bins for each feature
        and then getting the binned metrics for each target.
        """
        feature_bins_all_features: Dict[str, List[FeatureBin]] = {}
        for feature, feature_values in load_case_feature_values.items():
            feature_bins_all_features[feature.name] = get_feature_bins(
                feature=feature, feature_values=feature_values
            )

        binned_load_case_metrics: BinnedLoadCaseMetrics = {}
        for target_name in load_case_predictions:
            feature_binned_target_metrics = {
                feature: self._get_feature_binned_target_metrics(
                    target_actuals=load_case_actuals[target_name].values_as_np,
                    target_actuals_std=load_case_actuals[target_name].values_std_as_np,
                    target_predictions=load_case_predictions[target_name].values_as_np,
                    target_predictions_std=load_case_predictions[target_name].values_std_as_np,
                    feature_values=load_case_feature_values[feature],
                    feature_bins=feature_bins_all_features[feature.name],
                )
                for feature in load_case_feature_values.keys()
            }
            binned_load_case_metrics[target_name] = feature_binned_target_metrics
        return binned_load_case_metrics

    def get_standard_plots(
        self,
        load_case_actuals: LoadCaseActuals,
        load_case_predictions: LoadCasePredictions,
    ) -> Dict[str, BytesIO]:
        """Get evaluation plots."""

        target_metrics = self.get_standard_metrics(
            load_case_actuals=load_case_actuals,
            load_case_predictions=load_case_predictions,
        )

        plots_data: Dict[str, BytesIO] = {}
        max_load_evaluation_limit = self.max_load_evaluation_limit
        for target_name in load_case_predictions:
            target_actuals = load_case_actuals[target_name].values_as_np
            target_predictions = load_case_predictions[target_name].values_as_np
            pred_vs_actual_fig = plot_predictions_versus_actual(
                actuals=target_actuals,
                predictions=target_predictions,
                metrics=target_metrics[target_name],
                max_load_evaluation_limit=max_load_evaluation_limit,
                name_prefix=target_name,
                show=False,
            )
            pred_vs_actual_bytes = BytesIO()
            pred_vs_actual_fig.savefig(pred_vs_actual_bytes, format="png")
            plots_data[f"{target_name}_pred_vs_act_plot.png"] = pred_vs_actual_bytes

        return plots_data

    def get_coverage_plots(
        self,
        load_case_actuals: LoadCaseActuals,
        load_case_predictions: LoadCasePredictions,
    ) -> Dict[str, BytesIO]:
        """Get coverage plots."""

        plots_data: Dict[str, BytesIO] = {}
        max_load_evaluation_limit = self.max_load_evaluation_limit
        for target_name in load_case_predictions:
            target_actuals = load_case_actuals[target_name].values_as_np
            target_predictions = load_case_predictions[target_name].values_as_np
            target_predictions_std = load_case_predictions[target_name].values_std_as_np

            normalized_residual_fig = plot_normalized_residual(
                actuals=target_actuals,
                predictions=target_predictions,
                predictions_std=target_predictions_std,
                max_load_evaluation_limit=max_load_evaluation_limit,
                name_prefix=target_name,
                show=False,
            )
            normalized_residual_bytes = BytesIO()
            normalized_residual_fig.savefig(normalized_residual_bytes, format="png")
            plots_data[f"{target_name}_normalized_residual.png"] = normalized_residual_bytes

        return plots_data

    def get_binned_metrics_per_feature_heatmap(
        self,
        load_case_binned_metrics: BinnedLoadCaseMetrics,
    ) -> Dict[str, BytesIO]:
        binned_metrics_plots_data: Dict[str, BytesIO] = {}
        for var in self.feature_names:
            for metric in [Metric.MAE_NORM, Metric.E_MEAN_NORM]:
                metric_df, data_count_df = extract_binned_metrics_df_per_feature(
                    binned_metrics=load_case_binned_metrics,
                    evaluation_metric=metric,
                    binning_feature=var,
                )

                annotations_df = (
                    metric_df.round(2).astype(str) + "\n(" + data_count_df.astype(str) + ")"
                )

                fig = plot_heatmap_from_df(
                    plot_df=metric_df,
                    plot_title=metric_to_title(metric) + " (number of evaluation points)",
                    heatmap_config=metric,
                    annot_df=annotations_df,
                )
                binned_metrics_bytes = BytesIO()
                fig.savefig(binned_metrics_bytes, format="png")
                binned_metrics_plots_data[f"{metric}_heatmap_{var}.png"] = binned_metrics_bytes

            # Chi squared test

            metric = Metric.CHI_SQUARED_PASSED
            metric_df, data_count_df = extract_binned_metrics_df_per_feature(
                binned_metrics=load_case_binned_metrics,
                evaluation_metric=metric,
                binning_feature=var,
            )

            annotations_df = (
                metric_df.round(2).astype(str) + "\n(" + data_count_df.astype(str) + ")"
            )

            domain_test_df = metric_df

            plot_title = (
                f"Binned test results with a {(1-self.alpha_significance_level)*100:.0f}%"
                f" confidence Annotations: Normalized MAE (number of of evaluation points)."
            )

            fig = plot_heatmap_from_df(
                plot_df=domain_test_df,
                plot_title=plot_title,
                heatmap_config=metric,
                annot_df=annotations_df,
            )
            binned_metrics_bytes = BytesIO()
            fig.savefig(binned_metrics_bytes, format="png")
            binned_metrics_plots_data[f"{metric}_heatmap_{var}.png"] = binned_metrics_bytes

        return binned_metrics_plots_data
