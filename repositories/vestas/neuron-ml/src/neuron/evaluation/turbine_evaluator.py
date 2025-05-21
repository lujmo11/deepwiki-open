from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuron.evaluation.evaluation_utils import (
    CONDITION_NAMES,
    CONDITION_OPERATIONS,
    LoadBin,
    ModelAcceptanceCriteria,
    TargetMetrics,
    calculate_general_metrics,
    compute_aggregates,
    find_wohler_from_target_name,
    get_evaluation_groups,
    get_table_html,
    get_target_metrics,
    metric_to_title,
    plot_heatmap_from_df,
)
from neuron.evaluation.load_case_evaluator import LoadCaseEvaluator
from neuron.schemas.domain import (
    CalculationType,
    Metric,
    TargetValues,
    TurbineActuals,
    TurbinePredictions,
)
from neuron.schemas.training_run_config import (
    EvaluationConfig,
    LoadCaseTrainingRunConfig,
    TurbineConfig,
)

# Structure: [target][dlc][bin] metrics
BinnedTurbineRunMetrics = Dict[str, Dict[str, Dict[LoadBin, TargetMetrics]]]


class TurbineEvaluator:
    """Evaluates a collection of load case regression models on binned datasets"""

    def __init__(
        self,
        load_case_configs: List[LoadCaseTrainingRunConfig],
        eval_config: EvaluationConfig,
        turbine_config: TurbineConfig,
        apply_extreme_load_case_filter: bool = True,
    ):
        self.load_case_configs = load_case_configs
        self.eval_config = eval_config
        self.turbine_config = turbine_config

        self.alpha_significance_level = eval_config.alpha_significance_level
        self.no_of_bins = 6
        self.design_load_eval_margin = 0.2
        self.min_datapoints_in_bin = 50

        self.agg_dlcs_name = "all_dlcs"

        self.calculate_aggregated_metrics = any(
            load_case.calculate_aggregated_metrics is not None for load_case in load_case_configs
        )
        self.extreme_load_dlc_exist = any(
            load_case.calculation_type == CalculationType.EXTREME for load_case in load_case_configs
        )

        dlc_target_combinations = pd.DataFrame()

        if apply_extreme_load_case_filter:
            dlc_target_combinations["load_case_name"] = [
                load_case.name
                for load_case in load_case_configs
                for _ in load_case.target_list.targets
                if load_case.calculation_type == CalculationType.EXTREME
            ]
            dlc_target_combinations["target"] = [
                target
                for load_case in load_case_configs
                for target in load_case.target_list.targets
                if load_case.calculation_type == CalculationType.EXTREME
            ]
        else:
            dlc_target_combinations["load_case_name"] = [
                load_case.name
                for load_case in load_case_configs
                for _ in load_case.target_list.targets
            ]
            dlc_target_combinations["target"] = [
                target
                for load_case in load_case_configs
                for target in load_case.target_list.targets
            ]

        self.dlc_target_combinations = dlc_target_combinations
        self.eval_config = self.eval_config

    def get_turbine_aggregated_metrics(
        self, turbine_predictions: TurbinePredictions, turbine_actuals: TurbineActuals
    ) -> Dict[str, TargetMetrics]:
        """Get aggregated metrics per target across all load cases."""
        aggregated_turbine_metrics = defaultdict(lambda: defaultdict(TargetMetrics))

        agg_load_cases = [dlc for dlc in self.load_case_configs if dlc.calculate_aggregated_metrics]

        # Validation on the TrainingRunConfig class ensures that all load cases where
        # calculate_aggregated_metrics is True have the same target list
        target_list = turbine_actuals[agg_load_cases[0].name].keys()

        load_case_aggregated_actuals = {}
        load_case_aggregated_predictions = {}

        for target in target_list:
            full_target_actual_list = []
            full_target_actual_std_list = []
            full_target_prediction_list = []
            full_target_prediction_std_list = []
            group_list = []
            weight_vector = []

            exponent_coefficient = find_wohler_from_target_name(target_name=target)
            weight_vector = []
            for load_case in agg_load_cases:
                load_case_actuals = turbine_actuals[load_case.name]
                # only calculate metrics if the load case in
                # self.load_case_configs has the aggregation flag set

                load_case_predictions = turbine_predictions[load_case.name]

                target_actuals = load_case_actuals[target]
                target_predictions = load_case_predictions[target]

                evaluation_groups = get_evaluation_groups(target_actuals)

                aggregated_targets_lc, aggregated_targets_lc_std = compute_aggregates(
                    arrays_to_aggregate=[
                        target_actuals.values_as_np,
                        target_predictions.values_as_np,
                    ],
                    arrays_std_to_aggregate=[
                        target_actuals.values_std_as_np,
                        target_predictions.values_std_as_np,
                    ],
                    evaluation_groups=evaluation_groups,
                    exponent_coefficient=exponent_coefficient,
                )

                full_target_actual_list += target_actuals.value_list
                full_target_actual_std_list += target_actuals.value_list_std
                full_target_prediction_list += target_predictions.value_list
                full_target_prediction_std_list += target_predictions.value_list_std
                group_list += target_actuals.groupby
                weight_vector += target_actuals.weightby

                # capture individual dlc-target results
                aggregated_turbine_metrics[load_case.name][target] = get_target_metrics(
                    target_actuals=np.array(aggregated_targets_lc[0]),
                    target_actuals_std=np.array(aggregated_targets_lc_std[0]),
                    target_predictions=np.array(aggregated_targets_lc[1]),
                    target_predictions_std=np.array(aggregated_targets_lc_std[1]),
                    alpha_significance_level=self.alpha_significance_level,
                )

            target_aggregated_actuals = TargetValues(
                target_name=target,
                value_list=full_target_actual_list,
                value_list_std=full_target_actual_std_list,
                groupby=group_list,
                weightby=weight_vector,
            )

            evaluation_groups = get_evaluation_groups(target_aggregated_actuals)

            target_aggregated, target_aggregated_std = compute_aggregates(
                arrays_to_aggregate=[
                    np.array(full_target_actual_list),
                    np.array(full_target_prediction_list),
                ],
                arrays_std_to_aggregate=[
                    np.array(full_target_actual_std_list),
                    np.array(full_target_prediction_std_list),
                ],
                evaluation_groups=evaluation_groups,
                exponent_coefficient=exponent_coefficient,
            )

            # Create LoadCaseActuals and LoadCasePredictions structures
            # to enable using the LoadCaseEvaluator
            load_case_aggregated_actuals[target] = TargetValues(
                target_name=target,
                value_list=target_aggregated[0],
                value_list_std=target_aggregated_std[0],
            )
            load_case_aggregated_predictions[target] = TargetValues(
                target_name=target,
                value_list=target_aggregated[1],
                value_list_std=target_aggregated_std[1],
            )

        # Use LoadCaseEvaluator to calculate metrics
        aggregated_load_cases_evaluator = LoadCaseEvaluator(
            feature_names=[],
            max_load_evaluation_limit=0.0,
            alpha_significance_level=self.alpha_significance_level,
            calculate_aggregated_metrics=True,
        )

        aggregated_turbine_metrics[
            self.agg_dlcs_name
        ] = aggregated_load_cases_evaluator.get_standard_metrics(
            load_case_actuals=load_case_aggregated_actuals,
            load_case_predictions=load_case_aggregated_predictions,
        )

        return aggregated_turbine_metrics

    def get_turbine_binned_metrics(
        self, turbine_predictions: TurbinePredictions, turbine_actuals: TurbineActuals
    ) -> BinnedTurbineRunMetrics:
        """Compute metrics on a dataset that is binned on target level.

        The binning function does not contemplate filtering of data through the
        max_load_level_limit. The number of bins are not configurable and set to self.no_of_bins.
        """

        turbine_binned_metrics = {}
        if not self.dlc_target_combinations.empty:
            load_ranges = defaultdict(lambda: {"max_load": float("-inf"), "min_load": float("inf")})

            for load_case in self.load_case_configs:
                for target in load_case.target_list.targets:
                    load_ranges[target]["max_load"] = max(
                        max(turbine_predictions[load_case.name][target].value_list),
                        load_ranges[target]["max_load"],
                    )

                    load_ranges[target]["min_load"] = min(
                        min(turbine_predictions[load_case.name][target].value_list),
                        load_ranges[target]["min_load"],
                    )

            for target, load_range in load_ranges.items():
                bin_size = (load_range["max_load"] - load_range["min_load"]) / self.no_of_bins
                ranges = [
                    (load_range["min_load"] + i * bin_size) for i in range(self.no_of_bins + 1)
                ]

                dlc_metrics = {}

                target_dlc_list = self.dlc_target_combinations[
                    self.dlc_target_combinations["target"] == target
                ]

                if not target_dlc_list.empty:
                    for dlc_in_target in target_dlc_list["load_case_name"].to_list():
                        model_data = pd.DataFrame()
                        model_data["actuals"] = turbine_actuals[dlc_in_target][target].value_list
                        model_data["predictions"] = turbine_predictions[dlc_in_target][
                            target
                        ].value_list
                        model_data["bins"] = pd.cut(
                            model_data["predictions"], bins=ranges, include_lowest=True
                        )
                        load_groups = model_data.groupby("bins", observed=False)

                        dlc_metrics[dlc_in_target] = self._calculate_range_metrics(load_groups)

                    turbine_binned_metrics[target] = dlc_metrics

        return turbine_binned_metrics

    def _collect_metrics_data(
        self, turbine_binned_metrics: BinnedTurbineRunMetrics, target_name: str
    ):
        data_dict = {
            "dlc": [],
            "load_levels": [],
            "upper_bound": [],
            Metric.DATA_COUNT: [],
            Metric.MAE_NORM: [],
            Metric.E_MEAN_NORM: [],
        }

        for dlc in turbine_binned_metrics[target_name]:
            for load_range, metrics in turbine_binned_metrics[target_name][dlc].items():
                data_dict["dlc"].append(dlc)
                data_dict["load_levels"].append(load_range.bin_value)
                data_dict["upper_bound"].append(load_range.upper_bound)
                data_dict[Metric.DATA_COUNT].append(load_range.data_count)

                if metrics is not None:
                    data_dict[Metric.MAE_NORM].append(metrics.mae_norm)
                    data_dict[Metric.E_MEAN_NORM].append(metrics.e_mean_norm)
                else:
                    data_dict[Metric.MAE_NORM].append(np.nan)
                    data_dict[Metric.E_MEAN_NORM].append(np.nan)

        df_out = pd.DataFrame(data_dict)

        return df_out

    def get_metrics_heatmaps_for_target_bins_across_ext_load_cases(
        self, turbine_binned_metrics: BinnedTurbineRunMetrics
    ) -> Dict[str, BytesIO]:
        """Get heatmap of mae_norm and e_mean_norm for target
        bins as a function of extreme load cases. One plot per target."""

        design_loads_ext = self.turbine_config.design_loads_ext
        plots_data: Dict[str, BytesIO] = {}

        for target_name in turbine_binned_metrics:
            df_metrics = self._collect_metrics_data(turbine_binned_metrics, target_name)

            # Pivot the dataframe to create multi-level columns
            pivot_df = df_metrics.pivot_table(
                index="load_levels",
                columns="dlc",
                values=[Metric.MAE_NORM, Metric.E_MEAN_NORM, Metric.DATA_COUNT],
                dropna=False,
            )

            # Invert to have the highest load level at the top
            pivot_df = pivot_df.iloc[::-1]

            # Find the design load index
            design_load_index = -1
            if design_loads_ext is not None and target_name in design_loads_ext:
                design_load = design_loads_ext[target_name]
                for i, upper_bound in enumerate(df_metrics["upper_bound"].unique()):
                    if design_load < upper_bound:
                        design_load_index = len(df_metrics["upper_bound"].unique()) - i
                        break

            metrics = [Metric.MAE_NORM, Metric.E_MEAN_NORM]
            # Create heatmap
            for metric in metrics:
                plot_df = pivot_df[metric]
                title = metric_to_title(metric) + " for " + target_name
                fig = plot_heatmap_from_df(
                    plot_df=plot_df,
                    annot_df=pivot_df[Metric.DATA_COUNT],
                    plot_title=title,
                    heatmap_config=metric,
                )
                ax = fig.gca()
                if design_load_index >= 0:
                    ax.axhline(y=design_load_index - 1, color="r")
                    ax.axhline(y=design_load_index, color="r")
                plot_data_target = BytesIO()
                fig.savefig(plot_data_target, format="png")
                plt.close()
                plots_data[f"{target_name}_" + metric + "_dlc_overview.png"] = plot_data_target

        return plots_data

    def get_detailed_rel_design_error_plots(
        self, turbine_predictions: TurbinePredictions, turbine_actuals: TurbineActuals
    ) -> Dict[str, Dict[str, BytesIO]]:
        """Plot error vs relative difference to design load across load cases and targets."""

        output_plots = defaultdict(dict)

        design_loads_ext = self.turbine_config.design_loads_ext

        # Loop through all load cases and targets
        for dlc in self.dlc_target_combinations["load_case_name"].unique():
            dlc_targets = self.dlc_target_combinations[
                self.dlc_target_combinations["load_case_name"] == dlc
            ]

            for target in dlc_targets["target"].unique():
                actuals = np.array(turbine_actuals[dlc][target].value_list)

                if design_loads_ext is not None and target in design_loads_ext:
                    design_load = design_loads_ext[target]

                    actuals = np.array(turbine_actuals[dlc][target].value_list)
                    predictions = np.array(turbine_predictions[dlc][target].value_list)

                    # only create plots for targets with data inside the lower design load margin
                    limit = (1 - self.design_load_eval_margin) * design_load
                    make_plot = (
                        actuals.min() < limit if target.endswith("min") else actuals.max() > limit
                    )

                    if make_plot:
                        fig = self._rel_design_vs_error_plot(
                            actuals, predictions, design_load, target
                        )
                        plot_data_target = BytesIO()
                        fig.savefig(plot_data_target, format="png")
                        plt.close()
                        output_plots[dlc][target + "_err_rel_design.png"] = plot_data_target

        return output_plots

    def get_ext_design_load_bin_metrics(
        self, turbine_predictions: TurbinePredictions, turbine_actuals: TurbineActuals
    ) -> BinnedTurbineRunMetrics:
        """
        Calculate metrics considering datapoints that are within the design load bin across
        targets and load cases.
        """
        design_loads_ext = self.turbine_config.design_loads_ext

        binned_metrics: BinnedTurbineRunMetrics = defaultdict(lambda: defaultdict(dict))

        # Loop through all load cases and targets
        for dlc in self.dlc_target_combinations["load_case_name"].unique():
            dlc_targets = self.dlc_target_combinations[
                self.dlc_target_combinations["load_case_name"] == dlc
            ]

            target_masks = {}

            for target in dlc_targets["target"].unique():
                actuals = np.array(turbine_actuals[dlc][target].value_list)

                if design_loads_ext is not None and target in design_loads_ext:
                    design_load = design_loads_ext[target]
                    load_eval_threshold = (1 + self.design_load_eval_margin) * design_load

                    # Identifying the datapoints that result in loads exceeding the upper range
                    # of the design load bin. For these evaluation conditions, the turbine would
                    # not be suitable.
                    if target.endswith("min"):
                        target_masks[target] = actuals >= load_eval_threshold
                    else:
                        target_masks[target] = actuals <= load_eval_threshold

            # The combined mask represents the subset of data that is inside the
            # load_eval_threshold across all targets. If, for a specific target, the actual value
            # exceeds the upper range, then that target would govern the relative load.

            if target_masks:
                combined_mask = np.all(list(target_masks.values()), axis=0)
                index_array = np.where(combined_mask)
            else:
                index_array = []

            for target in dlc_targets["target"].unique():
                if design_loads_ext is not None and target in design_loads_ext:
                    actuals = np.array(turbine_actuals[dlc][target].value_list)[index_array]
                    actuals_std = np.array(turbine_actuals[dlc][target].value_list_std)[index_array]
                    predictions = np.array(turbine_predictions[dlc][target].value_list)[index_array]
                    predictions_std = np.array(turbine_predictions[dlc][target].value_list_std)[
                        index_array
                    ]
                    design_load = design_loads_ext[target]
                    # Calculate NMAE for the subset of data that is within +/- x% of the design load
                    (
                        subset_actuals,
                        subset_actuals_std,
                        subset_predictions,
                        subset_predictions_std,
                    ) = self._get_design_load_bin_subsets_for_actuals_and_predictions(
                        target, actuals, actuals_std, predictions, predictions_std, design_load
                    )

                    data_count = len(subset_actuals)
                    if len(subset_actuals) > 0:
                        bin_metrics = get_target_metrics(
                            subset_actuals,
                            subset_actuals_std,
                            subset_predictions,
                            subset_predictions_std,
                            alpha_significance_level=self.alpha_significance_level,
                        )
                    else:
                        bin_metrics = None

                else:  # Calculate NMAE for the full dataset
                    actuals = np.array(turbine_actuals[dlc][target].value_list)
                    actuals_std = np.array(turbine_actuals[dlc][target].value_list_std)
                    predictions = np.array(turbine_predictions[dlc][target].value_list)
                    predictions_std = np.array(turbine_predictions[dlc][target].value_list_std)

                    data_count = len(actuals)
                    design_load = np.mean(np.array(turbine_actuals[dlc][target].value_list))
                    if data_count > 0:
                        bin_metrics = get_target_metrics(
                            actuals,
                            actuals_std,
                            predictions,
                            predictions_std,
                        )
                    else:
                        bin_metrics = None

                bin = LoadBin(
                    bin_value=design_load,
                    data_count=data_count,
                    lower_bound=(1 - self.design_load_eval_margin) * design_load,
                    upper_bound=(1 + self.design_load_eval_margin) * design_load,
                )

                binned_metrics[target][dlc][bin] = bin_metrics

        return binned_metrics

    def find_excluded_targets_from_ext_bin_eval(
        self, binned_turbine_metrics: BinnedTurbineRunMetrics
    ) -> pd.DataFrame:
        """
        Determine the target and dlc combinations that have been fully disregarded (metrics=None)
        from the binned metrics calculation.
        """

        excluded_dlc_name = []
        excluded_target_name = []
        for dlc in self.dlc_target_combinations["load_case_name"].unique():
            dlc_targets = self.dlc_target_combinations[
                self.dlc_target_combinations["load_case_name"] == dlc
            ]

            for target in dlc_targets["target"].unique():
                metrics = list(binned_turbine_metrics[target][dlc].values())
                disregarded_all_bins = all(metric is None for metric in metrics)

                if disregarded_all_bins:
                    excluded_dlc_name += [dlc]
                    excluded_target_name += [target]

        filtered_target_models = pd.DataFrame(
            {"load_case_name": excluded_dlc_name, "target_name": excluded_target_name}
        )

        return filtered_target_models

    def create_design_load_bin_heatmap(
        self, binned_metrics: BinnedTurbineRunMetrics, metric: str
    ) -> Dict[str, BytesIO]:
        """Create heatmap of metric for design load bin across targets and load cases."""
        plots_data: Dict[str, BytesIO] = {}

        metrics_df_data = []
        for target, target_data in binned_metrics.items():
            for dlc, dlc_data in target_data.items():
                for bin, metrics in dlc_data.items():
                    if metrics is not None:
                        metrics_df_data.append(
                            {
                                "load_case_name": dlc,
                                "target": target,
                                metric: metrics.__getattribute__(metric),
                                Metric.DATA_COUNT: bin.data_count,
                                Metric.CHI_SQUARED_PASSED: metrics.chi_squared_passed,
                            }
                        )
                    else:
                        metrics_df_data.append(
                            {
                                "load_case_name": dlc,
                                "target": target,
                                metric: 0,
                                Metric.DATA_COUNT: bin.data_count,
                                Metric.CHI_SQUARED_PASSED: False,
                            }
                        )

        metrics_df = pd.DataFrame(metrics_df_data)

        plots_data: Dict[str, BytesIO] = {}

        metric_df = metrics_df.pivot_table(index="load_case_name", columns="target", values=metric)
        data_count_df = metrics_df.pivot_table(
            index="load_case_name", columns="target", values=Metric.DATA_COUNT
        )
        chi_squared_test_df = metrics_df.pivot_table(
            index="load_case_name", columns="target", values=Metric.CHI_SQUARED_PASSED
        ).astype(bool)

        data_count_df = data_count_df.replace([np.inf, np.nan], 0)
        data_count_df = data_count_df.astype(int)

        # Invert to have the highest load level at the top
        metric_df = metric_df.iloc[::-1]
        data_count_df = data_count_df.iloc[::-1]
        chi_squared_test_df = chi_squared_test_df.iloc[::-1]

        annotations_df = (
            chi_squared_test_df.astype(int).astype(str) + "\n(" + data_count_df.astype(str) + ")"
        )

        # Create metric heatmap
        title = metric_to_title(metric) + " for design load bin"
        fig = plot_heatmap_from_df(
            plot_df=metric_df, annot_df=annotations_df, plot_title=title, heatmap_config=metric
        )
        plot_data_metric = BytesIO()
        fig.savefig(plot_data_metric, format="png")
        plt.close()
        plots_data["design_load_bin_" + metric + "_overview.png"] = plot_data_metric

        return plots_data

    def get_turbine_metrics_overview(
        self,
        ext_design_load_bin_metrics: Union[BinnedTurbineRunMetrics, None],
        turbine_aggregated_metrics: Union[Dict[str, Dict[str, TargetMetrics]], None],
    ) -> List[pd.DataFrame]:
        ext_dlc_metrics = defaultdict(dict)
        if ext_design_load_bin_metrics is not None:
            for target, dlc_metrics in ext_design_load_bin_metrics.items():
                for dlc, bin_metrics in dlc_metrics.items():
                    # ext_design_load_bin_metrics only has the design load bin
                    metrics = list(bin_metrics.values())[0]
                    if metrics is not None:
                        ext_dlc_metrics[target][dlc] = metrics

        agg_all_metrics = defaultdict(dict)
        agg_dlc_metrics = defaultdict(dict)

        if turbine_aggregated_metrics is not None:
            for dlc in turbine_aggregated_metrics:
                if dlc == self.agg_dlcs_name:
                    self._add_metrics_to_dict(turbine_aggregated_metrics, dlc, agg_all_metrics)
                else:
                    self._add_metrics_to_dict(turbine_aggregated_metrics, dlc, agg_dlc_metrics)

        # df overview
        agg_all_df = self._create_df_overview(agg_all_metrics)
        agg_dlc_df = self._create_df_overview(agg_dlc_metrics)
        ext_dlc_df = self._create_df_overview(ext_dlc_metrics)

        return [agg_all_df, agg_dlc_df, ext_dlc_df]

    def get_eval_criteria_html(self, eval_criteria: List[ModelAcceptanceCriteria]) -> None:
        html_text = ""
        for criteria in eval_criteria:
            condition = CONDITION_NAMES[criteria.condition]
            if criteria.or_metric:
                condition_or_metric = CONDITION_NAMES[criteria.or_metric_condition]
                html_text += f"<li>{criteria.metric} {condition} {criteria.value} or \
                    {criteria.or_metric} {condition_or_metric} \
                    {criteria.or_metric_value} </li>\n"
            else:
                html_text += f"<li>{criteria.metric} {condition} {criteria.value}</li>\n"

        html_text += """</ul>
                    <p style="height: 10px;"></p>"""

        return html_text

    def filter_data_based_on_eval_criteria(
        self, metrics_df: pd.DataFrame, eval_criteria: List[ModelAcceptanceCriteria]
    ) -> pd.DataFrame:
        criteria_query = pd.Series([True] * len(metrics_df))
        for criteria in eval_criteria:
            operation = CONDITION_OPERATIONS[criteria.condition]

            if criteria.or_metric:
                either_operation = CONDITION_OPERATIONS[criteria.or_metric_condition]
                criteria_query &= operation(metrics_df[criteria.metric], criteria.value) | (
                    either_operation(metrics_df[criteria.or_metric], criteria.or_metric_value)
                )
            else:
                criteria_query &= operation(metrics_df[criteria.metric], criteria.value)

        filtered_data = metrics_df[~criteria_query].reset_index(drop=True)

        return filtered_data

    def get_turbine_overview_html(
        self,
        df_list: List[pd.DataFrame],
        filtered_ext_models: pd.DataFrame,
    ) -> str:
        agg_all_df = df_list[0]
        agg_dlc_df = df_list[1]
        ext_dlc_df = df_list[2]

        fat_acceptance_criteria = self.eval_config.fat_model_acceptance_criteria
        ext_acceptance_criteria = self.eval_config.ext_model_acceptance_criteria

        fat_acceptance_criteria_html = self.get_eval_criteria_html(fat_acceptance_criteria)
        ext_acceptance_criteria_html = self.get_eval_criteria_html(ext_acceptance_criteria)

        if agg_all_df.empty and agg_dlc_df.empty and ext_dlc_df.empty:
            return None

        # Create df with fat aggregated targets metrics
        if agg_all_df.empty:
            filtered_agg_all_df = pd.DataFrame(
                [""] * len(agg_all_df.columns), index=agg_all_df.columns
            ).transpose()
        else:
            filtered_agg_all_df = self.filter_data_based_on_eval_criteria(
                metrics_df=agg_all_df, eval_criteria=fat_acceptance_criteria
            )

        # Create df with fat load case targets metrics
        if agg_dlc_df.empty:
            filtered_agg_dlc_df = pd.DataFrame(
                [""] * len(agg_dlc_df.columns), index=agg_dlc_df.columns
            ).transpose()
        else:
            filtered_agg_dlc_df = self.filter_data_based_on_eval_criteria(
                metrics_df=agg_dlc_df, eval_criteria=fat_acceptance_criteria
            )

        # Create df with ext load case targets metrics
        if ext_dlc_df.empty:
            filtered_ext_dlc_df = pd.DataFrame(
                [""] * len(ext_dlc_df.columns), index=ext_dlc_df.columns
            ).transpose()
        else:
            filtered_ext_dlc_df = self.filter_data_based_on_eval_criteria(
                metrics_df=ext_dlc_df, eval_criteria=ext_acceptance_criteria
            )

        if not filtered_agg_all_df.empty:
            html_all_agg = get_table_html(filtered_agg_all_df)
        else:
            html_all_agg = "--- No fatigue aggregated load models fail the evaluation criteria ---"

        if not filtered_agg_dlc_df.empty:
            html_dlc_agg = get_table_html(filtered_agg_dlc_df)
        else:
            html_dlc_agg = "--- No fatigue load models fail the evaluation criteria ---"

        if not filtered_ext_dlc_df.empty:
            html_dlc_ext = get_table_html(filtered_ext_dlc_df)

            if not filtered_ext_models.empty:
                html_filtered_models = get_table_html(filtered_ext_models)
            else:
                html_filtered_models = "--- All models were captured in the design bin ---"
        else:
            html_dlc_ext = "--- No extreme load models fail the evaluation criteria ---"
            html_filtered_models = "--- No models were filtered ---"

        html = f"""          
        <html>        
        <head>        
        <style>        
        h3, body {{        
        font-family: Helvetica, Arial, sans-serif;        
        }}        
        .container {{  
        width: 800px;  
        margin: auto;  
        word-wrap: break-word;  
        }}  
        ul {{
            text-align: justify;
            line-style-type: disc;
             padding-left: 40px;
        }}
        </style>        
        </head>        
        <body>  
        <div class="container">         
        <h3>Fatigue load models </h3>          
        <p>The following table shows the targets that don't meet the acceptance criteria for
        the aggregated loads across all fatigue load cases. The acceptance criteria is described
        as follows:
        {fat_acceptance_criteria_html}   
        {html_all_agg}        
        <br><br>
        <p>The following table shows the targets that don't meet the acceptance criteria for     
        for the wohler aggregated loads for each individual load case.</p>        
        {html_dlc_agg}        
        <br><br>         
        <h3>Extreme load models </h3>          
        <p>The following table shows the targets that don't meet the acceptance criteria for   
        all extreme load cases. The acceptance criteria for extreme load cases is as follows:
        {ext_acceptance_criteria_html}    
        The metrics have been calculated based on the test data that is within +/- 20% of the
        design load. </p>        
        {html_dlc_ext}
        <br><br>
        <p>The following load case models are disregarded from the extreme overview as they 
        don't result in predictions within ithe mentioned design load range. </p>
        {html_filtered_models}
        <br><br>
        </div>        
        </body>          
        </html>          
        """

        return html

    def _create_df_overview(self, dictionary: Dict[str, Dict[str, TargetMetrics]]) -> pd.DataFrame:
        df_metrics = pd.DataFrame()
        for target, target_dict in dictionary.items():
            for dlc, metrics in target_dict.items():
                metrics_dict = metrics.dict()
                metrics_dict["target"] = target
                metrics_dict["dlc"] = dlc
                df_metrics = pd.concat(
                    [df_metrics, pd.DataFrame.from_records([metrics_dict])], ignore_index=True
                )

        if df_metrics.empty:
            return df_metrics

        columns = [
            "target",
            "dlc",
            Metric.MAE_NORM,
            Metric.E_MEAN_NORM,
            Metric.E_STD_NORM,
            Metric.SIGMA_MODEL_FORM,
            Metric.COV_MODEL_FORM,
        ]

        df_metrics = df_metrics.reindex(columns=columns)
        df_metrics = df_metrics.sort_values(by=Metric.MAE_NORM, ascending=False).reset_index(
            drop=True
        )
        df_metrics = df_metrics.round(4)

        return df_metrics

    def _get_design_load_bin_subsets_for_actuals_and_predictions(
        self,
        target: str,
        actuals: np.ndarray,
        actuals_std: np.ndarray,
        predictions: np.ndarray,
        predictions_std: np.ndarray,
        design_load: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the subset of actuals and predictions that are within the design load bin,
        defined by the design load and the design load margin. If the number of data points in
        the bin is less than the minimum number of data points in a bin, the function will
        return the minimum number of data points closest to the design load.
        """
        if target.endswith("min"):
            subset_index = (actuals <= (1 - self.design_load_eval_margin) * design_load) & (
                actuals > (1 + self.design_load_eval_margin) * design_load
            )

        else:
            subset_index = (actuals >= (1 - self.design_load_eval_margin) * design_load) & (
                actuals <= (1 + self.design_load_eval_margin) * design_load
            )

        subset_actuals = actuals[subset_index]
        subset_actuals_std = actuals_std[subset_index]
        subset_predictions = predictions[subset_index]
        subset_predictions_std = predictions_std[subset_index]

        if len(subset_actuals) < self.min_datapoints_in_bin and len(subset_actuals) > 0:
            sorted_indices = np.argsort(np.abs(actuals - design_load))
            if target.endswith("min"):
                sorted_indices = sorted_indices[
                    actuals[sorted_indices] > (1 + self.design_load_eval_margin) * design_load
                ]
            else:
                sorted_indices = sorted_indices[
                    actuals[sorted_indices] < (1 + self.design_load_eval_margin) * design_load
                ]
            subset_index = sorted_indices[: self.min_datapoints_in_bin]
            subset_actuals = actuals[subset_index]
            subset_actuals_std = actuals_std[subset_index]
            subset_predictions = predictions[subset_index]
            subset_predictions_std = predictions_std[subset_index]

        return subset_actuals, subset_actuals_std, subset_predictions, subset_predictions_std

    def _calculate_range_metrics(self, load_groups: pd.DataFrame) -> Dict[LoadBin, TargetMetrics]:
        """Calculate metrics for a range of load levels."""
        range_metrics = {}
        for load_group_data, load_group in load_groups:
            if len(load_group) > 3:
                general_metrics = calculate_general_metrics(
                    load_group["actuals"].to_numpy(),
                    load_group["predictions"].to_numpy(),
                )
                metrics_bin = TargetMetrics(**general_metrics)
            else:
                metrics_bin = None

            range_metrics[
                LoadBin(
                    bin_value=load_group_data.mid,
                    data_count=len(load_group),
                    lower_bound=load_group_data.left,
                    upper_bound=load_group_data.right,
                )
            ] = metrics_bin
        return range_metrics

    def _add_metrics_to_dict(
        self,
        turbine_aggregated_metrics: Union[Dict[str, Dict[str, TargetMetrics]], None],
        dlc: str,
        agg_metrics: Dict[str, Dict[str, TargetMetrics]],
    ) -> Dict[str, Dict[str, TargetMetrics]]:
        for target, metrics in turbine_aggregated_metrics[dlc].items():
            if metrics is not None:
                agg_metrics[target][dlc] = metrics
        return agg_metrics

    def _rel_design_vs_error_plot(
        self, actuals: np.ndarray, predictions: np.ndarray, design_load: float, target: str
    ) -> plt.Figure:
        """Plot error vs relative difference to design load"""

        target_df = pd.DataFrame()
        target_df["actuals"] = actuals
        target_df["predictions"] = predictions
        target_df["error"] = (target_df.predictions - target_df.actuals) / target_df.actuals
        target_df["rel_design"] = (target_df.actuals - design_load) / design_load

        # removing data lower than 40% of the design load
        target_df = target_df[target_df.rel_design > -0.4]

        target_df_filt = target_df[
            (target_df.rel_design > 0.0) & (target_df.error > -target_df.rel_design)
        ]

        fig, ax = plt.subplots()

        # Add shaded area
        lims_neg = np.arange(-0.3, 0, 0.05)
        lims_pos = np.arange(0.0, 0.30, 0.05)

        for lim in lims_neg:
            rect = plt.Rectangle((lim, lim), 0.05, 2 * abs(lim), facecolor="gray", alpha=0.2)
            ax.add_patch(rect)
        for lim in lims_pos:
            rect = plt.Rectangle(
                (lim, lim + 0.05), 0.05, -2 * (lim + 0.05), facecolor="gray", alpha=0.2
            )
            ax.add_patch(rect)

        # Plot data points
        ax.scatter(target_df.rel_design, target_df.error, marker=".")
        if not target_df_filt.empty:
            ax.scatter(
                target_df_filt.rel_design,
                target_df_filt.error,
                color="red",
                marker=".",
                label="(Actual > DL) & (Pred > DL) across all targets",
            )

        ax.set_xlabel("Relative difference to design Load (Actual - DL) / DL")
        ax.set_ylabel("Relative error (Pred - Actual) / Actual")
        ax.set_title(f"{target} - Relative error vs Relative difference to design Load")

        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-0.4, 0.4)

        lims = np.arange(-0.3, 0.35, 0.05)
        for lim in lims:
            ax.axvline(x=lim, color="black", linestyle=":", linewidth=1)
            ax.axhline(y=lim, color="black", linestyle=":", linewidth=1)

        ax.legend()

        return fig
