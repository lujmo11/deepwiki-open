"""module with functionality to evaluate a single load case model pipeline."""
import logging

from neuron.evaluation.evaluation_utils import get_aggregated_actuals_and_predictions
from neuron.evaluation.load_case_evaluator import LoadCaseEvaluator
from neuron.experiment_tracking.base import ExperimentTracker
from neuron.schemas.domain import (
    LoadCaseActuals,
    LoadCaseFeatureValues,
    LoadCasePredictions,
)

logger = logging.getLogger(__name__)


def evaluate_load_case_model(
    load_case_name: str,
    load_case_actuals: LoadCaseActuals,
    load_case_predictions: LoadCasePredictions,
    load_case_feature_values: LoadCaseFeatureValues,
    experiment_tracker: ExperimentTracker,
    load_case_evaluator: LoadCaseEvaluator,
) -> None:
    """Evaluate load case model predictions.

    Parameters
    ----------
    load_case_name: str
        Name of the load case.
    load_case_predictions: LoadCasePredictions
        Predictions for the load case.
    load_case_actuals: LoadCaseActuals
        Actual values for the load case.
    load_case_feature_values: LoadCaseFeatureValues
        Feature values for the load case.
    load_case: LoadCase
        Load case to evaluate.
    experiment_tracker : ExperimentTracker
        Experiment tracker.
    load_case_evaluator : LoadCaseEvaluator
        Load case evaluator.

    Returns
    -------
    None
    """
    load_case_standard_metrics = load_case_evaluator.get_standard_metrics(
        load_case_actuals=load_case_actuals,
        load_case_predictions=load_case_predictions,
    )

    load_case_binned_metrics = load_case_evaluator.get_binned_metrics(
        load_case_actuals=load_case_actuals,
        load_case_predictions=load_case_predictions,
        load_case_feature_values=load_case_feature_values,
    )

    logger.info("Logging load case metrics and plots.")
    experiment_tracker.log_load_case_metrics(
        load_case_name=load_case_name, metrics=load_case_standard_metrics, artifact_path="metrics"
    )

    experiment_tracker.log_binned_load_case_metrics(
        load_case_name=load_case_name, metrics=load_case_binned_metrics
    )

    plots_data = load_case_evaluator.get_standard_plots(
        load_case_actuals=load_case_actuals,
        load_case_predictions=load_case_predictions,
    )
    experiment_tracker.log_load_case_plots(
        load_case_name=load_case_name, plots=plots_data, artifact_path="full_data"
    )

    if load_case_evaluator.generate_coverage_plots:
        plots_coverage_data = load_case_evaluator.get_coverage_plots(
            load_case_actuals=load_case_actuals,
            load_case_predictions=load_case_predictions,
        )
        experiment_tracker.log_load_case_plots(
            load_case_name=load_case_name, plots=plots_coverage_data, artifact_path="coverage_plots"
        )

    logger.info("Generating load case overview.")
    load_case_binned_metrics_plots_data = (
        load_case_evaluator.get_binned_metrics_per_feature_heatmap(
            load_case_binned_metrics=load_case_binned_metrics,
        )
    )

    experiment_tracker.log_load_case_binned_metrics_plot(
        load_case_name=load_case_name, plots=load_case_binned_metrics_plots_data
    )


def evaluate_load_case_model_agg(
    load_case_name: str,
    load_case_agg_actuals: LoadCaseActuals,
    load_case_agg_predictions: LoadCasePredictions,
    experiment_tracker: ExperimentTracker,
    load_case_evaluator: LoadCaseEvaluator,
) -> None:
    "Evaluate load case model predictions"

    logger.info("Evaluating load_case_model performance on aggregated data.")
    (
        load_case_aggregated_actuals,
        load_case_aggregated_predictions,
    ) = get_aggregated_actuals_and_predictions(
        load_case_actuals=load_case_agg_actuals, load_case_predictions=load_case_agg_predictions
    )

    load_case_aggregated_metrics = load_case_evaluator.get_standard_metrics(
        load_case_actuals=load_case_aggregated_actuals,
        load_case_predictions=load_case_aggregated_predictions,
    )

    logger.info("Logging aggregated load case metrics and plots.")
    experiment_tracker.log_load_case_metrics(
        load_case_name=load_case_name,
        metrics=load_case_aggregated_metrics,
        artifact_path="aggregated_metrics",
    )

    aggregated_plot_data = load_case_evaluator.get_standard_plots(
        load_case_actuals=load_case_aggregated_actuals,
        load_case_predictions=load_case_aggregated_predictions,
    )

    experiment_tracker.log_load_case_plots(
        load_case_name=load_case_name,
        plots=aggregated_plot_data,
        artifact_path="aggregated_data",
    )
