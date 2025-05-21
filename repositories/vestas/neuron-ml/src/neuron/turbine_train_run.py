"""Main entrypoint for training load case models for a single turbine_variant configuration.

Trained load case models, parameters and metrics are logged to MLFlow.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from dobby.environment import is_running_on_databricks

from neuron.config import get_project_storage, settings
from neuron.data_splitting.registry import get_data_splitter
from neuron.evaluation.turbine_evaluator import TurbineEvaluator
from neuron.experiment_tracking.base import ExperimentTracker
from neuron.experiment_tracking.mlflow_tracker import MLFlowTracker
from neuron.io.training_data_repository import TrainingDataRepository
from neuron.load_case_model_training_pipeline import (
    evaluate_and_log_loadcase_data,
)
from neuron.models.load_case_model_pipeline import (
    LoadCaseModelPipeline,
)
from neuron.schemas.domain import (
    Metric,
    TurbineActuals,
    TurbinePredictions,
)
from neuron.schemas.training_run_config import (
    StorageType,
    TrainingRunConfig,
)
from neuron.turbine_training_pipeline import (
    DbxTurbineTrainingPipeline,
    LocalTurbineTrainingPipeline,
    TurbineTrainingPipeline,
)
from neuron.utils import set_seed

logger = logging.getLogger(__name__)


def build_turbine_data_repo(
    training_run_config: TrainingRunConfig,
) -> dict[str, TrainingDataRepository]:
    turbine_data_repo = {}
    for lc_training_run_config in training_run_config.load_case_training_runs:
        lc_name = lc_training_run_config.name
        data_splitter = get_data_splitter(
            name=lc_training_run_config.data_splitting.name,
            params=lc_training_run_config.data_splitting.params,
        )
        turbine_data_repo[lc_name] = TrainingDataRepository(
            training_file_uri=lc_training_run_config.data.training_data_file_uri,
            test_file_uri=lc_training_run_config.data.test_data_file_uri,
            agg_file_uri=lc_training_run_config.data.agg_data_file_uri,
            data_splitter=data_splitter,
            storage=get_project_storage(),
        )
    return turbine_data_repo


def run_turbine_training_pipeline(
    training_run_config: TrainingRunConfig,
    training_input_id: str,
    experiment_tracker: ExperimentTracker,
    turbine_training_pipeline: TurbineTrainingPipeline,
) -> None:
    """Run training pipeline for a single turbine_variant configuration.

    - Set up the turbine_evaluator at the turbine_variant level.
    - Log turbine level info to experiment tracker.
    - Use the provided turbine_training_pipeline to execute load case runs.
    - Evaluate and log load case and turbine level data.
    """
    turbine_evaluator = TurbineEvaluator(
        load_case_configs=training_run_config.load_case_training_runs,
        eval_config=training_run_config.evaluation,
        turbine_config=training_run_config.turbine,
    )

    logger.info("Logging runtime metadata.")
    experiment_tracker.log_runtime_metadata()

    logger.info("Setting seed and logging it to experiment tracker.")
    seed = set_seed()
    experiment_tracker.log_experiment_params(params={"seed": seed})

    logger.info("Logging the training_input_id to experiment tracker.")
    experiment_tracker.log_experiment_params(params={"training_input_id": training_input_id})

    logger.info("Logging turbine training run config to experiment tracker.")
    experiment_tracker.log_turbine_training_run_config(training_run_config=training_run_config)

    logger.info("Log turbine variant config.")
    experiment_tracker.log_turbine_variant(
        turbine_variant=training_run_config.turbine.turbine_variant
    )

    # Data structures to store load case predictions, actuals and model pipelines
    turbine_preds: TurbinePredictions = {}
    turbine_actuals: TurbineActuals = {}
    turbine_preds_agg: TurbinePredictions = {}
    turbine_actuals_agg: TurbineActuals = {}
    prediction_speed_test_results = {}

    lc_model_pipelines: List[LoadCaseModelPipeline] = []

    logger.info("Set up and run load case model pipeline for each load case training run config.")

    turbine_training_results = turbine_training_pipeline.train_load_cases()

    logger.info("Evaluate and log load case data for each load case training run.")

    for lc_train_config in training_run_config.load_case_training_runs:
        lc_name = lc_train_config.name

        trained_lc_model_pipeline = turbine_training_results[lc_name]

        logger.info(f"Evaluation and logging of load case: {lc_name}.")

        lc_eval_output = evaluate_and_log_loadcase_data(
            lc_train_config=lc_train_config,
            lc_model_pipeline=trained_lc_model_pipeline,
            experiment_tracker=experiment_tracker,
            alpha=turbine_evaluator.alpha_significance_level,
            data_repo=turbine_training_pipeline.turbine_data_repo[lc_name],
        )

        lc_model_pipelines.append(trained_lc_model_pipeline)

        turbine_preds[lc_name] = lc_eval_output.load_case_predictions
        turbine_actuals[lc_name] = lc_eval_output.load_case_actuals
        turbine_preds_agg[lc_name] = lc_eval_output.load_case_agg_predictions
        turbine_actuals_agg[lc_name] = lc_eval_output.load_case_agg_actuals
        prediction_speed_test_results[lc_name] = lc_eval_output.prediction_speed_test_results

    logger.info("Postprocess and log turbine level metrics and plots.")

    turbine_binned_metrics = turbine_evaluator.get_turbine_binned_metrics(
        turbine_predictions=turbine_preds, turbine_actuals=turbine_actuals
    )

    turbine_binned_plots = (
        turbine_evaluator.get_metrics_heatmaps_for_target_bins_across_ext_load_cases(
            turbine_binned_metrics=turbine_binned_metrics
        )
    )

    experiment_tracker.log_turbine_binned_plots(turbine_plots=turbine_binned_plots)
    experiment_tracker.log_prediction_speed_test_results(prediction_speed_test_results)

    turbine_aggregated_metrics = None
    if turbine_evaluator.calculate_aggregated_metrics:
        turbine_aggregated_metrics = turbine_evaluator.get_turbine_aggregated_metrics(
            turbine_predictions=turbine_preds_agg, turbine_actuals=turbine_actuals_agg
        )

        experiment_tracker.log_load_case_metrics(
            load_case_name="all_load_cases",
            metrics=turbine_aggregated_metrics[turbine_evaluator.agg_dlcs_name],
            artifact_path="aggregated_metrics",
        )

    # If we have any extreme load DLCs and we have design loads available
    ext_design_load_bin_metrics = None
    filtered_target_models = pd.DataFrame()
    if (
        turbine_evaluator.extreme_load_dlc_exist
        and training_run_config.turbine.design_loads_ext is not None
    ):
        ext_design_load_bin_metrics = turbine_evaluator.get_ext_design_load_bin_metrics(
            turbine_predictions=turbine_preds, turbine_actuals=turbine_actuals
        )

        filtered_target_models = turbine_evaluator.find_excluded_targets_from_ext_bin_eval(
            ext_design_load_bin_metrics
        )

        experiment_tracker.log_ext_design_load_bin_metrics(ext_design_load_bin_metrics)

        turbine_ex_loads_mae_norm_plot = turbine_evaluator.create_design_load_bin_heatmap(
            binned_metrics=ext_design_load_bin_metrics,
            metric=Metric.MAE_NORM.value,
        )
        turbine_ext_loads_e_mean_norm_plot = turbine_evaluator.create_design_load_bin_heatmap(
            binned_metrics=ext_design_load_bin_metrics,
            metric=Metric.E_MEAN_NORM.value,
        )

        turbine_extreme_loads_plots = {
            **turbine_ex_loads_mae_norm_plot,
            **turbine_ext_loads_e_mean_norm_plot,
        }

        experiment_tracker.log_turbine_extreme_load_plots(turbine_extreme_loads_plots)

        turbine_detailed_rel_design_plots = turbine_evaluator.get_detailed_rel_design_error_plots(
            turbine_predictions=turbine_preds,
            turbine_actuals=turbine_actuals,
        )

        experiment_tracker.log_detailed_rel_design_plots(turbine_detailed_rel_design_plots)

    turbine_metrics_overview_df_list = turbine_evaluator.get_turbine_metrics_overview(
        ext_design_load_bin_metrics=ext_design_load_bin_metrics,
        turbine_aggregated_metrics=turbine_aggregated_metrics,
    )

    experiment_tracker.log_turbine_metrics_overview(turbine_metrics_overview_df_list)

    logger.info("Log turbine run overview.")
    turbine_overview_html = turbine_evaluator.get_turbine_overview_html(
        df_list=turbine_metrics_overview_df_list,
        filtered_ext_models=filtered_target_models,
    )

    if turbine_overview_html is not None:
        experiment_tracker.log_turbine_overview_html(turbine_overview_html)

    logger.info("Log all turbine_variant models as a zipped folder.")
    experiment_tracker.log_turbine_variant_load_case_models(
        load_case_model_pipelines=lc_model_pipelines
    )


def main() -> None:
    """Main entrypoint for training load case models for a single turbine_variant configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-config-path",
        action="store",
        type=str,
        help="Path to yaml file with training run config. Overrides hydra config.",
        required=True,
    )
    args = parser.parse_args()
    train_config_path = args.train_config_path

    storage = get_project_storage()

    if not storage.exists(train_config_path):
        logger.info(f"Train config file {train_config_path} does not exist in storage {storage}.")
        sys.exit(1)

    logger.info(f"Reading training run config from {train_config_path}")
    config_dict = yaml.safe_load(storage.read(train_config_path))
    training_run_config = TrainingRunConfig(**config_dict)

    if training_run_config.storage_type != StorageType.INTERNAL:
        raise ValueError(
            f"Training run config storage type {training_run_config.storage_type} "
            f"is not supported inside the training pipeline."
            f"Expected {StorageType.INTERNAL}."
        )

    turbine_data_repo = build_turbine_data_repo(training_run_config)

    if is_running_on_databricks():
        training_input_id = Path(train_config_path).parent.as_posix()
        turbine_training_pipeline = DbxTurbineTrainingPipeline(
            load_case_configs=training_run_config.load_case_training_runs,
            turbine_data_repo=turbine_data_repo,
            turbine_variant=training_run_config.turbine.turbine_variant,
        )
    else:
        training_input_id = "local_run"
        turbine_training_pipeline = LocalTurbineTrainingPipeline(
            load_case_configs=training_run_config.load_case_training_runs,
            turbine_data_repo=turbine_data_repo,
            turbine_variant=training_run_config.turbine.turbine_variant,
        )

    logger.info(
        f"Run turbine training pipeline with training config {train_config_path} "
        f"in MLFlow experiment {settings.experiment_name}."
    )
    with MLFlowTracker(
        experiment_name=settings.experiment_name,
        tracking_uri=settings.mlflow_tracking_uri,
    ) as experiment_tracker:
        run_turbine_training_pipeline(
            training_run_config=training_run_config,
            training_input_id=training_input_id,
            experiment_tracker=experiment_tracker,
            turbine_training_pipeline=turbine_training_pipeline,
        )


if __name__ == "__main__":
    main()
