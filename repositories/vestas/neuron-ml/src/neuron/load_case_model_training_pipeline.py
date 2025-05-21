import logging
from typing import Dict, Union

import pandas as pd
from pydantic import BaseModel

from neuron.evaluate_load_case_model import evaluate_load_case_model, evaluate_load_case_model_agg
from neuron.evaluation.load_case_evaluator import LoadCaseEvaluator
from neuron.experiment_tracking.base import ExperimentTracker
from neuron.io.training_data_repository import TrainingDataRepository
from neuron.models.load_case_model_pipeline import (
    LoadCaseModelPipeline,
    initialize_load_case_model_from_config,
)
from neuron.schemas.domain import LoadCase, LoadCaseActuals, LoadCasePredictions, TurbineVariant
from neuron.schemas.training_run_config import LoadCaseTrainingRunConfig
from neuron.utils import (
    get_load_case_actuals_from_df,
    get_load_case_feature_values_df,
    plot_train_test_data,
)

logger = logging.getLogger(__name__)


class LoadCaseEvaluationOutput(BaseModel):
    """Output from a single load case model evaluation pipeline."""

    load_case_predictions: LoadCasePredictions
    load_case_actuals: LoadCaseActuals
    prediction_speed_test_results: Dict
    load_case_agg_predictions: Union[None, LoadCasePredictions] = None
    load_case_agg_actuals: Union[None, LoadCaseActuals] = None


def setup_and_run_load_case_model_training(
    load_case_training_run_config: LoadCaseTrainingRunConfig,
    train_df: pd.DataFrame,
    turbine_variant: TurbineVariant,
) -> LoadCaseModelPipeline:
    """Setup and run the load case model training pipeline for a single load case.
    - Initialize load case from load case training run config.
    - Initialize load case model pipeline from load case training run config.
    - Fit the load case model pipeline on the training data.
    """
    logger.info(
        "Initialize load case from load case config for load case "
        f"{load_case_training_run_config.name}."
    )
    load_case = LoadCase(
        name=load_case_training_run_config.name,
        feature_list=load_case_training_run_config.feature_list,
        target_list=load_case_training_run_config.target_list,
        postprocessor=load_case_training_run_config.postprocessor,
    )

    logger.info(f"Initialize load case model pipeline from config for load case {load_case.name}.")
    load_case_model = initialize_load_case_model_from_config(
        turbine_variant=turbine_variant,
        load_case=load_case,
        load_case_model_config=load_case_training_run_config.load_case_model,
    )

    logger.info(f"Run load case model training for load case {load_case.name}.")
    load_case_model.fit(df=train_df)

    return load_case_model


def evaluate_and_log_loadcase_data(
    lc_train_config: LoadCaseTrainingRunConfig,
    lc_model_pipeline: LoadCaseModelPipeline,
    experiment_tracker: ExperimentTracker,
    data_repo: TrainingDataRepository,
    alpha: float,
) -> LoadCaseEvaluationOutput:
    load_case = lc_model_pipeline.load_case

    load_case_evaluator = LoadCaseEvaluator(
        feature_names=[
            feature.name for feature in lc_model_pipeline.load_case.feature_list.features
        ],
        max_load_evaluation_limit=lc_train_config.max_load_evaluation_limit,
        alpha_significance_level=alpha,
        calculate_aggregated_metrics=lc_train_config.calculate_aggregated_metrics,
    )

    logger.info("Reading data for training pipeline.")
    train_df = data_repo.get_load_case_train_df()
    test_df = data_repo.get_load_case_test_df()

    logger.info(f"Logging target model parameters for {load_case.name}.")
    load_case_model_params = {
        target: lc_model_pipeline.models.get(target).get_params()
        for target in lc_model_pipeline.models.keys()
    }
    experiment_tracker.log_load_case_model_params(
        load_case_name=load_case.name, load_case_model_params=load_case_model_params
    )

    logger.info("Logging trained load case model.")
    experiment_tracker.log_load_case_model(load_case_model=lc_model_pipeline)

    logger.info(f"Predict with load case model on test set for {load_case.name}.")
    load_case_predictions_raw = lc_model_pipeline.predict(
        test_df,
        return_std=True,
    )

    feature_test_df = lc_model_pipeline.preprocess(test_df)
    feature_train_df = lc_model_pipeline.preprocess(train_df)
    load_case_feature_values = get_load_case_feature_values_df(
        df=feature_test_df, features=load_case.feature_list.features
    )

    load_case_actuals_raw = get_load_case_actuals_from_df(
        df=feature_test_df,
        target_names=load_case.target_list.targets,
        calculate_aggregated_metrics=None,
    )

    load_case_actuals = lc_model_pipeline.postprocess(load_case_actuals_raw)
    load_case_predictions = lc_model_pipeline.postprocess(load_case_predictions_raw)

    logger.info(f"Logging train and test data for load case {load_case.name}.")
    experiment_tracker.log_train_and_test_data(
        load_case_name=load_case.name,
        train_df=feature_train_df,
        test_df=feature_test_df,
    )

    logger.info(f"Plotting train and test data for load case {load_case.name}.")
    train_test_data_io = plot_train_test_data(
        train_df=train_df,
        test_df=test_df,
        load_case_features=load_case.feature_list.features,
        safe_domain=lc_model_pipeline.safe_domain_validator,
    )
    experiment_tracker.log_train_test_data_plots(
        load_case_name=load_case.name, data_plot_io=train_test_data_io, postfix="test"
    )

    logger.info(f"Evaluate load case model for load case {load_case.name}.")
    evaluate_load_case_model(
        load_case_actuals=load_case_actuals,
        load_case_predictions=load_case_predictions,
        load_case_feature_values=load_case_feature_values,
        load_case_name=load_case.name,
        experiment_tracker=experiment_tracker,
        load_case_evaluator=load_case_evaluator,
    )

    logger.info(f"Logging test data for load case {load_case.name}.")
    experiment_tracker.log_test_and_prediction_data(
        load_case_name=load_case.name,
        df=test_df,
        load_case_predictions=load_case_predictions,
        df_prefix="test",
    )

    logger.info(f"Running speed test and logging results for {load_case.name}.")

    prediction_speed_test_results = lc_model_pipeline.run_speed_test(test_df)

    load_case_agg_predictions = None
    load_case_agg_actuals = None

    if load_case_evaluator.calculate_aggregated_metrics:
        logger.info("Evaluating and logging load case model performance on aggregated data.")

        agg_df = data_repo.get_load_case_agg_df()
        load_case_agg_actuals_raw = get_load_case_actuals_from_df(
            df=agg_df,
            target_names=load_case.target_list.targets,
            calculate_aggregated_metrics=load_case_evaluator.calculate_aggregated_metrics,
        )
        load_case_agg_predictions_raw = lc_model_pipeline.predict(
            agg_df,
            return_std=True,
        )
        load_case_agg_actuals = lc_model_pipeline.postprocess(load_case_agg_actuals_raw)
        load_case_agg_predictions = lc_model_pipeline.postprocess(load_case_agg_predictions_raw)

        evaluate_load_case_model_agg(
            load_case_name=load_case.name,
            load_case_agg_actuals=load_case_agg_actuals,
            load_case_agg_predictions=load_case_agg_predictions,
            experiment_tracker=experiment_tracker,
            load_case_evaluator=load_case_evaluator,
        )

        experiment_tracker.log_test_and_prediction_data(
            load_case_name=load_case.name,
            df=agg_df,
            load_case_predictions=load_case_agg_predictions,
            df_prefix="agg",
        )

        train_agg_data_io = plot_train_test_data(
            train_df=train_df,
            test_df=agg_df,
            load_case_features=load_case.feature_list.features,
            safe_domain=lc_model_pipeline.safe_domain_validator,
        )
        experiment_tracker.log_train_test_data_plots(
            load_case_name=load_case.name, data_plot_io=train_agg_data_io, postfix="agg"
        )

    return LoadCaseEvaluationOutput(
        load_case_predictions=load_case_predictions,
        load_case_actuals=load_case_actuals,
        prediction_speed_test_results=prediction_speed_test_results,
        load_case_agg_predictions=load_case_agg_predictions,
        load_case_agg_actuals=load_case_agg_actuals,
    )
