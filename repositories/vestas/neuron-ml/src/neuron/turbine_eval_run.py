import argparse
import logging
import sys
from pathlib import Path

import yaml
from dobby.environment import is_running_on_databricks

from neuron.config import get_project_storage, settings
from neuron.experiment_tracking.mlflow_tracker import MLFlowTracker
from neuron.schemas.training_run_config import (
    TrainingRunConfig,
)
from neuron.turbine_train_run import build_turbine_data_repo, run_turbine_training_pipeline
from neuron.turbine_training_pipeline import (
    PreTrainedTurbinePipeline,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entrypoint for evaluating load case models for a pre-trained turbine configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-config-path",
        action="store",
        type=str,
        help="Path to yaml file config. Overrides hydra config.",
        required=True,
    )

    parser.add_argument(
        "--mlflow-run-id",
        action="store",
        type=str,
        help="MLFlow run ID of the pre-trained models",
        required=True,
    )

    args = parser.parse_args()
    train_config_path = args.train_config_path
    mlflow_run_id = args.mlflow_run_id

    storage = get_project_storage()

    if not storage.exists(train_config_path):
        logger.info(f"Config file {train_config_path} does not exist in storage {storage}.")
        sys.exit(1)

    logger.info(f"Reading config from {train_config_path}")
    config_dict = yaml.safe_load(storage.read(train_config_path))
    training_run_config = TrainingRunConfig(**config_dict)

    turbine_data_repo = build_turbine_data_repo(training_run_config)

    if is_running_on_databricks():
        training_input_id = Path(train_config_path).parent.as_posix()
    else:
        training_input_id = "local_run"

    turbine_training_pipeline = PreTrainedTurbinePipeline(
        load_case_configs=training_run_config.load_case_training_runs,
        turbine_data_repo=turbine_data_repo,
        turbine_variant=training_run_config.turbine.turbine_variant,
        mlflow_run_id=mlflow_run_id,
    )

    logger.info(
        f"Run turbine evaluation pipeline with config {train_config_path} "
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
