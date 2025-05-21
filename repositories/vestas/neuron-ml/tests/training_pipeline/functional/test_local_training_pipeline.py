"""Local end-to-end test of model training pipeline"""

from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize
from mlflow import MlflowClient
from pandas.testing import assert_frame_equal

from neuron.experiment_tracking.mlflow_tracker import MLFlowTracker
from neuron.models.load_case_model_pipeline import LoadCaseModelPipeline
from neuron.schemas.training_run_config import TrainingRunConfig
from neuron.turbine_train_run import (
    build_turbine_data_repo,
    run_turbine_training_pipeline,
)
from neuron.turbine_training_pipeline import LocalTurbineTrainingPipeline
from neuron_training_service.schemas import CLIUserTrainingRunConfig
from neuron_training_service.user_config_parsing import get_training_config_from_cli_user_config
from scripts.create_config_from_hydra import parse_hydra_config


@pytest.fixture(scope="function")
def test_mlflow_experiment_tracker(tmp_path_factory, monkeypatch) -> MLFlowTracker:  # noqa: ANN001
    """Create a local MLFlow experiment tracker and return it."""
    mlflow_artifacts_folder = tmp_path_factory.mktemp("mlflow_artifacts")
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    return MLFlowTracker(
        experiment_name="testing", tracking_uri=f"file:{str(mlflow_artifacts_folder)}"
    )


def get_test_training_run_config(hydra_overrides: list[str]) -> TrainingRunConfig:
    """Get a test training run config with the specified overwrites.

    Used to get different training run configurations for training.
    """
    with initialize(
        version_base="1.2",
        config_path="../../data/training_pipeline/training_run_hydra_config",
        job_name="test",
    ):
        hydra_config = compose(config_name="config", overrides=hydra_overrides)
        parsed_hydra_config = parse_hydra_config(hydra_config)
        return get_training_config_from_cli_user_config(
            CLIUserTrainingRunConfig(**parsed_hydra_config)
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "load_case_training_run_config_overwrite",
    [
        "named_ext",
        "named_with_overwrites_fat",
        "named_with_overwrites_ext",
        "custom",
        "named_with_agg_data",
        "named_with_agg_and_test_data",
    ],
)
def test_end_2_end_turbine_training_run_locally(
    load_case_training_run_config_overwrite: str,
    test_mlflow_experiment_tracker: MLFlowTracker,
    tmp_path_factory,  # noqa: ANN001
) -> None:
    """Test that the training pipeline runs e2e by executing the training pipeline
    scripts and checking that the expected artifacts are logged to MLFlow.
    We also download the trained load case models (from local MLFlow) and check that
    we can load them and make predictions.
    """
    # Arrange
    test_training_run_config = get_test_training_run_config(
        hydra_overrides=[f"load_case_training_runs={load_case_training_run_config_overwrite}"]
    )
    load_case_models_dir = tmp_path_factory.mktemp("load_case_models")
    mlflow_client = MlflowClient(tracking_uri=test_mlflow_experiment_tracker.tracking_uri)

    ########################################
    # Act / Assert TrainingRunConfig
    ########################################
    test_training_run_config_dict = test_training_run_config.model_dump()
    assert TrainingRunConfig(**test_training_run_config_dict) == test_training_run_config, (
        "Loading TrainingRunConfig from dict cannot be performed for a load case with"
        f"{load_case_training_run_config_overwrite}"
    )

    turbine_data_repo = build_turbine_data_repo(test_training_run_config)

    ########################################
    # Act / Assert Training Pipeline Results
    ########################################
    turbine_training_pipeline = LocalTurbineTrainingPipeline(
        load_case_configs=test_training_run_config.load_case_training_runs,
        turbine_data_repo=turbine_data_repo,
        turbine_variant=test_training_run_config.turbine.turbine_variant,
    )

    with test_mlflow_experiment_tracker as experiment_tracker:
        run_turbine_training_pipeline(
            training_run_config=test_training_run_config,
            training_input_id="dummy_path",
            experiment_tracker=experiment_tracker,
            turbine_training_pipeline=turbine_training_pipeline,
        )

    # Get artifacts from MLFlow
    experiment_id = mlflow_client.get_experiment_by_name(
        test_mlflow_experiment_tracker.experiment_name
    ).experiment_id
    runs = mlflow_client.search_runs(experiment_ids=[experiment_id])
    assert len(runs) == 1, "Expected only one run to be logged"

    run_id = runs[0].info.run_id

    artifacts = mlflow_client.list_artifacts(run_id)
    dir_artifacts = [artifact.path for artifact in artifacts if artifact.is_dir]
    expected_directories = {
        ".dobby",
        "binned_metrics",
        "data",
        "load_case_models",
        "metrics",
        "plots",
        "params",
        "full_training_run_metrics",
    }

    # Append aggregated_metrics artifact folder for test run where
    # calculate_aggregated_metrics is enabled
    if load_case_training_run_config_overwrite.startswith("named_with_agg"):
        expected_directories.add("aggregated_metrics")

    assert set(dir_artifacts) == expected_directories, (
        "Expected the following directories to be logged: " f"{', '.join(expected_directories)}"
    )

    # Download all load case models and check that we can load them and make predictions
    mlflow_client.download_artifacts(run_id, "load_case_models", str(load_case_models_dir))

    for load_case_config in test_training_run_config.load_case_training_runs:
        training_df_sample = pd.read_parquet(load_case_config.data.training_data_file_uri)
        load_case_model_dir = Path(
            f"{load_case_models_dir}/load_case_models/{load_case_config.name}_model"
        )

        load_case_model = LoadCaseModelPipeline.load_model(str(load_case_model_dir))
        load_case_model.predict(training_df_sample, return_std=True)


@pytest.mark.slow
def test_turbine_agg_metrics_calculation(
    test_mlflow_experiment_tracker: MLFlowTracker,
    tmp_path_factory,  # noqa: ANN001
) -> None:
    """Test that the turbine aggregated metrics are calculated correctly
    If the turbine aggregated metrics are calculated based on a single load case,
    it should give the same result as the load case aggregated metrics.
    """
    # Arrange
    test_training_run_config = get_test_training_run_config(
        hydra_overrides=["load_case_training_runs=named_with_agg_data"]
    )
    mlflow_client = MlflowClient(tracking_uri=test_mlflow_experiment_tracker.tracking_uri)
    tmp_dir = tmp_path_factory.mktemp("aggregated_metrics")

    turbine_data_repo = build_turbine_data_repo(test_training_run_config)

    turbine_training_pipeline = LocalTurbineTrainingPipeline(
        load_case_configs=test_training_run_config.load_case_training_runs,
        turbine_data_repo=turbine_data_repo,
        turbine_variant=test_training_run_config.turbine.turbine_variant,
    )
    # Act
    with test_mlflow_experiment_tracker as experiment_tracker:
        run_turbine_training_pipeline(
            training_run_config=test_training_run_config,
            training_input_id="dummy_id",
            experiment_tracker=experiment_tracker,
            turbine_training_pipeline=turbine_training_pipeline,
        )

    # Assert
    experiment_id = mlflow_client.get_experiment_by_name(
        test_mlflow_experiment_tracker.experiment_name
    ).experiment_id
    runs = mlflow_client.search_runs(experiment_ids=[experiment_id])
    assert len(runs) == 1, "Expected only one run to be logged"

    run_id = runs[0].info.run_id

    # download aggregated_metrics artifacts
    mlflow_client.download_artifacts(run_id, "aggregated_metrics", str(tmp_dir))

    load_case_agg_metrics = pd.read_csv(f"{tmp_dir}/aggregated_metrics/dlc11_target_metrics.csv")
    turbine_agg_metrics = pd.read_csv(
        f"{tmp_dir}/aggregated_metrics/all_load_cases_target_metrics.csv"
    )

    (assert_frame_equal(load_case_agg_metrics, turbine_agg_metrics, check_exact=False, atol=1e-8),)
    "Turbine aggregated metrics should be equal to load case aggregated "
    "metrics when calculated based on a single load case."
