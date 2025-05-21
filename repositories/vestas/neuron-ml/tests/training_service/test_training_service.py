import os
from pathlib import Path

import pytest
import yaml
from dobby.io.storages import FileSystemStorage
from dobby.io.storages.exceptions import FileDoesNotExistInStorageError

from neuron.data_validation.exceptions import DataValidationError
from neuron.schemas.training_run_config import StorageType, TrainingRunConfig
from neuron_training_service.schemas import CLIUserTrainingRunConfig
from neuron_training_service.training_service import TrainingService
from neuron_training_service.user_config_parsing import get_training_config_from_cli_user_config
from tests.test_doubles import DBXServiceTestDouble, SASURLBlobStorageTestDouble


def get_fake_training_service(target_folder_path: str) -> TrainingService:
    """Fixture with training service setup for testing."""
    neuron_source_storage = FileSystemStorage(mount_path="tests/data/training_service")
    simmesh_source_storage = SASURLBlobStorageTestDouble(
        storage_account_name="myaccount",
        container_name="mycontainer",
        available_data=Path("tests/data/training_service"),
        valid_sas_token="valid_token",
    )
    target_storage = FileSystemStorage()
    return TrainingService(
        neuron_source_storage=neuron_source_storage,
        simmesh_source_storage=simmesh_source_storage,
        target_storage=target_storage,
        training_job_service=DBXServiceTestDouble(),
        target_storage_base_path=target_folder_path,
    )


@pytest.mark.parametrize(
    ("storage_type", "data_config"),
    [
        ("simmesh", "agg"),
        ("simmesh", "no_agg"),
        ("internal", "agg"),
        ("internal", "no_agg"),
    ],
)
def test_training_service(storage_type, data_config, tmp_path_factory) -> None:  # noqa: ANN001
    """Test that the training service runs and correctly copies
    the training input data and config from the source storage to the target storage.
    """
    # Arrange
    target_folder_path = tmp_path_factory.mktemp("target")
    training_service = get_fake_training_service(str(target_folder_path))
    with open(
        f"tests/data/training_service/{storage_type}_storage_configs/training_run_config_example_{data_config}.yaml"
    ) as f:
        config_dict = yaml.safe_load(f)
    training_run_config = get_training_config_from_cli_user_config(
        CLIUserTrainingRunConfig(**config_dict)
    )

    # Act
    response = training_service.train(training_run_config)

    # Assert that the correct files where copied
    assert set(os.listdir(f"{target_folder_path}/{response.training_input_id}")) == {
        "sample_testing_data.parquet",
        "sample_training_data.parquet",
        "training_run_config.yaml",
        "original_training_run_config.yaml",
    }, " All training input data and the config was not copied to the target storage."

    # Assert that we return the correct Job id
    assert (
        response.job_run_id == 123
    ), "The run ID of the triggered Databricks job was not returned."

    # Assert that the data paths in the training run config was modified correctly
    with open(f"{target_folder_path}/{response.training_input_id}/training_run_config.yaml") as f:
        saved_config = yaml.safe_load(f)
        saved_training_run_config = TrainingRunConfig(**saved_config)
    assert os.path.normpath(
        saved_training_run_config.load_case_training_runs[0].data.training_data_file_uri
    ) == os.path.normpath(
        f"{target_folder_path}/{response.training_input_id}/sample_training_data.parquet"
    ), "The path to the training data in the config was not updated correctly."
    assert os.path.normpath(
        saved_training_run_config.load_case_training_runs[0].data.test_data_file_uri
    ) == os.path.normpath(
        f"{target_folder_path}/{response.training_input_id}/sample_testing_data.parquet"
    ), "The path to the test data in the config was not updated correctly."
    if data_config == "agg":
        assert os.path.normpath(
            saved_training_run_config.load_case_training_runs[0].data.agg_data_file_uri
        ) == os.path.normpath(
            f"{target_folder_path}/{response.training_input_id}/sample_testing_data.parquet"
        ), "The path to the aggregation data in the config was not updated correctly."

    # Assert that the saved config is the same as the original config, except for the modified
    # data paths
    saved_training_run_config.load_case_training_runs[0].data.training_data_file_uri = "dummy"
    saved_training_run_config.load_case_training_runs[0].data.test_data_file_uri = "dummy"
    training_run_config.load_case_training_runs[0].data.training_data_file_uri = "dummy"
    training_run_config.load_case_training_runs[0].data.test_data_file_uri = "dummy"
    assert (
        saved_training_run_config == training_run_config
    ), "The saved config does not match the original config."

    assert (
        training_run_config.storage_type == StorageType.INTERNAL
    ), "The storage type was not set correctly. In should be set to internal."


# TODO: Is this necessary if we test at the API level?
@pytest.mark.parametrize("storage_type", ["simmesh", "internal"])
def test_training_service_raises_data_does_not_exists_error(storage_type, tmp_path_factory) -> None:  # noqa: ANN001
    """Test that the training service throws a FileDoesNotExistInStorageError error
    when the training data does not exist in the source storage."""
    # Arrange
    training_service = get_fake_training_service(str(tmp_path_factory.mktemp("target")))
    with open(
        f"tests/data/training_service/{storage_type}_storage_configs/training_run_config_example_bad_data_path.yaml"
    ) as f:
        config_dict = yaml.safe_load(f)
    training_run_config = get_training_config_from_cli_user_config(
        CLIUserTrainingRunConfig(**config_dict)
    )
    with pytest.raises(FileDoesNotExistInStorageError):
        training_service.train(training_run_config)


@pytest.mark.parametrize("storage_type", ["simmesh", "internal"])
def test_training_service_raises_data_validation_error(storage_type, tmp_path_factory) -> None:  # noqa: ANN001
    training_service = get_fake_training_service(str(tmp_path_factory.mktemp("target")))
    """Test that the training service throws a TrainingDataValidationError
    when the training data is invalid."""
    with open(
        f"tests/data/training_service//{storage_type}_storage_configs/training_run_config_example_bad_data.yaml"
    ) as f:
        config_dict = yaml.safe_load(f)
    training_run_config = get_training_config_from_cli_user_config(
        CLIUserTrainingRunConfig(**config_dict)
    )
    with pytest.raises(DataValidationError):
        training_service.train(training_run_config)
