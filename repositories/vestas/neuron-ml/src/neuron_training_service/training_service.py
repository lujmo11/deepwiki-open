"""Module with Neuron Training Service"""

import datetime
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Protocol

import yaml
from dobby.io.storages import Storage

from neuron.data_validation.validation import (
    validate_data_all_load_cases,
    validate_test_data_for_agg_load_cases,
)
from neuron.io.storage_reader import StorageReader
from neuron.schemas.training_run_config import (
    LoadCaseTrainingRunConfig,
    StorageType,
    TrainingRunConfig,
)
from neuron_training_service.schemas import NeuronTrainingJobRun, TrainingServiceResponse
from neuron_training_service.simmesh_storage import BlobUrlComponents


class TrainingJobService(Protocol):
    def start_train_job_run(self, config_blob_path: str) -> int:
        ...

    def get_training_run_info(self, job_run_id: int) -> NeuronTrainingJobRun:
        ...


class TrainingService:
    """Service for
    - Starting a Neuron turbine training run.
    - Getting information about a Neuron turbine training run.
    """

    def __init__(
        self,
        neuron_source_storage: StorageReader,
        simmesh_source_storage: StorageReader,
        target_storage: Storage,
        training_job_service: TrainingJobService,
        target_storage_base_path: str,
    ):
        self.neuron_source_storage = neuron_source_storage
        self.simmesh_source_storage = simmesh_source_storage
        self.target_storage = target_storage
        self.training_job_service = training_job_service
        self.target_storage_base_path = target_storage_base_path
        """Initialize the training service.

        Parameters
        ----------
        neuron_source_storage : Storage
            The source storage for the training data and training curated by the Neuron team.
        simmesh_source_storage : Storage
            The source storage for the training data stored in SimMesh team.
        target_storage : Storage
            The target storage for the training data and training run configuration.
        training_job_service : TrainingJobService
            The service for starting and getting information about training job runs.
        target_storage_base_path : str
            The base path in the target storage where the training data and training 
            run configuration will be stored.
        """

    @staticmethod
    def _generate_unique_id() -> str:
        return f"{datetime.datetime.now(tz=datetime.UTC).date()}/{uuid.uuid4()}"

    @staticmethod
    def _get_blob_name(blob_uri: str, storage_type: StorageType) -> str:
        """Get the blob name from the blob URI based on the storage source."""
        if storage_type == StorageType.INTERNAL:
            blob_path = blob_uri
        elif storage_type == StorageType.SIMMESH:
            blob_path = BlobUrlComponents.from_url(blob_uri).blob_path
        else:
            raise ValueError(
                f"Invalid storage source {storage_type}. "
                f"Expected one of {['INTERNAL', 'SIMMESH']}"
            )
        return Path(blob_path).name

    def _copy_file(
        self, source_storage: StorageReader, source_dataset_uri: str, target_file_uri: str
    ) -> None:
        """Copy a file from a source to the target storage."""
        training_df_bytes = source_storage.read(source_dataset_uri)
        # Overwrite = True is set to avoid an error in the case that
        # test_data_file_uri and agg_data_file_uri are identical.
        self.target_storage.write(training_df_bytes, target_file_uri, overwrite=True)

    def _save_config(self, training_run_config: TrainingRunConfig, target_uri: str) -> None:
        """Save the training run configuration to the target storage."""
        training_run_config_dict = json.loads(training_run_config.model_dump_json())
        training_run_config_bytes = yaml.dump(training_run_config_dict, indent=4).encode()
        self.target_storage.write(training_run_config_bytes, target_uri)

    def _copy_files_and_modify_config(
        self,
        load_case_run: LoadCaseTrainingRunConfig,
        source_storage: StorageReader,
        source_storage_type: StorageType,
        target_storage_base_path: str,
    ):
        """Copy the training and test datasets from the source storage to the target storage
        and modify the training run config to point to the target storage."""
        source_training_file_name = self._get_blob_name(
            blob_uri=load_case_run.data.training_data_file_uri,
            storage_type=source_storage_type,
        )
        target_dataset_file_name = f"{target_storage_base_path}/{source_training_file_name}"
        self._copy_file(
            source_storage=source_storage,
            source_dataset_uri=load_case_run.data.training_data_file_uri,
            target_file_uri=target_dataset_file_name,
        )
        # Modify the path of the training dataset in the training run config to the target path
        load_case_run.data.training_data_file_uri = target_dataset_file_name

        if load_case_run.data.test_data_file_uri:
            source_test_data_file_name = self._get_blob_name(
                blob_uri=load_case_run.data.test_data_file_uri,
                storage_type=source_storage_type,
            )
            target_dataset_file_name = f"{target_storage_base_path}/{source_test_data_file_name}"
            self._copy_file(
                source_storage=source_storage,
                source_dataset_uri=load_case_run.data.test_data_file_uri,
                target_file_uri=target_dataset_file_name,
            )
            # Modify the path of the test dataset in the training run config to the target path
            load_case_run.data.test_data_file_uri = target_dataset_file_name

        if load_case_run.data.agg_data_file_uri:
            source_agg_data_file_name = self._get_blob_name(
                blob_uri=load_case_run.data.agg_data_file_uri,
                storage_type=source_storage_type,
            )
            target_dataset_file_name = f"{target_storage_base_path}/{source_agg_data_file_name}"
            self._copy_file(
                source_storage=source_storage,
                source_dataset_uri=load_case_run.data.agg_data_file_uri,
                target_file_uri=target_dataset_file_name,
            )
            load_case_run.data.agg_data_file_uri = target_dataset_file_name

    def prepare_training(self, training_run_config: TrainingRunConfig) -> Dict[str, Any]:
        """Prepare files and configuration for training.

        - Validates training data against the configuration
        - Copies training configuration and datasets to the target storage
        - Creates a uniquely named directory for the training run

        Parameters
        ----------
        training_run_config : TrainingRunConfig
            The user supplied training run configuration.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - training_input_id: Unique identifier for the training run
            - config_path: Path to the uploaded configuration file
        """
        if training_run_config.storage_type == StorageType.INTERNAL:
            source_storage = self.neuron_source_storage
        elif training_run_config.storage_type == StorageType.SIMMESH:
            source_storage = self.simmesh_source_storage
        else:
            raise ValueError(
                f"Invalid storage source {training_run_config.storage_type}. "
                f"Expected one of {['NEURON', 'SIMMESH']}"
            )
        validate_data_all_load_cases(
            training_run_config.load_case_training_runs, storage=source_storage
        )
        validate_test_data_for_agg_load_cases(
            training_run_config.load_case_training_runs, storage=source_storage
        )

        training_input_id = self._generate_unique_id()
        modified_config_target_path = (
            f"{self.target_storage_base_path}/{training_input_id}/training_run_config.yaml"
        )
        original_config_target_path = (
            f"{self.target_storage_base_path}/{training_input_id}/original_training_run_config.yaml"
        )
        original_training_run_config = training_run_config.model_copy(deep=True)

        for load_case_run in training_run_config.load_case_training_runs:
            self._copy_files_and_modify_config(
                load_case_run=load_case_run,
                source_storage=source_storage,
                source_storage_type=training_run_config.storage_type,
                target_storage_base_path=f"{self.target_storage_base_path}/{training_input_id}",
            )

        # We always modify the storage type to be internal as
        # the DBX job only supports internal Neuron storage
        training_run_config.storage_type = StorageType.INTERNAL
        self._save_config(
            training_run_config=training_run_config, target_uri=str(modified_config_target_path)
        )
        self._save_config(
            training_run_config=original_training_run_config,
            target_uri=str(original_config_target_path),
        )

        return {
            "training_input_id": training_input_id,
            "config_path": modified_config_target_path,
        }

    def train(self, training_run_config: TrainingRunConfig) -> TrainingServiceResponse:
        """Start a training job run on Databricks based on the provided configuration.

        - The training files and configuration are prepared and copied to the target storage
        - The training job run is triggered on Databricks

        Parameters
        ----------
        training_run_config : TrainingRunConfig
            The user supplied training run configuration.

        Returns
        -------
        TrainingServiceResponse
            The response from the training service.
        """
        # Prepare the training files and configuration
        prep_result = self.prepare_training(training_run_config)

        # Submit the training job
        job_run_id = self.training_job_service.start_train_job_run(prep_result["config_path"])

        return TrainingServiceResponse(
            job_run_id=job_run_id,
            training_input_id=prep_result["training_input_id"],
        )

    def get_training_run_info(self, job_run_id: int) -> NeuronTrainingJobRun:
        return self.training_job_service.get_training_run_info(job_run_id)
