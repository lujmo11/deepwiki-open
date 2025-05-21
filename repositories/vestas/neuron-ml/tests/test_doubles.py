from pathlib import Path
from typing import List, Union

import pandas as pd
from azure.core.exceptions import ClientAuthenticationError
from databricks.sdk.service.jobs import RunLifeCycleState
from dobby.io.storages.exceptions import FileDoesNotExistInStorageError

from neuron.data_validation.exceptions import DataValidationError
from neuron.safe_domain.validators.exceptions import ValidatorFittingError
from neuron.schemas.training_run_config import TrainingRunConfig
from neuron_training_service.dbx_service.exceptions import DBXJobDoesNotExistError
from neuron_training_service.schemas import JobRun, NeuronTrainingJobRun, TrainingServiceResponse
from neuron_training_service.simmesh_storage import (
    BlobUrlComponents,
    SimMeshAuthenticationError,
    WrongBlobURIError,
    WrongStorageAccountError,
)


class DBXServiceTestDouble:
    def __init__(self, job_run_id: int = 123):
        self.job_run_id = job_run_id

    def start_train_job_run(self, config_blob_path: str) -> int:
        return self.job_run_id

    def get_training_run_info(self, job_run_id: int) -> NeuronTrainingJobRun:
        return NeuronTrainingJobRun(
            job_run=JobRun(
                id=self.job_run_id,
                url="dummy_url",
                result_state=None,
                life_cycle_state=RunLifeCycleState.RUNNING,
            ),
            mlflow_run=None,
        )


class SASURLBlobStorageTestDouble:
    def __init__(
        self,
        storage_account_name: str,
        container_name: str,
        available_data: Path,
        valid_sas_token: str,
    ):
        self.storage_account_name = storage_account_name
        self.container_name = container_name
        self.available_data_folder = Path(available_data)
        self.valid_sas_token = valid_sas_token

    def read(self, path: str) -> bytes:
        components = BlobUrlComponents.from_url(path)
        if components.storage_account_name != self.storage_account_name:
            raise WrongStorageAccountError("Invalid storage account.")
        if components.blob_path not in [f.name for f in self.available_data_folder.iterdir()]:
            raise FileDoesNotExistInStorageError(path)
        return self.available_data_folder.joinpath(components.blob_path).read_bytes()


class TrainingServiceTestDouble:
    def __init__(
        self,
        available_data_paths: Union[List[str], None] = None,
        available_job_run_ids: Union[List[int], None] = None,
        throw_data_validation_error: bool = False,
        throw_safe_domain_validator_error: bool = False,
        throw_bad_blob_uri_error: bool = False,
        throw_wrong_storage_account_error: bool = False,
        throw_client_authentication_error: bool = False,
        throw_simmesh_authentication_error: bool = False,
        data_validation_error_message: Union[str, None] = None,
        safe_domain_validator_error_message: Union[str, None] = None,
        wrong_blob_uri_error_message: Union[str, None] = None,
        wrong_storage_account_error_message: Union[str, None] = None,
        simmesh_authentication_error_message: Union[str, None] = None,
    ):
        self.available_data_paths = available_data_paths or []
        self.available_job_run_ids = available_job_run_ids or []
        self.throw_data_validation_error = throw_data_validation_error
        self.throw_safe_domain_validator_error = throw_safe_domain_validator_error
        self.throw_bad_blob_uri_error = throw_bad_blob_uri_error
        self.throw_wrong_storage_account_error = throw_wrong_storage_account_error
        self.throw_client_authentication_error = throw_client_authentication_error
        self.throw_simmesh_authentication_error = throw_simmesh_authentication_error
        self.data_validation_error_message = data_validation_error_message
        self.safe_domain_validator_error_message = safe_domain_validator_error_message
        self.wrong_blob_uri_error_message = wrong_blob_uri_error_message
        self.wrong_storage_account_error_message = wrong_storage_account_error_message
        self.simmesh_authentication_error_message = simmesh_authentication_error_message

        if self.throw_data_validation_error and not self.data_validation_error_message:
            raise ValueError(
                "data_validation_error_message must be provided if "
                "throw_data_validation_error is True."
            )
        if self.throw_safe_domain_validator_error and not self.safe_domain_validator_error_message:
            raise ValueError(
                "safe_domain_validator_error_message must be provided if "
                "throw_safe_domain_validator_error is True."
            )
        if self.throw_bad_blob_uri_error and not self.wrong_blob_uri_error_message:
            raise ValueError(
                "wrong_blob_uri_error_message must be provided if "
                "throw_bad_blob_uri_error is True."
            )
        if self.throw_wrong_storage_account_error and not self.wrong_storage_account_error_message:
            raise ValueError(
                "wrong_storage_account_error_message must be provided if "
                "throw_wrong_storage_account_error is True."
            )

        if (
            self.throw_simmesh_authentication_error
            and not self.simmesh_authentication_error_message
        ):
            raise ValueError(
                "simmesh_authentication_error_message must be provided if "
                "throw_simmesh_authentication_error is True."
            )

    def train(self, training_run_config: TrainingRunConfig) -> TrainingServiceResponse:  # noqa: C901
        if self.throw_data_validation_error:
            raise DataValidationError(self.data_validation_error_message)
        if self.throw_safe_domain_validator_error:
            raise ValidatorFittingError(self.safe_domain_validator_error_message)
        if self.throw_bad_blob_uri_error:
            raise WrongBlobURIError(self.wrong_blob_uri_error_message)
        if self.throw_wrong_storage_account_error:
            raise WrongStorageAccountError(self.wrong_storage_account_error_message)
        if self.throw_client_authentication_error:
            raise ClientAuthenticationError
        if self.throw_simmesh_authentication_error:
            raise SimMeshAuthenticationError(self.simmesh_authentication_error_message)
        training_data_paths = [
            lc.data.training_data_file_uri for lc in training_run_config.load_case_training_runs
        ]
        test_data_paths = [
            lc.data.test_data_file_uri
            for lc in training_run_config.load_case_training_runs
            if lc.data.test_data_file_uri is not None
        ]
        for p in training_data_paths:
            if p not in self.available_data_paths:
                raise FileDoesNotExistInStorageError(path=p)
            for p in test_data_paths:
                if p not in self.available_data_paths:
                    raise FileDoesNotExistInStorageError(path=p)
        return TrainingServiceResponse(job_run_id=123, training_input_id="test_training_input_id")

    def get_training_run_info(self, job_run_id: int) -> NeuronTrainingJobRun:
        if job_run_id not in self.available_job_run_ids:
            raise DBXJobDoesNotExistError(job_run_id)
        return NeuronTrainingJobRun(
            job_run=JobRun(
                id=job_run_id,
                url="dummy_url",
                result_state=None,
                life_cycle_state=RunLifeCycleState.RUNNING,
            ),
            mlflow_run=None,
        )


class TrainingDataRepositoryTestDouble:
    """Test double for TrainingDataRepository."""

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, agg_df: pd.DataFrame) -> None:
        self.training_df = train_df
        self.test_df = test_df
        self.agg_df = agg_df

    def get_load_case_train_df(self) -> pd.DataFrame:
        return self.training_df

    def get_load_case_test_df(self) -> pd.DataFrame:
        return self.test_df

    def get_load_case_agg_df(self) -> pd.DataFrame:
        return self.agg_df
