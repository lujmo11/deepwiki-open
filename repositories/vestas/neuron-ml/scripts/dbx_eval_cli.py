"""CLI for submitting evaluation jobs to Databricks."""
import os

import typer
import yaml
from azure.identity import DefaultAzureCredential
from databricks.sdk import WorkspaceClient
from dobby.io.storages import BlobStorage, FileSystemStorage

from neuron import logger
from neuron_training_service.config import settings
from neuron_training_service.dbx_service.dbx_service import DBXService
from neuron_training_service.schemas import CLIUserTrainingRunConfig, NeuronTrainingJobRun
from neuron_training_service.simmesh_storage import SASURLBlobStorage
from neuron_training_service.training_service import TrainingJobService, TrainingService
from neuron_training_service.user_config_parsing import get_training_config_from_cli_user_config

app = typer.Typer()

client = WorkspaceClient()


_raw_id = os.getenv("DATABRICKS_EVAL_JOB_ID")
if _raw_id is None or not _raw_id.isdigit():
    raise ValueError("DATABRICKS_EVAL_JOB_ID must be set and must be an integer.")
EVAL_JOB_ID: int = int(_raw_id)


class EvalJobService(TrainingJobService):
    """Service for submitting evaluation jobs to Databricks."""

    def __init__(self, dbx_host: str, dbx_job_id: int, mlflow_run_id: str):
        """Initialize the evaluation job service.

        Args:
            dbx_host: Databricks host URL
            dbx_job_id: Databricks job ID for evaluation
            mlflow_run_id: MLflow run ID to evaluate
        """
        self.dbx_host = dbx_host
        self.dbx_job_id = dbx_job_id
        self.client = WorkspaceClient()
        self.mlflow_run_id = mlflow_run_id

    def start_train_job_run(self, config_blob_path: str) -> int:
        """Start an evaluation job run on Databricks.

        Args:
            config_blob_path: Path to the training configuration blob

        Returns:
            int: The run ID of the triggered job run
        """

        job_trigger_response = self.client.jobs.run_now(
            job_id=self.dbx_job_id,
            python_params=[
                "--train-config-path",
                config_blob_path,
                "--mlflow-run-id",
                self.mlflow_run_id,
            ],
        )
        return job_trigger_response.response.run_id

    def get_training_run_info(self, job_run_id: int) -> NeuronTrainingJobRun:
        """Get information about a job run."""
        dbx_service = DBXService(
            dbx_host=self.dbx_host,
            dbx_job_id=self.dbx_job_id,
        )
        return dbx_service.get_training_run_info(job_run_id)


@app.command()
def eval(
    train_config_path: str,
    mlflow_run_id: str,
    username: str,  # Added username parameter
    use_local_source_storage: bool = False,
) -> None:
    """Submit an evaluation job to Databricks.

    Args:
        train_config_path: Path to the training config file.
        mlflow_run_id: MLflow run ID for the model to evaluate.
        username: Username to use as base directory.
        use_local_source_storage: Whether to use local source storage.
    """
    logger.info("Submitting an evaluation job to Databricks")
    logger.info(f"Using training config path: {train_config_path}")
    logger.info(f"Using MLflow run ID: {mlflow_run_id}")
    logger.info(f"Using username: {username}")
    logger.info(f"Using job ID: {EVAL_JOB_ID}")

    # Set up the storage and services similar to training CLI
    if use_local_source_storage:
        neuron_source_storage = FileSystemStorage()
    else:
        neuron_source_storage = BlobStorage(
            storage_account_name=settings.source_internal_storage_account_name,
            container_name=settings.source_internal_storage_container_name,
            credentials=DefaultAzureCredential(),
        )
    logger.info(f"Using {neuron_source_storage} for neuron source data.")

    simmesh_source_storage = SASURLBlobStorage(
        storage_account_name=settings.source_simmesh_storage_account_name
    )
    logger.info(f"Using {simmesh_source_storage} for simmesh source data.")

    target_storage = BlobStorage(
        storage_account_name=settings.target_internal_storage_account_name,
        container_name=settings.target_internal_storage_container_name,
        credentials=DefaultAzureCredential(),
    )
    logger.info(f"Using {target_storage} for target data storage.")

    # Create evaluation job service with the MLflow run ID
    eval_job_service = EvalJobService(
        dbx_host=settings.databricks_host, dbx_job_id=EVAL_JOB_ID, mlflow_run_id=mlflow_run_id
    )

    training_service = TrainingService(
        training_job_service=eval_job_service,
        neuron_source_storage=neuron_source_storage,
        simmesh_source_storage=simmesh_source_storage,
        target_storage=target_storage,
        target_storage_base_path=username,
    )

    # Read and parse config
    logger.info(f"Reading training run config from {train_config_path}")
    with open(train_config_path, "r") as f:
        user_training_run_config = yaml.safe_load(f)

    logger.info("Parsing user training run config")
    user_training_run_config = CLIUserTrainingRunConfig(**user_training_run_config)
    user_training_run_config = get_training_config_from_cli_user_config(user_training_run_config)

    logger.info("Starting evaluation preparation")
    prep_result = training_service.prepare_training(user_training_run_config)

    eval_job_run_id = eval_job_service.start_train_job_run(prep_result["config_path"])

    logger.info(f"Submitted evaluation job with run ID: {eval_job_run_id}")


@app.command()
def status(job_run_id: int) -> None:
    eval_job_service = EvalJobService(
        dbx_host=settings.databricks_host, dbx_job_id=EVAL_JOB_ID, mlflow_run_id=""
    )
    training_job_info = eval_job_service.get_training_run_info(job_run_id)
    logger.info(training_job_info.model_dump())


if __name__ == "__main__":
    app()
