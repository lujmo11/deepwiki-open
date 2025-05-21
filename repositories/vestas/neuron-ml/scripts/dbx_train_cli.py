"""CLI for starting a training job on Databricks based on a provided `CLIUserTrainingRunConfig`."""


import yaml
from azure.identity import DefaultAzureCredential
from dobby.io.storages import BlobStorage, FileSystemStorage
from typer import Typer

from neuron import logger
from neuron_training_service.config import settings
from neuron_training_service.dbx_service.dbx_service import DBXService
from neuron_training_service.schemas import CLIUserTrainingRunConfig
from neuron_training_service.simmesh_storage import SASURLBlobStorage
from neuron_training_service.training_service import TrainingService
from neuron_training_service.user_config_parsing import get_training_config_from_cli_user_config

app = Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    training_run_config_path: str, base_target_dir: str, use_local_source_storage: bool = False
) -> None:
    """Start a training job on Databricks based on a provided training run configuration."""

    # Set up the training service with the required storage and job trigger services.
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

    training_service = TrainingService(
        training_job_service=DBXService(
            dbx_host=settings.databricks_host, dbx_job_id=settings.databricks_job_id
        ),
        neuron_source_storage=neuron_source_storage,
        simmesh_source_storage=simmesh_source_storage,
        target_storage=target_storage,
        target_storage_base_path=base_target_dir,
    )

    logger.info(f"Reading training run config from {training_run_config_path}")
    with open(training_run_config_path, "r") as f:
        user_training_run_config = yaml.safe_load(f)

    logger.info("parsing user training run config")
    user_training_run_config = CLIUserTrainingRunConfig(**user_training_run_config)
    user_training_run_config = get_training_config_from_cli_user_config(user_training_run_config)

    logger.info("Starting training job.")
    training_service_response = training_service.train(user_training_run_config)
    logger.info(
        f"Started Databricks job with run ID {training_service_response.job_run_id} "
        f"and training data input ID "
        f"{training_service_response.training_input_id}."
    )
    logger.info("Getting status of the training job run.")
    training_run_info = training_service.get_training_run_info(training_service_response.job_run_id)

    logger.info(
        f"Training triggered with following bob run info response: {training_run_info.model_dump()}"
    )


@app.command()
def status(job_run_id: str) -> None:
    """Get the status of a training job run on Databricks."""
    dbx_job_service = DBXService(
        dbx_host=settings.databricks_host,
        dbx_job_id=settings.databricks_job_id,
    )
    training_job_info = dbx_job_service.get_training_run_info(int(job_run_id))
    logger.info(training_job_info.model_dump())


if __name__ == "__main__":
    app()
