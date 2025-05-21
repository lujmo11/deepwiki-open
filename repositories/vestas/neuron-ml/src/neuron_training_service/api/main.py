"""
FastAPI app for neuron training service.
"""


import structlog
from azure.identity import DefaultAzureCredential
from dobby.io.storages import BlobStorage
from prometheus_fastapi_instrumentator import Instrumentator, metrics

from neuron_training_service.api import create_app
from neuron_training_service.config import settings
from neuron_training_service.dbx_service.dbx_service import DBXService
from neuron_training_service.simmesh_storage import SASURLBlobStorage
from neuron_training_service.training_service import TrainingService

logger = structlog.get_logger()

logger.info(
    "Setting up source internal Neuron storage with blob storage for "
    f"{settings.source_internal_storage_account_name}/{settings.source_internal_storage_container_name}"
)
neuron_source_storage = BlobStorage(
    storage_account_name=settings.source_internal_storage_account_name,
    container_name=settings.source_internal_storage_container_name,
    credentials=DefaultAzureCredential(),
)
logger.info("Setting up source SimMesh storage with SAS URL reader.")
simmesh_source_storage = SASURLBlobStorage(
    storage_account_name=settings.source_simmesh_storage_account_name
)

logger.info(
    "Setting up target interal Neuron storage with blob storage for "
    f"{settings.target_internal_storage_account_name}/{settings.target_internal_storage_container_name}"
)
target_storage = BlobStorage(
    storage_account_name=settings.target_internal_storage_account_name,
    container_name=settings.target_internal_storage_container_name,
    credentials=DefaultAzureCredential(),
)

logger.info(
    "Setting up job trigger service with Databricks job ID "
    f"{settings.databricks_job_id} on host {settings.databricks_host} and MLFlow experiment id's "
    f"{settings.searchable_mlflow_experiment_ids}."
)
dbx_service = DBXService(
    dbx_host=settings.databricks_host,
    dbx_job_id=settings.databricks_job_id,
    experiment_ids=settings.searchable_mlflow_experiment_ids,
)

training_service = TrainingService(
    neuron_source_storage=neuron_source_storage,
    simmesh_source_storage=simmesh_source_storage,
    target_storage=target_storage,
    training_job_service=dbx_service,
    target_storage_base_path=settings.training_api_target_base_dir,
)

app = create_app(training_service, build_number=settings.build_number, git_sha=settings.git_sha)

# Add Prometheus metrics endpoint
instrumentator = Instrumentator(
    excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json", "/version"]
)
# Measure the latency of requests, with buckets for different response times
instrumentator.add(metrics.latency(buckets=[0.1, 1.0, 2.0, 5.0, 10.0]))

instrumentator.instrument(app).expose(app)
