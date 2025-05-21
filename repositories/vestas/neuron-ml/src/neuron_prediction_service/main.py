"""
FastAPI app for neuron prediction service.
"""

import json
from io import BytesIO
from typing import Annotated, List, Union

import pandas as pd
import structlog
from azure.identity import DefaultAzureCredential
from dobby.io.storages import BlobStorage, FileSystemStorage, Storage
from fastapi import Body, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from neuron.api_utils import LogMiddleware
from neuron_prediction_service.config import StorageBackend, settings
from neuron_prediction_service.exception_handling import register_exceptions
from neuron_prediction_service.model_cache import ModelCache
from neuron_prediction_service.model_store_service.model_store import ModelStore
from neuron_prediction_service.model_store_service.schemas import (
    TurbineModelVersionArtifactMetadata,
)
from neuron_prediction_service.schemas import (
    LOAD_CASE_MODEL_INPUT_EXAMPLES,
    LoadCaseModelInput,
    LoadCaseModelMetadata,
    PredictionMetadata,
    Predictions,
    PredictionServiceResponse,
)

app = FastAPI(description="Neuron Prediction Service")

logger = structlog.get_logger()
if settings.storage_backend == StorageBackend.local:
    logger.info("Using local disk model store")
    model_store_storage = FileSystemStorage(settings.model_store_folder_path)
elif settings.storage_backend == StorageBackend.azure:
    logger.info("Using Azure Blob model store")
    model_store_storage = BlobStorage(
        storage_account_name=settings.storage_account_name_models,
        container_name=settings.container_name,
        credentials=settings.model_store_token or DefaultAzureCredential(),
        mount_path=settings.build_number,
    )
else:
    raise ValueError(f"Unknown storage backend: {settings.storage_backend}")


def get_turbine_version_artifacts(storage: Storage) -> List[TurbineModelVersionArtifactMetadata]:
    with BytesIO(storage.read("turbine_model_version_artifacts.json")) as f:
        return [
            TurbineModelVersionArtifactMetadata.from_dict(artifact) for artifact in json.load(f)
        ]


turbine_model_artifacts = get_turbine_version_artifacts(storage=model_store_storage)
model_store = ModelStore(
    storage_reader=model_store_storage,
    turbine_model_artifacts=turbine_model_artifacts,
)

model_cache = ModelCache(model_cache_size=settings.model_cache_size)


@app.get("/")
async def root() -> str:
    """Root endpoint"""
    logger.info("Get request on '/'")
    return f"Hello World from Neuron Prediction Service. Build number = '{settings.build_number}'."


@app.get("/turbine_variant_ids")
def get_turbine_variant_ids() -> List[str]:
    """Get list of turbine variant ids"""
    return model_store.turbine_variant_ids


@app.get("/load_case_model_ids")
def get_model_ids() -> List[str]:
    """Get list of models ids"""
    return model_store.get_load_case_model_ids()


@app.get("/load_case_model_ids/{turbine_variant_id}")
def get_turbine_model_ids(
    turbine_variant_id: Union[str, None] = None,
) -> List[str]:
    """Get list of model ids for turbine_variant"""
    return model_store.get_load_case_model_ids(turbine_variant_id=turbine_variant_id)


@app.get("/load_case_model_ids/{turbine_variant_id}/{version}")
def get_turbine_version_model_ids(
    turbine_variant_id: Union[str, None] = None,
    version: Union[int, None] = None,
) -> List[str]:
    """Get list of model ids for turbine_variant"""
    return model_store.get_load_case_model_ids(
        turbine_variant_id=turbine_variant_id, version=version
    )


@app.get("/load_case_model_metadata/{load_case_model_id}")
def get_model_metadata(load_case_model_id: str) -> LoadCaseModelMetadata:
    """Get model metadata for a specific model"""
    turbine_variant_build_config = model_store.get_turbine_variant_build_config(
        load_case_model_id=load_case_model_id
    )
    load_case = model_store.get_load_case(load_case_model_id=load_case_model_id)
    return LoadCaseModelMetadata(
        turbine_variant_build_config=turbine_variant_build_config,
        load_case=load_case,
    )


@app.post("/predict/{load_case_model_id}")
def predict(
    load_case_model_id: str,
    body: Annotated[LoadCaseModelInput, Body(examples=LOAD_CASE_MODEL_INPUT_EXAMPLES)],
) -> PredictionServiceResponse:
    """Make predictions with a specific model"""
    load_case_model = model_cache.get(model_id=load_case_model_id)
    if load_case_model is None:
        load_case_model = model_store.get_load_case_model(load_case_model_id=load_case_model_id)
        model_cache.add(model_id=load_case_model_id, model=load_case_model)

    turbine_variant_build_config = model_store.get_turbine_variant_build_config(
        load_case_model_id=load_case_model_id
    )
    load_case = model_store.get_load_case(load_case_model_id=load_case_model_id)
    df_raw = pd.DataFrame.from_dict(data=body.data)

    df_preprocessed = load_case_model.preprocess(df=df_raw)
    interp_domain_validation = load_case_model.interpolation_domain_validation(df=df_preprocessed)
    extrap_domain_validation = load_case_model.extrapolation_domain_validation(df=df_preprocessed)

    load_case_predictions = load_case_model.predict(
        df=df_raw,
        return_std=True,
        targets=body.targets,
        grad_features=body.grad_features,
    )
    load_case_prediction_response = {
        target: Predictions(
            averages=pred.value_list,
            standard_deviations=pred.value_list_std,
            gradients=pred.gradients_dict,
        )
        for target, pred in load_case_predictions.items()
    }
    return PredictionServiceResponse(
        predictions=load_case_prediction_response,
        interpolation_domain_validation=interp_domain_validation,
        extrapolation_domain_validation=extrap_domain_validation,
        meta=PredictionMetadata(
            build_number=settings.build_number,
            turbine_variant_build_config=turbine_variant_build_config,
            load_case_name=load_case.name,
        ),
    )


@app.get("/version")
async def get_version() -> dict:
    return {
        "git_sha": settings.git_sha,
        "build_id": settings.build_number,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


app.add_middleware(LogMiddleware)
app = register_exceptions(app)
# Add Prometheus metrics endpoint
Instrumentator(
    excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json", "/version"]
).instrument(app).expose(app)
