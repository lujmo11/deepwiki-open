"""
FastAPI app for neuron training service.
"""

import os
from typing import Annotated

import structlog
from fastapi import Body, FastAPI, Security

from neuron.api_utils import LogMiddleware, configure_logger, strtobool
from neuron.schemas.training_run_config import TrainingRunConfig
from neuron_training_service.api.exception_handling import register_exception_handlers
from neuron_training_service.api.security import validate_api_key
from neuron_training_service.schemas import (
    TRAIN_JOB_INPUT_EXAMPLES,
    APIUserTrainingRunConfig,
    NeuronTrainingJobRun,
    TrainingServiceResponse,
)
from neuron_training_service.training_service import TrainingService
from neuron_training_service.user_config_parsing import (
    get_training_config_from_api_user_config,
)

logger = structlog.get_logger()


def create_app(
    training_service: TrainingService,
    build_number: str,
    git_sha: str,
) -> FastAPI:
    """Create the FastAPI app for the neuron training service.

    We use the build pattern to allow for easier testing of the FastAPI app.
    """
    use_json_logging = strtobool(os.getenv("USE_JSON_LOGGING", "False"))
    configure_logger(use_json_logging=use_json_logging)

    app = FastAPI(description="Neuron Training Service")

    @app.get("/")
    def root() -> str:
        """Root endpoint"""
        logger.info("Get request on '/'")
        return f"Hello World from Neuron Training Service. Build number = '{build_number}'."

    @app.get("/version")
    def get_version() -> dict:
        return {
            "git_sha": git_sha,
            "build_id": build_number,
        }

    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy"}

    @app.post("/train_job", dependencies=[Security(validate_api_key)])
    def start_train_job(
        train_config: Annotated[APIUserTrainingRunConfig, Body(examples=TRAIN_JOB_INPUT_EXAMPLES)],
    ) -> TrainingServiceResponse:
        """Start a training job based on the provided training run configuration."""
        logger.info(f"Parsed training config: {train_config.model_dump()}.")
        parsed_train_config = get_training_config_from_api_user_config(train_config)
        training_service_response = training_service.train(training_run_config=parsed_train_config)
        return TrainingServiceResponse(
            job_run_id=training_service_response.job_run_id,
            training_input_id=training_service_response.training_input_id,
        )

    @app.post("/train_job_full_spec", dependencies=[Security(validate_api_key)])
    def start_train_job_based_on_full_training_config(
        train_config: TrainingRunConfig,
    ) -> TrainingServiceResponse:
        """Start a training job based on a full training run configuration.

        Only for internal neuron use! Used for re-running jobs based on full training run config
        from old runs or experimentation in DEV environment.
        """
        training_service_response = training_service.train(training_run_config=train_config)
        return TrainingServiceResponse(
            job_run_id=training_service_response.job_run_id,
            training_input_id=training_service_response.training_input_id,
        )

    @app.get("/train_job/{job_run_id}", dependencies=[Security(validate_api_key)])
    def get_train_job_status(job_run_id: int) -> NeuronTrainingJobRun:
        """Get info on a training job based on the provided job run ID."""
        return training_service.get_training_run_info(job_run_id=job_run_id)

    app.add_middleware(LogMiddleware)
    register_exception_handlers(app)

    return app
