"""Exception handling for neuron_training_service.

This module sets up error handling for the FastAPI app.
We translate different exceptions raised by the neuron code into appropriate HTTP responses.
"""
import logging

from azure.core.exceptions import ClientAuthenticationError
from dobby.io.storages.exceptions import FileDoesNotExistInStorageError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from neuron.data_validation.exceptions import DataValidationError
from neuron.safe_domain.validators.exceptions import ValidatorFittingError
from neuron_training_service.dbx_service.exceptions import DBXJobDoesNotExistError
from neuron_training_service.schemas import APIUserConfigParsingError, TrainingRungConfigError
from neuron_training_service.simmesh_storage import (
    SimMeshAuthenticationError,
    WrongBlobURIError,
    WrongStorageAccountError,
)

logger = logging.getLogger(__name__)


def dbx_job_does_not_exist_exception_handler(
    _: Request, exception: DBXJobDoesNotExistError
) -> JSONResponse:
    """Exception handler for DBXJobDoesNotExistError."""
    return JSONResponse(
        status_code=404,
        content={"detail": f"Job run with ID {exception.job_run_id} does not exist."},
    )


def file_does_not_exist_exception_handler(
    _: Request, exception: FileDoesNotExistInStorageError
) -> JSONResponse:
    """Exception handler for dobby exception FileDoesNotExistInStorageError."""
    return JSONResponse(
        status_code=404,
        content={
            "detail": f"File {exception.path} in training config body does not exist in storage."
        },
    )


def client_authentication_exception_handler(
    _: Request,
    exception: ClientAuthenticationError,  # noqa: ARG001
) -> JSONResponse:
    """Exception handler for Azure ClientAuthenticationError."""
    return JSONResponse(
        status_code=401,
        content={
            "detail": (
                "Client authentication error. "
                "Check the credentials used to access the storage or the name of the container."
            )
        },
    )


def data_validation_exception_handler(_: Request, exception: DataValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": (f"Data validation error: {exception}")},
    )


def validator_fitting_exception_handler(
    _: Request, exception: ValidatorFittingError
) -> JSONResponse:
    """Exception handler for ValidatorFittingError."""
    return JSONResponse(
        status_code=400,
        content={"detail": (f"Problem fitting the SafeDomainValidator: {exception}")},
    )


def wrong_blob_uri_exception_handler(_: Request, exception: WrongBlobURIError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exception)},
    )


def wrong_storage_account_exception_handler(
    _: Request, exception: WrongStorageAccountError
) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": str(exception)},
    )


def api_user_config_parsing_exception_handler(
    _: Request, exception: APIUserConfigParsingError
) -> JSONResponse:
    """Exception handler for APIUserConfigParsingError."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exception)},
    )


def training_run_config_error_exception_handler(
    _: Request, exception: TrainingRungConfigError
) -> JSONResponse:
    """Exception handler for TrainingRunConfigError."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exception)},
    )


def sim_mesh_authentication_exception_handler(
    _: Request, exception: SimMeshAuthenticationError
) -> JSONResponse:
    """Exception handler for SimMeshAuthenticationError."""
    return JSONResponse(
        status_code=401,
        content={"detail": str(exception)},
    )


def unknown_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unknown exception: {exc!r}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                f"Failed method {request.method} at URL {request.url}."
                f" Exception message is {exc!r}."
                "Please contact Neuron team if this persists."
            )
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DBXJobDoesNotExistError, dbx_job_does_not_exist_exception_handler)
    app.add_exception_handler(FileDoesNotExistInStorageError, file_does_not_exist_exception_handler)
    app.add_exception_handler(ClientAuthenticationError, client_authentication_exception_handler)
    app.add_exception_handler(DataValidationError, data_validation_exception_handler)
    app.add_exception_handler(ValidatorFittingError, validator_fitting_exception_handler)
    app.add_exception_handler(WrongBlobURIError, wrong_blob_uri_exception_handler)
    app.add_exception_handler(WrongStorageAccountError, wrong_storage_account_exception_handler)
    app.add_exception_handler(SimMeshAuthenticationError, sim_mesh_authentication_exception_handler)
    app.add_exception_handler(APIUserConfigParsingError, api_user_config_parsing_exception_handler)
    app.add_exception_handler(TrainingRungConfigError, training_run_config_error_exception_handler)
    app.add_exception_handler(Exception, unknown_exception_handler)
