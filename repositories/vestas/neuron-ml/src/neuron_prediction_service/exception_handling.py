"""Exception handling for neuron_prediction_service.

This module sets up error handling for the FastAPI app.
We translate different exceptions raised by the neuron code into appropriate HTTP responses.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from neuron.data_validation.exceptions import DataValidationError
from neuron_prediction_service.model_store_service.exceptions import (
    LoadCaseModelIDNotCorrect,
    LoadCaseModelNotInModelStore,
    TurbineVariantNotInModelStore,
    TurbineVariantVersionNotInModelStore,
)

logger = logging.getLogger(__name__)


def data_validation_exception_handler(_: Request, exception: DataValidationError) -> JSONResponse:
    """Exception handler for DataValidationError."""
    return JSONResponse(status_code=400, content={"details": f"Bad model input data: {exception}"})


def turbine_variant_not_in_model_store_exception_handler(
    _: Request, exception: TurbineVariantNotInModelStore
) -> JSONResponse:
    """Exception handler for TurbineVariantNotInModelStore error."""
    return JSONResponse(
        status_code=404,
        content={"details": str(exception)},
    )


def turbine_version_not_in_model_store_exception_handler(
    _: Request, exception: TurbineVariantVersionNotInModelStore
) -> JSONResponse:
    """Exception handler for TurbineVersionNotInModelStore error."""
    return JSONResponse(
        status_code=404,
        content={"details": str(exception)},
    )


def load_case_model_not_in_model_store_exception_handler(
    _: Request, exception: LoadCaseModelNotInModelStore
) -> JSONResponse:
    """Exception handler for LoadCaseModelNotInModelStore error."""
    return JSONResponse(
        status_code=404,
        content={"details": str(exception)},
    )


def load_case_model_id_not_correct_exception_handler(
    _: Request, exception: LoadCaseModelIDNotCorrect
) -> JSONResponse:
    """Exception handler for LoadCaseModelIDNotCorrect error."""
    return JSONResponse(
        status_code=422,
        content={"details": str(exception)},
    )


def unknown_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unknown exception: {exc!r}")
    return JSONResponse(
        status_code=500,
        content={
            "details": (
                f"Failed method {request.method} at URL {request.url}."
                f" Exception message is {exc!r}."
            )
        },
    )


def register_exceptions(app: FastAPI) -> FastAPI:
    """Register exception handlers for FastAPI app."""
    app.add_exception_handler(DataValidationError, data_validation_exception_handler)
    app.add_exception_handler(
        TurbineVariantNotInModelStore, turbine_variant_not_in_model_store_exception_handler
    )
    app.add_exception_handler(
        TurbineVariantVersionNotInModelStore, turbine_version_not_in_model_store_exception_handler
    )
    app.add_exception_handler(
        LoadCaseModelNotInModelStore, load_case_model_not_in_model_store_exception_handler
    )
    app.add_exception_handler(
        LoadCaseModelIDNotCorrect, load_case_model_id_not_correct_exception_handler
    )
    app.add_exception_handler(Exception, unknown_exception_handler)
    return app
