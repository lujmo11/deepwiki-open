"""Shared utilities for both Prediction and Training APIs.
Mostly helper functions related to logging"""

import json
import logging
import uuid

import pydantic
import structlog
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    return pydantic.TypeAdapter(bool).validate_python(val)


class StructlogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Capture the log message and other record data, and pass it to structlog
        logger = structlog.get_logger(record.name)
        level = record.levelname.lower()

        # Use `structlog` methods to log at the appropriate level
        log_method = getattr(logger, level, logger.info)
        log_method(
            record.getMessage(),
            original_logger_name=record.name,
            original_filename=record.pathname,
            original_lineno=record.lineno,
            original_func_name=record.funcName,
            original_module=record.module,
            original_level=record.levelname,
        )


def configure_logger(use_json_logging: bool = False) -> None:
    # Disable uvicorn logging
    logging.getLogger("uvicorn.error").disabled = True
    logging.getLogger("uvicorn.access").disabled = True

    json_renderers = [
        structlog.processors.EventRenamer("msg"),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ]
    renderer = json_renderers if use_json_logging else [structlog.dev.ConsoleRenderer()]

    # Structlog configuration
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.PATHNAME,
                ]
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ]
        + renderer,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    # Configure loggers for "neuron" and "dobby" packages
    # We want to redirect logs from these packages to structlog,
    # so we can format as json for server logs
    for package in ["neuron", "dobby"]:
        package_logger = logging.getLogger(package)
        package_logger.setLevel(logging.INFO)  # Ensure that the log level is set
        package_logger.handlers = []
        package_logger.addHandler(StructlogHandler())  # Redirect logs to structlog


class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: callable) -> Response:
        """
        This middleware intercepts all requests and logs details from the request and response.
        """
        # Clear previous context variables
        structlog.contextvars.clear_contextvars()

        # Retrieve the request ID from the request headers
        trace_id: str = request.headers.get("x-vestas-request-id", str(uuid.uuid4()))

        # Bind new variables identifying the request and a generated UUID
        structlog.contextvars.bind_contextvars(
            request_path=request.url.path,
            request_method=request.method,
            request_id=str(trace_id),
            request_headers=dict(request.headers),
            query_params=dict(request.query_params),
        )

        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            structlog.contextvars.bind_contextvars(request_body=body.decode("utf-8"))
        # Make the request and receive a response
        response = await call_next(request)

        # If this is a health or metrics endpoint, don't log anything
        ignored_endpoints = ["/health", "/metrics", "/openapi.json"]
        if request.url.path in ignored_endpoints and response.status_code == 200:
            return response

        # Log the response if there is any error
        if response.status_code >= 400:
            try:
                response = await self.log_error_response(response)
            except Exception as e:
                logger.error(f"Failed to log response: {str(e)}")

        # Bind the status code of the response
        structlog.contextvars.bind_contextvars(status_code=response.status_code)

        if 400 <= response.status_code < 500:
            logger.warn("Client error")
        elif response.status_code >= 500:
            logger.error("Server error")
        else:
            logger.info("OK")

        return response

    async def log_error_response(self, response: Response) -> Response:
        """Log the response body for error responses"""
        if isinstance(response, StreamingResponse):
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            body = b"".join(chunks)
            response = StreamingResponse(
                iter([body]), status_code=response.status_code, headers=dict(response.headers)
            )
            response_body = json.loads(body.decode("utf-8"))

        else:
            try:
                response_body = response.body.decode("utf-8")
            except Exception:
                logger.warn(f"Unsupported response type for logging: {type(response)}")
                return response

        try:
            structlog.contextvars.bind_contextvars(response_body=response_body)
        except Exception:
            structlog.contextvars.bind_contextvars(response_body="[unable to decode response body]")

        return response
