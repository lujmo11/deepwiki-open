import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="x-api-key")


def validate_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Validate the API key from the request header."""

    # TRAIN_API_ALLOWED_API_KEYS will look like: `secure_key,another_key`
    allowed_api_keys = os.environ["TRAIN_API_ALLOWED_API_KEYS"].split(",")

    if api_key in allowed_api_keys:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )
