FROM python:3.12-slim as base
# Stage 1: Build
FROM base AS builder
WORKDIR /app
ENV POETRY_VERSION=1.7.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN apt-get update && apt-get install -y \
    build-essential && \
    apt-get clean && \
    pip install poetry==${POETRY_VERSION}

# Copy relevant files and folders
COPY pyproject.toml poetry.lock README.md ./

# Install all dependencies, but not the project itself
ARG AZURE_FEED_TOKEN
RUN poetry config http-basic.MLPlatform docker ${AZURE_FEED_TOKEN} && \
    poetry install --no-root --with api

# Copy the rest of the files and install the project. 
# This is split into two steps to avoid invalidating the cache when the code changes
COPY src src
RUN poetry install

# Stage 2: Final
FROM base as final
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Set build_number as environment variable. Do this late in the Dockerfile to avoid invalidating the cache
ARG BUILD_NUMBER
ENV BUILD_NUMBER=${BUILD_NUMBER}
ARG GIT_SHA="unknown"
ENV GIT_SHA=${GIT_SHA}

EXPOSE 8000
# Execute uvicorn with the fastapi app
ENTRYPOINT ["uvicorn", "neuron_training_service.api.main:app", "--proxy-headers", "--forwarded-allow-ips", "*", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]