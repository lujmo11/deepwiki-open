"""
Test the logging middleware of the API. """
import json
import os

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def valid_api_key() -> str:
    return "secure_key"


@pytest.fixture(autouse=True)
def setup_env(monkeypatch: pytest.MonkeyPatch, valid_api_key: str) -> None:
    monkeypatch.setenv("USE_JSON_LOGGING", "true")
    monkeypatch.setenv("TRAIN_API_ALLOWED_API_KEYS", f"some_key,{valid_api_key}")


@pytest.fixture()
def client(setup_env: None) -> TestClient:  # noqa: ARG001
    import structlog

    structlog.reset_defaults()

    from neuron_training_service.api import create_app
    from tests.test_doubles import TrainingServiceTestDouble

    app = create_app(
        training_service=TrainingServiceTestDouble(),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )

    return TestClient(app)


@pytest.mark.usefixtures("setup_env")
def test_info_logging(client: TestClient, capsys: pytest.CaptureFixture[str]) -> None:
    response = client.get("/version")
    assert response.status_code == 200

    assert os.getenv("USE_JSON_LOGGING") == "true"

    # Read and parse the logs
    captured = capsys.readouterr()
    log_entries = [json.loads(line) for line in captured.out.splitlines()]

    # Since the response is successful, I expect only one entry, and it should be info(OK)
    assert len(log_entries) == 1
    log_entry = log_entries[0]
    assert log_entry["status_code"] == 200
    assert log_entry["level"] == "info"
    assert log_entry["request_path"] == "/version"
    # We should not capture the response body for successful requests
    assert "response_body" not in log_entry


@pytest.mark.usefixtures("setup_env")
def test_error_logging_404(client: TestClient, capsys: pytest.CaptureFixture[str]) -> None:
    response = client.post("/non-existant", json={"wrong_field": "value"})
    assert response.status_code == 404

    assert os.getenv("USE_JSON_LOGGING") == "true"

    # Read and parse the logs
    captured = capsys.readouterr()
    log_entries = [json.loads(line) for line in captured.out.splitlines()]

    assert len(log_entries) == 1
    log_entry = log_entries[0]
    assert log_entry["request_path"] == "/non-existant"
    assert log_entry["status_code"] == 404
    # Since this is an error, we should capture the response body
    assert "response_body" in log_entry
    assert len(log_entry["response_body"]) > 0


@pytest.mark.usefixtures("setup_env")
def test_error_logging_bad_input(
    client: TestClient, capsys: pytest.CaptureFixture[str], valid_api_key: str
) -> None:
    response = client.post(
        "/train_job", json={"wrong_field": "value"}, headers={"x-api-key": valid_api_key}
    )
    assert response.status_code == 422

    assert os.getenv("USE_JSON_LOGGING") == "true"

    # Read and parse the logs
    captured = capsys.readouterr()
    log_entries = [json.loads(line) for line in captured.out.splitlines()]

    assert len(log_entries) == 1
    log_entry = log_entries[0]
    assert log_entry["request_path"] == "/train_job"
    assert log_entry["status_code"] == 422
    # Since this is an error, we should capture the response body
    assert "response_body" in log_entry
    assert len(log_entry["response_body"]) > 0
    # This is part of the pydantic validation error message
    assert "detail" in log_entry["response_body"]
