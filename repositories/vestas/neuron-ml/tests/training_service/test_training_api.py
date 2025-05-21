import pytest
from fastapi.testclient import TestClient

from neuron_training_service.api import create_app
from neuron_training_service.schemas import APIUserTrainingRunConfig
from neuron_training_service.user_config_parsing import get_training_config_from_api_user_config
from tests.test_doubles import TrainingServiceTestDouble


@pytest.fixture(scope="function")
def train_job_payload() -> dict:
    """A train job payload for testing."""
    return {
        "load_case_training_runs": [
            {"name": "dlc11", "data": {"training_data_file_uri": "data.parquet"}}
        ],
        "turbine": {
            "turbine_variant": {
                "rotor_diameter": "150",
                "rated_power": "4000",
                "mk_version": "mk3f",
            }
        },
        "evaluation": {
            "alpha_significance_level": 0,
            "generate_coverage_plots": True,
            "fat_model_acceptance_criteria": [
                {"metric": "e_mean_norm", "value": 0.02, "condition": "lt"}
            ],
            "ext_model_acceptance_criteria": [
                {"metric": "e_mean_norm", "value": 0.02, "condition": "lt"}
            ],
        },
    }


@pytest.fixture(scope="function")
def train_job_payload_full_spec(train_job_payload: dict) -> dict:
    """A train job payload for the full spec endpoint for testing."""
    return get_training_config_from_api_user_config(
        APIUserTrainingRunConfig(**train_job_payload)
    ).model_dump()


@pytest.fixture(scope="function")
def train_job_payload_simmesh_storage(train_job_payload: dict) -> dict:
    """A train job payload for testing with simmesh storage."""
    train_job_payload["storage_type"] = "simmesh"
    train_job_payload["load_case_training_runs"][0]["data"][
        "training_data_file_uri"
    ] = "https://myaccount.blob.core.windows.net/mycontainer/sample_testing_data.parquet?valid_token"
    return train_job_payload


@pytest.fixture
def valid_api_key() -> str:
    return "secure_key"


@pytest.fixture
def invalid_api_key() -> str:
    return "invalid_key"


@pytest.fixture
def valid_api_key_header(valid_api_key: str) -> dict:
    return {"x-api-key": valid_api_key}


@pytest.fixture(autouse=True)
def set_api_keys(monkeypatch: pytest.MonkeyPatch, valid_api_key: str) -> None:
    monkeypatch.setenv("TRAIN_API_ALLOWED_API_KEYS", f"some_key,{valid_api_key}")


@pytest.mark.api
def test_ping() -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == (
        "Hello World from Neuron Training Service. Build number = 'test_build_number'."
    )


def test_health() -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_version() -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.get("/version")
    assert response.status_code == 200
    response_body = response.json()
    assert response_body["git_sha"] == "test_git_sha"
    assert response_body["build_id"] == "test_build_number"


def test_train_job_post(train_job_payload: dict, valid_api_key_header: dict) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(available_data_paths=["data.parquet"]),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post("/train_job", json=train_job_payload, headers=valid_api_key_header)
    assert response.status_code == 200
    response_body = response.json()
    assert response_body["job_run_id"] == 123
    assert response_body["training_input_id"] == "test_training_input_id"


def test_train_job_bad_load_case_name(train_job_payload: dict, valid_api_key_header: dict) -> None:
    """Test that the API handles an error when the load case name is not valid."""
    train_job_payload_bad = train_job_payload.copy()
    train_job_payload_bad["load_case_training_runs"][0]["name"] = "bad_load_case_name"
    app = create_app(
        training_service=TrainingServiceTestDouble(available_data_paths=["data.parquet"]),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post("/train_job", json=train_job_payload, headers=valid_api_key_header)
    assert response.status_code == 422
    response_body = response.json()
    assert (
        "Load case name bad_load_case_name not in default load case names."
        in response_body["detail"]
    ), (
        "Expected error message to contain 'Load case name bad_load_case_name not in default "
        "load case names.', "
    )


def test_train_job_full_spec_post(
    train_job_payload_full_spec: dict, valid_api_key_header: dict
) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(available_data_paths=["data.parquet"]),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post(
        "/train_job_full_spec",
        json=train_job_payload_full_spec,
        headers=valid_api_key_header,
    )
    assert response.status_code == 200
    response_body = response.json()
    assert response_body["job_run_id"] == 123
    assert response_body["training_input_id"] == "test_training_input_id"


def test_train_job_post_returns_404_for_non_existing_data(
    train_job_payload: dict, valid_api_key_header: dict
) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(available_data_paths=["data.parquet"]),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)

    # Replace the training data path with a non-existing path
    train_job_payload["load_case_training_runs"][0]["data"][
        "training_data_file_uri"
    ] = "non_existing_data_path.parquet"
    response = client.post("/train_job", json=train_job_payload, headers=valid_api_key_header)
    assert response.status_code == 404
    response_body = response.json()
    assert (
        response_body["detail"]
        == "File non_existing_data_path.parquet in training config body does not exist in storage."
    )


def test_train_job_post_returns_400_for_data_validation_error(
    train_job_payload: dict, valid_api_key_header: dict
) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(  # type: ignore
            throw_data_validation_error=True,
            data_validation_error_message="Some data validation error occurred.",
        ),
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post("/train_job", json=train_job_payload, headers=valid_api_key_header)
    assert response.status_code == 400
    response_body = response.json()
    assert response_body["detail"] == "Data validation error: Some data validation error occurred."


def test_train_job_post_returns_400_for_safe_domain_validator_error(
    train_job_payload: dict,
    valid_api_key_header: dict,
) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(  # type: ignore
            throw_safe_domain_validator_error=True,
            safe_domain_validator_error_message="Some safe domain validator error occurred.",
        ),
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post("/train_job", json=train_job_payload, headers=valid_api_key_header)
    assert response.status_code == 400
    response_body = response.json()
    assert (
        response_body["detail"]
        == "Problem fitting the SafeDomainValidator: Some safe domain validator error occurred."
    )


def test_train_job_post_returns_401_for_client_authentication_error(
    train_job_payload_simmesh_storage: dict,
    valid_api_key_header: dict,
) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(throw_client_authentication_error=True),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post(
        "/train_job", json=train_job_payload_simmesh_storage, headers=valid_api_key_header
    )
    assert response.status_code == 401
    response_body = response.json()
    assert response_body["detail"] == (
        "Client authentication error. "
        "Check the credentials used to access the storage or the name of the container."
    )


def test_train_job_post_returns_401_for_wrong_storage_account(
    train_job_payload_simmesh_storage: dict,
    valid_api_key_header: dict,
) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(  # type: ignore
            throw_wrong_storage_account_error=True,
            wrong_storage_account_error_message="Invalid storage account.",
        ),
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post(
        "/train_job", json=train_job_payload_simmesh_storage, headers=valid_api_key_header
    )
    assert response.status_code == 401
    response_body = response.json()
    assert response_body["detail"] == "Invalid storage account."


def test_train_job_post_returns_422_for_bad_blob_url(
    train_job_payload_simmesh_storage: dict,
    valid_api_key_header: dict,
) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(  # type: ignore
            throw_bad_blob_uri_error=True,
            wrong_blob_uri_error_message="Invalid blob URI.",
        ),
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post(
        "/train_job", json=train_job_payload_simmesh_storage, headers=valid_api_key_header
    )
    assert response.status_code == 422
    response_body = response.json()
    assert response_body["detail"] == "Invalid blob URI."


def test_train_job_get(valid_api_key_header: dict) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(available_job_run_ids=[123]),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.get("/train_job/123", headers=valid_api_key_header)
    assert response.status_code == 200
    response_body = response.json()
    assert response_body == {
        "job_run": {
            "id": 123,
            "url": "dummy_url",
            "result_state": None,
            "life_cycle_state": "RUNNING",
        },
        "mlflow_run": None,
    }


def test_train_job_get_returns_404_for_non_existing_job(valid_api_key_header: dict) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(available_job_run_ids=[123]),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.get("/train_job/666", headers=valid_api_key_header)
    assert response.status_code == 404
    response_body = response.json()
    assert response_body["detail"] == "Job run with ID 666 does not exist."


def test_train_job_get_returns_401_for_invalid_api_key(invalid_api_key: str) -> None:
    app = create_app(
        training_service=TrainingServiceTestDouble(),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.get("/train_job/123", headers={"x-api-key": invalid_api_key})
    assert response.status_code == 401
    response_body = response.json()
    assert response_body["detail"] == "Invalid or missing API Key"


def test_train_job_get_returns_422_for_api_config_that_cannot_be_parsed(
    train_job_payload: dict, valid_api_key_header: dict
) -> None:
    """Test that the API handles an error where the API Config cannot be parsed
    into a training config.
    """
    train_job_payload_duplicate_load_case_name = train_job_payload.copy()
    train_job_payload_duplicate_load_case_name["load_case_training_runs"].append(
        {
            "name": "dlc11",
            "data": {"training_data_file_uri": "data.parquet"},
        }
    )
    app = create_app(
        training_service=TrainingServiceTestDouble(),  # type: ignore
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post("/train_job", json=train_job_payload, headers=valid_api_key_header)
    assert response.status_code == 422
    response_body = response.json()
    assert (
        "Training run config is not valid: 1 validation error for TrainingRunConfig"
        in response_body["detail"]
    )


def test_train_job_returns_401_for_simmesh_authentication_error(
    train_job_payload: dict, valid_api_key_header: dict
) -> None:
    """Test that the API SimMesh authentication error is handled correctly."""
    app = create_app(
        training_service=TrainingServiceTestDouble(  # type: ignore
            throw_simmesh_authentication_error=True,
            simmesh_authentication_error_message="SimMesh authentication error occurred.",
        ),
        build_number="test_build_number",
        git_sha="test_git_sha",
    )
    client = TestClient(app)
    response = client.post("/train_job", json=train_job_payload, headers=valid_api_key_header)
    assert response.status_code == 401
    response_body = response.json()
    assert (
        response_body["detail"] == "SimMesh authentication error occurred."
    ), "Wrong error message returned. "
