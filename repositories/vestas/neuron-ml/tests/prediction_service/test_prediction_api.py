import re
from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUILD_NUMBER", "test_build_number")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("MODEL_STORE_FOLDER_PATH", "tests/data/prediction_service/model_store")


@pytest.fixture
def client(test_env: None) -> TestClient:  # noqa: ARG001
    from src.neuron_prediction_service.main import app

    return TestClient(app)


@pytest.fixture(scope="function")
def load_case_model_input_examples() -> list[dict[str, Any]]:
    from src.neuron_prediction_service.schemas import LOAD_CASE_MODEL_INPUT_EXAMPLES

    return deepcopy(LOAD_CASE_MODEL_INPUT_EXAMPLES)


@pytest.mark.api
def test_ping(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == (
        "Hello World from Neuron Prediction Service. Build number = 'test_build_number'."
    )


@pytest.mark.api
def test_get_turbine_variant_ids(client: TestClient) -> None:
    response = client.get("/turbine_variant_ids")
    assert response.status_code == 200
    response_body = response.json()
    assert set(response_body) == {"150_4000_mk3e", "162_5600_mk0a"}


@pytest.mark.api
def test_get_load_case_model_ids(client: TestClient) -> None:
    response = client.get("/load_case_model_ids")
    assert response.status_code == 200
    response_body = response.json()
    assert set(response_body) == {
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=dlc11",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=dlc11",
        "turbine_variant_id=162_5600_mk0a---version=1---load_case=custom_load_case",
        "turbine_variant_id=162_5600_mk0a---version=1---load_case=dlc11",
        "turbine_variant_id=162_5600_mk0a---version=2---load_case=custom_load_case",
        "turbine_variant_id=162_5600_mk0a---version=2---load_case=dlc11",
    }


@pytest.mark.api
def test_get_load_case_model_ids_for_turbine(client: TestClient) -> None:
    response = client.get("/load_case_model_ids/150_4000_mk3e")
    assert response.status_code == 200
    response_body = response.json()
    assert set(response_body) == {
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=dlc11",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=dlc11",
    }


@pytest.mark.api
def test_get_load_case_model_missing_turbine_error(client: TestClient) -> None:
    response = client.get("/load_case_model_ids/I_DO_NOT_EXIST")
    assert response.status_code == 404, "Response status code is not 404 for missing turbine."
    assert response.json() == {
        "details": "Turbine variant I_DO_NOT_EXIST does not exist in model store."
    }


@pytest.mark.api
def test_get_load_case_model_ids_for_turbine_and_version(client: TestClient) -> None:
    response = client.get("/load_case_model_ids/150_4000_mk3e/2")
    assert response.status_code == 200
    response_body = response.json()
    assert set(response_body) == {
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=dlc11",
    }


@pytest.mark.api
def test_get_load_case_model_missing_turbine_version_error(client: TestClient) -> None:
    response = client.get("/load_case_model_ids/150_4000_mk3e/3")
    assert (
        response.status_code == 404
    ), "Response status code is not 404 for missing turbine version."
    assert response.json() == {
        "details": "Version 3 for turbine variant 150_4000_mk3e does not exist in model store."
    }


@pytest.mark.api
def test_get_load_case_model_metadata_happy_path(client: TestClient) -> None:
    response = client.get(
        "/load_case_model_metadata/turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case"
    )
    assert response.status_code == 200
    response_body = response.json()
    assert response_body["turbine_variant_build_config"]["turbine_variant"] == {
        "rotor_diameter": 150,
        "rated_power": 4000,
        "mk_version": "mk3e",
    }, "turbine_variant not correct in response body"
    assert (
        response_body["load_case"]["name"] == "custom_load_case"
    ), "load_case not correct in response body"
    assert (
        response_body["turbine_variant_build_config"]["version"] == 1
    ), "model_version not correct in response body"


@pytest.mark.api
def test_get_load_case_model_metadata_handles_bad_model_id_format(client: TestClient) -> None:
    response = client.get("/load_case_model_metadata/I_AM_NOT_A_VALID_MODEL_ID")
    assert response.status_code == 422, "Response status code is not 404 for bad model id."
    expected_error_message = (
        "Could not parse load case model id from I_AM_NOT_A_VALID_MODEL_ID. "
        "Expected 'turbine_variant_id', 'version' and 'load_case' in query string,"
        "in the format 'turbine_variant_id=<turbine variant id>"
        "---version=<version>---load_case=<load case name>'."
    )
    assert response.json() == {"details": expected_error_message}


@pytest.mark.api
def test_get_load_case_model_metadata_handles_model_does_not_exist_error(
    client: TestClient,
) -> None:
    response = client.get(
        "/load_case_model_metadata/turbine_variant_id=150_4000_mk3e---version=1---load_case=I_DO_NOT_EXIST"
    )
    assert response.status_code == 404, "Response status code is not 404 for bad model id."
    assert response.json() == {
        "details": (
            "Load case model with id turbine_variant_id=150_4000_mk3e"
            "---version=1---load_case=I_DO_NOT_EXIST "
            "does not exist in model store."
        )
    }


@pytest.mark.api
def test_prediction_happy_path(
    client: TestClient, load_case_model_input_examples: list[dict[str, Any]]
) -> None:
    request_body = load_case_model_input_examples[0]
    response = client.post(
        url="predict/turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        json=request_body,
    )
    assert response.status_code == 200
    response_body = response.json()

    assert "predictions" in response_body, "predictions not in response body"
    assert (
        "interpolation_domain_validation" in response_body
    ), "interpolation_domain_validation not in response body"
    assert (
        "extrapolation_domain_validation" in response_body
    ), "extrapolation_domain_validation not in response body"
    # Ensure predictions are made for all targets specified in the request
    for target in request_body["targets"]:
        assert "averages" in response_body["predictions"][target], "averages not in response body"
        assert (
            "standard_deviations" in response_body["predictions"][target]
        ), "standard_deviations not in response body"
    assert "meta" in response_body, "meta not in response body"
    assert (
        response_body["meta"]["build_number"] == "test_build_number"
    ), "build_number number is not correct."
    assert (
        response_body["meta"]["load_case_name"] == "custom_load_case"
    ), "load_case_name is not correct."
    assert response_body["meta"]["turbine_variant_build_config"]["turbine_variant"] == {
        "rotor_diameter": 150,
        "rated_power": 4000,
        "mk_version": "mk3e",
    }, "turbine_variant not correct in response body"


@pytest.mark.api
def test_prediction_handles_bad_model_id_format(
    client: TestClient, load_case_model_input_examples: list[dict[str, Any]]
) -> None:
    request_body = load_case_model_input_examples[0]
    response = client.post(
        url="predict/I_AM_NOT_A_VALID_MODEL_ID",
        json=request_body,
    )
    assert response.status_code == 422, "Response status code is not 422 for bad model id."
    expected_error_message = (
        "Could not parse load case model id from I_AM_NOT_A_VALID_MODEL_ID. "
        "Expected 'turbine_variant_id', 'version' and 'load_case' in query string,"
        "in the format 'turbine_variant_id=<turbine variant id>"
        "---version=<version>---load_case=<load case name>'."
    )
    assert response.json() == {"details": expected_error_message}


@pytest.mark.api
def test_prediction_handles_model_id_does_not_exist_error(
    client: TestClient, load_case_model_input_examples: list[dict[str, Any]]
) -> None:
    request_body = load_case_model_input_examples[0]
    response = client.post(
        url="predict/turbine_variant_id=150_4000_mk3e---version=1---load_case=I_DO_NOT_EXIST",
        json=request_body,
    )
    assert response.status_code == 404, "Response status code is not 404 for missing model."
    assert response.json() == {
        "details": (
            "Load case model with id turbine_variant_id=150_4000_mk3e---version=1"
            "---load_case=I_DO_NOT_EXIST "
            "does not exist in model store."
        )
    }


@pytest.mark.api
def test_prediction_handles_data_validation_error(
    client: TestClient, load_case_model_input_examples: list[dict[str, Any]]
) -> None:
    request_body = load_case_model_input_examples[0]
    del request_body["data"]["ws"]
    response = client.post(
        url="predict/turbine_variant_id=150_4000_mk3e---version=1---load_case=dlc11",
        json=request_body,
    )
    assert response.status_code == 400, "Response status code is not 400 for missing feature."


@pytest.mark.api
def test_prediction_handles_missing_targets(
    client: TestClient, load_case_model_input_examples: list[dict[str, Any]]
) -> None:
    """It should be ok to not pass the targets explicitly in the request body."""
    request_body = load_case_model_input_examples[0]
    del request_body["targets"]
    response = client.post(
        url="predict/turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        json=request_body,
    )
    assert response.status_code == 200
    response_body = response.json()
    # Ensure predictions are made for all targets available in the model when targets list is empty
    expected_targets = {"MxHub_m800", "MxBldRoot_m1000", "MyHub_m400"}
    assert (
        set(response_body["predictions"].keys()) == expected_targets
    ), "Predictions not made for all targets when targets are not specified in the request body."


@pytest.mark.api
def test_prediction_handles_empty_targets_list(
    client: TestClient, load_case_model_input_examples: list[dict[str, Any]]
) -> None:
    # Create a request body with an empty targets list
    request_body = load_case_model_input_examples[0]
    request_body["targets"] = []
    response = client.post(
        url="predict/turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        json=request_body,
    )
    assert response.status_code == 200, "Response status code is not 200 for an empty targets list."

    response_body = response.json()
    # Ensure predictions are made for all targets available in the model when targets list is empty
    expected_targets = {"MxHub_m800", "MxBldRoot_m1000", "MyHub_m400"}
    assert (
        set(response_body["predictions"].keys()) == expected_targets
    ), "Predictions not made for all targets when targets list is empty."


@pytest.mark.api
def test_prediction_handles_missing_grad_features(
    client: TestClient, load_case_model_input_examples: list[dict[str, Any]]
) -> None:
    request_body = load_case_model_input_examples[0]
    del request_body["grad_features"]
    response = client.post(
        url="predict/turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        json=request_body,
    )
    assert response.status_code == 200

    response_body = response.json()
    # check that gradients are not calculated when grad_features is empty
    for target in request_body["targets"]:
        assert (
            response_body["predictions"][target]["gradients"] is None
        ), "Gradients are not None (null) when grad_features is missing."


@pytest.mark.api
def test_prediction_handles_empty_grad_features_list(
    client: TestClient, load_case_model_input_examples: list[dict[str, Any]]
) -> None:
    request_body = load_case_model_input_examples[0]
    request_body["grad_features"] = []
    response = client.post(
        url="predict/turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        json=request_body,
    )
    assert (
        response.status_code == 200
    ), "Response status code is not 200 for an empty grad_features list."

    response_body = response.json()
    # check that gradients are not calculated when grad_features is empty
    for target in request_body["targets"]:
        assert (
            response_body["predictions"][target]["gradients"] is None
        ), "Gradients are not None (null) when grad_features is empty."


@pytest.mark.api
def test_prediction_handles_vexp_only(
    client: TestClient, load_case_model_input_examples: list[dict[str, Any]]
) -> None:
    request_body = load_case_model_input_examples[0]
    del request_body["data"]["wnd_grad"]
    request_body["data"]["vexp"] = [0.1, 0.2, 0.3]
    response = client.post(
        url="predict/turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        json=request_body,
    )
    assert (
        response.status_code == 200
    ), "Response status code is not 200 for a case where vexp is provided as input."


@pytest.mark.api
def test_prediction_api_captures_request_id(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that when a "x-vestas-request-id" header is provided in the request, that values is
    used as the request_id in the log messages."""
    client.get("/", headers={"x-vestas-request-id": "test-request-id"})

    # Capture all the log messages that were printed to the terminal
    log_messages = capsys.readouterr().out.splitlines()
    # Clean out all the terminal color codes ("\x1b[0m", "\x1b[31m", etc.)
    log_messages = [re.sub(r"\x1b\[\d+m", "", message) for message in log_messages]

    # All log messages should have a bound context where the request_id is "test-request-id"
    assert all("request_id=test-request-id" in message for message in log_messages)
