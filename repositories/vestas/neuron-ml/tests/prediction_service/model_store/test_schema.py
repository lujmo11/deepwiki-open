import json

import pytest

from neuron.schemas.domain import LoadCase
from neuron_prediction_service.model_store_service.exceptions import LoadCaseModelIDNotCorrect
from neuron_prediction_service.model_store_service.schemas import (
    LoadCaseModelId,
    TurbineModelVersionArtifactMetadata,
)


@pytest.fixture(scope="module")
def turbine_model_version_artifact_metadata() -> TurbineModelVersionArtifactMetadata:
    with open(
        "tests/data/prediction_service/example_turbine_model_version_artifact.json", "r"
    ) as f:
        return TurbineModelVersionArtifactMetadata.from_dict(json.load(f))


def test_load_case_model_id_from_str_happy_path() -> None:
    load_case_model_id = LoadCaseModelId.from_str(
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case"
    )
    assert (
        load_case_model_id.turbine_variant_id == "150_4000_mk3e"
    ), "Expected turbine variant id to be '150_4000_mk3e'"
    assert load_case_model_id.version == 1, "Expected version to be 1"
    assert (
        load_case_model_id.load_case_name == "custom_load_case"
    ), "Expected load case name to be 'custom_load_case'"


def test_load_case_model_id_from_str_bad_format() -> None:
    with pytest.raises(LoadCaseModelIDNotCorrect):
        LoadCaseModelId.from_str("invalid_model_id_format")


def test_load_case_model_id_from_str_validation_error() -> None:
    with pytest.raises(LoadCaseModelIDNotCorrect):
        LoadCaseModelId.from_str(
            "turbine_variant_id=150_4000_mk3e---version='not-an-int'"
            "---load_case=custom_load_case"
        )


def test_turbine_model_version_artifact_metadata_load_case_model_ids(
    turbine_model_version_artifact_metadata: TurbineModelVersionArtifactMetadata,
) -> None:
    expected_load_case_model_ids = {
        LoadCaseModelId(
            turbine_variant_id="150_4000_mk3e", version=1, load_case_name="custom_load_case"
        ),
        LoadCaseModelId(turbine_variant_id="150_4000_mk3e", version=1, load_case_name="dlc11"),
    }
    actual_load_case_model_ids = turbine_model_version_artifact_metadata.load_case_model_ids
    assert isinstance(actual_load_case_model_ids, list), "Expected load case model ids to be a list"
    assert set(actual_load_case_model_ids) == expected_load_case_model_ids, (
        f"Expected load case model ids: {expected_load_case_model_ids}, "
        f"but got: {actual_load_case_model_ids}"
    )


def test_turbine_model_version_artifact_metadata_get_load_case(
    turbine_model_version_artifact_metadata: TurbineModelVersionArtifactMetadata,
) -> None:
    load_case = turbine_model_version_artifact_metadata.get_load_case(
        load_case_model_id=LoadCaseModelId(
            turbine_variant_id="150_4000_mk3e", version=1, load_case_name="custom_load_case"
        )
    )
    assert isinstance(load_case, LoadCase), "Expected load case to be of type LoadCase"
    assert load_case.name == "custom_load_case", "Expected load case name to be 'custom_load_case'"


def test_turbine_model_version_artifact_metadata_get_load_case_model_artifact_path(
    turbine_model_version_artifact_metadata: TurbineModelVersionArtifactMetadata,
) -> None:
    expected_load_case_model_artifact_path = "150_4000_mk3e_1/load_case_models/custom_load_case.zip"
    load_case_artifact_path = (
        turbine_model_version_artifact_metadata.get_load_case_model_artifact_path(
            LoadCaseModelId(
                turbine_variant_id="150_4000_mk3e", version=1, load_case_name="custom_load_case"
            )
        )
    )
    assert (
        load_case_artifact_path == expected_load_case_model_artifact_path
    ), f"Expected load case artifact path to be '{expected_load_case_model_artifact_path}'"
