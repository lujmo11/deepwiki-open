import json

import pytest
from dobby.io.storages import FileSystemStorage

from neuron.schemas.domain import LoadCase
from neuron_prediction_service.model_store_service.exceptions import (
    LoadCaseModelIDNotCorrect,
    LoadCaseModelNotInModelStore,
)
from neuron_prediction_service.model_store_service.model_store import ModelStore
from neuron_prediction_service.model_store_service.schemas import (
    TurbineModelVersionArtifactMetadata,
    TurbineVariantBuildConfig,
)


@pytest.fixture
def model_store() -> ModelStore:
    folder = "tests/data/prediction_service/model_store"
    storage = FileSystemStorage(mount_path=folder)
    with open(f"{folder}/turbine_model_version_artifacts.json") as f:
        turbine_model_artifacts = [
            TurbineModelVersionArtifactMetadata.from_dict(d) for d in json.load(f)
        ]
    model_store = ModelStore(
        storage_reader=storage, turbine_model_artifacts=turbine_model_artifacts
    )
    return model_store


def test_model_store_get_load_case_model_ids_happy_paths(model_store: ModelStore) -> None:
    expected_full_set_of_load_case_model_ids = {
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=dlc11",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=dlc11",
        "turbine_variant_id=162_5600_mk0a---version=1---load_case=custom_load_case",
        "turbine_variant_id=162_5600_mk0a---version=1---load_case=dlc11",
        "turbine_variant_id=162_5600_mk0a---version=2---load_case=custom_load_case",
        "turbine_variant_id=162_5600_mk0a---version=2---load_case=dlc11",
    }
    all_load_case_model_ids = model_store.get_load_case_model_ids()
    assert isinstance(all_load_case_model_ids, list)
    assert set(all_load_case_model_ids) == expected_full_set_of_load_case_model_ids, (
        f"Expected load case model ids: {expected_full_set_of_load_case_model_ids}, "
        f"but got: {all_load_case_model_ids}"
    )

    expected_150_4000_mk3e_load_case_model_ids = {
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=dlc11",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=dlc11",
    }
    load_case_model_ids_150_4000_mk3e = model_store.get_load_case_model_ids(
        turbine_variant_id="150_4000_mk3e"
    )
    assert isinstance(load_case_model_ids_150_4000_mk3e, list)
    assert set(load_case_model_ids_150_4000_mk3e) == expected_150_4000_mk3e_load_case_model_ids, (
        f"Expected load case model ids for 150_4000_mk3e: "
        f"{expected_150_4000_mk3e_load_case_model_ids}, "
        f"but got: {load_case_model_ids_150_4000_mk3e}"
    )

    expected_150_4000_mk3e_version_2_load_case_model_ids = {
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=custom_load_case",
        "turbine_variant_id=150_4000_mk3e---version=2---load_case=dlc11",
    }
    load_case_model_ids_150_4000_mk3e_version_2 = model_store.get_load_case_model_ids(
        turbine_variant_id="150_4000_mk3e", version=2
    )
    assert isinstance(load_case_model_ids_150_4000_mk3e_version_2, list)
    assert (
        set(load_case_model_ids_150_4000_mk3e_version_2)
        == expected_150_4000_mk3e_version_2_load_case_model_ids
    ), (
        f"Expected load case model ids for 150_4000_mk3e version 2: "
        f"{expected_150_4000_mk3e_version_2_load_case_model_ids}, "
        f"but got: {load_case_model_ids_150_4000_mk3e_version_2}"
    )


def test_get_turbine_model_version_artifact_happy_path(model_store: ModelStore) -> None:
    load_case_model_id = "turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case"
    turbine_model_version_artifact = model_store.get_turbine_model_version_artifact(
        load_case_model_id
    )
    assert isinstance(turbine_model_version_artifact, TurbineModelVersionArtifactMetadata)
    assert (
        turbine_model_version_artifact.turbine_variant_build_config.turbine_variant.id
        == "150_4000_mk3e"
    )
    assert turbine_model_version_artifact.turbine_variant_build_config.version == 1


def test_get_turbine_model_version_artifact_load_case_model_not_in_model_store(
    model_store: ModelStore,
) -> None:
    load_case_model_id = (
        "turbine_variant_id=150_4000_mk3e---version=1---load_case=not_in_model_store"
    )
    with pytest.raises(LoadCaseModelNotInModelStore):
        model_store.get_turbine_model_version_artifact(load_case_model_id)


def test_get_turbine_model_version_artifact_bad_load_case_model_id_format(
    model_store: ModelStore,
) -> None:
    load_case_model_id = "invalid_model_id_format"
    with pytest.raises(LoadCaseModelIDNotCorrect):
        model_store.get_turbine_model_version_artifact(load_case_model_id)


def test_model_store_get_load_case(model_store: ModelStore) -> None:
    load_case_model_id = "turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case"
    load_case = model_store.get_load_case(load_case_model_id)
    assert load_case is not None
    assert load_case.name == "custom_load_case"
    assert isinstance(load_case, LoadCase)


# def test_model_store_get_load_case_bad_model_id_format(
#     model_store: ModelStore,
# ) -> None:
#     load_case_model_id = "invalid_model_id_format"
#     with pytest.raises(LoadCaseModelIDNotCorrect):
#         model_store.get_load_case(load_case_model_id)
#
#
# def test_model_store_get_load_case_load_case_model_not_in_model_store(
#     model_store: ModelStore,
# ) -> None:
#     load_case_model_id = (
#         "turbine_variant_id=150_4000_mk3e---version=1---load_case=not_in_model_store"
#     )
#     with pytest.raises(LoadCaseModelNotInModelStore):
#         model_store.get_load_case(load_case_model_id)


def test_get_turbine_variant_build_config(model_store: ModelStore) -> None:
    load_case_model_id = "turbine_variant_id=150_4000_mk3e---version=1---load_case=custom_load_case"
    turbine_variant_build_config = model_store.get_turbine_variant_build_config(load_case_model_id)
    assert isinstance(turbine_variant_build_config, TurbineVariantBuildConfig)
    assert turbine_variant_build_config.turbine_variant.id == "150_4000_mk3e"
    assert turbine_variant_build_config.version == 1
