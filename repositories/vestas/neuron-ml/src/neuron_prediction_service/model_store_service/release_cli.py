"""CLI for releasing model store artifacts for the prediction API by:
 - downloading turbine training run artifacts from MLFlow
 - packaging them in the correct format
 - uploading them to Azure Blob Storage.

The CLI is intended to be used in a CI/CD pipeline to release a model store artifact,
but has arguments to allow for downloading and packaging the artifacts locally without
uploading them to Azure Blob Storage.

A model store artifact consists of:
    - A folder for each turbine variant model artifact:
    - A `turbine_model_version_artifacts.json` file containing a list of
        `TurbineModelVersionArtifactMetadata` objects. This objects contains the
        metadata for each of the turbine variant model artifact, including the paths to the
        load case model artifacts.

A turbine variant model artifact consists of:
    - A `build_config.json` file containing the `TurbineVariantBuildConfig`
    - A `turbine_variant.json` file containing the `TurbineVariant`
    - A folder called `load_case_models` containing a zip file for each load case model artifact.
"""

import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import List, Union

import mlflow
import typer
import yaml
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from typing_extensions import Annotated

from neuron.experiment_tracking.mlflow_auth import with_databricks_mlflow_auth
from neuron.schemas.domain import LoadCase, TurbineVariant
from neuron.utils import zip_folder
from neuron_prediction_service.model_store_service.schemas import (
    LoadCaseModelArtifactMetadata,
    TurbineModelVersionArtifactMetadata,
    TurbineVariantBuildConfig,
)

MLFLOW_RUN_LOAD_CASE_MODEL_ARTIFACT_PATH = "load_case_models.zip"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
TURBINE_VARIANT_ARTIFACT_PATH = "turbine_variant.json"
TURBINE_VARIANT_BUILD_CONFIG_FILE_NAME = "build_config.json"
LOAD_CASE_FILE_NAME = "load_case.json"

app = typer.Typer()


@with_databricks_mlflow_auth
def download_mlflow_artifact(
    mlflow_run_id: Annotated[str, typer.Argument(help="MLFlow run id for the training run.")],
    mlflow_artifact_path: Annotated[
        str, typer.Argument(help="File or folder path of the MLFlow artifact.")
    ],
    local_dir: Annotated[
        Path, typer.Argument(help="Local directory were the artifact(s) will be downloaded.")
    ],
) -> None:
    """Download an artifact from an MLFlow run to a local directory."""
    mlflow_client = mlflow.tracking.MlflowClient()
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    mlflow_client.download_artifacts(mlflow_run_id, mlflow_artifact_path, str(local_dir))


def download_and_extract_turbine_training_run_model_artifacts(
    turbine_variant_build_config: TurbineVariantBuildConfig, local_dir: Path
) -> Path:
    """Download turbine training run model artifacts from MLFlow experiment tracking server based on
    the config defined in a `TurbineVariantBuildConfig`.

    Additional to downloading and extracting the turbine training run model artifacts
    the following happens:

    - The turbine variant config used to identify the run is downloaded saved to
    the local directory as json.
    - it is checked that the turbine_variant in the turbine variant model build config
    matches the turbine_variant in the MLFlow run.
    - The turbine variant model build config is saved to the local directory as json.
    """

    # Download the turbine variant model artifacts zip folder
    download_mlflow_artifact(
        mlflow_run_id=turbine_variant_build_config.mlflow_run_id,
        mlflow_artifact_path=MLFLOW_RUN_LOAD_CASE_MODEL_ARTIFACT_PATH,
        local_dir=local_dir,
    )

    # Download the turbine variant config json
    download_mlflow_artifact(
        mlflow_run_id=turbine_variant_build_config.mlflow_run_id,
        mlflow_artifact_path=TURBINE_VARIANT_ARTIFACT_PATH,
        local_dir=local_dir,
    )

    # extract using zipfile and keep only the folders and files in the extracted folder
    zipfile_path = Path(local_dir) / MLFLOW_RUN_LOAD_CASE_MODEL_ARTIFACT_PATH
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        zip_ref.extractall(local_dir)
    os.remove(zipfile_path)

    validate_downloaded_turbine_config_matches_turbine_id(
        turbine_variant_id=turbine_variant_build_config.turbine_variant.id,
        turbine_config_path=Path(local_dir) / TURBINE_VARIANT_ARTIFACT_PATH,
    )
    for load_case_model_folder in (Path(local_dir) / "load_case_models").iterdir():
        validate_downloaded_turbine_config_matches_turbine_id(
            turbine_variant_id=turbine_variant_build_config.turbine_variant.id,
            turbine_config_path=Path(load_case_model_folder) / "turbine_variant.json",
        )

    turbine_variant_model_config_path = Path(local_dir) / "build_config.json"
    with open(turbine_variant_model_config_path, "w") as f:
        json.dump(turbine_variant_build_config.model_dump(), f)

    return Path(local_dir)


def create_turbine_model_version_artifact(
    turbine_model_build_config: TurbineVariantBuildConfig,
    local_dir: Path,
    turbine_model_version_artifact_metadata_path: Path = None,
) -> TurbineModelVersionArtifactMetadata:
    """Create a local turbine model version artifact folder from a turbine model build config.

    Returns the metadata for the turbine model version artifact.
    """

    download_and_extract_turbine_training_run_model_artifacts(
        turbine_variant_build_config=turbine_model_build_config,
        local_dir=local_dir,
    )
    turbine_version_model_artifact_metadata = get_turbine_model_version_artifact_metadata(local_dir)
    zip_load_case_model_artifacts(local_dir)
    if turbine_model_version_artifact_metadata_path is not None:
        with open(turbine_model_version_artifact_metadata_path, "w") as f:
            json.dump(turbine_version_model_artifact_metadata.to_dict(), f)

    return turbine_version_model_artifact_metadata


@app.command()
def model_release(
    local_dir: Annotated[
        Path, typer.Option(help="Local directory were the artifact(s) will be downloaded.")
    ],
    turbine_model_config_path: Annotated[
        Path,
        typer.Option(help="Path to yaml config with trained turbine_variant model configuration."),
    ] = None,
    turbine_model_config_folder: Annotated[
        Path,
        typer.Option(
            help="Path to folder containing yaml configs with "
            "trained turbine_variant model configuration."
        ),
    ] = None,
    storage_account_name: Annotated[str, typer.Option(help="Azure storage account name")] = None,
    container_name: Annotated[str, typer.Option(help="Azure container name")] = None,
    blob_base_path: Annotated[str, typer.Option(help="Blob base path")] = None,
    delete_downloaded_turbine_model_version_folders: Annotated[
        bool,
        typer.Option(
            help="Delete the local turbine model folders after uploading the model artifacts."
        ),
    ] = False,
    upload: Annotated[
        bool,
        typer.Option(help="Upload artifacts to Azure Blob Storage container."),
    ] = False,
) -> None:
    """Do a release of a model store artifact for the prediction API.

    Exactly one of --turbine_variant-model-config-path
    or --turbine_variant-model-config-folder should be specified.

    All the turbine variant model artifacts are downloaded to the directory
    `local_dir` and extracted.

    If `upload` is True, the turbine variant model artifacts are uploaded to an Azure Blob Storage.
    Not uploading the artifacts is useful for testing the release process locally.

    if `delete_downloaded_turbine_model_version_folders` is True, the local folders containing the
    turbine variant model version artifacts are deleted after uploading the artifacts.
    """
    validate_turbine_model_build_config_path_params(
        turbine_model_config_folder=turbine_model_config_folder,
        turbine_model_config_path=turbine_model_config_path,
    )
    validate_upload_option_params(
        storage_account_name=storage_account_name,
        container_name=container_name,
        blob_base_path=blob_base_path,
        upload=upload,
    )
    if turbine_model_config_path:
        turbine_model_configs = [
            get_turbine_build_config_from_yaml(config_path=turbine_model_config_path)
        ]
    else:
        turbine_model_configs = get_turbine_build_configs_from_folder(turbine_model_config_folder)

    validate_turbine_variants_build_configs_ids_are_unique(turbine_model_configs)

    turbine_model_version_artifacts: List[TurbineModelVersionArtifactMetadata] = []
    for turbine_model_config in turbine_model_configs:
        turbine_artifact_folder_path = (
            local_dir / f"{turbine_model_config.turbine_variant.id}_{turbine_model_config.version}"
        )
        turbine_model_version_artifact = create_turbine_model_version_artifact(
            turbine_model_build_config=turbine_model_config,
            local_dir=turbine_artifact_folder_path,
            turbine_model_version_artifact_metadata_path=None,
        )

        turbine_model_version_artifacts.append(turbine_model_version_artifact)
        if upload:
            upload_folder_to_azure_storage_container(
                local_model_folder=str(turbine_artifact_folder_path),
                storage_account_name=storage_account_name,
                container_name=container_name,
                blob_base_path=blob_base_path + f"/{turbine_artifact_folder_path.name}",
            )
        if delete_downloaded_turbine_model_version_folders:
            shutil.rmtree(turbine_artifact_folder_path)

    turbine_model_version_artifacts_path = Path(local_dir) / "turbine_model_version_artifacts.json"
    with open(turbine_model_version_artifacts_path, "w") as f:
        json.dump(
            [
                turbine_model_version_artifact.to_dict()
                for turbine_model_version_artifact in turbine_model_version_artifacts
            ],
            f,
        )
    if upload:
        container_client = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=DefaultAzureCredential(),
        ).get_container_client(container=container_name)
        blob_client = container_client.get_blob_client(
            blob=f"{blob_base_path}/turbine_model_version_artifacts.json"
        )
        with open(turbine_model_version_artifacts_path, "rb") as f:
            blob_client.upload_blob(f)


def validate_turbine_model_build_config_path_params(
    turbine_model_config_folder: Union[Path, None], turbine_model_config_path: Union[Path, None]
) -> None:
    both_path_and_folder_passed = (
        turbine_model_config_folder is not None and turbine_model_config_path is not None
    )
    neither_of_path_or_folder_passed = (
        turbine_model_config_folder is None and turbine_model_config_path is None
    )
    if both_path_and_folder_passed or neither_of_path_or_folder_passed:
        print(
            "Error: Exactly one of --turbine_variant-model-config-path or "
            "--turbine_variant-model-config-folder should be specified."
        )
        typer.Exit(code=1)

    if turbine_model_config_path:
        file_exists = os.path.isfile(turbine_model_config_path)
        file_is_yaml = (turbine_model_config_path.suffix in [".yml", ".yaml"]) or (
            turbine_model_config_path.suffix in [".yml", ".yaml"]
        )
        if not file_exists or not file_is_yaml:
            print("Error: --turbine_variant-model-config-path should be a path to a yaml file.")
        typer.Exit(code=1)

    if turbine_model_config_folder and not turbine_model_config_folder.is_dir():
        print("Error: --turbine_variant-model-config-folder should be a path to a directory")
        typer.Exit(code=1)


def validate_upload_option_params(
    storage_account_name: Union[str, None],
    container_name: Union[str, None],
    blob_base_path: Union[str, None],
    upload: bool,
) -> None:
    """Validate that the upload parameters are correct."""
    all_blob_params_passed = all(
        [storage_account_name is not None, container_name is not None, blob_base_path is not None]
    )
    any_of_blob_params_passed = any(
        [storage_account_name is not None, container_name is not None, blob_base_path is not None]
    )
    if not all_blob_params_passed and upload:
        raise ValueError(
            "Error: All of storage_account_name, container_name and blob_base_path "
            "should be specified if upload is True."
        )
    if any_of_blob_params_passed and not upload:
        raise ValueError(
            "Error: If upload is False, none of storage_account_name, "
            "container_name and blob_base_path should be specified."
        )


def validate_turbine_variants_build_configs_ids_are_unique(
    turbine_variant_build_configs: list[TurbineVariantBuildConfig],
) -> None:
    turbine_variant_versions = [
        (turbine_variant_build_config.turbine_variant.id, turbine_variant_build_config.version)
        for turbine_variant_build_config in turbine_variant_build_configs
    ]
    turbine_variant_version_counts = Counter(turbine_variant_versions)
    duplicates = [el for el, count in turbine_variant_version_counts.items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Turbine variant build config ids should be unique Found duplicates: {duplicates}"
        )


def validate_downloaded_turbine_config_matches_turbine_id(
    turbine_variant_id: str, turbine_config_path: Path
) -> None:
    """Check that the turbine_variant in the turbine variant model build config
    matches the turbine_variant in the MLFlow run.
    """
    with open(turbine_config_path, "r") as f:
        downloaded_turbine_variant = TurbineVariant(**json.load(f))
    if downloaded_turbine_variant.id != turbine_variant_id:
        raise ValueError(
            "Turbine variant id in the turbine variant model build config "
            f"{turbine_variant_id} does not match the "
            f"turbine variant id in the MLFlow run {downloaded_turbine_variant.id}."
        )


def upload_folder_to_azure_storage_container(
    local_model_folder: Annotated[str, typer.Option(help="Local model folder path")],
    storage_account_name: Annotated[str, typer.Option(help="Azure storage account name")],
    container_name: Annotated[str, typer.Option(help="Azure container name")],
    blob_base_path: Annotated[str, typer.Option(help="Blob base path")],
) -> None:
    """Upload a local folder to an Azure Blob Storage container."""

    local_model_folder = Path(local_model_folder)
    if not local_model_folder.exists() or not local_model_folder.is_dir():
        print(f"Error: Folder {local_model_folder} does not exist.")
        typer.Exit(code=1)
    container_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    ).get_container_client(container=container_name)

    # glob searches recursively for all files in the folder and subfolders
    for file in local_model_folder.glob("**/*"):
        relative_path = file.relative_to(local_model_folder)
        if not file.is_file():
            continue
        blob_client = container_client.get_blob_client(blob=f"{blob_base_path}/{relative_path}")
        with open(file, "rb") as data:
            blob_client.upload_blob(data)
        typer.echo(f"Uploaded {file} to {blob_client.url}")


def zip_load_case_model_artifacts(turbine_model_artifact_folder: Union[str, Path]) -> None:
    """Zip the load case model artifacts in a turbine model artifact folder.

    Each load case model folder is zipped and the folder is removed after zipping.
    """
    turbine_model_artifact_folder = Path(turbine_model_artifact_folder)
    load_case_models_root_dor = Path(turbine_model_artifact_folder) / "load_case_models"
    for p in load_case_models_root_dor.iterdir():
        if p.is_dir():
            zip_folder(folder_path=p, zip_artifact_path=load_case_models_root_dor / f"{p.name}.zip")
            shutil.rmtree(p)


def get_turbine_build_config_from_yaml(config_path: Path) -> TurbineVariantBuildConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TurbineVariantBuildConfig(**config_dict)


def get_turbine_build_configs_from_folder(
    turbine_model_config_folder: Path,
) -> List[TurbineVariantBuildConfig]:
    """Traverse and try to load all .yaml and .yml files in folder
    as `TurbineVariantBuildConfig`s.
    """
    turbine_model_configs = []
    for file in turbine_model_config_folder.rglob("*.yaml"):
        turbine_model_configs.append(get_turbine_build_config_from_yaml(file))
    for file in turbine_model_config_folder.rglob("*.yml"):
        turbine_model_configs.append(get_turbine_build_config_from_yaml(file))
    return turbine_model_configs


def get_turbine_variant_build_config_from_json(
    turbine_variant_build_config_folder: Path,
) -> TurbineVariantBuildConfig:
    with open(turbine_variant_build_config_folder / TURBINE_VARIANT_BUILD_CONFIG_FILE_NAME) as f:
        return TurbineVariantBuildConfig(**json.load(f))


def get_turbine_model_version_artifact_metadata(
    folder: Union[str, Path],
) -> TurbineModelVersionArtifactMetadata:
    """Get turbine model version artifact from a folder containing the model artifact.

    The function assumes that the folder contains a json file with
    a turbine variant build config and a folder called `load_case_models`
    with sub-folders for each load case model.
    """

    turbine_variant_build_config = get_turbine_variant_build_config_from_json(folder)
    load_case_model_metadata_map = {}
    for model_folder_dir in (folder / "load_case_models").iterdir():
        load_case_model_metadata_map[model_folder_dir.name] = get_load_case_model_artifact_metadata(
            turbine_model_version_path=folder.name, load_case_model_model_path=model_folder_dir
        )
    return TurbineModelVersionArtifactMetadata(
        turbine_variant_build_config=turbine_variant_build_config,
        load_case_model_artifacts=load_case_model_metadata_map,
    )


def get_load_case_model_artifact_metadata(
    turbine_model_version_path: Union[str, Path],
    load_case_model_model_path: Union[Path, str],
) -> LoadCaseModelArtifactMetadata:
    load_case_model_model_path = Path(load_case_model_model_path)
    with open(load_case_model_model_path / LOAD_CASE_FILE_NAME) as f:
        load_case = LoadCase.model_validate(json.load(f))
    return LoadCaseModelArtifactMetadata(
        load_case=load_case,
        load_case_model_artifact_path=f"{turbine_model_version_path}/load_case_models/{load_case.name}.zip",
    )


if __name__ == "__main__":
    app()
