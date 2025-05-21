import tempfile
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union

from neuron.io.storage_reader import StorageReader
from neuron.models.load_case_model_pipeline import LoadCaseModelPipeline
from neuron.schemas.domain import LoadCase
from neuron_prediction_service.model_store_service.exceptions import (
    LoadCaseModelNotInModelStore,
    TurbineVariantNotInModelStore,
    TurbineVariantVersionNotInModelStore,
)
from neuron_prediction_service.model_store_service.schemas import (
    LoadCaseModelId,
    TurbineModelVersionArtifactMetadata,
    TurbineVariantBuildConfig,
)


class ModelStore:
    """Model store service.

    Manages retrieval of load case models for different turbine variants.

    Args:
        storage_reader: storage reader instance that implements the StorageReader interface.
        turbine_model_artifacts: dictionary containing turbine model versions metadata. Has the
            following structure: {turbine_variant_id: {version: TurbineModelVersionMetadata}}.
    """

    def __init__(
        self,
        storage_reader: StorageReader,
        turbine_model_artifacts: List[TurbineModelVersionArtifactMetadata],
    ) -> None:
        self.storage_reader = storage_reader
        self.turbine_model_version_artifacts_map = self._get_turbine_model_version_artifacts_map(
            turbine_model_artifacts
        )

    @staticmethod
    def _get_turbine_model_version_artifacts_map(
        turbine_model_artifacts: List[TurbineModelVersionArtifactMetadata],
    ) -> Dict[str, Dict[int, TurbineModelVersionArtifactMetadata]]:
        """Get turbine model artifacts map.

        Args:
            turbine_model_artifacts: list of turbine model version artifacts.

        Returns:
            Dictionary containing turbine model versions metadata. Has the following structure:
            {turbine_variant_id: {version: TurbineModelVersionMetadata}}.
        """
        turbine_model_artifacts_map = defaultdict(dict)
        for turbine_model_artifact in turbine_model_artifacts:
            turbine_variant_id = (
                turbine_model_artifact.turbine_variant_build_config.turbine_variant.id
            )
            version = turbine_model_artifact.turbine_variant_build_config.version
            if version in turbine_model_artifacts_map[turbine_variant_id]:
                raise ValueError(
                    f"Duplicate version {version} found for turbine variant {turbine_variant_id}"
                )
            turbine_model_artifacts_map[turbine_variant_id][version] = turbine_model_artifact
        return turbine_model_artifacts_map

    @property
    def turbine_variant_ids(self) -> List[str]:
        """Get list of turbine variant ids."""
        return list(self.turbine_model_version_artifacts_map.keys())

    def get_load_case_model_ids(
        self, turbine_variant_id: Union[str, None] = None, version: Union[int, None] = None
    ) -> List[str]:
        """Get list of load case model ids.

        If turbine_variant_id is provided, only load case model ids for that turbine variant are
        returned. If version is provided, only load case model ids for that version are returned.
        """
        turbine_variant_id_passed = turbine_variant_id is not None
        version_passed = version is not None

        if version_passed and not turbine_variant_id_passed:
            raise ValueError("Version cannot be provided without turbine variant id.")

        if (
            turbine_variant_id_passed
            and turbine_variant_id not in self.turbine_model_version_artifacts_map
        ):
            raise TurbineVariantNotInModelStore(turbine_variant_id=turbine_variant_id) from None

        if (
            version_passed
            and version not in self.turbine_model_version_artifacts_map[turbine_variant_id]
        ):
            raise TurbineVariantVersionNotInModelStore(
                model_version=version, turbine_variant_id=turbine_variant_id
            ) from None

        if turbine_variant_id_passed and version_passed:
            return [
                str(load_case_model_id)
                for load_case_model_id in self.turbine_model_version_artifacts_map[
                    turbine_variant_id
                ][version].load_case_model_ids
            ]

        elif turbine_variant_id_passed and not version_passed:
            return [
                str(load_case_model_id)
                for version in self.turbine_model_version_artifacts_map[turbine_variant_id]
                for load_case_model_id in self.turbine_model_version_artifacts_map[
                    turbine_variant_id
                ][version].load_case_model_ids
            ]

        else:
            return [
                str(load_case_model_id)
                for turbine_variant_id in self.turbine_model_version_artifacts_map
                for version in self.turbine_model_version_artifacts_map[turbine_variant_id]
                for load_case_model_id in self.turbine_model_version_artifacts_map[
                    turbine_variant_id
                ][version].load_case_model_ids
            ]

    def _validate_load_case_model_id(self, load_case_model_id: LoadCaseModelId) -> None:
        if str(load_case_model_id) not in self.get_load_case_model_ids():
            raise LoadCaseModelNotInModelStore(load_case_model_id=load_case_model_id)

    def get_turbine_model_version_artifact(
        self, load_case_model_id: str
    ) -> TurbineModelVersionArtifactMetadata:
        """Get model metadata for a specific model"""
        load_case_model_id = LoadCaseModelId.from_str(load_case_model_id)
        self._validate_load_case_model_id(load_case_model_id)
        return self.turbine_model_version_artifacts_map[load_case_model_id.turbine_variant_id][
            load_case_model_id.version
        ]

    def get_load_case(self, load_case_model_id: str) -> LoadCase:
        """Get model metadata for a specific model"""
        turbine_model_version_artifact = self.get_turbine_model_version_artifact(load_case_model_id)
        load_case_model_id = LoadCaseModelId.from_str(load_case_model_id)
        return turbine_model_version_artifact.get_load_case(load_case_model_id=load_case_model_id)

    def get_turbine_variant_build_config(
        self, load_case_model_id: str
    ) -> TurbineVariantBuildConfig:
        """Get model metadata for a specific model"""
        return self.get_turbine_model_version_artifact(
            load_case_model_id
        ).turbine_variant_build_config

    def get_load_case_model(self, load_case_model_id: str) -> LoadCaseModelPipeline:
        """Get load case model for a specific model"""
        load_case_model_id = LoadCaseModelId.from_str(load_case_model_id)
        self._validate_load_case_model_id(load_case_model_id)

        turbine_model_version = self.turbine_model_version_artifacts_map[
            load_case_model_id.turbine_variant_id
        ][load_case_model_id.version]
        load_case_model_artifact = turbine_model_version.load_case_model_artifacts[
            load_case_model_id.load_case_name
        ]
        with tempfile.TemporaryDirectory() as temp_dir_turbine_model_folder:
            zip_bytes = self.storage_reader.read(
                load_case_model_artifact.load_case_model_artifact_path
            )
            with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zip_ref:
                zip_ref.extractall(temp_dir_turbine_model_folder)
            return LoadCaseModelPipeline.load_model(
                folder_path=str(
                    Path(temp_dir_turbine_model_folder) / load_case_model_artifact.load_case.name
                )
            )
