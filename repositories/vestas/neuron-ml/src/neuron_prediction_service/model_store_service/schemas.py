from dataclasses import dataclass
from typing import Dict, List, Self
from urllib.parse import parse_qsl

from pydantic import BaseModel

from neuron.schemas.domain import LoadCase, TurbineVariant
from neuron_prediction_service.model_store_service.exceptions import LoadCaseModelIDNotCorrect


class TurbineVariantBuildConfig(BaseModel, frozen=True):
    """Config for a turbine variant training run that will be used in the prediction service.

    A training run will contain multiple load case models.
    """

    turbine_variant: TurbineVariant
    version: int
    mlflow_run_id: str


@dataclass
class LoadCaseModelId:
    """Identifier for a load case model."""

    turbine_variant_id: str
    version: int
    load_case_name: str

    def __str__(self):
        return (
            f"turbine_variant_id={self.turbine_variant_id}"
            f"---version={self.version}"
            f"---load_case={self.load_case_name}"
        )

    @classmethod
    def from_str(cls, s: str):
        dict_repr = dict(parse_qsl(s, separator="---"))
        try:
            turbine_variant_id = dict_repr["turbine_variant_id"]
            version = dict_repr["version"]
            load_case_name = dict_repr["load_case"]
        except KeyError as e:
            raise LoadCaseModelIDNotCorrect(
                f"Could not parse load case model id from {s}. "
                "Expected 'turbine_variant_id', 'version' and 'load_case' in query string,"
                "in the format "
                "'turbine_variant_id=<turbine variant id>"
                "---version=<version>---load_case=<load case name>'."
            ) from e
        try:
            version = int(version)
        except ValueError:
            raise LoadCaseModelIDNotCorrect(
                f"Could not parse version from {version}. Expected an integer."
            ) from None
        return cls(
            turbine_variant_id=turbine_variant_id,
            version=version,
            load_case_name=load_case_name,
        )

    @property
    def turbine_variant_model_id(self) -> str:
        return f"turbine_variant_id={self.turbine_variant_id}---version={self.version}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


class LoadCaseModelArtifactMetadata(BaseModel, frozen=True):
    """Metadata about a load case model artifact.

    Attributes:
        load_case: The load case used in the Load Case Model training
        load_case_model_artifact_path: The path to the model artifact,
            relative to the turbine model version artifact folder
    """

    load_case: LoadCase
    load_case_model_artifact_path: str


class TurbineModelVersionArtifactMetadata:
    """Metadata for a turbine model version artifact."""

    def __init__(
        self,
        turbine_variant_build_config: TurbineVariantBuildConfig,
        load_case_model_artifacts: Dict[str, LoadCaseModelArtifactMetadata],
    ):
        self.turbine_variant_build_config = turbine_variant_build_config
        self.load_case_model_artifacts = load_case_model_artifacts

    @property
    def load_case_model_ids(self) -> List[LoadCaseModelId]:
        load_case_model_ids = []
        for load_case_model_name in self.load_case_model_artifacts:
            load_case_model_ids.append(
                LoadCaseModelId(
                    turbine_variant_id=self.turbine_variant_build_config.turbine_variant.id,
                    version=self.turbine_variant_build_config.version,
                    load_case_name=load_case_model_name,
                )
            )
        return load_case_model_ids

    def get_load_case_model_artifact_path(self, load_case_model_id: LoadCaseModelId) -> str:
        """Get the artifact path for a specific load case model."""
        if load_case_model_id not in self.load_case_model_ids:
            raise ValueError(
                f"Load case model id {load_case_model_id} not in turbine model version."
            )
        return self.load_case_model_artifacts[
            load_case_model_id.load_case_name
        ].load_case_model_artifact_path

    def get_load_case(self, load_case_model_id: LoadCaseModelId) -> LoadCase:
        """Get load case for a specific load case model."""
        if load_case_model_id not in self.load_case_model_ids:
            raise ValueError(
                f"Load case model id {load_case_model_id} not in turbine model version."
            )
        return self.load_case_model_artifacts[load_case_model_id.load_case_name].load_case

    def to_dict(self) -> Dict:
        return {
            "turbine_variant_build_config": self.turbine_variant_build_config.model_dump(),
            "load_case_model_artifacts": {
                k: v.model_dump() for k, v in self.load_case_model_artifacts.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            turbine_variant_build_config=TurbineVariantBuildConfig(
                **d["turbine_variant_build_config"]
            ),
            load_case_model_artifacts={
                k: LoadCaseModelArtifactMetadata(**v)
                for k, v in d["load_case_model_artifacts"].items()
            },
        )
