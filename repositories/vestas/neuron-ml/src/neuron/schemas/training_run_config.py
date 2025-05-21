"""Schemas for training run configuration
and helper functions to load the configuration from a hydra config.

The main turbine level training run config is the `TrainingRunConfig` object,
"""
from enum import StrEnum
from typing import Any, Dict, List, Self, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from neuron.data_splitting import registry as data_splitting_registry
from neuron.evaluation.evaluation_utils import ModelAcceptanceCriteria
from neuron.models.target_models import registry as target_model_registry
from neuron.schemas.domain import (
    AggregationMethod,
    CalculationType,
    FeatureList,
    PostprocessorName,
    Target,
    TargetList,
    TurbineVariant,
)


class StorageType(StrEnum):
    """Type of data storage.

    INTERNAL: Data is stored in the internal Neuron storage account, curated by the Neuron team.
    This can be both local file storage or blob storage.
    SIMMESH: Data is stored in SimMesh.
    """

    INTERNAL = "internal"
    SIMMESH = "simmesh"


class LoadCaseDataConfig(BaseModel):
    """Data configuration for a single Load Case training run

    If test_data_file_uri is None, the training data is split
    into train and test data in the training pipeline.
    """

    training_data_file_uri: str
    test_data_file_uri: Union[str, None] = None
    agg_data_file_uri: Union[str, None] = None


class TargetModelConfig(BaseModel):
    """Configuration for a target model

    The name should correspond to a registered target model class in the registry.
    The params should correspond to the parameters for initializing the target model class
    (not including the feature and target parameters).
    """

    name: str
    params: Dict[str, Any]

    @field_validator("name")
    @classmethod
    def validate_target_model_name(cls, value: str) -> str:
        _ = target_model_registry.get_registered_model_class_from_name(value)
        return value

    @model_validator(mode="after")
    def validate_model_can_be_initialized(self) -> Self:
        init_config = target_model_registry.TargetModelInitializationConfig(
            name=self.name, params=self.params, features=["test"], target_col="test"
        )
        try:
            _ = target_model_registry.initialize_target_model(init_config)
        except Exception as e:
            raise ValueError("Bad params. Model can not be initialized with supplied params") from e
        return self


class LoadCaseModelConfig(TargetModelConfig):
    """Configuration for a load case model

    By default, the name and params are used for all load cases,
    except when model_override is specified for specific target names.
    """

    model_override: Union[Dict[str, TargetModelConfig], None] = None

    # To avoid warning of clash between `model_override`and Pydantic protected namespace.
    model_config = ConfigDict(protected_namespaces=())

    def get_target_model_config(self, target_name: str) -> TargetModelConfig:
        if self.model_override is None:
            return TargetModelConfig(name=self.name, params=self.params)
        try:
            return self.model_override[target_name]
        except KeyError:
            return TargetModelConfig(name=self.name, params=self.params)


class DataSplitConfig(BaseModel):
    """Configuration for splitting the data into train and test sets

    The name should correspond to a registered data splitting class in the registry.
    The params should correspond to the parameters for initializing data splitting class.
    """

    name: str
    params: Dict[str, Any]

    @field_validator("name")
    @classmethod
    def validate_target_model_name(cls, value: str) -> str:
        if value not in data_splitting_registry.DATA_SPLITTING_REGISTRY.keys():
            raise ValueError(
                "Data splitting method not in method registry. Available methods are: "
                f"{list(data_splitting_registry.DATA_SPLITTING_REGISTRY.keys())}"
            )
        return value

    @model_validator(mode="after")
    def validate_model_can_be_initialized(self) -> Self:
        try:
            splitter = data_splitting_registry.get_data_splitter(name=self.name, params=self.params)
            # Just because the class can be initialized doesn't mean the params are valid
            # They might fail later when the data is actually split
            if splitter.validate_params() is False:
                raise ValueError(
                    "Bad params. Data splitter can not be initialized with supplied params"
                )
        except Exception as e:
            raise ValueError(
                "Bad params. Data splitter can not be initialized with supplied params"
            ) from e

        return self


class LoadCaseTrainingRunConfig(BaseModel):
    name: str
    data: LoadCaseDataConfig
    data_splitting: DataSplitConfig
    load_case_model: LoadCaseModelConfig
    postprocessor: PostprocessorName
    feature_list: FeatureList
    target_list: TargetList
    max_load_evaluation_limit: float
    calculation_type: CalculationType
    calculate_aggregated_metrics: Union[AggregationMethod, None] = None

    @field_validator("feature_list")
    @classmethod
    def no_duplicate_feature_names(cls, value: FeatureList) -> FeatureList:
        feature_names = [feature.name for feature in value.features]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Load case includes repeated feature names.")
        return value

    @field_validator("target_list")
    @classmethod
    def no_duplicate_target_names(cls, value: TargetList) -> TargetList:
        if len(value.targets) != len(set(value.targets)):
            raise ValueError("Load case includes repeated target names.")
        return value

    @field_validator("max_load_evaluation_limit")
    @classmethod
    def validate_max_load_evaluation_limit(cls, value: float) -> float:
        if not (0 <= value < 1):
            raise ValueError("'max_load_evaluation_limit' should be in the range [0, 1).")
        return value

    @model_validator(mode="after")
    def data_needed_for_aggregated_metrics(self) -> Self:
        if self.calculate_aggregated_metrics and not self.data.agg_data_file_uri:
            raise ValueError(
                "Separate aggregation data set is required to compute aggregated metrics"
            )

        if self.calculate_aggregated_metrics and self.calculation_type != CalculationType.FATIGUE:
            raise ValueError(
                "The aggregation method contemplates 1Hz fatigue loads only."
                "Extreme loads are not allowed."
            )
        return self


class EvaluationConfig(BaseModel):
    alpha_significance_level: float
    generate_coverage_plots: bool
    fat_model_acceptance_criteria: List[ModelAcceptanceCriteria]
    ext_model_acceptance_criteria: List[ModelAcceptanceCriteria]


class TurbineConfig(BaseModel):
    turbine_variant: TurbineVariant
    design_loads_ext: Union[Dict[Target, float], None] = None

    @field_validator("design_loads_ext", mode="before")
    @classmethod
    def check_design_loads_ext(cls, v):  # noqa: ANN001
        if v is not None:
            for key in v:
                if key not in Target.__members__:
                    raise ValueError(
                        f"'{key}' is not a valid member of the ExtremeTarget enum list"
                    )
        return v


class TrainingRunConfig(BaseModel):
    load_case_training_runs: List[LoadCaseTrainingRunConfig]
    turbine: TurbineConfig
    evaluation: EvaluationConfig
    storage_type: StorageType = StorageType.INTERNAL

    @field_validator("load_case_training_runs")
    @classmethod
    def load_cases_names_must_be_unique(cls, v):  # noqa: ANN001
        load_case_names = [load_case.name for load_case in v]
        if len(set(load_case_names)) != len(load_case_names):
            raise ValueError(
                "Load case names must be unique within a single turbine_variant training run."
            )
        return v

    @field_validator("load_case_training_runs")
    @classmethod
    def check_agg_load_case_target_lists(cls, v):  # noqa: ANN001
        agg_load_cases = [lc for lc in v if lc.calculate_aggregated_metrics is not None]
        if len(agg_load_cases) > 0:
            # target set for the first agg load case
            target_set = set(agg_load_cases[0].target_list.targets)

            for lc in agg_load_cases:
                # check that target sets are identical across all agg load cases
                if set(lc.target_list.targets) != target_set:
                    raise ValueError(
                        "All load cases with calculate_aggregated_metrics = True "
                        "must have the same target list"
                    )
        return v
