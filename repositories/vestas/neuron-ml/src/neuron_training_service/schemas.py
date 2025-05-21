"""Schemas for user input to the training service.

The user input can be provided as a CLI input or as an API input.
The CLI is used by the developers of the project, and the schema has a lot more flexibility.
The API is used by the end users of the project, and the schema is more restricted.
"""
from typing import List, Union

from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState
from pydantic import BaseModel, field_validator

from neuron.schemas.domain import (
    AggregationMethod,
    CalculationType,
    Feature,
    FeatureList,
    PostprocessorName,
    TargetList,
)
from neuron.schemas.training_run_config import (
    DataSplitConfig,
    EvaluationConfig,
    LoadCaseDataConfig,
    LoadCaseModelConfig,
    StorageType,
    TurbineConfig,
)
from neuron.training_run_default_configs.default_config import DEFAULT_LOAD_CASE_TRAINING_RUN_NAMES


class CLIUserLoadCaseBaseConfig(BaseModel, extra="forbid"):
    """The base load case training run config for a CLI user."""

    name: str
    data: LoadCaseDataConfig
    data_splitting: Union[DataSplitConfig, None] = None
    load_case_model: Union[LoadCaseModelConfig, None] = None
    postprocessor: Union[PostprocessorName, None] = None
    feature_list: Union[FeatureList, None] = None
    target_list: Union[TargetList, None] = None
    max_load_evaluation_limit: Union[float, None] = None
    calculation_type: Union[CalculationType, None] = None
    calculate_aggregated_metrics: Union[AggregationMethod, None] = None


class APIUserConfigParsingError(Exception):
    """Exception raised when the API user config cannot be parsed."""

    pass


class TrainingRungConfigError(Exception):
    """Exception raised when the training run config is not valid."""

    pass


class APIUserLoadCaseBaseConfig(BaseModel, extra="forbid"):
    """The base load case training run config for an API user."""

    name: str
    data: LoadCaseDataConfig
    data_splitting: Union[DataSplitConfig, None] = None
    calculate_aggregated_metrics: Union[AggregationMethod, None] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        """Ensure the load case name is in the default load case names."""
        if name not in DEFAULT_LOAD_CASE_TRAINING_RUN_NAMES:
            raise APIUserConfigParsingError(
                f"Load case name {name} not in default load case names."
                f"Valid names are {DEFAULT_LOAD_CASE_TRAINING_RUN_NAMES}"
            )
        return name


class UserLoadCaseChanges(BaseModel, extra="forbid"):
    """User specified changes to a load case training run config."""

    add_features: Union[List[Feature], None] = None
    drop_features: Union[List[str], None] = None
    add_targets: Union[List[str], None] = None
    drop_targets: Union[List[str], None] = None
    data_splitting: Union[DataSplitConfig, None] = None
    calculate_aggregated_metrics: Union[AggregationMethod, None] = None


class CLIUserLoadCaseConfig(CLIUserLoadCaseBaseConfig, UserLoadCaseChanges):
    """Load case training run config for a CLI user."""


class APIUserLoadCaseConfig(APIUserLoadCaseBaseConfig, UserLoadCaseChanges):
    """Load case training run config for an API user."""


class CLIUserTrainingRunConfig(BaseModel, extra="forbid"):
    """The training run config for a CLI user."""

    load_case_training_runs: List[CLIUserLoadCaseConfig]
    turbine: TurbineConfig
    evaluation: EvaluationConfig
    storage_type: StorageType = StorageType.INTERNAL


# Use strict validation for the API user
class APIUserTrainingRunConfig(BaseModel, extra="forbid"):
    """The training run config for an API user."""

    load_case_training_runs: List[APIUserLoadCaseConfig]
    turbine: TurbineConfig
    evaluation: EvaluationConfig
    storage_type: StorageType = StorageType.INTERNAL


class TrainingServiceResponse(BaseModel):
    """Response from the training service.

    Attributes
    ----------
    job_run_id : int
        The run ID of the triggered Databricks job run.
    training_input_id : str
        The training input ID. This is a unique identifier for the training input.
        It is the root directory of the training input data in the target storage.
    """

    job_run_id: int
    training_input_id: str


class JobRun(BaseModel):
    """Information about a DBX job run."""

    id: int
    url: str
    result_state: Union[RunResultState, None] = None
    life_cycle_state: RunLifeCycleState


class MLFlowRun(BaseModel):
    """Information about an MLFlow run."""

    id: str
    url: str


class NeuronTrainingJobRun(BaseModel):
    """Information about a Neuron training run.
    This includes the DBX job run information and the MLFlow run information if it exists.

    Attributes
    ----------
    job_run : neuron_training_service.schemas.JobRun
        The job run information.
    mlflow_run : Union[MLFlowRun, None]
        The MLFlow run information. If the MLFlow run does not exist, this is None.

    """

    job_run: JobRun
    mlflow_run: Union[MLFlowRun, None] = None


TRAIN_JOB_INPUT_EXAMPLES = [
    {
        "load_case_training_runs": [
            {
                "name": "dlc11",
                "data": {"training_data_file_uri": "test/data_reduced_dlc11.parquet"},
                "data_splitting": {"name": "random_test_train_split", "params": {"test_size": 0.2}},
            }
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
                {
                    "metric": "e_mean_norm",
                    "value": 0.01,
                    "condition": "le",
                }
            ],
            "ext_model_acceptance_criteria": [
                {
                    "metric": "e_mean_norm",
                    "value": 0.01,
                    "condition": "le",
                }
            ],
        },
    }
]
