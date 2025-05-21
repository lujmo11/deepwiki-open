"""Module for parsing user training run configs to generate a `TrainingRunConfig`."""
import argparse
from typing import List

import yaml

from neuron.schemas.domain import Feature
from neuron.schemas.training_run_config import (
    LoadCaseTrainingRunConfig,
    TrainingRunConfig,
)
from neuron.training_run_default_configs.default_config import (
    DEFAULT_LOAD_CASE_TRAINING_RUN_NAMES,
    get_default_load_case_training_run_config,
)
from neuron_training_service.schemas import (
    APIUserTrainingRunConfig,
    CLIUserLoadCaseConfig,
    CLIUserTrainingRunConfig,
    TrainingRungConfigError,
    UserLoadCaseChanges,
)


def get_training_config_from_api_user_config(
    api_user_config: APIUserTrainingRunConfig,
) -> TrainingRunConfig:
    """Get a TrainingRunConfig from a user training run config.

    For each load case training run config we
    - get the default load case training run config based on the load case name.
    - Update the default config with the user specified changes.

    The Turbine and Evaluation configs are just copied over from the user config.
    """
    load_case_training_run_configs = []
    for user_lc_run_config in api_user_config.load_case_training_runs:
        load_case_training_run_config = get_default_load_case_training_run_config(
            load_case_name=user_lc_run_config.name, data_config=user_lc_run_config.data
        )
        load_case_training_run_config = update_load_case_training_run_config_with_user_changes(
            lc_training_run_config=load_case_training_run_config, user_changes=user_lc_run_config
        )
        load_case_training_run_configs.append(load_case_training_run_config)
    try:
        return TrainingRunConfig(
            load_case_training_runs=load_case_training_run_configs,
            evaluation=api_user_config.evaluation,
            turbine=api_user_config.turbine,
            storage_type=api_user_config.storage_type,
        )
    except ValueError as err:
        raise TrainingRungConfigError(f"Training run config is not valid: {err}") from err


def get_training_config_from_cli_user_config(
    cli_user_config: CLIUserTrainingRunConfig,
) -> TrainingRunConfig:
    """Get a TrainingRunConfig from a user training run config.

    For each load case training run config we
    - Use the values as they are if the load case name is not one of the default load case names.
    - Get the default load case training run config based on the load case name.
    - Update the default config with the user specified changes.
    - Overwrite the default config with any user overwrites in the CLI config.
    """
    load_case_training_run_configs = []
    for user_lc_run_config in cli_user_config.load_case_training_runs:
        if user_lc_run_config.name not in DEFAULT_LOAD_CASE_TRAINING_RUN_NAMES:
            try:
                load_case_training_run_config = LoadCaseTrainingRunConfig.model_validate(
                    user_lc_run_config, from_attributes=True
                )
            except ValueError as err:
                raise ValueError(
                    f"Load case training run config for {user_lc_run_config.name} is not valid."
                ) from err
        else:
            load_case_training_run_config = get_default_load_case_training_run_config(
                load_case_name=user_lc_run_config.name, data_config=user_lc_run_config.data
            )
            load_case_training_run_config = update_load_case_training_run_config_with_user_changes(
                lc_training_run_config=load_case_training_run_config,
                user_changes=user_lc_run_config,
            )
            load_case_training_run_config = make_user_overwrites(
                load_case_training_run_config, user_lc_run_config
            )
        load_case_training_run_configs.append(load_case_training_run_config)

    return TrainingRunConfig(
        load_case_training_runs=load_case_training_run_configs,
        evaluation=cli_user_config.evaluation,
        turbine=cli_user_config.turbine,
        storage_type=cli_user_config.storage_type,
    )


def make_user_overwrites(
    load_case_training_run_config: LoadCaseTrainingRunConfig,
    cli_load_case_user_config: CLIUserLoadCaseConfig,
) -> LoadCaseTrainingRunConfig:
    """Overwrite the load case training run config with the user overwrites in the CLI config.

    Any values from the CLI config will take precedence.
    """
    lc_training_run_config_dict = load_case_training_run_config.model_dump()
    cli_load_case_user_config_dict = cli_load_case_user_config.model_dump()

    for key, value in cli_load_case_user_config_dict.items():
        if value is not None:
            lc_training_run_config_dict[key] = value
    return LoadCaseTrainingRunConfig(**lc_training_run_config_dict)


def update_load_case_training_run_config_with_user_changes(
    lc_training_run_config: LoadCaseTrainingRunConfig, user_changes: UserLoadCaseChanges
) -> LoadCaseTrainingRunConfig:
    if user_changes.add_features:
        lc_training_run_config = add_features(lc_training_run_config, user_changes.add_features)
    if user_changes.drop_features:
        lc_training_run_config = drop_features(lc_training_run_config, user_changes.drop_features)
    if user_changes.add_targets:
        lc_training_run_config = add_targets(lc_training_run_config, user_changes.add_targets)
    if user_changes.drop_targets:
        lc_training_run_config = drop_targets(lc_training_run_config, user_changes.drop_targets)
    if user_changes.data_splitting:
        lc_training_run_config.data_splitting = user_changes.data_splitting
    if user_changes.calculate_aggregated_metrics:
        lc_training_run_config.calculate_aggregated_metrics = (
            user_changes.calculate_aggregated_metrics
        )
    return lc_training_run_config


def add_features(
    load_case_training_run_config: LoadCaseTrainingRunConfig, features: List[Feature]
) -> LoadCaseTrainingRunConfig:
    """Add features to the load case training run config."""
    load_case_training_run_config_dict = load_case_training_run_config.model_dump()
    load_case_training_run_config_dict["feature_list"]["features"].extend(features)
    return LoadCaseTrainingRunConfig(**load_case_training_run_config_dict)


def add_targets(
    load_case_training_run_config: LoadCaseTrainingRunConfig, targets: List[str]
) -> LoadCaseTrainingRunConfig:
    """Add targets to the load case training run config."""
    load_case_training_run_config_dict = load_case_training_run_config.model_dump()
    load_case_training_run_config_dict["target_list"]["targets"].extend(targets)
    return LoadCaseTrainingRunConfig(**load_case_training_run_config_dict)


def drop_targets(
    load_case_training_run_config: LoadCaseTrainingRunConfig, targets: List[str]
) -> LoadCaseTrainingRunConfig:
    """Drop targets from the load case training run config."""
    load_case_training_run_config_dict = load_case_training_run_config.model_dump()
    load_case_training_run_config_dict["target_list"]["targets"] = [
        target
        for target in load_case_training_run_config_dict["target_list"]["targets"]
        if target not in targets
    ]
    return LoadCaseTrainingRunConfig(**load_case_training_run_config_dict)


def drop_features(
    load_case_training_run_config: LoadCaseTrainingRunConfig, features: List[str]
) -> LoadCaseTrainingRunConfig:
    """Drop features from the load case training run config."""
    load_case_training_run_config_dict = load_case_training_run_config.model_dump()
    load_case_training_run_config_dict["feature_list"]["features"] = [
        feature
        for feature in load_case_training_run_config_dict["feature_list"]["features"]
        if feature["name"] not in features
    ]
    return LoadCaseTrainingRunConfig(**load_case_training_run_config_dict)


if __name__ == "__main__":
    """Parse user training run config and output parsed yaml file.
    This is used when training locally. The input YAML file is the user config 
    and must adhere to the CLIUserTrainingRunConfig schema. 
    The output YAML file is the parsed training run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user-train-config-path",
        action="store",
        type=str,
        help=(
            "Path to yaml file with user training run config. "
            "Must adhere to the CLIUserTrainingRunConfig schema."
        ),
        required=True,
    )
    parser.add_argument(
        "--parsed-train-config-path",
        action="store",
        type=str,
        help="Output path for parsed yaml file with training run config.",
        required=True,
    )
    args = parser.parse_args()
    with open(args.user_train_config_path, "r") as f:
        user_train_config = yaml.safe_load(f)

    user_train_config = CLIUserTrainingRunConfig(**user_train_config)
    parsed_train_config = get_training_config_from_cli_user_config(user_train_config)
    parsed_config_dict = parsed_train_config.model_dump(mode="json")
    with open(args.parsed_train_config_path, "w") as f:
        yaml.dump(parsed_config_dict, f)
