"""Test the parsing of `CLIUserTrainingRunConfig` objects to `TrainingRunConfig`objects
using the get_training_config_from_cli_user_config function.
"""

import pytest

from neuron.schemas.training_run_config import TrainingRunConfig
from neuron_training_service.schemas import CLIUserTrainingRunConfig
from neuron_training_service.user_config_parsing import get_training_config_from_cli_user_config


@pytest.fixture
def cli_training_run_config_with_named_load_case() -> CLIUserTrainingRunConfig:
    config = {
        "evaluation": {
            "alpha_significance_level": 0.05,
            "generate_coverage_plots": False,
            "fat_model_acceptance_criteria": [
                {"metric": "e_mean_norm", "value": 0.02, "condition": "lt"}
            ],
            "ext_model_acceptance_criteria": [
                {"metric": "e_mean_norm", "value": 0.02, "condition": "lt"}
            ],
        },
        "load_case_training_runs": [
            {
                "name": "dlc11",
                "data": {
                    "training_data_file_uri": "some_training_data.parquet",
                },
            }
        ],
        "turbine": {
            "turbine_variant": {
                "mk_version": "mk3e",
                "rated_power": "4000",
                "rotor_diameter": "150",
            }
        },
    }
    return CLIUserTrainingRunConfig(**config)


@pytest.fixture
def cli_training_run_config_with_custom_load_case(
    cli_training_run_config_with_named_load_case: CLIUserTrainingRunConfig,
) -> CLIUserTrainingRunConfig:
    config_with_custom_load_case = cli_training_run_config_with_named_load_case.dict()
    config_with_custom_load_case["load_case_training_runs"][0] = {
        "name": "a-load-case",
        "data": {
            "training_data_file_uri": "some-uri",
        },
        "data_splitting": {
            "name": "random_test_train_split",
            "params": {"test_size": 0.2},
        },
        "postprocessor": "no_postprocessor",
        "load_case_model": {
            "name": "linear_regression",
            "params": {"fit_intercept": False},
        },
        "feature_list": {
            "name": "custom_feature_name",
            "features": [
                {
                    "name": "some-feature",
                    "feature_type": "raw",
                    "feature_value_type": "continuous",
                    "safe_domain_method": "range_validation",
                    "is_model_input": True,
                },
            ],
        },
        "target_list": {
            "name": "custom-sensor-list",
            "targets": ["MxBld_0000_max", "MxBld_0000_min"],
        },
        "max_load_evaluation_limit": 0.1,
        "calculation_type": "fatigue",
    }
    return CLIUserTrainingRunConfig(**config_with_custom_load_case)


def test_named_config_is_parsed_correctly(
    cli_training_run_config_with_named_load_case: CLIUserTrainingRunConfig,
) -> None:
    parsed_config = get_training_config_from_cli_user_config(
        cli_training_run_config_with_named_load_case
    )
    assert parsed_config.load_case_training_runs[0].name == "dlc11"


def test_completely_custom_config_is_parsed_correctly(
    cli_training_run_config_with_custom_load_case: CLIUserTrainingRunConfig,
) -> None:
    parsed_config = get_training_config_from_cli_user_config(
        cli_training_run_config_with_custom_load_case
    )
    assert (
        TrainingRunConfig(**cli_training_run_config_with_custom_load_case.model_dump())
        == parsed_config
    )


def test_named_config_with_custom_overrides_is_parsed_correctly(
    cli_training_run_config_with_named_load_case: CLIUserTrainingRunConfig,
) -> None:
    """Test that we can parse a named load case with custom overwrites.

    We expect the overwrites to take precedence over the default values.
    """
    parsed_config_no_overwrites = get_training_config_from_cli_user_config(
        cli_training_run_config_with_named_load_case
    )

    cli_training_run_config_with_overwrites = cli_training_run_config_with_named_load_case.dict()

    # Set up a default load case name, but with custom overwrites
    load_case_default_model_overwrite = {
        "name": "linear_regression",
        "params": {"fit_intercept": True},
    }
    load_case_default_feature_list_overwrite = {
        "name": "custom_feature_name",
        "features": [
            {
                "name": "some-feature",
                "feature_type": "raw",
                "feature_value_type": "continuous",
                "is_model_input": True,
                "safe_domain_method": "range_validation",
            },
        ],
    }
    cli_training_run_config_with_overwrites["load_case_training_runs"][0][
        "load_case_model"
    ] = load_case_default_model_overwrite
    cli_training_run_config_with_overwrites["load_case_training_runs"][0][
        "feature_list"
    ] = load_case_default_feature_list_overwrite

    # The expected config should have the default load case values, but with the custom overwrites
    expected_config = parsed_config_no_overwrites.dict()
    expected_config["load_case_training_runs"][0][
        "load_case_model"
    ] = load_case_default_model_overwrite
    expected_config["load_case_training_runs"][0][
        "feature_list"
    ] = load_case_default_feature_list_overwrite

    # Act
    parsed_config = get_training_config_from_cli_user_config(
        CLIUserTrainingRunConfig(**cli_training_run_config_with_overwrites)
    )

    # Assert
    assert TrainingRunConfig(**expected_config) == parsed_config


def test_config_with_features_added(
    cli_training_run_config_with_named_load_case: CLIUserTrainingRunConfig,
) -> None:
    """Test that we can add features to a load case."""
    config_with_features_added = cli_training_run_config_with_named_load_case.dict()
    config_with_features_added["load_case_training_runs"][0]["add_features"] = [
        {
            "name": "test_feature1",
            "feature_type": "raw",
            "feature_value_type": "discrete",
            "safe_domain_method": "discrete_validation",
            "is_model_input": True,
        },
        {
            "name": "test_feature2",
            "feature_type": "raw",
            "feature_value_type": "discrete",
            "safe_domain_method": "discrete_validation",
            "is_model_input": True,
        },
    ]
    parsed_config = get_training_config_from_cli_user_config(
        CLIUserTrainingRunConfig(**config_with_features_added)
    )
    assert parsed_config.load_case_training_runs[0].name == "dlc11"
    assert parsed_config.load_case_training_runs[0].target_list.name == "fat"
    assert (
        parsed_config.load_case_training_runs[0].feature_list.name
        == "features_climate_dependent_fat"
    )
    assert parsed_config.load_case_training_runs[0].max_load_evaluation_limit == 0.01
    assert parsed_config.load_case_training_runs[0].postprocessor == "directional_twr"
    assert parsed_config.load_case_training_runs[0].load_case_model.name == "deep_gpr"

    feature_names = [
        feat.name for feat in parsed_config.load_case_training_runs[0].feature_list.features
    ]
    assert all([feat in feature_names for feat in ["test_feature1", "test_feature2"]])


def test_config_with_features_dropped(
    cli_training_run_config_with_named_load_case: CLIUserTrainingRunConfig,
) -> None:
    """Test that we can drop features from a load case."""
    config_with_features_dropped = cli_training_run_config_with_named_load_case.dict()
    config_with_features_dropped["load_case_training_runs"][0]["drop_features"] = ["yaw"]
    parsed_config = get_training_config_from_cli_user_config(
        CLIUserTrainingRunConfig(**config_with_features_dropped)
    )
    feature_names = [
        feat.name for feat in parsed_config.load_case_training_runs[0].feature_list.features
    ]
    assert "yaw" not in feature_names


def test_config_with_targets_added(
    cli_training_run_config_with_named_load_case: CLIUserTrainingRunConfig,
) -> None:
    """Test that we can add targets to a load case."""
    config_with_targets_added = cli_training_run_config_with_named_load_case.dict()
    config_with_targets_added["load_case_training_runs"][0]["add_targets"] = [
        "MxBld_0000_max",
        "MxBld_0000_min",
    ]
    parsed_config = get_training_config_from_cli_user_config(
        CLIUserTrainingRunConfig(**config_with_targets_added)
    )
    target_names = parsed_config.load_case_training_runs[0].target_list.targets
    assert all([target in target_names for target in ["MxBld_0000_min", "MxBld_0000_min"]])


def test_config_with_targets_dropped(
    cli_training_run_config_with_named_load_case: CLIUserTrainingRunConfig,
) -> None:
    """Test that we can drop targets from a load case."""
    config_with_targets_dropped = cli_training_run_config_with_named_load_case.dict()
    config_with_targets_dropped["load_case_training_runs"][0]["drop_targets"] = [
        "MxTwrBot_Fnd_m400",
        "MxTwrBot_Fnd_m800",
    ]
    parsed_config = get_training_config_from_cli_user_config(
        CLIUserTrainingRunConfig(**config_with_targets_dropped)
    )
    target_names = parsed_config.load_case_training_runs[0].target_list.targets
    assert "MxTwrBot_Fnd_m400" not in target_names
    assert "MxTwrBot_Fnd_m800" not in target_names
