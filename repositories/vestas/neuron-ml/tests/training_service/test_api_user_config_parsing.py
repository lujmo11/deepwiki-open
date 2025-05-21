"""Test the parsing of `APIUserTrainingRunConfig` objects to `TrainingRunConfig`objects
using the get_training_config_from_api_user_config function.
"""

import pytest

from neuron_training_service.schemas import APIUserTrainingRunConfig
from neuron_training_service.user_config_parsing import get_training_config_from_api_user_config


@pytest.fixture
def api_training_run_config() -> APIUserTrainingRunConfig:
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
    return APIUserTrainingRunConfig(**config)


def test_named_config_is_parsed_correctly(
    api_training_run_config: APIUserTrainingRunConfig,
) -> None:
    """Test that a named load case config is parsed correctly."""
    parsed_config = get_training_config_from_api_user_config(api_training_run_config)
    assert parsed_config.load_case_training_runs[0].name == "dlc11"


def test_adding_features(
    api_training_run_config: APIUserTrainingRunConfig,
) -> None:
    """Test that adding features works as expected."""
    parsed_config_no_additions = get_training_config_from_api_user_config(api_training_run_config)

    api_training_run_config_with_additions = api_training_run_config.dict()
    api_training_run_config_with_additions["load_case_training_runs"][0]["add_features"] = [
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
    parsed_config_with_additions = get_training_config_from_api_user_config(
        APIUserTrainingRunConfig(**api_training_run_config_with_additions)
    )

    feature_names_no_additions = set(
        feat.name
        for feat in parsed_config_no_additions.load_case_training_runs[0].feature_list.features
    )
    feature_names_with_additions = set(
        feat.name
        for feat in parsed_config_with_additions.load_case_training_runs[0].feature_list.features
    )
    # Assert that feature_names_with_additions contains the two
    # added features and the original features
    assert feature_names_no_additions.issubset(feature_names_with_additions)
    assert "test_feature1" in feature_names_with_additions
    assert "test_feature2" in feature_names_with_additions
    assert len(feature_names_with_additions) == len(feature_names_no_additions) + 2


def test_dropping_features(
    api_training_run_config: APIUserTrainingRunConfig,
) -> None:
    """Test that dropping features works as expected."""
    parsed_config_no_additions = get_training_config_from_api_user_config(api_training_run_config)

    api_training_run_config_with_drops = api_training_run_config.dict()
    api_training_run_config_with_drops["load_case_training_runs"][0]["drop_features"] = ["yaw"]
    parsed_config = get_training_config_from_api_user_config(
        APIUserTrainingRunConfig(**api_training_run_config_with_drops)
    )

    feature_names_with_drops = set(
        feat.name for feat in parsed_config.load_case_training_runs[0].feature_list.features
    )
    feature_names_with_no_drops = set(
        feat.name
        for feat in parsed_config_no_additions.load_case_training_runs[0].feature_list.features
    )
    assert feature_names_with_drops.issubset(feature_names_with_no_drops)
    assert "yaw" not in feature_names_with_drops
    assert len(feature_names_with_drops) == len(feature_names_with_no_drops) - 1


def test_adding_targets(
    api_training_run_config: APIUserTrainingRunConfig,
) -> None:
    """Test that adding targets works as expected."""
    parsed_config_no_additions = get_training_config_from_api_user_config(api_training_run_config)

    api_training_run_config_with_additions = api_training_run_config.dict()
    api_training_run_config_with_additions["load_case_training_runs"][0]["add_targets"] = [
        "MxBld_0000_max",
        "MxBld_0000_min",
    ]
    parsed_config = get_training_config_from_api_user_config(
        APIUserTrainingRunConfig(**api_training_run_config_with_additions)
    )

    target_names_with_additions = set(
        target for target in parsed_config.load_case_training_runs[0].target_list.targets
    )
    target_names_with_no_additions = set(
        target
        for target in parsed_config_no_additions.load_case_training_runs[0].target_list.targets
    )
    assert target_names_with_no_additions.issubset(target_names_with_additions)
    assert "MxBld_0000_max" in target_names_with_additions
    assert "MxBld_0000_min" in target_names_with_additions
    assert len(target_names_with_additions) == len(target_names_with_no_additions) + 2


def test_dropping_targets(
    api_training_run_config: APIUserTrainingRunConfig,
) -> None:
    """Test that dropping targets works as expected."""
    parsed_config_no_additions = get_training_config_from_api_user_config(api_training_run_config)

    api_training_run_config_with_drops = api_training_run_config.dict()
    api_training_run_config_with_drops["load_case_training_runs"][0]["drop_targets"] = [
        "MxTwrBot_Fnd_m400",
        "MxTwrBot_Fnd_m800",
    ]
    parsed_config = get_training_config_from_api_user_config(
        APIUserTrainingRunConfig(**api_training_run_config_with_drops)
    )

    target_names_with_drops = set(
        target for target in parsed_config.load_case_training_runs[0].target_list.targets
    )
    target_names_with_no_drops = set(
        target
        for target in parsed_config_no_additions.load_case_training_runs[0].target_list.targets
    )
    assert target_names_with_drops.issubset(target_names_with_no_drops)
    assert "MxTwrBot_Fnd_m400" not in target_names_with_drops
    assert "MxTwrBot_Fnd_m800" not in target_names_with_drops
    assert len(target_names_with_drops) == len(target_names_with_no_drops) - 2


def test_validating_data_splitter() -> None:
    """Test that the data splitter can validate valid configs and reject invalid ones."""
    from neuron_training_service.schemas import DataSplitConfig

    data_split_config = {
        "name": "random_test_train_split",
        "params": {
            "test_size": 0.2,
        },
    }
    # Test that a valid config is accepted
    DataSplitConfig.model_validate(data_split_config)

    # Test that an invalid config is rejected
    data_split_config["params"]["test_size"] = "abc"
    with pytest.raises(ValueError):
        DataSplitConfig.model_validate(data_split_config)


def test_calculate_aggregated_metrics(
    api_training_run_config: APIUserTrainingRunConfig,
) -> None:
    """Test that the calculation of aggregated metrics works as expected."""

    api_training_run_config_with_agg_data = api_training_run_config.model_dump()
    api_training_run_config_with_agg_data["load_case_training_runs"][0][
        "calculate_aggregated_metrics"
    ] = {"groupby": "a_grouping_column", "weightby": "a_weighting_column"}

    api_training_run_config_with_agg_data["load_case_training_runs"][0]["data"] = {
        "training_data_file_uri": "some_training_data.parquet",
        "agg_data_file_uri": "grouped_training_data.parquet",
    }

    APIUserTrainingRunConfig.model_validate(api_training_run_config_with_agg_data)
