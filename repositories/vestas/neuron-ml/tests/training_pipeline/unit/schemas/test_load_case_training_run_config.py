import pytest
from pydantic import ValidationError

from neuron.schemas.training_run_config import LoadCaseTrainingRunConfig


@pytest.fixture
def test_config() -> dict:
    return {
        "name": "a-load-case",
        "data": {
            "training_data_file_uri": "some-uri",
        },
        "data_splitting": {
            "name": "random_test_train_split",
            "params": {"test_size": 0.2},
        },
        "preprocessor": "wind_gradient",
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
                    "feature_value_type": "continuous",
                    "feature_type": "raw",
                    "is_model_input": True,
                    "safe_domain_method": "range_validation",
                },
            ],
        },
        "target_list": {
            "name": "custom-target-list",
            "targets": ["MxBld_0000_max", "MxBld_0000_min"],
        },
        "max_load_evaluation_limit": 0.1,
        "calculation_type": "fatigue",
    }


@pytest.mark.parametrize(
    "max_load_evaluation_limit",
    [-0.1, 1.1],
)
def test_bad_max_load_evaluation_limit(test_config: dict, max_load_evaluation_limit: float) -> None:
    """Test that a limit outside the range [0, 1) raises an error."""
    with pytest.raises(ValidationError) as exc_info:
        test_config["max_load_evaluation_limit"] = max_load_evaluation_limit
        LoadCaseTrainingRunConfig(**test_config)

    assert len(exc_info.value.errors()) == 1
    assert exc_info.value.errors()[0]["loc"] == ("max_load_evaluation_limit",)
    assert exc_info.value.errors()[0]["type"] == "value_error"
    assert (
        exc_info.value.errors()[0]["msg"]
        == "Value error, 'max_load_evaluation_limit' should be in the range [0, 1)."
    )


def test_duplicate_features(test_config: dict) -> None:
    """Test that duplicate feature names raises an error."""
    with pytest.raises(ValidationError) as exc_info:
        test_config["feature_list"]["features"].append(test_config["feature_list"]["features"][0])
        LoadCaseTrainingRunConfig(**test_config)

    assert len(exc_info.value.errors()) == 1
    assert exc_info.value.errors()[0]["loc"] == ("feature_list",)
    assert exc_info.value.errors()[0]["type"] == "value_error"
    assert (
        exc_info.value.errors()[0]["msg"]
        == "Value error, Load case includes repeated feature names."
    )


def test_duplicate_targets(test_config: dict) -> None:
    """Test that duplicate feature names raises an error."""
    with pytest.raises(ValidationError) as exc_info:
        test_config["target_list"]["targets"].append(test_config["target_list"]["targets"][0])
        LoadCaseTrainingRunConfig(**test_config)

    assert len(exc_info.value.errors()) == 1
    assert exc_info.value.errors()[0]["loc"] == ("target_list",)
    assert exc_info.value.errors()[0]["type"] == "value_error"
    assert (
        exc_info.value.errors()[0]["msg"]
        == "Value error, Load case includes repeated target names."
    )


def test_bad_target_name(test_config: dict) -> None:
    """Test that a bad target name raises an error."""
    with pytest.raises(ValidationError) as exc_info:
        test_config["target_list"]["targets"] = ["bad_target_name"]
        LoadCaseTrainingRunConfig(**test_config)

    assert len(exc_info.value.errors()) == 1
    assert exc_info.value.errors()[0]["loc"] == ("target_list", "targets", 0)
    assert exc_info.value.errors()[0]["type"] == "enum"
