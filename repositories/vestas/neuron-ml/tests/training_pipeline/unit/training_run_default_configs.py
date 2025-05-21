from neuron.schemas.training_run_config import LoadCaseDataConfig
from neuron.training_run_default_configs.default_config import (
    DEFAULT_LOAD_CASE_TRAINING_RUN_NAMES,
    get_default_load_case_training_run_config,
)


def test_default_load_case_configs() -> None:
    """Test that we can load all the default load case configurations."""
    for default_load_case_name in DEFAULT_LOAD_CASE_TRAINING_RUN_NAMES:
        try:
            data_config = LoadCaseDataConfig(training_data_file_uri="some-uri")
            _ = get_default_load_case_training_run_config(default_load_case_name, data_config)
        except Exception as e:
            raise AssertionError(
                f"Failed to load default load case {default_load_case_name}"
            ) from e
