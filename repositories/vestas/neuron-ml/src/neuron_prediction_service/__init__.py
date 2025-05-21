import os

from neuron.api_utils import configure_logger, strtobool

# I have to parse an environment variable and convert it to a boolean value.
# I would do it in the pydantic settings, but if I initialize the settings in this file,
# I run into trouble with overriding the settings in the tests.
use_json_bool = strtobool(os.getenv("USE_JSON_LOGGING", "False"))

configure_logger(
    use_json_logging=use_json_bool,
)
