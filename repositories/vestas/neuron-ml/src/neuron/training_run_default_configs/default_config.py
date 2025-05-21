"""Default Configuration module for neuron load cases.

Contains information on default named load cases.
"""

import os
from copy import deepcopy
from typing import Any, Dict, List

import yaml

from neuron.schemas.training_run_config import (
    LoadCaseDataConfig,
    LoadCaseTrainingRunConfig,
)

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "default_training_run_config.yml")


def flatten_dict_config(config: Dict[str, Any]) -> Dict[str, Any]:
    adjusted_config = {}

    for list_name in config:
        flat_list = []
        for value in config[list_name]:
            if isinstance(value, List):
                flat_list += value
            else:
                flat_list.append(value)
        adjusted_config[list_name] = flat_list

    return adjusted_config


with open(DEFAULT_CONFIG_FILE, "r") as configyamlfile:
    default_conf_dict = yaml.safe_load(configyamlfile)

_LOAD_CASE_TRAINING_RUN_DICTS = default_conf_dict["load_cases"]
DEFAULT_LOAD_CASE_TRAINING_RUN_NAMES = list(_LOAD_CASE_TRAINING_RUN_DICTS.keys())


def get_default_load_case_training_run_config(
    load_case_name: str, data_config: LoadCaseDataConfig
) -> LoadCaseTrainingRunConfig:
    load_case_training_run_dict = deepcopy(_LOAD_CASE_TRAINING_RUN_DICTS[load_case_name])
    load_case_training_run_dict["data"] = data_config.dict()
    return LoadCaseTrainingRunConfig(**load_case_training_run_dict)
