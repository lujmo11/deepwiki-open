"""Logic for migrating old training runs with outdated training configurations

The changes are due to
- Changes in the evaluation configuration to include the model acceptance criteria in the PR: https://dev.azure.com/vestas/Neuron/_git/neuron-ml/commit/afa492fd23259940ace01219459a58ba54cc697d?refName=refs%2Fheads%2Fmain
"""

from copy import deepcopy


def _introduce_model_acceptance_criteria(eval_config: dict) -> None:
    eval_config["fat_model_acceptance_criteria"] = [
        {"metric": "mae_norm", "value": 0.03, "condition": "le"}
    ]
    eval_config["ext_model_acceptance_criteria"] = [
        {"metric": "mae_norm", "value": 0.03, "condition": "le"}
    ]
    eval_config.pop("mae_norm_fat_threshold")
    eval_config.pop("mae_norm_ext_threshold")


def migrate_train_config(train_config: dict) -> dict:
    train_config = deepcopy(train_config)
    eval_config = train_config["evaluation"]

    _introduce_model_acceptance_criteria(eval_config)

    return train_config
