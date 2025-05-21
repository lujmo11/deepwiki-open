"""Logic for migrating old training runs with outdated training configurations

The changes are due to
- Schema changes to Features and preprocessing in the TrainingRunConfig in the PR: https://vestas.visualstudio.com/Neuron/_git/neuron-ml/commit/a6c6e45d19e680dd82547fc03334bda50341a501?refName=refs%2Fheads%2Fmain:
- Changes to a feature name in the data and default configs: https://vestas.visualstudio.com/Neuron/_git/neuron-ml/commit/47581eb743930a0ed16cd7228cd3965766c2ac0c?refName=refs%2Fheads%2Fmain
- The addition of a postprocessor in the code and schema: https://vestas.visualstudio.com/Neuron/_git/neuron-ml/commit/e533b5a565ee885920b3f78a39ac1eb053c8bc2a?refName=refs%2Fheads%2Fmain
"""
from copy import deepcopy
from typing import List


def _fix_single_feature_attributes(feature: dict) -> None:
    if feature["name"] == "phi_h":
        feature["name"] = "slope"
    feature["feature_value_type"] = feature.pop("feature_type")
    feature["feature_type"] = feature.pop("feature_construction")
    feature["is_model_input"] = True
    if feature["feature_type"] == "engineered":
        del feature["safe_domain_method"]


def _add_wind_gradient_features_if_missing(feature_list: List[dict]) -> None:
    feature_names = [f["name"] for f in feature_list]
    if "wnd_grad" not in feature_names:
        # This is an error in the previous run. We are modifying such a run by assuming that
        # the wind gradient feature was meant to be included.
        feature_list.append(
            {
                "name": "wnd_grad",
                "feature_type": "engineered",
                "feature_value_type": "continuous",
                "is_model_input": True,
            }
        )
    if "vexp" not in feature_names:
        feature_list.append(
            {
                "name": "vexp",
                "feature_type": "raw",
                "feature_value_type": "continuous",
                "is_model_input": False,
                "safe_domain_method": "range_validation",
            }
        )
    if "ws" not in feature_names:
        feature_list.append(
            {
                "name": "ws",
                "feature_type": "raw",
                "feature_value_type": "continuous",
                "is_model_input": False,
                "safe_domain_method": "range_validation",
            }
        )
    if "twr_hh" not in feature_names:
        feature_list.append(
            {
                "name": "twr_hh",
                "feature_type": "raw",
                "feature_value_type": "continuous",
                "is_model_input": False,
                "safe_domain_method": "range_validation",
            }
        )


def _change_model_to_deep_gpr(load_case: dict) -> None:
    """Change the model to deep_gpr for all load case runs"""
    load_case["load_case_model"] = {"name": "deep_gpr", "params": {}}


def migrate_train_config(train_config: dict) -> dict:
    train_config = deepcopy(train_config)
    for load_case in train_config["load_case_training_runs"]:
        _change_model_to_deep_gpr(load_case)

        for f in load_case["feature_list"]["features"]:
            _fix_single_feature_attributes(f)
        if load_case["preprocessor"] == "wind_gradient":
            _add_wind_gradient_features_if_missing(load_case["feature_list"]["features"])
        del load_case["preprocessor"]

        if "fat" in load_case["feature_list"]["name"]:
            load_case["postprocessor"] = "directional_twr"
        else:
            load_case["postprocessor"] = "no_postprocessor"
    return train_config
