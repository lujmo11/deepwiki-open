from typing import List, Tuple

from neuron.safe_domain.validators.base import Validator


def get_range_domain(
    feature_x: str,
    feature_y: str,
    safe_domain_x: Validator,
    safe_domain_y: Validator,
    method: str = "interp",
) -> Tuple[List, List]:
    x_range = safe_domain_x._get_ranges(feature_x, method=method)
    y_range = safe_domain_y._get_ranges(feature_y, method=method)

    vertices = [
        (x_range[0], y_range[0]),
        (x_range[1], y_range[0]),
        (x_range[1], y_range[1]),
        (x_range[0], y_range[1]),
        (x_range[0], y_range[0]),
    ]

    x, y = zip(*vertices, strict=False)

    return x, y


def get_hull_domain(
    feature_x: str, feature_y: str, safe_domain: Validator, method: str = "interp"
) -> Tuple[List, List]:
    if method == "interp":
        method_suffix = "dict_interp"
    elif method == "extrap":
        method_suffix = "dict_extrap"

    model_dict = safe_domain.get_model_data()["features_hull_" + method_suffix]

    if (feature_x + "_combined_with_" + feature_y) in model_dict:
        vertices = model_dict[feature_x + "_combined_with_" + feature_y]
        x, y = zip(*vertices, strict=False)
    elif (feature_y + "_combined_with_" + feature_x) in model_dict:
        vertices = model_dict[feature_y + "_combined_with_" + feature_x]
        y, x = zip(*vertices, strict=False)

    return x, y
