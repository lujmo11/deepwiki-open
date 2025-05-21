import pandas as pd
import pytest

from neuron.safe_domain.validators.exceptions import ValidatorFittingError
from neuron.safe_domain.validators.hull_validator import HullValidator
from neuron.schemas.domain import Feature, FeatureType, FeatureValueType, SafeDomainGroup


def test_hull_validator() -> None:
    df_test = pd.DataFrame({"x": [1, 3, 4, 5, 5, 4, 3, 2, 1], "y": [1, 3, 2, 1, 5, 6, 5, 2, 1]})

    features = [
        Feature(
            name="x",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.WS_TI_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
        Feature(
            name="y",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.WS_TI_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
    ]

    validator = HullValidator()
    validator.fit(df_test, features)

    test_points = pd.DataFrame({"x": [1, 2, 3, 3, 6], "y": [1, 2, 4, 2, 6]})

    results_interp = validator.validate_interpolation(test_points)
    results_extrap = validator.validate_extrapolation(test_points)

    print(results_interp)

    assert results_interp.iloc[0, 0] == 1  # point equal to a vertex
    assert results_interp.iloc[1, 0] == 1  # point exactly on the line between two vertices
    assert results_interp.iloc[2, 0] == 1
    assert results_interp.iloc[3, 0] == 1
    assert results_interp.iloc[4, 0] == 0

    # when the extrapolation_domain_offset is not defined, the extrapolation
    # results should be the same as the interpolation results
    assert results_extrap.iloc[0, 0] == 1  # point equal to a vertex
    assert results_extrap.iloc[1, 0] == 1  # point exactly on the line between two vertices
    assert results_extrap.iloc[2, 0] == 1
    assert results_extrap.iloc[3, 0] == 1
    assert results_extrap.iloc[4, 0] == 0

    # For visual confirmation, uncomment the below
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.plot(df_test['x'], df_test['y'], 'o')
    # for vertices in validator.features_hull_dict_interp.values():
    #     vertices = np.array(vertices)
    #     plt.plot(vertices[:, 0], vertices[:, 1], 'r--')
    #     plt.plot([vertices[0, 0], vertices[-1, 0]], [vertices[0, 1], vertices[-1, 1]], 'r--')
    # plt.plot(test_points['x'], test_points['y'], 'go')
    # plt.savefig('test_convex_hull.png')


def test_hull_validator_extrap_domain() -> None:
    df_test = pd.DataFrame({"x": [1, 3, 4, 5, 5, 4, 3, 2, 1], "y": [1, 3, 2, 1, 5, 6, 5, 2, 1]})

    features = [
        Feature(
            name="x",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.WS_TI_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
            extrapolation_domain_offset=1,
        ),
        Feature(
            name="y",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.WS_TI_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
            extrapolation_domain_offset=2,
        ),
    ]

    validator = HullValidator()
    validator.fit(df_test, features)

    test_points = pd.DataFrame({"x": [0, 2, 3, 3, 20], "y": [1, -1, 4, 0, 20]})

    results_extrap = validator.validate_extrapolation(test_points)

    assert results_extrap.iloc[0, 0] == 1  # point equal to a vertex
    assert results_extrap.iloc[1, 0] == 1  # point exactly on the line between two vertices
    assert results_extrap.iloc[2, 0] == 1
    assert results_extrap.iloc[3, 0] == 1
    assert results_extrap.iloc[4, 0] == 0

    # # For visual confirmation, uncomment the below
    # import matplotlib.pyplot as plt
    # import numpy as np

    # plt.plot(df_test["x"], df_test["y"], "o")
    # for vertices in validator.features_hull_dict_interp.values():
    #     vertices = np.array(vertices)
    #     plt.plot(vertices[:, 0], vertices[:, 1], "r--")
    #     plt.plot([vertices[0, 0], vertices[-1, 0]], [vertices[0, 1], vertices[-1, 1]], "r--")
    # for vertices in validator.features_hull_dict_extrap.values():
    #     vertices = np.array(vertices)
    #     plt.plot(vertices[:, 0], vertices[:, 1], "r--")
    #     plt.plot([vertices[0, 0], vertices[-1, 0]], [vertices[0, 1], vertices[-1, 1]], "r--")
    # plt.plot(test_points["x"], test_points["y"], "go")
    # plt.savefig("test_convex_hull.png")


def test_correct_alpha_parameter_assignment() -> None:
    df_test = pd.DataFrame({"x": [1, 3, 4, 5, 5, 4, 3, 2, 1], "y": [1, 3, 2, 1, 5, 6, 5, 2, 1]})
    test_points = pd.DataFrame({"x": [2, 3, 4, 5], "y": [3, 1, 1, 4]})

    features = [
        Feature(
            name="x",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.WS_TI_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
        Feature(
            name="y",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.WS_TI_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
    ]

    validator_convex_hull = HullValidator(alpha_parameter=0.0)
    validator_convex_hull.fit(df_test, features)
    results_cv_interp = validator_convex_hull.validate_interpolation(test_points)

    validator_alpha = HullValidator(alpha_parameter=0.6)
    validator_alpha.fit(df_test, features)
    results_alpha_interp = validator_alpha.validate_interpolation(test_points)

    assert (results_cv_interp.iloc[0, 0] == 1) & (results_alpha_interp.iloc[0, 0] == 0)
    assert (results_cv_interp.iloc[1, 0] == 1) & (results_alpha_interp.iloc[1, 0] == 0)
    assert (results_cv_interp.iloc[2, 0] == 1) & (results_alpha_interp.iloc[2, 0] == 0)
    assert (results_cv_interp.iloc[3, 0] == 1) & (results_alpha_interp.iloc[3, 0] == 0)


def test_that_multi_polygon_raises_error() -> None:
    df_test = pd.DataFrame({"x": [1, 3, 4, 5, 5, 4, 3, 2, 1], "y": [1, 3, 2, 1, 5, 6, 5, 2, 1]})
    validator = HullValidator(alpha_parameter=0.7)

    features = [
        Feature(
            name="x",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.WS_TI_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
        Feature(
            name="y",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.WS_TI_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
    ]

    with pytest.raises(ValidatorFittingError):
        validator.fit(df_test, features)
