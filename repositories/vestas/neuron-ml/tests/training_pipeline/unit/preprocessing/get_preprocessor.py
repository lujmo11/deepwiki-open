import pytest

from neuron.preprocessing.registry import get_preprocessor
from neuron.schemas.domain import REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION, TurbineVariant


def test_all_allowed_engineered_features_have_preprocessors() -> None:
    for engineered_feature_name in REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION:
        try:
            get_preprocessor(
                engineered_feature_name=engineered_feature_name,
                turbine_variant=TurbineVariant(
                    rotor_diameter=150, rated_power="4000", mk_version="mk3e"
                ),
            )
        except ValueError:
            pytest.fail(
                f"Preprocessor for {engineered_feature_name} "
                f"in ALLOWED_ENGINEERED_FEATURE_NAMES is not found in the preprocessor registry."
            )
