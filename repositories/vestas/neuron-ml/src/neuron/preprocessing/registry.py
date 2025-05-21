from neuron.preprocessing.base import Preprocessor
from neuron.preprocessing.wind_gradient import WindGradientPreprocessor
from neuron.schemas.domain import TurbineVariant


def get_preprocessor(engineered_feature_name: str, turbine_variant: TurbineVariant) -> Preprocessor:
    """Get preprocessor class for a given engineered feature

    Some preprocessor classes might take the turbine_variant as an argument,
    so we need to pass it to the function.
    """
    preprocessing_registry = {
        "wnd_grad": WindGradientPreprocessor(turbine_variant=turbine_variant),
    }
    try:
        return preprocessing_registry[engineered_feature_name]
    except KeyError as e:
        raise ValueError(
            f"Engineered feature {engineered_feature_name} not found in the registry."
            f"Available features: {list(preprocessing_registry.keys())}"
        ) from e
