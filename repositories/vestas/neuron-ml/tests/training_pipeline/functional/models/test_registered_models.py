from typing import Any, Dict, List

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from neuron.models.scaler_registry import DATA_SCALER_REGISTRY
from neuron.models.target_models.base import Model
from neuron.models.target_models.deep_gpr import DeepGPRPytorch
from neuron.models.target_models.feature_range_gpr import FeatureRangeGPRPytorch
from neuron.models.target_models.gpr import GPRPytorch
from neuron.models.target_models.gpr_sklearn import GPR, KERNEL_CLASS_REGISTRY
from neuron.models.target_models.registry import TARGET_MODEL_REGISTER


def get_model_config(model_class: Model) -> List[Dict[str, Any]]:
    """Given a Model, return a list of initialization parameters to be tested"""
    if model_class == GPR:
        # Test all kernels we have registered
        kernel_test_list = [{"kernel_name": name} for name in KERNEL_CLASS_REGISTRY.keys()]
        # Test all scalers we have registered
        scaler_test_list = [
            {"kernel_name": "Matern", "feature_scaler_name": name, "target_scaler_name": name}
            for name in DATA_SCALER_REGISTRY.keys()
        ]
        full_test_list = kernel_test_list + scaler_test_list
        return full_test_list
    elif model_class == FeatureRangeGPRPytorch:
        # The synthetic features are just named "feature_0", "feature_1" etc
        test_list = [
            {
                "range_feature": "feature_0",
                "range_upper_bounds": [0.0],
                "feature_scaler_name": name,
                "target_scaler_name": name,
            }
            for name in DATA_SCALER_REGISTRY.keys()
        ]
        return test_list
    elif model_class == DeepGPRPytorch:
        return [
            {"n_inducing_points": 20, "feature_scaler_name": name, "target_scaler_name": name}
            for name in DATA_SCALER_REGISTRY.keys()
        ]
    elif model_class == GPRPytorch:
        return [
            {"n_inducing_points": 20, "feature_scaler_name": name, "target_scaler_name": name}
            for name in DATA_SCALER_REGISTRY.keys()
        ]
    else:
        # Default case, return a list with no parameters
        return [{}]


@pytest.mark.slow
@pytest.mark.parametrize(argnames="model_class", argvalues=TARGET_MODEL_REGISTER.values())
def test_registered_model(
    model_class: Model,
    mlflow_artifacts_folder: str,
    training_df_fixed_data: pd.DataFrame,
) -> None:
    """Test all models defined in TARGET_MODEL_REGISTER.

    Test that they can :
        - be initialized, potentially with custom parameters.
        - train with fake data
        - serialize and deserialize
        - predict from the deserialized model
    """

    df_train, df_test = train_test_split(training_df_fixed_data, test_size=0.2, random_state=42)

    # Loop over initialization parameters to test
    for init_params in get_model_config(model_class=model_class):
        # Train the model
        model = model_class(
            features=[f"feature_{i}" for i in range(5)], target_col="target", **init_params
        )
        model.fit(df_train)

        # Save the model
        model.save_model(folder_path=mlflow_artifacts_folder)
        # Load the saved model
        loaded_model = model_class.load_model(folder_path=mlflow_artifacts_folder)

        # Make predictions using the loaded model
        preds = loaded_model.predict(df_test, return_std=False)
        assert len(preds.value_list) == len(df_test), "the number of predictions is not correct"
        assert all(
            isinstance(p, float) for p in preds.value_list
        ), "The predictions should be a float"
