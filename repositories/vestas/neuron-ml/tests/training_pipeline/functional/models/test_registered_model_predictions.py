import pickle
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd
import pytest

from neuron.models.target_models.base import Model
from neuron.models.target_models.registry import TARGET_MODEL_REGISTER
from neuron.utils import set_seed
from scripts.generate_fixed_predictions_for_model_testing import model_init_params_to_str

FIXED_PREDICTIONS_FOLDER = "tests/data/training_pipeline/fixed_predictions"
PARAMETER_PREDICTIONS_MAP_FILE_NAME = "parameter_predictions_map.pkl"

model_classes_to_test = TARGET_MODEL_REGISTER.values()


@pytest.mark.parametrize(argnames="model_class", argvalues=model_classes_to_test)
def test_registered_model_predictions(
    model_class: Type[Model],
    training_df_fixed_data: pd.DataFrame,
    model_class_sample_configs_fixture: Dict[Type[Model], List[Dict[str, Any]]],
) -> None:
    """Test that all models defined in TARGET_MODEL_REGISTER make the expected predictions
    for fixed synthetic data. If any changes are made to a model class that should result in new
    predictions, you need to rerun the script
    scripts/generate_fixed_predictions_for_model_testing.py to generate
    the new predictions for the specific model.
    """
    # Arrange
    set_seed(42)
    expected_predictions_file_path = (
        f"tests/data/training_pipeline/fixed_predictions/"
        f"{model_class.name}/parameter_predictions_map.pkl"
    )
    with open(expected_predictions_file_path, "rb") as f:
        expected_parameter_prediction_map = pickle.load(f)

    trained_models = {}
    for init_params in model_class_sample_configs_fixture[model_class]:
        # Load trained model
        trained_models[str(init_params)] = model_class.load_model(
            folder_path=f"{FIXED_PREDICTIONS_FOLDER}/{model_class.name}/{model_init_params_to_str(init_params)}_model"
        )

    # Act / Assert
    for init_params in model_class_sample_configs_fixture[model_class]:
        predictions = trained_models[str(init_params)].predict(
            training_df_fixed_data, return_std=True
        )
        np.testing.assert_allclose(
            predictions.value_list,
            expected_parameter_prediction_map[str(init_params)].value_list,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            predictions.value_list_std,
            expected_parameter_prediction_map[str(init_params)].value_list_std,
            rtol=1e-4,
        )
