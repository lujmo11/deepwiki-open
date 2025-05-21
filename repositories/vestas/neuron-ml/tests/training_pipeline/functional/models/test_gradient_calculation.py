import numpy as np
import pandas as pd
import pytest

from neuron.models.target_models.registry import (
    TARGET_MODEL_REGISTER,
    TargetModelInitializationConfig,
    initialize_target_model,
)

models_to_disregard = ["constant_regression", "linear_regression"]
models_to_test = [
    model for model in TARGET_MODEL_REGISTER.keys() if model not in models_to_disregard
]


def generate_sine_data(num_points: int) -> pd.DataFrame:
    x = np.linspace(0, 2 * np.pi, num_points)
    y = np.sin(x)
    y_std = 0.0 * y
    df_d = pd.DataFrame({"x": x, "y": y, "y_std": y_std})
    return df_d


@pytest.mark.slow
@pytest.mark.parametrize(argnames="model_name", argvalues=models_to_test)
def test_model_gradients(model_name: str) -> None:
    """
    Test of gradient calculation accuracy for a single feature model.

    Fitting a model to a sine wave, calculating the model gradients
    and comparing them to the analytical gradients.
    """

    if model_name == "feature_range_gpr":
        params = {"range_feature": "x", "range_upper_bounds": [np.pi]}
    else:
        params = {}

    model_config = TargetModelInitializationConfig(
        name=model_name,
        params=params,
        features=["x"],
        target_col="y",
    )

    model = initialize_target_model(model_config)

    train_df = generate_sine_data(num_points=100)
    test_df = generate_sine_data(num_points=10)

    model.fit(train_df)

    # Calculate analytical gradients
    test_df["analytical_grad"] = np.cos(test_df["x"])

    model_gradients = np.array(model.predict(test_df, return_grads=True).gradients_dict["x"])

    analytical_gradients = test_df["analytical_grad"].to_numpy()

    # removing first and last points to avoid model fit inaccuracy at the edges
    np.testing.assert_allclose(model_gradients[1:-1], analytical_gradients[1:-1], rtol=0.1)

    # looser requirement to gradient accurcy at the edges as model fit is less accurate there
    np.testing.assert_allclose(model_gradients[0], analytical_gradients[0], rtol=0.2)
    np.testing.assert_allclose(model_gradients[-1], analytical_gradients[-1], rtol=0.2)


@pytest.mark.slow
@pytest.mark.parametrize(argnames="model_name", argvalues=models_to_test)
def test_model_gradients_multiple_features(model_name: str) -> None:
    """
    Test of gradient calculation for models with multiple features.

    Fitting a model to a 2D sine surface, calculating the model gradients
    and comparing them to the analytical gradients.

    To keep the test relatively fast, the training data is of small size, so the model fit and
    thereby the gradients are allowed to be less accurate compared to the single feature case.
    """

    if model_name == "feature_range_gpr":
        params = {
            "range_feature": "x",
            "range_upper_bounds": [np.pi * 0.5],
            "n_inducing_points": 40,
            "training_iter": 200,
        }
    elif model_name == "deep_gpr":
        params = {"n_inducing_points": 20, "num_hidden_dims": 2, "training_iter": 200}
    elif model_name == "gpr":
        params = {"n_inducing_points": 40, "training_iter": 200}
    else:
        params = {}

    model_config = TargetModelInitializationConfig(
        name=model_name,
        params=params,
        features=["x", "y"],
        target_col="z",
    )
    model = initialize_target_model(model_config)
    x = np.linspace(0, np.pi, 40)
    y = np.linspace(0, np.pi, 40)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    df_train = pd.DataFrame(
        {"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel(), "z_std": 0.0001 * Z.ravel()}
    )
    model.fit(df_train)
    df_test = df_train.sample(10, random_state=0)

    # Calculate analytical gradients
    analytical_grad_x = np.cos(df_test["x"]) * np.cos(df_test["y"])
    analytical_grad_y = -np.sin(df_test["x"]) * np.sin(df_test["y"])

    # Remove points where analytical gradients are close to zero to avoid large relative errors
    mask_x = abs(analytical_grad_x) > 0.1
    mask_y = abs(analytical_grad_y) > 0.1
    mask = mask_x & mask_y

    assert len(mask) > 5

    analytical_grad_x = analytical_grad_x[mask]
    analytical_grad_y = analytical_grad_y[mask]

    df_test = df_test[mask]
    preds = model.predict(df_test, return_grads=True)

    np.testing.assert_allclose(preds.gradients_dict["x"], analytical_grad_x, rtol=0.1)
    np.testing.assert_allclose(preds.gradients_dict["y"], analytical_grad_y, rtol=0.1)
