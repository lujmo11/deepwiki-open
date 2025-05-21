# test_wind_gradient_preprocessor.py

import pandas as pd
import pytest

from neuron.data_validation.exceptions import DataValidationError
from neuron.preprocessing.wind_gradient import TurbineVariant, WindGradientPreprocessor


@pytest.fixture
def preprocessor() -> WindGradientPreprocessor:
    """Fixture to create a WindGradientPreprocessor instance."""
    return WindGradientPreprocessor(
        turbine_variant=TurbineVariant(rotor_diameter=150, rated_power=5000, mk_version="test")
    )


def test_preprocess_with_wnd_grad(preprocessor: WindGradientPreprocessor) -> None:
    """Test the preprocess method when 'wnd_grad' is provided in the DataFrame."""
    data = {
        "twr_hh": [80],
        "ws": [10],
        "wnd_grad": [0.1],
        "vexp": [0.0],
    }
    dataframe = pd.DataFrame(data)

    # Run the preprocessor
    df_processed = preprocessor.preprocess(dataframe)

    # Check if 'vexp' column is added
    assert "vexp" in df_processed.columns

    # check that wnd_grad takes precedence over vexp
    assert df_processed["wnd_grad"][0] == 0.1


def test_preprocess_without_wnd_grad(preprocessor: WindGradientPreprocessor) -> None:
    """Test the preprocess method when 'wnd_grad' is not provided in the DataFrame."""
    data = {
        "twr_hh": [80],
        "ws": [10],
        "vexp": [0.2],
    }
    df_data = pd.DataFrame(data)
    df_processed = preprocessor.preprocess(df_data)

    # Check if 'wnd_grad' column is added
    assert "wnd_grad" in df_processed.columns

    # Manually calculate expected wnd_grad for comparison
    diameter = preprocessor.turbine.rotor_diameter
    r1 = ((df_data["twr_hh"][0] + diameter / 2) / df_data["twr_hh"][0]) ** df_data["vexp"][0]
    r2 = ((df_data["twr_hh"][0] - diameter / 2) / df_data["twr_hh"][0]) ** df_data["vexp"][0]
    expected_wnd_grad = (r1 - r2) / diameter * df_data["ws"][0]

    assert df_processed["wnd_grad"][0] == pytest.approx(
        expected_wnd_grad
    ), "Calculated wnd_grad does not match expected value."


def test_calculate_vexp_estimate_out_of_range(preprocessor: WindGradientPreprocessor) -> None:
    """Test that DataValidationError is raised when the estimated vexp is unreasonable."""
    twr_hh = 100
    ws = 1
    wnd_grad = 10  # Unreasonable value

    with pytest.raises(DataValidationError) as excinfo:
        preprocessor._calculate_vexp(twr_hh, ws, wnd_grad)
    assert "The wind shear exponent cannot be calculated with the given parameters" in str(
        excinfo.value
    )
