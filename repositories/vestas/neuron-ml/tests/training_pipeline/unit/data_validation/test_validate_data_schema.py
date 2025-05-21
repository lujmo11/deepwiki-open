import pandas as pd
import pandera
import pytest

from neuron.data_validation.validation import validate_data_schema


def test_validate_data_schema_misconfigured_feature_type_error(
    training_df_sample_fat: pd.DataFrame,
) -> None:
    """Use a load case config with a mis-specified feature type.
    This should throw an error when validating the data.
    """

    # Make the wind speed negative. This is not allowed, and should throw an error
    training_df_sample_fat["ws"] *= -1

    with pytest.raises(pandera.errors.SchemaError):
        validate_data_schema(training_df_sample_fat, columns=["ws"])


def test_validate_data_schema_bad_target_error(
    training_df_sample_fat: pd.DataFrame,
) -> None:
    """Use a load case config with a mis-specified target name.
    This should throw an error when validating the data.
    """

    # Make sure the expected target name is not present in the dataframe
    training_df_sample_fat = training_df_sample_fat.rename(
        columns={"MxBldRoot_m1000": "not_a_target_name"}
    )

    with pytest.raises(pandera.errors.SchemaError):
        validate_data_schema(training_df_sample_fat, columns=["MxBldRoot_m1000"])
