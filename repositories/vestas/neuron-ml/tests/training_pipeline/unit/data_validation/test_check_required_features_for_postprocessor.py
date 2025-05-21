import pandas as pd
import pytest

from neuron.data_validation.exceptions import DataValidationError
from neuron.data_validation.validation import (
    POSTPROCESSOR_REQUIRED_TARGET_MAP,
    check_required_targets_for_postprocessor,
)
from neuron.schemas.domain import PostprocessorName, Target


def test_check_required_targets_for_postprocessor_happy_path() -> None:
    test_df = pd.DataFrame.from_dict(
        {
            Target.MyTwrBot_Twr_m400.value: [1, 2, 3, 4],
            Target.MxTwrBot_Twr_m400.value: [1, 2, 3, 4],
            Target.MyTwrTop_Twr_m400.value: [1, 2, 3, 4],
            Target.MxTwrTop_Twr_m400.value: [1, 2, 3, 4],
        }
    )
    check_required_targets_for_postprocessor(
        df=test_df, postprocessor_name=PostprocessorName.DIRECTIONAL_TWR, dataset_name="my_df"
    )


def test_check_required_targets_for_postprocessor_throws_error_when_required_features_missing() -> (
    None
):
    test_df = pd.DataFrame.from_dict(
        {
            Target.MyTwrBot_Twr_m400.value: [1, 2, 3, 4],
        }
    )
    with pytest.raises(DataValidationError) as exc_info:
        check_required_targets_for_postprocessor(
            df=test_df, postprocessor_name=PostprocessorName.DIRECTIONAL_TWR, dataset_name="my_df"
        )
    expected_error_message = (
        f"The postprocessor '{PostprocessorName.DIRECTIONAL_TWR.value}' requires the columns "
        f"{POSTPROCESSOR_REQUIRED_TARGET_MAP[PostprocessorName.DIRECTIONAL_TWR.value]} "
        f"but only the columns {[f for f in test_df.columns]} are "
        f"present in the my_df dataset."
    )
    assert str(exc_info.value) == expected_error_message
