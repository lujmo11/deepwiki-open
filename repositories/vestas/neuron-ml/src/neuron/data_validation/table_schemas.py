import logging
from typing import Dict

import pandera as pa
from pandera import Check, Column, Float, Int

from neuron.data_validation.exceptions import DataValidationError
from neuron.schemas.domain import Target

logger = logging.getLogger(__name__)

FEATURE_COLUMN_REGISTRY_BASE: Dict[str, Column] = {
    "ws": Column(
        Float,
        coerce=True,
        nullable=False,
        checks=[Check(lambda s: s >= 0), Check(lambda s: s < 200)],
    ),
    "iref": Column(
        Float,
        coerce=True,
        nullable=False,
        checks=[Check(lambda s: s >= 0), Check(lambda s: s < 10)],
    ),
    "ti": Column(
        Float,
        coerce=True,
        nullable=False,
        checks=[Check(lambda s: s >= 0), Check(lambda s: s < 10)],
    ),
    "slope": Column(
        Float,
        coerce=True,
        nullable=False,
        checks=[Check(lambda s: s > -90), Check(lambda s: s < 90)],
    ),
    "vexp": Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s > -5), Check(lambda s: s < 5)]
    ),
    "yaw": Column(
        Float,
        coerce=True,
        nullable=False,
        checks=[Check(lambda s: s > -360), Check(lambda s: s < 360)],
    ),
    "rho": Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0), Check(lambda s: s < 5)]
    ),
    "wnd_grad": Column(Float, coerce=True, nullable=False, checks=[]),
    "twr_frq1": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    "twr_frq2": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    "twr_hh": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 20)]),
}
FEATURE_COLUMN_REGISTRY_OLE = {
    "MxBld_0000_std": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    "Power_mean": Column(Float, coerce=True, nullable=False, checks=[]),
    "Power_std": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    "Pitch_mean": Column(Float, coerce=True, nullable=False, checks=[]),
    "Pitch_std": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    "AccX_std": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    "AccY_std": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    "Fthr_mean": Column(Float, coerce=True, nullable=False, checks=[]),
    "PTD_m100": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
}
FEATURE_COLUMN_REGISTRY_DLC_VARIATIONS = {
    "horz_grad": Column(Float, coerce=True, nullable=False, checks=[]),
    "vert_grad": Column(Float, coerce=True, nullable=False, checks=[]),
    "yaw_offset": Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s <= 360)]),
    "eog_trigger_time": Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    "coherence_category": Column(Int, coerce=False, nullable=False, checks=[]),
}
FEATURE_COLUMN_REGISTRY_CUSTOM = {
    "epb": Column(Int, coerce=False, nullable=False, checks=[]),
    "vtl": Column(Int, coerce=False, nullable=False, checks=[]),
    "eco": Column(Int, coerce=False, nullable=False, checks=[]),
    "power_rating": Column(Int, coerce=True, nullable=False, checks=[]),
    "rotor_diameter": Column(Float, coerce=True, nullable=False, checks=[]),
}
TARGET_COLUMN_REGISTRY_FATIGUE: Dict[str, Column] = {
    Target.MxBldRoot_m1000: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBldRoot_m1000: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxBldMid_m1000: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBldMid_m1000: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxHub_m400: Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    Target.MxHub_m800: Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    Target.MyHub_m400: Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    Target.MyHub_m800: Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    Target.MxMbRot_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxMbRot_m800: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_m800: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMbRot_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMbRot_m800: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxMbFix_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxMbFix_m800: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMbFix_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMbFix_m800: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrTop_Nac_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrTop_Nac_m800: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzTwrTop_Nac_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzTwrTop_Nac_m800: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrTop_Twr_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzTwrTop_Twr_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.Fytt_m400: Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    Target.Fytt_m800: Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    Target.MxTwrBot_Twr_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyTwrTop_Twr_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyTwrBot_Twr_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrBot_Fnd_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrBot_Fnd_m800: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyTwrBot_Fnd_m400: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyTwrBot_Fnd_m800: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_LRD_m330: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_LRD_m570: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_LRD_m870: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxHub_LDD_m300: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxHub_LDD_m330: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    # Std
    Target.MxBldRoot_m1000_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBldRoot_m1000_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxBldMid_m1000_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBldMid_m1000_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxHub_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxHub_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyHub_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyHub_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxMbRot_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxMbRot_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMbRot_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMbRot_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxMbFix_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxMbFix_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMbFix_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMbFix_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrTop_Nac_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrTop_Nac_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzTwrTop_Nac_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzTwrTop_Nac_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrTop_Twr_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzTwrTop_Twr_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.Fytt_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.Fytt_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrBot_Twr_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyTwrTop_Twr_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyTwrBot_Twr_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrBot_Fnd_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrBot_Fnd_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyTwrBot_Fnd_m400_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyTwrBot_Fnd_m800_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_LRD_m330_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_LRD_m570_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMbRot_LRD_m870_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxHub_LDD_m300_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxHub_LDD_m330_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
}
TARGET_COLUMN_REGISTRY_EXTREME = {
    Target.MxBld_0000_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxBld_0000_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxBld_0250_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxBld_0250_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxBld_0500_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxBld_0500_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxBld_0750_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxBld_0750_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyBld_0000_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyBld_0000_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyBld_0250_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyBld_0250_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyBld_0500_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyBld_0500_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyBld_0750_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyBld_0750_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxHub_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyHub_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MPitch_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MPitch_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxMBFix_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyMBRot_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MyMBRot_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxMBRot_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MzMBFix_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MzMBRot_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.FyMBRot_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.FyMBRot_min: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MxTwrTop_Nac_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MzTwrTop_Nac_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MrTwrTop_Nac_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MrTwrTop_Twr_abs: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MrTwrBot_Twr_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.MrTwrBot_Fnd_max: Column(Float, coerce=True, nullable=False, checks=[]),
    Target.bldefl_max: Column(Float, coerce=True, nullable=False, checks=[]),
    # Std
    Target.MxBld_0000_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxBld_0000_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxBld_0250_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxBld_0250_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxBld_0500_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxBld_0500_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxBld_0750_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxBld_0750_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBld_0000_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBld_0000_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBld_0250_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBld_0250_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBld_0500_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBld_0500_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBld_0750_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyBld_0750_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxHub_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyHub_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MPitch_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MPitch_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxMBFix_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMBRot_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MyMBRot_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxMBRot_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMBFix_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzMBRot_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.FyMBRot_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.FyMBRot_min_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MxTwrTop_Nac_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MzTwrTop_Nac_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MrTwrTop_Nac_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MrTwrTop_Twr_abs_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MrTwrBot_Twr_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.MrTwrBot_Fnd_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
    Target.bldefl_max_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
}

TARGET_COLUMN_REGISTRY_PERFORMANCE = {
    Target.Power_mean: Column(Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]),
    Target.Power_mean_std: Column(
        Float, coerce=True, nullable=False, checks=[Check(lambda s: s >= 0)]
    ),
}


def create_schema_from_columns(columns: list[str]) -> pa.DataFrameSchema:
    base_schema = pa.DataFrameSchema({}, strict="filter")

    combined_registry = FEATURE_COLUMN_REGISTRY_BASE.copy()
    combined_registry.update(FEATURE_COLUMN_REGISTRY_OLE)
    combined_registry.update(FEATURE_COLUMN_REGISTRY_DLC_VARIATIONS)
    combined_registry.update(FEATURE_COLUMN_REGISTRY_CUSTOM)
    combined_registry.update(TARGET_COLUMN_REGISTRY_FATIGUE)
    combined_registry.update(TARGET_COLUMN_REGISTRY_EXTREME)
    combined_registry.update(TARGET_COLUMN_REGISTRY_PERFORMANCE)

    for column_name in columns:
        try:
            base_schema = base_schema.add_columns({column_name: combined_registry[column_name]})
        except KeyError as e:
            raise DataValidationError(f"{column_name} not present in the registry.") from e
    return base_schema
