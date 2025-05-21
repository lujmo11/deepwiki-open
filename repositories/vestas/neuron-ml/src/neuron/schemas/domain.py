"""Schemas (class definitions) for central domain objects for the Neuron project"""

from enum import StrEnum
from typing import Dict, List, Self, Union

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION = {
    "wnd_grad": ["twr_hh", "vexp", "ws"],
    "vexp": ["twr_hh", "wnd_grad", "ws"],
}


class Target(StrEnum):
    """List of allowed targets across load cases"""

    # Extreme target list
    MxBld_0000_max = "MxBld_0000_max"
    MxBld_0000_min = "MxBld_0000_min"
    MxBld_0250_max = "MxBld_0250_max"
    MxBld_0250_min = "MxBld_0250_min"
    MxBld_0500_max = "MxBld_0500_max"
    MxBld_0500_min = "MxBld_0500_min"
    MxBld_0750_max = "MxBld_0750_max"
    MxBld_0750_min = "MxBld_0750_min"
    MyBld_0000_max = "MyBld_0000_max"
    MyBld_0000_min = "MyBld_0000_min"
    MyBld_0250_max = "MyBld_0250_max"
    MyBld_0250_min = "MyBld_0250_min"
    MyBld_0500_max = "MyBld_0500_max"
    MyBld_0500_min = "MyBld_0500_min"
    MyBld_0750_max = "MyBld_0750_max"
    MyBld_0750_min = "MyBld_0750_min"
    MxHub_abs = "MxHub_abs"
    MyHub_abs = "MyHub_abs"
    MPitch_max = "MPitch_max"
    MPitch_min = "MPitch_min"
    MxMBFix_abs = "MxMBFix_abs"
    MyMBRot_max = "MyMBRot_max"
    MyMBRot_min = "MyMBRot_min"
    MxMBRot_abs = "MxMBRot_abs"
    MzMBFix_abs = "MzMBFix_abs"
    MzMBRot_abs = "MzMBRot_abs"
    FyMBRot_max = "FyMBRot_max"
    FyMBRot_min = "FyMBRot_min"
    MxTwrTop_Nac_abs = "MxTwrTop_Nac_abs"
    MzTwrTop_Nac_abs = "MzTwrTop_Nac_abs"
    MrTwrTop_Nac_abs = "MrTwrTop_Nac_abs"
    MrTwrTop_Twr_abs = "MrTwrTop_Twr_abs"
    MrTwrBot_Twr_max = "MrTwrBot_Twr_max"
    MrTwrBot_Fnd_max = "MrTwrBot_Fnd_max"
    bldefl_max = "bldefl_max"
    MrMB_max = "MrMB_max"
    # "MS" extreme targets
    FyMSRot_max = "FyMSRot_max"
    MrMS_max = "MrMS_max"
    MyMSRot_max = "MyMSRot_max"
    FyMSRot_min = "FyMSRot_min"
    MyMSRot_min = "MyMSRot_min"
    MxMSFix_abs = "MxMSFix_abs"
    MzMSFix_abs = "MzMSFix_abs"
    MxMSRot_abs = "MxMSRot_abs"
    MzMSRot_abs = "MzMSRot_abs"
    # Extreme standard error target list
    MxBld_0000_max_std = "MxBld_0000_max_std"
    MxBld_0000_min_std = "MxBld_0000_min_std"
    MxBld_0250_max_std = "MxBld_0250_max_std"
    MxBld_0250_min_std = "MxBld_0250_min_std"
    MxBld_0500_max_std = "MxBld_0500_max_std"
    MxBld_0500_min_std = "MxBld_0500_min_std"
    MxBld_0750_max_std = "MxBld_0750_max_std"
    MxBld_0750_min_std = "MxBld_0750_min_std"
    MyBld_0000_max_std = "MyBld_0000_max_std"
    MyBld_0000_min_std = "MyBld_0000_min_std"
    MyBld_0250_max_std = "MyBld_0250_max_std"
    MyBld_0250_min_std = "MyBld_0250_min_std"
    MyBld_0500_max_std = "MyBld_0500_max_std"
    MyBld_0500_min_std = "MyBld_0500_min_std"
    MyBld_0750_max_std = "MyBld_0750_max_std"
    MyBld_0750_min_std = "MyBld_0750_min_std"
    MxHub_abs_std = "MxHub_abs_std"
    MyHub_abs_std = "MyHub_abs_std"
    MPitch_max_std = "MPitch_max_std"
    MPitch_min_std = "MPitch_min_std"
    MxMBFix_abs_std = "MxMBFix_abs_std"
    MyMBRot_max_std = "MyMBRot_max_std"
    MyMBRot_min_std = "MyMBRot_min_std"
    MxMBRot_abs_std = "MxMBRot_abs_std"
    MzMBFix_abs_std = "MzMBFix_abs_std"
    MzMBRot_abs_std = "MzMBRot_abs_std"
    FyMBRot_max_std = "FyMBRot_max_std"
    FyMBRot_min_std = "FyMBRot_min_std"
    MxTwrTop_Nac_abs_std = "MxTwrTop_Nac_abs_std"
    MzTwrTop_Nac_abs_std = "MzTwrTop_Nac_abs_std"
    MrTwrTop_Nac_abs_std = "MrTwrTop_Nac_abs_std"
    MrTwrTop_Twr_abs_std = "MrTwrTop_Twr_abs_std"
    MrTwrBot_Twr_max_std = "MrTwrBot_Twr_max_std"
    MrTwrBot_Fnd_max_std = "MrTwrBot_Fnd_max_std"
    bldefl_max_std = "bldefl_max_std"
    # "MS" extreme standard error targets
    FyMSRot_max_std = "FyMSRot_max_std"
    MrMS_max_std = "MrMS_max_std"
    MyMSRot_max_std = "MyMSRot_max_std"
    FyMSRot_min_std = "FyMSRot_min_std"
    MyMSRot_min_std = "MyMSRot_min_std"
    MxMSFix_abs_std = "MxMSFix_abs_std"
    MzMSFix_abs_std = "MzMSFix_abs_std"
    MxMSRot_abs_std = "MxMSRot_abs_std"
    MzMSRot_abs_std = "MzMSRot_abs_std"
    # Fatigue target list
    MxBldRoot_m1000 = "MxBldRoot_m1000"
    MxBldMid_m1000 = "MxBldMid_m1000"
    MyBldRoot_m1000 = "MyBldRoot_m1000"
    MyBldMid_m1000 = "MyBldMid_m1000"
    MxHub_m400 = "MxHub_m400"
    MxHub_m800 = "MxHub_m800"
    MyHub_m400 = "MyHub_m400"
    MyHub_m800 = "MyHub_m800"
    MxHub_LDD_m300 = "MxHub_LDD_m300"
    MxHub_LDD_m330 = "MxHub_LDD_m330"
    MyMbRot_m400 = "MyMbRot_m400"
    MyMbRot_m800 = "MyMbRot_m800"
    MyMbRot_LRD_m330 = "MyMbRot_LRD_m330"
    MyMbRot_LRD_m570 = "MyMbRot_LRD_m570"
    MyMbRot_LRD_m870 = "MyMbRot_LRD_m870"
    MxMbRot_m400 = "MxMbRot_m400"
    MxMbRot_m800 = "MxMbRot_m800"
    MzMbRot_m400 = "MzMbRot_m400"
    MzMbRot_m800 = "MzMbRot_m800"
    MxMbFix_m400 = "MxMbFix_m400"
    MxMbFix_m800 = "MxMbFix_m800"
    MzMbFix_m400 = "MzMbFix_m400"
    MzMbFix_m800 = "MzMbFix_m800"
    MxTwrTop_Nac_m400 = "MxTwrTop_Nac_m400"
    MxTwrTop_Nac_m800 = "MxTwrTop_Nac_m800"
    MzTwrTop_Nac_m400 = "MzTwrTop_Nac_m400"
    MzTwrTop_Nac_m800 = "MzTwrTop_Nac_m800"
    MxTwrTop_Twr_m400 = "MxTwrTop_Twr_m400"
    MyTwrTop_Twr_m400 = "MyTwrTop_Twr_m400"
    MzTwrTop_Twr_m400 = "MzTwrTop_Twr_m400"
    MxTwrBot_Twr_m400 = "MxTwrBot_Twr_m400"
    MyTwrBot_Twr_m400 = "MyTwrBot_Twr_m400"
    MxTwrBot_Fnd_m400 = "MxTwrBot_Fnd_m400"
    MxTwrBot_Fnd_m800 = "MxTwrBot_Fnd_m800"
    MyTwrBot_Fnd_m400 = "MyTwrBot_Fnd_m400"
    MyTwrBot_Fnd_m800 = "MyTwrBot_Fnd_m800"
    Fytt_m400 = "Fytt_m400"
    Fytt_m800 = "Fytt_m800"
    # Post-processed fatigue target list
    Mr1TwrTop_Twr_m400 = "Mr1TwrTop_Twr_m400"
    Mr2TwrTop_Twr_m400 = "Mr2TwrTop_Twr_m400"
    Mr3TwrTop_Twr_m400 = "Mr3TwrTop_Twr_m400"
    Mr1TwrBot_Twr_m400 = "Mr1TwrBot_Twr_m400"
    Mr2TwrBot_Twr_m400 = "Mr2TwrBot_Twr_m400"
    Mr3TwrBot_Twr_m400 = "Mr3TwrBot_Twr_m400"
    # Performance target list
    Power_mean = "Power_mean"
    # Fatigue standard error target list
    MxBldRoot_m1000_std = "MxBldRoot_m1000_std"
    MyBldRoot_m1000_std = "MyBldRoot_m1000_std"
    MxBldMid_m1000_std = "MxBldMid_m1000_std"
    MyBldMid_m1000_std = "MyBldMid_m1000_std"
    MxHub_m400_std = "MxHub_m400_std"
    MxHub_m800_std = "MxHub_m800_std"
    MyHub_m400_std = "MyHub_m400_std"
    MyHub_m800_std = "MyHub_m800_std"
    MxMbRot_m400_std = "MxMbRot_m400_std"
    MxMbRot_m800_std = "MxMbRot_m800_std"
    MyMbRot_m400_std = "MyMbRot_m400_std"
    MyMbRot_m800_std = "MyMbRot_m800_std"
    MzMbRot_m400_std = "MzMbRot_m400_std"
    MzMbRot_m800_std = "MzMbRot_m800_std"
    MxMbFix_m400_std = "MxMbFix_m400_std"
    MxMbFix_m800_std = "MxMbFix_m800_std"
    MzMbFix_m400_std = "MzMbFix_m400_std"
    MzMbFix_m800_std = "MzMbFix_m800_std"
    MxTwrTop_Nac_m400_std = "MxTwrTop_Nac_m400_std"
    MxTwrTop_Nac_m800_std = "MxTwrTop_Nac_m800_std"
    MzTwrTop_Nac_m400_std = "MzTwrTop_Nac_m400_std"
    MzTwrTop_Nac_m800_std = "MzTwrTop_Nac_m800_std"
    MxTwrTop_Twr_m400_std = "MxTwrTop_Twr_m400_std"
    MzTwrTop_Twr_m400_std = "MzTwrTop_Twr_m400_std"
    MyTwrTop_Twr_m400_std = "MyTwrTop_Twr_m400_std"
    Fytt_m400_std = "Fytt_m400_std"
    Fytt_m800_std = "Fytt_m800_std"
    MxTwrBot_Twr_m400_std = "MxTwrBot_Twr_m400_std"
    MyTwrBot_Twr_m400_std = "MyTwrBot_Twr_m400_std"
    MxTwrBot_Fnd_m400_std = "MxTwrBot_Fnd_m400_std"
    MxTwrBot_Fnd_m800_std = "MxTwrBot_Fnd_m800_std"
    MyTwrBot_Fnd_m400_std = "MyTwrBot_Fnd_m400_std"
    MyTwrBot_Fnd_m800_std = "MyTwrBot_Fnd_m800_std"
    MyMbRot_LRD_m330_std = "MyMbRot_LRD_m330_std"
    MyMbRot_LRD_m570_std = "MyMbRot_LRD_m570_std"
    MyMbRot_LRD_m870_std = "MyMbRot_LRD_m870_std"
    MxHub_LDD_m300_std = "MxHub_LDD_m300_std"
    MxHub_LDD_m330_std = "MxHub_LDD_m330_std"
    # Performance standard error
    Power_mean_std = "Power_mean_std"


POSTPROCESSOR_REQUIRED_TARGETS_MAP = {
    "directional_twr": [
        Target.MyTwrTop_Twr_m400,
        Target.MxTwrTop_Twr_m400,
        Target.MyTwrBot_Twr_m400,
        Target.MxTwrBot_Twr_m400,
    ]
}


class TargetList(BaseModel, frozen=True):
    """TargetList list metadata

    A target list has a name and a list of targets. The targets are the targets that we want to
    predict.
    """

    name: str
    targets: List[Target]


class SafeDomainGroup(StrEnum):
    """Enum used for grouping features which enables assigning a group
    to a specific safe domain validator"""

    RANGE_VALIDATION = "range_validation"
    DISCRETE_VALIDATION = "discrete_validation"
    TOWER_FEATURES_VALIDATION = "twr_features_validation"
    WS_TI_VALIDATION = "ws_ti_validation"


class CalculationType(StrEnum):
    """Estimation method for each load case.

    The calculation type defines how the load case evaluation is carried out and is
    linked to the expected target list defined for the load case.
    """

    FATIGUE = "fatigue"
    EXTREME = "extreme"


class Metric(StrEnum):
    # Standard Metrics
    E_MEAN_NORM = "e_mean_norm"
    E_STD_NORM = "e_std_norm"
    E_MAX_NORM = "max_norm_error"
    MAE = "mae"
    MAE_NORM = "mae_norm"
    MANE = "mane"
    R2 = "r2"
    # Coverage Metrics
    PRED_COVERAGE_ERROR_1STD = "pred_coverage_error_1std"
    PRED_COVERAGE_ERROR_2STD = "pred_coverage_error_2std"
    # Statistical test derived metrics
    CHI_SQUARED = "chi_squared"
    CRITICAL_VALUE = "critical_value"
    SIGMA_MODEL_FORM = "sigma_model_form"
    COV_MODEL_FORM = "cov_model_form"
    CHI_SQUARED_PASSED = "chi_squared_passed"
    # MISC
    DATA_COUNT = "data_count"

    # Return the string value of the enum
    def __str__(self):
        return self.value


class FeatureValueType(StrEnum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class FeatureType(StrEnum):
    """The type of the feature

    A feature can be either raw or engineered.
    - A Raw feature is available directly in the training data.
    - An Engineered feature is a feature that is calculated from raw features. If the feature
    allready exists in the feature space, it is not calculated/enigneered.
    """

    RAW = "raw"
    ENGINEERED = "engineered"


class Feature(BaseModel, frozen=True):
    """A feature for a load case model.

    If the feature has feature_type `engineered`, it will be calculated from the raw features. Only
    certain engineered features are allowed, and they require specific raw features to be present.

    If the feature has `model_input` set to True, it will be used directly as input to the model.
    If the feature has `model_input` set to False, it will only be used for calculating other
    engineered features.
    """

    name: str
    feature_type: FeatureType
    feature_value_type: FeatureValueType
    is_model_input: bool
    safe_domain_method: Union[SafeDomainGroup, None] = None
    extrapolation_domain_offset: Union[float, None] = None

    @model_validator(mode="after")
    def validate_engineered_feature_name(self) -> Self:
        """Validate that the name of an engineered feature is allowed"""
        if (
            self.feature_type == FeatureType.ENGINEERED
            and self.name not in REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION.keys()
        ):
            raise ValueError(
                f"Feature name {self.name} is not allowed for feature type `engineered`. "
                f"Allowed names are: {REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION.keys()}."
            )
        return self

    @model_validator(mode="after")
    def validate_model_input_and_feature_type(self) -> Self:
        if self.feature_type == FeatureType.ENGINEERED and not self.is_model_input:
            raise ValueError("A feature of type engineered can only be model input")
        return self

    @model_validator(mode="after")
    def validate_safe_domain_method(self) -> Self:
        if self.feature_type == FeatureType.ENGINEERED and self.safe_domain_method is not None:
            raise ValueError("A feature of type engineered can not have a safe_domain_method.")
        if self.feature_type == FeatureType.RAW and self.safe_domain_method is None:
            raise ValueError("A feature of type raw needs to have a safe_domain_method.")
        return self

    @model_validator(mode="after")
    def check_consistency_feature_type_safe_domain(self) -> Self:
        if (
            self.feature_value_type == FeatureValueType.DISCRETE
            and self.safe_domain_method != SafeDomainGroup.DISCRETE_VALIDATION
        ):
            raise ValueError(
                f"Feature {self.name} does not have the correct assignment of "
                "the safe domain method. "
                "Discrete feature value types require the discrete_validation method."
            )

        if (
            self.safe_domain_method == SafeDomainGroup.DISCRETE_VALIDATION
            and self.feature_value_type != FeatureValueType.DISCRETE
        ):
            raise ValueError(
                f"Feature {self.name} does not have the correct assignment of "
                "the safe domain method."
                f"{self.feature_value_type} features cannot have "
                f"the discrete_validation method assigned."
            )

        if (
            self.feature_value_type == FeatureValueType.DISCRETE
            and self.extrapolation_domain_offset is not None
        ):
            raise ValueError(
                f"Feature {self.name} has the feature type set to discrete, but the"
                " extrapolation_domain_offset is not None."
                " An extrapolation_domain_offset is not allowed for discrete features."
            )

        return self


class AggregationMethod(BaseModel, frozen=True):
    """A method to be used to aggregate data"""

    groupby: str
    weightby: str


class PostprocessorName(StrEnum):
    """Postprocessors available for a load case"""

    NO_POSTPROCESSOR = "no_postprocessor"
    DIRECTIONAL_TWR = "directional_twr"


class FeatureList(BaseModel, frozen=True):
    """Feature list metadata

    A feature list has a name and a list of features.
    """

    name: str
    features: List[Feature]

    @model_validator(mode="after")
    def validate_required_features_for_preprocessing(self) -> Self:
        """Validate that the required raw features are present for all engineered feature"""
        engineered_feature_names = [
            f.name for f in self.features if f.feature_type == FeatureType.ENGINEERED
        ]
        raw_feature_names = [f.name for f in self.features if f.feature_type == FeatureType.RAW]
        for engineered_feature_name in engineered_feature_names:
            required_raw_features = REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION[engineered_feature_name]
            if not all(f in raw_feature_names for f in required_raw_features):
                raise ValueError(
                    f"Engineered feature {engineered_feature_name} requires the following "
                    f"raw features: {required_raw_features}."
                )
        return self

    @model_validator(mode="after")
    def validate_all_non_model_inputs_required(self) -> Self:
        """Validate that any raw features with model_input=False are required for at least
        one engineered feature
        """
        engineered_feature_names = [
            f.name for f in self.features if f.feature_type == FeatureType.ENGINEERED
        ]
        required_for_engineering = set(
            [
                raw_feature
                for feature in engineered_feature_names
                for raw_feature in REQUIRED_INPUT_FOR_FEATURE_CONSTRUCTION[feature]
            ]
        )
        non_model_features = [
            f for f in self.features if f.feature_type == FeatureType.RAW and not f.is_model_input
        ]
        for non_model_feature in non_model_features:
            if non_model_feature.name not in required_for_engineering:
                raise ValueError(
                    f"Raw feature {non_model_feature.name} is not used as model input "
                    "and is not required for any engineered feature."
                )
        return self


class LoadCase(BaseModel, frozen=True):
    """Load case metadata"""

    name: str
    feature_list: FeatureList
    target_list: TargetList
    postprocessor: PostprocessorName

    @model_validator(mode="after")
    def validate_required_targets_for_postprocessor(self) -> Self:
        """Validate that the required targets for the postprocessor are in target_list"""
        if self.postprocessor in POSTPROCESSOR_REQUIRED_TARGETS_MAP:
            required_targets = POSTPROCESSOR_REQUIRED_TARGETS_MAP[self.postprocessor]
            target_set = set(self.target_list.targets)
            missing_targets = [target for target in required_targets if target not in target_set]
            if missing_targets:
                raise ValueError(
                    f"The following required targets for the postprocessor '{self.postprocessor}' "
                    f"are missing from the target_list: {missing_targets}"
                )
        return self

    def get_feature_name_list(self) -> List[str]:
        return [feature.name for feature in self.feature_list.features]


class TurbineVariant(BaseModel, frozen=True):
    rotor_diameter: int
    rated_power: int
    mk_version: str

    def __repr__(self):
        return self.id

    @field_validator("mk_version")
    @classmethod
    def _transform_mk_version(cls, v: str):
        v = v.lower()
        return v

    @property
    def id(self) -> str:
        return f"{self.rotor_diameter}_{self.rated_power}_{self.mk_version}"

    @classmethod
    def from_id(cls, turbine_id: str) -> Self:
        turbine_id_parts = turbine_id.split("_")
        if len(turbine_id_parts) != 3:
            raise ValueError(
                f"Could not parse turbine id {turbine_id}. Expected 3 parts separated by '_'."
            )
        return cls(
            rotor_diameter=turbine_id_parts[0],
            rated_power=turbine_id_parts[1],
            mk_version=turbine_id_parts[2],
        )


class TargetValues(BaseModel, frozen=True):
    """
    Class for storing target values and optional aggregation attributes.
    Used for both predictions and actuals
    """

    target_name: str
    value_list: List[float]
    value_list_std: Union[List[float], None] = None
    gradients_dict: Union[Dict[str, List[float]], None] = None

    groupby: Union[List[Union[int, str]], None] = None
    weightby: Union[List[float], None] = None

    @property
    def values_as_np(self) -> np.ndarray:
        return np.array(self.value_list).reshape(-1, 1)

    @property
    def values_std_as_np(self) -> np.ndarray:
        if self.value_list_std:
            return np.array(self.value_list_std).reshape(-1, 1)
        return None


# Predictions for a load case consists of a dictionary mapping targets to TargetValues
LoadCasePredictions = Dict[str, TargetValues]
# Actuals for a load case consists of a dictionary mapping targets to TargetValues
LoadCaseActuals = Dict[str, TargetValues]
# Feature values for a load case consists of a dictionary mapping features to numpy arrays
LoadCaseFeatureValues = Dict[Feature, np.ndarray]

# Predictions for a turbine_variant consists of a dictionary mapping
# load case names to LoadCasePredictions
TurbinePredictions = Dict[str, LoadCasePredictions]
# Actuals for a turbine_variant consists of a dictionary mapping load case names to LoadCaseActuals
TurbineActuals = Dict[str, LoadCaseActuals]
