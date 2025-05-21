from typing import Dict, List, Union

from pydantic import BaseModel

from neuron.data_validation.table_schemas import Target
from neuron.schemas.domain import LoadCase
from neuron_prediction_service.model_store_service.schemas import TurbineVariantBuildConfig


class Predictions(BaseModel):
    averages: List[float]
    standard_deviations: List[float]
    gradients: Union[Dict[str, List[float]], None] = None


class LoadCaseModelMetadata(BaseModel):
    turbine_variant_build_config: TurbineVariantBuildConfig
    load_case: LoadCase


class PredictionMetadata(BaseModel):
    build_number: str
    turbine_variant_build_config: TurbineVariantBuildConfig
    load_case_name: str


class PredictionServiceResponse(BaseModel):
    predictions: Dict[str, Predictions]
    interpolation_domain_validation: List[int]
    extrapolation_domain_validation: List[int]
    meta: PredictionMetadata


class LoadCaseModelInput(BaseModel):
    data: Dict[str, List[float]]
    targets: Union[List[Target], None] = None
    grad_features: Union[List[str], None] = None


LOAD_CASE_MODEL_INPUT_EXAMPLES = [
    {
        "data": {
            "ws": [10, 10, 10],
            "yaw": [0, 0, 0],
            "rho": [1.225, 1.225, 1.225],
            "slope": [0, 0, 0],
            "twr_frq1": [0.17, 0.17, 0.17],
            "twr_frq2": [1.4, 1.4, 1.4],
            "ti": [0.2, 0.2, 0.2],
            "twr_hh": [130, 130, 130],
            "epb": [0, 0, 0],
            "vtl": [0, 0, 0],
            "eco": [0, 0, 0],
            "power_rating": [0, 0, 0],
            "wnd_grad": [0.0, 0.0, 0.0],
        },
        "targets": ["MyHub_m400", "MxHub_m800"],
        "grad_features": ["ws"],
    }
]
