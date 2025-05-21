from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuron_prediction_service.model_store_service.schemas import LoadCaseModelId


class TurbineVariantNotInModelStore(Exception):
    def __init__(self, turbine_variant_id: str):
        self.turbine_variant_id = turbine_variant_id

    def __str__(self):
        return f"Turbine variant {self.turbine_variant_id} does not exist in model store."


class TurbineVariantVersionNotInModelStore(Exception):
    def __init__(self, turbine_variant_id: str, model_version: int):
        self.turbine_variant_id = turbine_variant_id
        self.model_version = model_version

    def __str__(self):
        return (
            f"Version {self.model_version} for turbine variant "
            f"{self.turbine_variant_id} does not exist in model store."
        )


class LoadCaseModelNotInModelStore(Exception):
    def __init__(self, load_case_model_id: "LoadCaseModelId"):
        self.load_case_model_id = load_case_model_id

    def __str__(self) -> str:
        return f"Load case model with id {self.load_case_model_id} does not exist in model store."


class LoadCaseModelIDNotCorrect(Exception):
    pass
