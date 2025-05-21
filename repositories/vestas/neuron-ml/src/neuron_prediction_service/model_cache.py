from collections import OrderedDict
from typing import Union

from neuron.models.load_case_model_pipeline import LoadCaseModelPipeline


class ModelCache:
    def __init__(self, model_cache_size: int):
        self._models = OrderedDict()
        self._model_cache_size = model_cache_size

    def get(self, model_id: str) -> Union[LoadCaseModelPipeline, None]:
        return self._models.get(model_id)

    def add(self, model_id: str, model: LoadCaseModelPipeline) -> None:
        if model_id in self._models:
            self._update_model_position_in_cache(model_id=model_id)
        else:
            if len(self._models) >= self._model_cache_size:
                self._remove_oldest_model()
            self._models[model_id] = model

    def _remove_oldest_model(self) -> None:
        self._models.popitem(last=False)

    def _update_model_position_in_cache(self, model_id: str) -> None:
        model = self._models.pop(model_id)
        self._models[model_id] = model
