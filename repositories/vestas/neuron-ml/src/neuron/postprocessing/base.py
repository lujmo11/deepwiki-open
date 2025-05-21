from abc import ABC, abstractmethod
from typing import Dict

from neuron.schemas.domain import TargetValues


class Postprocessor(ABC):
    """Base class for postprocessors

    Any prototype class should have this interface.
    """

    name: str

    @abstractmethod
    def postprocess(self, dict_target_values: Dict[str, TargetValues]) -> Dict[str, TargetValues]:
        pass
