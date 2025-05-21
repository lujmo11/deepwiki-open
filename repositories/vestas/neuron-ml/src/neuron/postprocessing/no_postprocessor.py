import logging
from typing import Dict

from neuron.postprocessing.base import Postprocessor
from neuron.schemas.domain import TargetValues

logger = logging.getLogger(__name__)


class NoPostprocessor(Postprocessor):
    name = "no_postprocessor"

    def postprocess(self, dict_target_values: Dict[str, TargetValues]) -> Dict[str, TargetValues]:
        logger.info("Not postprocessing data")
        return dict_target_values
