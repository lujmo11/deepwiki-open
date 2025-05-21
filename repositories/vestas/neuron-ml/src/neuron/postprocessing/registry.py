from neuron.postprocessing.base import Postprocessor
from neuron.postprocessing.directional_twr import DirectionalTwr
from neuron.postprocessing.no_postprocessor import NoPostprocessor
from neuron.schemas.domain import PostprocessorName


def get_postprocessor(
    postprocessor_name: PostprocessorName,
) -> Postprocessor:
    """Get postprocessor class for a given load case.

    If the postprocessor is not found in the registry, we return a NoPostprocessor.
    """
    postrocessing_registry = {
        DirectionalTwr.name: DirectionalTwr(),
    }
    return postrocessing_registry.get(postprocessor_name, NoPostprocessor())
