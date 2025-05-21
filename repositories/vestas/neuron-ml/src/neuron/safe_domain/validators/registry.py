from typing import Any, Dict, Type

from neuron.safe_domain.validators.base import Validator
from neuron.safe_domain.validators.discrete_validator import DiscreteValidator
from neuron.safe_domain.validators.hull_validator import HullValidator
from neuron.safe_domain.validators.range_validator import RangeValidator
from neuron.schemas.domain import SafeDomainGroup

VALIDATOR_MAP: Dict[SafeDomainGroup, Type[Validator]] = {
    SafeDomainGroup.RANGE_VALIDATION: RangeValidator,
    SafeDomainGroup.DISCRETE_VALIDATION: DiscreteValidator,
    SafeDomainGroup.TOWER_FEATURES_VALIDATION: HullValidator,
    SafeDomainGroup.WS_TI_VALIDATION: HullValidator,
}

VALIDATOR_PARAMS: Dict[SafeDomainGroup, Dict[str, Any]] = {
    SafeDomainGroup.RANGE_VALIDATION: {},
    SafeDomainGroup.DISCRETE_VALIDATION: {},
    SafeDomainGroup.TOWER_FEATURES_VALIDATION: {"alpha_parameter": 0.0},
    SafeDomainGroup.WS_TI_VALIDATION: {"alpha_parameter": 0.04},
}


def get_validator(safe_domain_method: SafeDomainGroup) -> Validator:
    """Gets the registered validator from the safe domain method name."""
    try:
        ValidatorClass = VALIDATOR_MAP[safe_domain_method]
        return ValidatorClass(**VALIDATOR_PARAMS[safe_domain_method])
    except KeyError as e:
        raise ValueError(
            f"Safe domain method {safe_domain_method} not in method registry. "
            f"Available models are: {list(VALIDATOR_MAP.keys())}"
        ) from e
