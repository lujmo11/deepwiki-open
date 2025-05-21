import os

from neuron.logs import initialize_logger

logger = initialize_logger(__name__, log_level=os.getenv("NEURON_LOG_LEVEL", "INFO"))

logger.info("Neuron package initialized")
