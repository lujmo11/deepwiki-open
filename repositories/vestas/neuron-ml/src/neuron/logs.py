import logging

from rich.logging import RichHandler


def initialize_logger(name: str, log_level: str) -> logging.Logger:
    """Initialize root logger. All other loggers in the module will inherit the configuration."""

    logger = logging.getLogger(name)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)
    logging.getLogger("py4j").setLevel(logging.ERROR)

    logger.setLevel(level=log_level)

    # create console handler and set level to debug
    ch = RichHandler(
        level=log_level,
        omit_repeated_times=False,
        log_time_format="%Y-%m-%dT%H:%M:%S",
    )

    # add ch to logger
    logger.addHandler(ch)
    return logger
