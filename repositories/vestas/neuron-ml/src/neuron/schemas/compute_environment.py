from enum import StrEnum


class ComputeEnvironment(StrEnum):
    """Enum for the different compute targets."""

    LOCAL = "local"
    DATABRICKS = "databricks"
