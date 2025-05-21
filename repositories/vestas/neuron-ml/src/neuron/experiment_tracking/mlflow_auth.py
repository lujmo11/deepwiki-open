"""Module for authenticating to MLFlow tracking server on Databricks.

This module provides a decorator to authenticate to MLFlow tracking server on Databricks,
using standard Databricks authentication. The decorator should only be used for building
CLI's or scripts that need to query an MLFlow tracking server on Databricks.
Not when logging to mlflow runs.
"""
import os

from databricks.sdk import WorkspaceClient


def with_databricks_mlflow_auth(func: callable) -> callable:
    """Decorator to authenticate to MLFlow tracking server on Databricks, using standard
    Databricks authentication.
    See: https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/#--default
    -order-of-evaluation-for-client-unified-authentication-methods-and-credentials

    Use it to decorate functions that need to query an MLFlow tracking server on Databricks.


    The decorator requires that
    - The environment variable DATABRICKS_HOST is set to the hostname of the Databricks workspace
    with the MLFlow tracking server.
    - The credentials of the user or Service Principal are configured in one of the standard ways
    (e.g .Databricks configuration profile or environment variables).

    Example usage:
    -------------
    ```python
      @with_databricks_mlflow_auth
        def get_run(run_id: str)-> str:
            return mlflow.get_run(run_id=run_id).info.run_id
    ```
    """

    def wrapper(*args, **kwargs):
        try:
            databricks_host = os.environ["DATABRICKS_HOST"]
        except KeyError as e:
            raise ValueError(
                "DATABRICKS_HOST environment variable must be set when using the "
                "with_databricks_mlflow_auth decorator."
            ) from e
        dbx_ws_client = WorkspaceClient(host=databricks_host)
        token_create_response = dbx_ws_client.tokens.create(
            lifetime_seconds=120, comment="mlflow-short-lived-token"
        )
        original_mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        original_dbx_token = os.environ.get("DATABRICKS_TOKEN")
        os.environ["DATABRICKS_TOKEN"] = token_create_response.token_value
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        try:
            result = func(*args, **kwargs)
        finally:
            del os.environ["DATABRICKS_TOKEN"]
            del os.environ["MLFLOW_TRACKING_URI"]
            if original_dbx_token:
                os.environ["DATABRICKS_TOKEN"] = original_dbx_token
            if original_mlflow_tracking_uri:
                os.environ["MLFLOW_TRACKING_URI"] = original_mlflow_tracking_uri
            dbx_ws_client.tokens.delete(token_create_response.token_info.token_id)
        return result

    return wrapper
