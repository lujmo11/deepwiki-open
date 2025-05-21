"""Module with service for interacting with Databricks training Job/Workflow and the
associated MLFlow experiments."""
import os
from typing import List, Union

import structlog
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import InvalidParameterValue
from mlflow import MlflowClient

from neuron_training_service.dbx_service.exceptions import DBXJobDoesNotExistError
from neuron_training_service.schemas import JobRun, MLFlowRun, NeuronTrainingJobRun

logger = structlog.get_logger()
#  We are using a client for the Databricks workspace that uses default authentication.
#  This means that locally it will work with you Databricks profile while in production,
#  it will use the service principal set through environment variables.
#  See: https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/azure-sp#python
dbx_ws_client = WorkspaceClient()
mlflow_client = MlflowClient(tracking_uri="databricks")


def with_mlflow_token_auth(ws_client: WorkspaceClient) -> callable:
    """Decorator function to set databricks token before function call and delete it after

    We need this when using running as a service principal, because the MLFlow client uses a
    Databricks token to authenticate with the databricks workspace.
    """

    def decorator(function: callable):
        def wrapper(*args, **kwargs):
            if "DATABRICKS_TOKEN" in os.environ:
                result = function(*args, **kwargs)
                return result
            token = ws_client.tokens.create(
                comment="Short lived token for training API", lifetime_seconds=120
            )
            os.environ["DATABRICKS_TOKEN"] = token.token_value
            result = function(*args, **kwargs)
            ws_client.token_management.delete(token.token_info.token_id)
            del os.environ["DATABRICKS_TOKEN"]
            return result

        return wrapper

    return decorator


class DBXService:
    """The class provides methods interacting with a Databricks Neuron Job/workflow. It allows us to
    - start a training job run for the job/workflow
    - get information about a training job run for the job/workflow
    - get information about the MLFlow run associated with a training job run for the job/workflow
    """

    def __init__(
        self,
        dbx_host: str,
        dbx_job_id: int,
        experiment_ids: Union[List[str], None] = None,
    ):
        """Initialize the Databricks service.

        Parameters
        ----------
        dbx_host : str
            The Databricks host URL.
        dbx_job_id : str
            The Databricks job ID.
        experiment_ids : Union[List[str], None], optional
            The list of MLFlow experiment IDs to search for the MLFlow run associated with the
            training job run, by default None. If None, all experiments are searched.
        """
        self.dbx_host = dbx_host
        self.dbx_job_id = dbx_job_id
        if experiment_ids is None:
            experiment_ids = [
                exp.experiment_id for exp in dbx_ws_client.experiments.list_experiments()
            ]
        self.experiment_ids = experiment_ids

    def start_train_job_run(self, config_blob_path: str) -> int:
        """Start a Neuron training job run on Databricks.

        Parameters
        ----------
        config_blob_path : str
            The path to the training configuration blob in the Azure storage container.

        Returns
        -------
        int
            The run ID of the triggered job run.
        """
        job_trigger_response = dbx_ws_client.jobs.run_now(
            job_id=self.dbx_job_id, python_params=["--train-config-path", config_blob_path]
        )
        return job_trigger_response.response.run_id

    @staticmethod
    def _get_dbx_job_run(job_run_id: int) -> Union[JobRun, None]:
        try:
            job_run = dbx_ws_client.jobs.get_run(run_id=job_run_id)
        except InvalidParameterValue as e:
            raise DBXJobDoesNotExistError(job_run_id) from e
        return JobRun(
            id=job_run.run_id,
            url=job_run.run_page_url,
            result_state=job_run.state.result_state,
            life_cycle_state=job_run.state.life_cycle_state,
        )

    @staticmethod
    def _get_mlflow_run_url(dbx_host: str, experiment_id: str, mlflow_run_id: str) -> str:
        return f"{dbx_host}/#mlflow/experiments/{experiment_id}/runs/{mlflow_run_id}"

    @with_mlflow_token_auth(ws_client=dbx_ws_client)
    def _get_mlflow_run(self, job_run_id: int) -> Union[MLFlowRun, None]:
        try:
            job_run = dbx_ws_client.jobs.get_run(run_id=job_run_id)
        except InvalidParameterValue as e:
            raise DBXJobDoesNotExistError(job_run_id) from e
        task_run_id = job_run.tasks[0].run_id
        runs = mlflow_client.search_runs(
            self.experiment_ids,
            filter_string=(
                f"tags.mlflow.databricks.jobID = '{self.dbx_job_id}' "
                f"AND tags.mlflow.databricks.jobRunID = '{task_run_id}'"
            ),
        )
        if len(runs) > 1:
            raise ValueError(
                f"Found multiple runs with job run ID {task_run_id}. This should not be possible."
            )
        elif len(runs) == 0:
            return None
        else:
            run = runs[0]
            return MLFlowRun(
                id=run.info.run_id,
                url=self._get_mlflow_run_url(
                    dbx_host=self.dbx_host,
                    experiment_id=run.info.experiment_id,
                    mlflow_run_id=run.info.run_id,
                ),
            )

    def get_training_run_info(self, job_run_id: int) -> NeuronTrainingJobRun:
        """Get information about a Neuron training run."""
        job_run = self._get_dbx_job_run(job_run_id)
        mlflow_run = self._get_mlflow_run(job_run_id)
        return NeuronTrainingJobRun(job_run=job_run, mlflow_run=mlflow_run)
