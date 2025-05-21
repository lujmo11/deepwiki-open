"""CLI to rerun a Neuron training run.

You can use this script to rerun a training run in the same or a different environment.
This might be relevant if
- There has been changes to the training pipeline and the old load case models need to be re-trained
- A model trained in DEV needs to be retrained in PROD to have it available in the
production environment

The script will use the saved original configuration from a training run. That means that the same
input data needs to be available with the same name in the new environment.
"""

import os
from typing import Literal

import mlflow
import requests
import yaml
from azure.identity import DefaultAzureCredential
from dobby.io.storages import BlobStorage

# Uncomment this to migrate the config. Potentially you will have to do some
# new migrating logic.
# from config_migrations.migration_20241030 import migrate_train_config
from neuron.experiment_tracking.mlflow_auth import with_databricks_mlflow_auth
from neuron.schemas.training_run_config import TrainingRunConfig

dbx_host_map = {
    "dev": "https://adb-1760994821569850.10.azuredatabricks.net",
    "staging": "https://adb-1123038963413504.4.azuredatabricks.net",
    "prod": "https://adb-1123038963413504.4.azuredatabricks.net",
}
train_api_url_map = {
    "dev": "https://training-neuron-dev.neuron-azweu-dev.vks.vestas.net",
    "staging": "https://training-neuron-staging.neuron-azweu-dev.vks.vestas.net",
    "prod": "https://training-neuron-prod.neuron-azweu-dev.vks.vestas.net",
}
storage_map = {
    "dev": BlobStorage(
        storage_account_name="lacneurondevsa",
        container_name="neuron-training-run-input",
        credentials=DefaultAzureCredential(),
    ),
    "staging": BlobStorage(
        storage_account_name="lacneuronprodsa",
        container_name="neuron-training-run-input",
        credentials=DefaultAzureCredential(),
    ),
    "prod": BlobStorage(
        storage_account_name="lacneuronprodsa",
        container_name="neuron-training-run-input",
        credentials=DefaultAzureCredential(),
    ),
}


@with_databricks_mlflow_auth
def get_training_input_id(mlflow_run_id: str) -> str:
    mlflow_run = mlflow.get_run(run_id=mlflow_run_id)
    return mlflow_run.data.params.get("training_input_id")


def get_train_api_key_from_env_var(env: Literal["dev", "staging", "prod"]) -> str:
    API_KEY_ENV_VAR_NAME = {
        "dev": "TRAIN_API_KEY_DEV",
        "staging": "TRAIN_API_KEY_STAGING",
        "prod": "TRAIN_API_KEY_PROD",
    }[env]
    try:
        return os.environ[API_KEY_ENV_VAR_NAME]
    except KeyError as e:
        raise ValueError(
            f"{API_KEY_ENV_VAR_NAME} environment variable needs "
            f"to be set if the new env is '{env}'. "
        ) from e


def main(
    mlflow_run_id: str,
    original_env: Literal["dev", "staging", "prod"],
    new_env: Literal["dev", "staging", "prod"],
) -> None:
    original_databricks_host_env_var = os.environ.get("DATABRICKS_HOST")
    os.environ["DATABRICKS_HOST"] = dbx_host_map[original_env]
    train_storage = storage_map[original_env]
    train_api_url = train_api_url_map[new_env]
    api_key = get_train_api_key_from_env_var(new_env)

    print(
        f"Getting training input id from run {mlflow_run_id} "
        f"on {original_env} MLFlow tracking server "
        f"at {os.environ['DATABRICKS_HOST']}"
    )
    training_input_id = get_training_input_id(mlflow_run_id)

    # If the original DATABRICKS_HOST environment variable was set,
    # we set it back to the original value
    if original_databricks_host_env_var:
        os.environ["DATABRICKS_HOST"] = original_databricks_host_env_var

    print(f"Getting original training config from {training_input_id} on {original_env} storage")
    original_config_path = training_input_id + "/original_training_run_config.yaml"
    train_config = yaml.safe_load(train_storage.read(original_config_path))

    # Uncomment this to migrate the config. Potentially you will have to do some
    # new migrating logic.
    # train_config = migrate_train_config(train_config)

    # If migrating the config, you will need to comment out the line below that
    # validates the training config with the current schema
    print("Validating the train config schema before triggering new training run")
    _ = TrainingRunConfig(**train_config)

    print(f"Triggering new training run in {new_env} on {train_api_url}")
    url = train_api_url + "/train_job_full_spec"
    headers = {"User-Agent": "Avoid Automatic Firewall", "x-api-key": api_key}
    response = requests.post(url, verify=False, headers=headers, json=train_config)
    response.raise_for_status()
    print(f"New training run triggered with response: {response.json()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-run-id", type=str, required=True)
    parser.add_argument(
        "--original-env",
        type=str,
        required=False,
        choices=["dev", "staging", "prod"],
        default="dev",
    )
    parser.add_argument(
        "--new-env",
        type=str,
        required=False,
        choices=["dev", "staging", "prod"],
        default="dev",
    )
    args = parser.parse_args()
    main(
        mlflow_run_id=args.mlflow_run_id,
        original_env=args.original_env,
        new_env=args.new_env,
    )
