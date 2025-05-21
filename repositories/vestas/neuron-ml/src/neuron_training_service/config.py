from typing import List, Union

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    build_number: str = "dev"
    source_internal_storage_account_name: str = "lacneurondevsa"
    source_internal_storage_container_name: str = "neuron-training-data"
    source_simmesh_storage_account_name: str = "meshlindodev"
    target_internal_storage_account_name: str = "lacneurondevsa"
    target_internal_storage_container_name: str = "neuron-training-run-input"
    git_sha: str = "unknown"
    searchable_mlflow_experiment_ids: Union[List[str], None] = None
    training_api_target_base_dir: str
    databricks_host: str
    databricks_job_id: str


settings = Settings()
