from enum import StrEnum
from typing import Self, Union

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class StorageBackend(StrEnum):
    local = "local"
    azure = "azure"


class Settings(BaseSettings):
    model_store_folder_path: str = "../tests/data/api/model_store"
    build_number: str = "dev/2024-04-18"
    storage_account_name_models: str = "lacneurondevmodelsa"
    container_name: str = "neuron-models"
    model_cache_size: int = 1
    storage_backend: StorageBackend = StorageBackend.local
    model_store_token: Union[str, None] = None
    git_sha: str = "unknown"

    # To avoid warning of clash between `model_store_folder_path`,
    # model_store_folder_path`model_store_token`, `model_cache_size`
    # and Pydantic protected namespace.
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="after")
    def validate_model_store_token(self) -> Self:
        """If model_store_token='' is provided, set it to None"""
        if self.model_store_token == "":
            self.model_store_token = None
        return self


settings = Settings()
