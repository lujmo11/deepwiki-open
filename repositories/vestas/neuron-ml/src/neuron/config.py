from dobby.environment import ComputeTarget, get_compute_target
from dobby.io.storages import BlobStorage, FileSystemStorage, Storage
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    experiment_name: str = "turbine-training-runs"
    mlflow_tracking_uri: str | None = None


settings = Settings()


def get_project_storage() -> Storage:
    """Get storage for the project based on the compute target"""
    compute_target = get_compute_target()

    if compute_target == ComputeTarget.LOCAL:
        return FileSystemStorage(mount_path=".")

    elif compute_target == ComputeTarget.DBX_DEV:
        return BlobStorage.create_using_mlproduct_credentials(
            storage_account_name="lacneurondevsa",
            container_name="neuron-training-run-input",
        )

    elif get_compute_target() == ComputeTarget.DBX_PROD:
        return BlobStorage.create_using_mlproduct_credentials(
            storage_account_name="lacneuronprodsa",
            container_name="neuron-training-run-input",
        )
    else:
        raise ValueError(f"Unsupported compute target: {compute_target}")
