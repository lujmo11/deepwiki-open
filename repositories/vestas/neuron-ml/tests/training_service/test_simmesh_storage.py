import pytest
from azure.core.exceptions import ClientAuthenticationError
from dobby.io.storages import BlobStorage
from dobby.io.storages.exceptions import FileDoesNotExistInStorageError

from neuron_training_service.simmesh_storage import (
    BlobUrlComponents,
    SASURLBlobStorage,
    SimMeshAuthenticationError,
    WrongBlobURIError,
    WrongStorageAccountError,
)


def test_blob_url_components_from_url_happy() -> None:
    blob_url = "https://mystorageaccount.blob.core.windows.net/mycontainer/myblobpath?mysastoken"
    blob_url_components = BlobUrlComponents.from_url(blob_url)
    assert blob_url_components.storage_account_name == "mystorageaccount"
    assert blob_url_components.container_name == "mycontainer"
    assert blob_url_components.blob_path == "myblobpath"


@pytest.mark.parametrize(
    "blob_url",
    [
        "https://mystorageaccount.blob.core.windows.net/mycontainer/myblobpath",
        "https://mystorageaccount.blob.core.windows.net/mycontainer/myblobpath?",
        "mystorageaccount.blob.core.windows.net/mycontainer/myblobpath?mysastoken",
        "https://mystorageaccount.blob.core/mycontainer/myblobpath?mysastoken&otherstuff",
    ],
)
def test_blob_url_components_from_url_no_match(blob_url: str) -> None:
    """Test that the BlobUrlComponents raises a WrongBlobURIError when the blob URL is not in the
    correct format."""
    with pytest.raises(WrongBlobURIError) as e:
        BlobUrlComponents.from_url(blob_url)
    assert str(e.value) == (
        f"Blob url {blob_url} is not in the correct format. It should be in the format "
        "https://<storage_account_name>.blob.core.windows.net/<container_name>/<blob_name>?<sas_token>"
    )


def test_sas_url_blob_storage(monkeypatch) -> None:  # noqa: ANN001
    """Test that the SAS URL blob storage reads the correct data."""

    # Mock the read method of the BlobStorage class
    def mock_read(self, file_uri: str) -> bytes:  # noqa: ARG001, ANN001
        if file_uri != "valid_blob":
            raise FileDoesNotExistInStorageError(file_uri)
        return b"dummy data"

    monkeypatch.setattr(BlobStorage, "read", mock_read)

    sas_url_storage = SASURLBlobStorage(storage_account_name="mystorageaccount")
    valid_blob_url = (
        "https://mystorageaccount.blob.core.windows.net/mycontainer/valid_blob?mysastoken"
    )
    invalid_blob_url = (
        "https://mystorageaccount.blob.core.windows.net/mycontainer/invalid_blob?mysastoken"
    )
    assert (
        sas_url_storage.read(file_uri=valid_blob_url) == b"dummy data"
    ), "The correct data was not read."

    with pytest.raises(FileDoesNotExistInStorageError):
        sas_url_storage.read(file_uri=invalid_blob_url)


def test_sas_url_blob_storage_wrong_storage_account() -> None:
    """Test that the SAS URL blob storage raises a WrongStorageAccountError when the storage account
    in the URL is different from the one that the storage is initialized with."""
    sas_url_storage = SASURLBlobStorage(storage_account_name="mystorageaccount")
    blob_url = "https://wrongstorageaccount.blob.core.windows.net/mycontainer/myblobpath?mysastoken"
    with pytest.raises(WrongStorageAccountError):
        sas_url_storage.read(file_uri=blob_url)


def test_sas_url_blob_catches_authentication_error(monkeypatch) -> None:  # noqa: ANN001
    """Test that the SAS URL blob storage catches
    azure.core.exceptions.ClientAuthenticationError
    """

    # Mock the read and methods of the BlobStorage class
    def mock_read(self, file_uri: str) -> bytes:  # noqa: ARG001, ANN001
        raise ClientAuthenticationError

    monkeypatch.setattr(BlobStorage, "read", mock_read)

    sas_url_storage = SASURLBlobStorage(storage_account_name="mystorageaccount")
    blob_url = "https://mystorageaccount.blob.core.windows.net/mycontainer/valid_blob?mysastoken"

    with pytest.raises(SimMeshAuthenticationError):
        sas_url_storage.read(file_uri=blob_url)
