import re
from dataclasses import dataclass
from typing import Self, Union

from azure.core.exceptions import ClientAuthenticationError
from dobby.io.storages import BlobStorage

from neuron import logger


class WrongStorageAccountError(Exception):
    """Error raised when the blob url does not belong to the storage account."""


class WrongBlobURIError(Exception):
    """Error raised when the blob uri is not in the correct format."""


class SimMeshAuthenticationError(Exception):
    """Authentication to sim mesh fails"""


@dataclass
class BlobUrlComponents:
    storage_account_name: str
    container_name: str
    blob_path: str
    sas_token: str

    @classmethod
    def from_url(cls, blob_url: str) -> Self:
        """Parse a blob url into its components.

        Parameters
        ----------
        blob_url : str
            Blob url.

        Returns
        -------
        BlobUrlComponents
            Blob url components.
        """
        pattern = re.compile(
            r"https://(?P<storage_account_name>[^.]+)\.blob\.core\.windows\.net/(?P<container_name>[^/]+)/(?P<blob_name>[^?]+)\?(?P<sas_token>.+)"
        )
        match = pattern.match(blob_url)
        if not match:
            raise WrongBlobURIError(
                f"Blob url {blob_url} is not in the correct format. "
                f"It should be in the format https://<storage_account_name>.blob.core.windows.net/<container_name>/<blob_name>?<sas_token>"
            )

        components = match.groupdict()
        return cls(
            storage_account_name=components["storage_account_name"],
            container_name=components["container_name"],
            blob_path=components["blob_name"],
            sas_token=components["sas_token"],
        )


class SASURLBlobStorage:
    """Implements the Neuron StorageReader interface,
    but overloads the file_uri parameter to also allow a URL with credentials.

    The class wraps the class dobby.io.storage.BlobStorage,
    but extracts the container name and credentials from the blob URL
    to dynamically set the blob storage for each operation.
    """

    def __init__(self, storage_account_name: str):
        self._storage_account_name = storage_account_name
        self._blob_storage: Union[BlobStorage, None] = None

    def _set_blob_storage_from_blob_url(self, blob_url: str) -> None:
        """Sets the blob storage by extracting the blob path and container name from a blob URL.

        The blob URL should be in the format:
        https://<storage_account_name>.blob.core.windows.net/<container_name>/<blob_name>?<sas_token>
        """
        blob_url_components = BlobUrlComponents.from_url(blob_url)
        if blob_url_components.storage_account_name != self._storage_account_name:
            raise WrongStorageAccountError(
                f"Blob storage account {blob_url_components.storage_account_name} is not allowed."
                f"The storage account should be {self._storage_account_name}."
            )
        self._blob_storage = BlobStorage(
            storage_account_name=self._storage_account_name,
            container_name=blob_url_components.container_name,
            credentials=blob_url_components.sas_token,
        )

    def read(self, file_uri: str) -> bytes:
        """Read data from file_uri that is a blob URL, including the SAS token.

        Parameters
        ----------
        file_uri : str
            Blob URL in the format
            https://<storage_account_name>.blob.core.windows.net/<container_name>/<blob_name>?<sas_token>

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the given path.

        Returns
        -------
        bytes
            Data read from path.
        """
        logger.info(f"Reading data from {file_uri}.")
        self._set_blob_storage_from_blob_url(file_uri)
        try:
            return self._blob_storage.read(file_uri=BlobUrlComponents.from_url(file_uri).blob_path)
        except ClientAuthenticationError as e:
            raise SimMeshAuthenticationError(
                "Authentication to sim mesh failed. "
                "Check that the SAS token in the URL is correct and not expired."
            ) from e

    def __repr__(self):
        return f"SASURLBlobStorage(storage_account_name={self._storage_account_name})"
