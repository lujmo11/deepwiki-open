from typing import Protocol


class StorageReader(Protocol):
    def read(self, file_uri: str) -> bytes:
        """Read data from path.

        Args:
            file_uri: Path to data.

        Returns:
            Data read from path.
        """
        ...
