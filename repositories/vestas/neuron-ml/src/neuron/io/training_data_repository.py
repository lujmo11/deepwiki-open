"""The training data repository is used to get data for training and evaluation
for a single load case in the Neuron training pipeline."""

import logging
from io import BytesIO
from typing import Union

import pandas as pd

from neuron.data_splitting.base import DataSplitter
from neuron.io.storage_reader import StorageReader

logger = logging.getLogger(__name__)


class TrainingDataRepository:
    """Training data repository for load case."""

    def __init__(
        self,
        training_file_uri: str,
        storage: StorageReader,
        data_splitter: DataSplitter,
        test_file_uri: Union[str, None] = None,
        agg_file_uri: Union[str, None] = None,
    ):
        """Initialize the training data repository.

        Parameters
        ----------

        """
        self.training_file_uri = training_file_uri
        self.test_file_uri = test_file_uri
        self.agg_file_uri = agg_file_uri
        self.storage = storage
        self.data_splitter = data_splitter
        self._train_df: Union[pd.DataFrame, None] = None
        self._test_df: Union[pd.DataFrame, None] = None
        self._agg_df: Union[pd.DataFrame, None] = None

        # Running test train split when initializing to ensure spilt is only done once
        self.get_load_case_train_df()

    @staticmethod
    def _read_df_from_bytes(data: bytes) -> pd.DataFrame:
        df_io = BytesIO()
        df_io.write(data)
        df_io.seek(0)
        return pd.read_parquet(df_io)

    def get_load_case_train_df(self) -> pd.DataFrame:
        """Get training dataframe for the load case.

        Returns
        -------
        pd.DataFrame
            Load case training dataframe.
        """
        if self._train_df is not None:
            return self._train_df
        if self.test_file_uri:
            logger.info(
                "test_file_uri supplied for load case. "
                "Using training_file_uri to load training data."
            )
            self._train_df = self._read_df_from_bytes(self.storage.read(self.training_file_uri))
            return self._train_df
        else:
            logger.info(
                "test_file_uri NOT supplied for load case. "
                "Splitting data in training_file_uri to get "
                f"training data using data splitter: {self.data_splitter}."
            )
            self._train_df, self._test_df = self.data_splitter.train_test_split(
                self._read_df_from_bytes(self.storage.read(self.training_file_uri))
            )
            return self._train_df

    def get_load_case_test_df(self) -> pd.DataFrame:
        """Get test dataframe for the load case.

        Returns
        -------
        pd.DataFrame
            Returns a load case test dataframe if it exists, else None.
        """
        if self._test_df is not None:
            return self._test_df
        if self.test_file_uri:
            logger.info(
                "test_file_uri supplied for load case. Using test_file_uri to load test data."
            )
            self._test_df = self._read_df_from_bytes(self.storage.read(self.test_file_uri))
            return self._test_df
        else:
            logger.info(
                "test_file_uri NOT supplied for load case. "
                "Splitting data in training_file_uri to get "
                f"test data using data splitter: {self.data_splitter}."
            )
            self._train_df, self._test_df = self.data_splitter.train_test_split(
                self._read_df_from_bytes(self.storage.read(self.training_file_uri))
            )
            return self._test_df

    def get_load_case_agg_df(self) -> pd.DataFrame:
        """Get agg test dataframe for the load case.

        Returns
        -------
        pd.DataFrame
            Returns a load case agg test dataframe if it exists, else None.
        """
        if self._agg_df is not None:
            return self._agg_df
        if self.agg_file_uri:
            logger.info(
                "agg_file_uri supplied for load case. Using agg_file_uri to load agg test data."
            )
            self._agg_df = self._read_df_from_bytes(self.storage.read(self.agg_file_uri))
            return self._agg_df
