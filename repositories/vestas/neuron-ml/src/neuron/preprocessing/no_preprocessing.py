import logging

import pandas as pd

from neuron.preprocessing.base import Preprocessor

logger = logging.getLogger(__name__)


class NoPreprocessor(Preprocessor):
    name = "no_preprocessor"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Not preprocessing data")
        return df
