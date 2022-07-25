__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import pandas as pd
import numpy as np
from models.encoder.featureencoder import FeatureEncoder

"""
    Implementation of the conversion of numerical value
    :param text_col_name: Name of column for which values have to be encoded
"""


class NumericalEncoder(FeatureEncoder):
    def __init__(self, text_col_name: str):
        super(NumericalEncoder, self).__init__(text_col_name)

    def name(self) -> str:
        return 'Numerical encoder'

    @staticmethod
    def to_numeric(feature_df: pd.DataFrame) -> np.array:
        """
            Normalize this data frame of values
            :return: Array of normalized values
        """
        assert len(feature_df.columns) == 1, 'Encoder: it should be a single column feature'
        feature_array = feature_df.to_numpy().squeeze(1)
        max_value = np.max(feature_array)
        # A simple broadcast
        return feature_array / max_value


    def encode(self, feature_df: pd.DataFrame) -> np.array:
        return NumericalEncoder.to_numeric(feature_df)
