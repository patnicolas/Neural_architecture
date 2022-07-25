__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import pandas as pd
from abc import abstractmethod
import numpy as np

"""
    Abstract class for any type of encoder (Numeric, Grid, Categorical....)
    Sub-classes
        CategoricalEncoder,
        NumericalEncoder
        GridTfIdfEncoder
        LaplacianEncoder
        CoFrequencyEncoder
        
    :param text_col_name: Name of column for which values have to be encoded
"""


class FeatureEncoder(object):
    laplacian_encoder_label = 'laplacian'
    tfidf_encoder_label = 'tfidf'
    covariance_encoder_label = 'covariance'

    def __init__(self, text_col_name: str):
        self.text_col_name = text_col_name

    @abstractmethod
    def encode(self, feature_df: pd.DataFrame) -> np.array:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
