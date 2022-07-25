__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from models.encoder.featureencoder import FeatureEncoder

"""
    Implementation of the conversion of categorical encoder into 
    - normalized numeric value
    - one hot encoding
    - tf-idf encoding
    - sparse td-idf encoding
    Note: Difference between tf-idf and sparse tf-idf encoding
    tf-idf
        [0, 0, 0.67, 0.0, 0.78, ...]
    sparse tf-idf
        [(0.67, 2), (0.78, 4), ...]
        
    :param text_col_name: Name of the feature to be encoded
    :param ctx_encoding_scheme: Label for the encoding method
"""


class CategoricalEncoder(FeatureEncoder):
    def __init__(self, text_col_name: str, encoding_scheme: str):
        super(CategoricalEncoder, self).__init__(text_col_name)
        assert encoding_scheme in ['numeric', 'one_hot', 'tfidf'], \
            f'Feature encoder, encoding scheme {encoding_scheme} not supported'
        self.encoding_scheme = encoding_scheme

    def name(self) -> str:
        return 'Categorical encoder'

    @staticmethod
    def to_numeric(feature_df: pd.DataFrame) -> np.array:
        """
            Convert the data frame of categorical values into a normalized numerical values
            :return: Numpy array of normalized numerical values
        """
        # Encode into integer
        label_encoder = LabelEncoder()
        label_encoded = label_encoder.fit_transform(feature_df)
        # Normalize the numpy array and convert to a data frame
        max_value = np.max(label_encoded)
        return label_encoded / max_value


    @staticmethod
    def to_one_hot_encoded(feature_df: pd.DataFrame) -> np.array:
        """
             Convert the data frame of categorical values into one hot encoded values
             :return: Numpy array one hot encoded values
         """
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded = encoder.fit_transform(feature_df)
        return encoded.toarray()


    @staticmethod
    def to_tfidf(feature_df: pd.DataFrame, is_squeeze: bool = False) -> np.array:
        """
            Compute tfidf on the single column data set which contains text
            :param feature_df: Input data frame ["line 1", "line 2", ...]
            :return: Numpy array of  tf-idf weights
        """
        tfidf_encoded_np = CategoricalEncoder.__tfidf_encode_feature(feature_df, is_squeeze)
        return tfidf_encoded_np


    def encode(self, feature_df: pd.DataFrame) -> np.array:
        """
            Generic wrapper for encoding a single column (or feature) data frame
            :param feature_df: Single feature data frame
            :return: Numpy array of encoded values
        """
        assert self.encoding_scheme in ['numeric', 'one_hot', 'tfidf'], \
            f'Feature encoder, encoding scheme {self.encoding_scheme} not supported'
        if self.encoding_scheme == 'numeric':
            encoded_np = CategoricalEncoder.to_numeric(feature_df)
        elif self.encoding_scheme == 'tfidf':
            encoded_np = CategoricalEncoder.to_tfidf(feature_df)
        else:
            encoded_np = CategoricalEncoder.to_one_hot_encoded(feature_df)
        return encoded_np


    @staticmethod
    def __tfidf_encode_feature(feature_df: pd.DataFrame, is_squeeze: bool = False) -> np.array:
        if is_squeeze:
            corpus = feature_df.values.squeeze(1).tolist()
        else:
            corpus = feature_df.values.tolist()

        # Initialize and train using input_tensor data frame
        vectorized = TfidfVectorizer()
        vectorized.fit(corpus)
        acc = []
        [acc.append(vectorized.transform([entry]).toarray()) for entry in corpus]
        #  Remove unnecessary num_tfidf_features
        tfidf_encoded_np = np.squeeze(np.array(acc), 1)
        del acc
        return tfidf_encoded_np



