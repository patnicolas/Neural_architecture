__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from models.encoder.featureencoder import FeatureEncoder

"""
    Define a 2 dimension encoder for categorical values. For a given vocabulary {a, b, c, d,...}
    Vocabulary doc_terms_str are extracted from a text... as {b, c, t, b, d...}
    The grid is generated as 
        b, c, t, b, d
    b   X           X
    c      X        
    d               X
    t         X

    :param text_col_name: Name of the column containing input_tensor text
    :param num_features: Number of encoder (or dimension of the model)
"""


class GridTfIdfEncoder(FeatureEncoder):
    def __init__(self, text_col_name: str, num_features: int):
        super(GridTfIdfEncoder, self).__init__(text_col_name)
        assert num_features > 0, \
            f'Grid categorical encoder tfidf num_context_features {num_features} should be > 0'
        self.num_features = num_features

    def name(self) -> str:
        return FeatureEncoder.tfidf_encoder_label

    @staticmethod
    def to_tfidf(feature_df: pd.DataFrame, num_features: int) -> list:
        """
            Extract a tfidf weights from a corpus defined in the feature data frame with column name, 'text_col_name'
            (i.e. text_col_name: termsText)
            :param feature_df: Data Frame of test1 from which to compute tf-idf weights
            :param num_features: Total number of encoder
            :return: List of Numpy arrays (one per text)
        """
        corpus = feature_df.values.squeeze(1).tolist()
        # Initialize and train using input_tensor data frame
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(corpus)

        list_2d_array = []
        for entry in corpus:
            grid_tfidf = GridTfIdfEncoder.__generate_grid(tfidf_vectorizer, entry, num_features)
            list_2d_array.append(grid_tfidf)
        return list_2d_array

    def encode(self, feature_df: pd.DataFrame) -> list:
        """
            Generic Polymorphic wrapper for encoding a single column (or feature) data frame
            :param feature_df: Single feature data frame
            :return: List of tf-idf num_features x num_features grid
        """
        return GridTfIdfEncoder.to_tfidf(feature_df, self.num_features)

    # -------------------------  Supporting method -------------------------
    @staticmethod
    def __generate_grid(vectorized: TfidfVectorizer, doc_terms: str, num_features: int) -> np.array:
        from models.encoder.tfidfencoder import TfIdfEncoder

        top_tfidf_weights = TfIdfEncoder.extract(vectorized, doc_terms, num_features)
        # Initialize the convolutional grid
        grid_tfidf = np.zeros(shape=(num_features, num_features))
        row = 0
        col = 0
        prev_non_zero_idx = 0
        # Populate the grid
        for non_zero_idx, weight in top_tfidf_weights:
            grid_tfidf[row][col] = weight
            if prev_non_zero_idx != non_zero_idx:
                row += 1
            col += 1
            prev_non_zero_idx = non_zero_idx
        del top_tfidf_weights
        return grid_tfidf

