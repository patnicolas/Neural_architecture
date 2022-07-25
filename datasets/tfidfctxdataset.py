__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import constants
from datasets.s3datasetloader import S3DatasetLoader
from models.encoder.featureencoder import FeatureEncoder

"""
    Create a Pytorch dataset from a Panda data frame for a column containing a text and a id column.
    It is assumed that the input_tensor data frame contains only the encoder column with id and text column.
    A typical row (schema) of the input_tensor data frame
         [ id, text for tf-idf, age, gender, taxonomu, .....]
    Output format 
        [ term1_tfidf, term2_tfidf, .. age, gender, taxonomy,....]
    The constructor uses the combination of a generator and comprehensive-list

    :param df: Data frame containing all the input_tensor.
    :param ctx_encoding_scheme: List of encoding scheme for each column
"""


class TfidfCtxDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoding_scheme: list):
        super(TfidfCtxDataset, self).__init__()
        self.encoding_scheme = encoding_scheme
        features_df =  self.create_df(df)
        # Generate the input_tensor
        self.data = torch.from_numpy(features_df.to_numpy())
        # Clean up
        del features_df, df


    def create_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Generate the dataframe of encoder extracted from the original data loaded from files
            :param df: Input data frame
            :return: Output data frame
        """
        def generator(feature_encoder: FeatureEncoder) -> np.array:
            feature_df = df[feature_encoder.text_col_name]
            encoded_array = feature_encoder.encode(feature_df)
            return encoded_array[0]

        acc = []
        [acc.append(generator(feature_encoder)) for feature_encoder in self.encoding_scheme]
        features_df = pd.concat(acc)
        del acc
        return features_df

    @classmethod
    def init_from_s3(cls,
                     s3_bucket: str,
                     s3_folder: str,
                     is_nested: bool,
                     encoding_scheme: list,
                     num_samples: int = -1) -> Dataset:
        """
            Generate a Torch dataset from data loaded from S3
            :param s3_bucket: Name of the S3 bucket
            :param s3_folder: Name of the S3 folder
            :param encoding_scheme: Encoding scheme
            :param is_nested: Flag to specify the columns to be extracted are nested in JSON
            :param num_samples: Number of records to be processed (all records if -1)
            :return: Dataset
        """
        col_names = [encoder.text_col_name for encoder in encoding_scheme]
        s3_dataset_loader = S3DatasetLoader(s3_bucket, s3_folder, col_names, is_nested, '.json', num_samples)
        return cls(s3_dataset_loader.df, encoding_scheme)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, row):
        """
            Implement the subscript operator [] for this data set.
            :param row: index of the row
            :type row: int
            :returns: a Row or record or None for index out of bounds
        """
        try:
            return self.data[row]
        except IndexError as e:
            constants.log_error(str(e))