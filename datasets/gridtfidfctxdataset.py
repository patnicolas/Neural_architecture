__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from datasets.tfidfctxdataset import TfidfCtxDataset
from datasets.s3datasetloader import S3DatasetLoader
from torch.utils.data import Dataset
import torch
import constants
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
    Create a Pytorch dataset from a Panda data frame for a column containing a text and a id column.
    It is assumed that the input_tensor data frame contains only the encoder column with id and text column.
    A typical row (schema) of the input_tensor data frame
         [ id, text for tf-idf, age, gender, taxonomy, .....]
    Output format 
        b, c, t, b, d
    b   X           X
    c      X        
    d               X
    t         X
       
    The constructor uses the combination of a generator and comprehensive-list

    :param df: Data frame containing all the input_tensor.
    :param encoding_scheme: List of encoding scheme for each column
"""


class GridTfidfCtxDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoding_scheme: list):
        super(GridTfidfCtxDataset, self).__init__()
        self.encoding_scheme = encoding_scheme
        features_array = self.__create_tensor(df)
        # Aggregate grid input_tensor
        dataset_tensor = np.stack(features_array, axis=0)
        # Generate the input_tensor
        self.data = torch.from_numpy(dataset_tensor)
        # Clean up
        del df

    @classmethod
    def init_from_s3(cls,
                     s3_bucket: str,
                     s3_folder: str,
                     is_nested: bool,
                     encoding_scheme: list,
                     num_samples: int = -1) -> TfidfCtxDataset:
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
            return None

            # ------------------  Supporting private method ----------------------
    def __create_tensor(self, df: pd.DataFrame) -> list:
        """
            Generate a list of 2D numpy arrays
            :param df: Input data frame
            :return: List 2D grid np.array
        """
        # Generate the numpy array for the tfidf weights
        tfidf_encoder = self.encoding_scheme[0]
        tfidf_input_df = df[tfidf_encoder.text_col_name].to_frame()
        tfidf_encoded_np_list = tfidf_encoder.encode(tfidf_input_df)

        def context_vector_generator(idx: int) -> np.array:
            feature_encoder = self.encoding_scheme[idx]
            # Convert pdSeries to a formal Data frame (is is really needed?)
            feature_df = df[feature_encoder.text_col_name].to_frame()
            return feature_encoder.encode(feature_df)

        encoded_context_features_list = []
        [encoded_context_features_list.append(context_vector_generator(idx)) for idx in range(1, len(self.encoding_scheme))]
        # Aggregate the various contextual data to be added to each of the elements of the grid
        context_features_np = np.stack(encoded_context_features_list, axis=1)

        # Generate Grid for tfidf and contextual data
        def grid_generator(idx: int) -> np.array:
            tfidf_encoded_np = tfidf_encoded_np_list[idx].flatten()
            acc = []
            # Concatenate the various encoder in the grid
            [acc.append(np.concatenate([[grid], context_features_np[idx]], axis = 0))
             for grid in tfidf_encoded_np]
            ar = np.stack(acc, axis=0)\
                .reshape((tfidf_encoded_np_list[idx].shape[0], tfidf_encoded_np_list[idx].shape[1], -1))
            del acc
            return ar

        # Aggregate the tfidf values with the encoded, contextual data (age, ...)
        accumulate = []
        [accumulate.append(grid_generator(idx)) for idx in tqdm(range(len(tfidf_encoded_np_list)))]
        del encoded_context_features_list
        return accumulate
