__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import constants
from models.encoder.featureencoder import FeatureEncoder


class FeatureEncoderDataFrame(object):
    def __init__(self, encoder: FeatureEncoder, df: pd.DataFrame):
        self.encoder = encoder
        self.df = df

    def name(self):
        return self.feature_encoder.param_name()

    def encode(self) -> np.array:
        return self.encoder.encoder(self.df)


"""
    Dataset of a stack of features encoders. This class assembles, stacks, various encoders in a 3D tensor
    :param feature_encoders: List of feature encoder to be aggregated
    :param df: Data frame containing the values of featuree
"""


class StackedEncoderDataset(Dataset):
    def __init__(self, feature_encoders_df: list, df: pd.DataFrame):
        super(StackedEncoderDataset, self).__init__()
        self.data_tensor = StackedEncoderDataset.init(feature_encoders, df)

    @staticmethod
    def init(feature_encoders: list, df: pd.DataFrame) -> Dataset:
        """
            Instantiate a dataset of type StackedEncoderDataset from a given data frame, column name for text from
            which the relevant terms have to be extracted, a list of feature encoder (Co-occurence frequency,
            Graph Laplacian,..)
            :param feature_encoders: List of feature encoders. The list of feature encoders include (Co-occurence
                frequency, Graph Laplacian,..)
            :param df: Input data frame
            :return: Data set of shape max_num_items x max_num_items x num_features
        """
        dataset_array = []
        for feature_encoder in feature_encoders:
            assert feature_encoder.param_name() in \
                   [FeatureEncoder.laplacian_encoder_label, FeatureEncoder.covariance_encoder_label],\
                    f'Feature encoder {feature_encoder.param_name()} is not supported'
            feature_array = feature_encoder.encode(df)
            dataset_array.append(feature_array)

        # Aggregate the various n x n x feature_dim Numpy array
        concat_array = np.concatenate(dataset_array, axis=2)
        # Finally converts into a PyTorch Tensor
        dataset_tensor = torch.from_numpy(concat_array, constants.torch_FloatTensor)
        del dataset_array
        return dataset_tensor


    @staticmethod
    def stack(feature_encoders: list) -> np.array:
        for feature_encoder in feature_encoders:
            assert len(feature_encoder.shape) == 3, \
                f'{feature_encoder.param_name()} shape, {feature_encoder.shapes} should be 3'
        return np.concatenate(feature_encoders, axis=2)

    def __repr__(self):
        return repr(self.feature_encoders)


    def __getitem__(self, row: int) -> torch.Tensor:
        """
            Implement the subscript operator [] for this data set. It is assume that this data set is
            flat and each row has the following layout
                     [feature1, feature2, .., label]
            :param row:  index of the row
            :return: tuple encoder input_tensor, label input_tensor
        """
        try:
            return self.data_tensor[row]
        except IndexError as e:
            constants.log_error(str(e))
            return None

    def __len__(self) -> int:
        return len(self.data_tensor)


