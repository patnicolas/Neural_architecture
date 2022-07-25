__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from datasets.s3datasetloader import S3DatasetLoader
import constants

"""
    Dataset for mentions extracted by NLP models. This dataset relies on a featurizer
    :param df: Pandas dataframe 
    :param separator: Separator string used to extract tokens
"""


class TokenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, separator: str = ','):
        super(TokenDataset, self).__init__()

        features_df = df.transform(lambda x: x.split(separator))
        encoder = MultiLabelBinarizer()
        features_val = encoder.fit_transform(features_df)
        # One-hot encoding
        values_t = torch.tensor(features_val)
        self.encoded = F.one_hot(values_t)

    def __len__(self):
        return self.encoded.shape[0]


    def __getitem__(self, idx: int) -> torch.Tensor:
        """
            Implement the subscript operator [] for this data set.
            :param  idx: index of the encoded variable
            :returns: The encoded variable at index client_id or None for index out of bounds
        """
        try:
            return self.encoded[idx]
        except IndexError as e:
            constants.log_error(str(e))
            return None


    '''
        Alternative constructor using a Torch Tensor as input_tensor: (Decorator design pattern). The layout of the input_tensor
        should be [feature, label]
        :param actual: Pytorch input_tensor with at least two entries [feature, label]
        :type actual: Tensor
        :param column_names: Optional list of column_names name
        :type column_names: List of str
        :returns: Dataset
    '''
    @classmethod
    def from_tensor(cls, x: torch.Tensor, column_names: list = None) -> Dataset:
        return cls(pd.DataFrame(x, column_names))


    '''
        Alternative constructor using the content of a file as input_tensor. (Decorator design pattern)
        :param input_path: Path of file containing input_tensor data
        :param feature_name: Optional feature_name to extract numerical data
        :param label_name: Optional label key
        :returns: Dataset with numerical values
    '''
    @classmethod
    def _from_json_file(cls, input_path: str, feature_name: str, label_name: str) -> Dataset:
        return cls.from_json_file(input_path, [feature_name], label_name)


    '''
        Alternative constructor using the content of a file as input_tensor. (Decorator design pattern)
        :param input_path: Path of file containing input_tensor data
        :param feature_names: Optional list of feature_name to extract numerical data
        :param label_name: Optional label key
        :returns: Dataset
    '''
    @classmethod
    def from_json_files(cls, input_path: str, feature_names: list = None, label_name: str = '') -> Dataset:
        df = pd.read_json(input_path)
        if feature_names is None:
            return cls(df)
        else:
            return cls(df[[*feature_names, label_name]])


    '''
        Alternative constructor using the content of a  S3 file as input_tensor. (Decorator design pattern)
        :param s3_bucket: S3 bucket containing the input_tensor data
        :param s3_folder: Path of file containing input_tensor data
        :param feature_names: Name of column_names containing the encoder and label
        :param file_extension: Extension of the files from which the encoder and labels have to be extracted
        :returns: Dataset
    '''
    @classmethod
    def from_s3(cls, s3_bucket: str, s3_folder: str, col_names: list, file_extension: str) -> Dataset:
        dataset_s3_loader = S3DatasetLoader(s3_bucket, s3_folder, col_names, False, file_extension)
        return cls(dataset_s3_loader.df)

