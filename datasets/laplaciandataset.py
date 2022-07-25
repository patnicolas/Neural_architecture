__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from torch.utils.data import Dataset
import pandas as pd
import torch
import constants
from models.encoder.vocabulary import Vocabulary
from models.encoder import LaplacianEncoder

"""
    Dataset for Graph Laplacian encoder. It is a wrapper 
    :param text_col_name: Name of the column containing the document
    :param vocabulary: Formal vocabulary
    :param max_num_items: Maximum number of items considered 
    :param df: Data frame of input data
"""


class LaplacianDataset(Dataset):
    def __init__(self, text_col_name: str, vocabulary: Vocabulary, max_num_items: int, df: pd.DataFrame):
        super(LaplacianDataset, self).__init__()
        laplacian_encoder = LaplacianEncoder(text_col_name, vocabulary, max_num_items)
        self.data_tensor = laplacian_encoder.encode(df)

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


