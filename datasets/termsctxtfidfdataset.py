__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from torch.utils.data import Dataset
from models.encoder.termsctxtfidfencoder import TermsCtxTfIdfEncoder
import torch
import constants
import numpy

"""
     Data set for terms context TF-IDF data for a given source
     :param s3_data_source: Data source (relative folder) on S3
     :param sample_size Number of samples used for training and evaluation. -1 if all the data set is used
 """


class TermsCtxTfIdfDataset(Dataset):
    def __init__(self, s3_data_source: str, sample_size: int = -1):
        super(TermsCtxTfIdfDataset, self).__init__()
        termsCtxTfIdfEncoder = TermsCtxTfIdfEncoder(s3_data_source, 2000, sample_size)
        self.training_data = termsCtxTfIdfEncoder.to_list()
        self.num_labels = termsCtxTfIdfEncoder.count
        self.label_index_map = termsCtxTfIdfEncoder.label_index_map
        self.dimension = self.__get_dimension()


    def __getitem__(self, row: int) -> (torch.Tensor, int):
        """
            Implement the subscript operator [] for this data set. It is assume that this data set is
            flat and each row has the following layout
            (dense one-hot-encoded features, label_index)
            :param row:  index of the row
            :return: tuple encoder input_tensor, label_index_tensor
        """
        try:
            record = self.training_data[row]
            features = TermsCtxTfIdfDataset.to_dense(record.fields, record.indices, record.size)
            label_index = self.label_index_map[record.label]
            return torch.from_numpy(features), label_index
        except IndexError as e:
            constants.log_error(str(e))
            return None
        except StopIteration as e:
            constants.log_error(str(e))
            return None

    def __len__(self) -> int:
        return len(self.training_data)

    def dimension(self) -> int:
        return self.dimension


    @staticmethod
    def to_dense(fields: list, indices: list, size: int) -> numpy.array:
        dense_vector = numpy.zeros(size).astype(numpy.float32)
        for index in range(len(indices)):
            dense_vector[indices[index]] = float(fields[index])
        # dense_vector[len(dense_vector)-1] = dense_vector[len(dense_vector)-1]*100.0
        return dense_vector

    def __get_dimension(self) -> int:
        """
            Compute the size of the dense features vector
        :   return: Number of features in the dense representation of input data
        """
        record = self.training_data[0]
        return len(TermsCtxTfIdfDataset.to_dense(record.fields, record.indices, record.size))

