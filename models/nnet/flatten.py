__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from torch import nn

"""
    Simple module ot flatten a PyTorch tensor. It is usually invoked for the last convolutional block 
    prior to the fully connected (RBM) layers
    :param start_dim: Starting dimension for flattening the tensor
"""


class Flatten(nn.Module):
    def __init__(self, start_dim: int):
        assert start_dim >= 0, f'Flatten - start_dim {start_dim} should be >= 0'
        super(Flatten, self).__init__()
        self.start_dim = start_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self.start_dim < len(input.size()), f'Flatten - start_dim should be < number of shapes'
        return input.flatten(start_dim = self.start_dim)


import constants
if __name__ == "__main__":
    X = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=torch.float32, device = constants.torch_device)
    print(X)
    Y = X.view(1, 3, 4)   # Add an extra num_tfidf_features
    # [[ [1., 2., 3., 4.],
    #    [5., 6., 7., 8.],
    #    [9., 10., 11., 12.] ]]
    Z = X.view(-1, 3)   # 3 column_names and whatever number of rows
    #  [[1., 2., 3.],
    #   [4., 5., 6.],
    #   [7., 8., 9.],
    #   [10., 11., 12.]]
    W = X.view(4, -1)   # 4 rows and whatever number of cols
    # [[1., 2., 3.],
    #  [4., 5., 6.],
    #  [7., 8., 9.],
    #  [10., 11., 12.]]
    V = Y.squeeze()       # Remove the extract num_tfidf_features (if == 1)
    T = Y.flatten()       # [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.]
    S = X.unsqueeze(0)    # Equivalent to view(1, 3, 4)
    #  [[[1., 2., 3., 4.],
    #    [5., 6., 7., 8.],
    #    [9., 10., 11., 12.]]]
    S = X.unsqueeze(1)    # Add extra num_tfidf_features  Same as view(3, 1, 4)
    S = X.unsqueeze(2)    # Same as view(3, 4, 1)

