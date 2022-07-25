from unittest import TestCase
import torch
import constants


class TestFlatten(TestCase):
    def test_tensor_views(self):
        try:
            X = torch.tensor([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=torch.float32, device=constants.torch_device)
            constants.log_info(X)
            Y = X.view(1, 3, 4)  # Add an extra num_tfidf_features
            # [[ [1., 2., 3., 4.],
            #    [5., 6., 7., 8.],
            #    [9., 10., 11., 12.] ]]
            Z = X.view(-1, 3)  # 3 column_names and whatever number of rows
            #  [[1., 2., 3.],
            #   [4., 5., 6.],
            #   [7., 8., 9.],
            #   [10., 11., 12.]]
            W = X.view(4, -1)  # 4 rows and whatever number of cols
            # [[1., 2., 3.],
            #  [4., 5., 6.],
            #  [7., 8., 9.],
            #  [10., 11., 12.]]
            V = Y.squeeze()  # Remove the extract num_tfidf_features (if == 1)
            T = Y.flatten()  # [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.]
            S = X.unsqueeze(0)  # Equivalent to view(1, 3, 4)
            #  [[[1., 2., 3., 4.],
            #    [5., 6., 7., 8.],
            #    [9., 10., 11., 12.]]]
            S = X.unsqueeze(1)  # Add extra num_tfidf_features  Same as view(3, 1, 4)
            S = X.unsqueeze(2)  # Same as view(3, 4, 1)
        except Exception as e:
            self.fail(str(e))

    def test_forward(self):
        try:
            from models.nnet.flatten import Flatten
            X = torch.tensor([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=torch.float32, device=constants.torch_device)
            start_dim = 1
            flatten = Flatten(start_dim)
            Y = flatten.forward(X)
            constants.log_info(f'Shape for start_dim {start_dim} {list(Y.size())}\n{Y}')

            start_dim = 0
            flatten = Flatten(start_dim)
            Y = flatten.forward(X)
            constants.log_info(f'Shape for start_dim {start_dim} {list(Y.size())}\n{Y}')
        except Exception as e:
            self.fail(str(e))
