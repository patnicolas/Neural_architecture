__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import scipy.linalg
import constants
from torch.distributions import MultivariateNormal

"""
    Class that wraps the computation of the Fréchet inception distance (FID) in the case of multivariate 
    Gaussian distributions N1, N2 as:
        FID = ||mean(N1) - mean(N2)||^2 +TR(sigma(N1) + sigma(N2) - 2 sqrt(sigma(N1)*sigma(N2)))
    :param mean_x: mean of the first Gaussian, (n_features)
    :param mean_y: mean of the second Gaussian, (n_features)
    :param sigma_x: covariance matrix of the first Gaussian, (n_features, n_features)
    :param sigma_y: covariance matrix of the second Gaussian, (n_features, n_features)
"""


class FrechetDistance(object):
    def __init__(self, mean_x: torch.Tensor, mean_y: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor):
        assert mean_x.shape == mean_y.shape, f'Shape mean x {mean_x.shape} should == shape mean y {mean_y.shape}'
        assert sigma_x.shape == sigma_y.shape, f'Shape sigma x {sigma_x.shape} should == shape sigma y {sigma_y.shape}'

        multivariate_normal_x = MultivariateNormal(mean_x, sigma_x)
        multivariate_normal_y = MultivariateNormal(mean_y, sigma_y)
        self.mean_x = multivariate_normal_x.mean
        self.sigma_x = multivariate_normal_x.covariance_matrix
        self.mean_y = multivariate_normal_y.mean
        self.sigma_y = multivariate_normal_y.covariance_matrix

    def distance(self):
        """
            Function for returning the Fréchet distance between multivariate Gaussian distributions
            parameterized by their means and covariance matrices.
            :return: The Frechet inception distance
        """
        delta_mean = torch.norm(self.mean_x - self.mean_y, 'fro')
        prod = self.sigma_x @ self.sigma_y
        cov_mul = FrechetDistance.__sqrt_matrix(prod)
        cov = self.sigma_x + self.sigma_y - 2*cov_mul
        return delta_mean * delta_mean + torch.trace(cov)


    def get_multivariate_normal(self) -> (MultivariateNormal, MultivariateNormal):
        """
            return the two multi-variate normal distribution associated with the means and standard deviations
            :return: Pair of Multi-variate normal distributions
        """
        return MultivariateNormal(self.mean_x, self.sigma_x), MultivariateNormal(self.mean_y, self.sigma_y)

    @staticmethod
    def __sqrt_matrix(mat: torch.Tensor) -> torch.Tensor:
        """
            Function that takes in a matrix and returns the square root of that matrix.
            For an input_tensor matrix A, the output matrix B would be such that B @ B is the matrix A.
            :param mat: Matrix as a n x n PyTorch tensor
            :return: A matrix with square root of elements
        """
        y = mat.to(constants.torch_device).detach().numpy()
        y = scipy.linalg.sqrtm(y)
        return torch.Tensor(y.real, device = constants.torch_device)