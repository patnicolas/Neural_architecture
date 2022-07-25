__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
from models.gan.generator import Generator
from models.gan.frechetdistance import FrechetDistance
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
    Implement the Frechet Inception Distance (FID) to evaluate the accuracy of the generator extracted from
    the training of a Generative Adversarial Network. We assume that the distribution in Z-space follows
    a multi-variate normal distribution.
    Given the Gaussian distributions Nreal, Nfake of z-vectors the FID is computed as
     FID = ||mean(Nreal) - mean(Nfake)||^2 +TR(sigma(Nreal) + sigma(Nfake) - 2 sqrt(sigma(Nreal)*sigma(Nfake)))
     
    :param gen: Generator trained as part of the GAN
    :param inception_model: Convert real/labeled input into features.. It is assumed it is a convolutional model
"""


class FrechetInception(object):
    def __init__(self, gen: Generator, inception_model: torch.nn.Module):
        self.gen = gen
        self.inception_model = inception_model

    def eval(self, test_loader: DataLoader) -> float:
        """
            Compute the Frechet distance using a test1 loader for real examples and n_samples
            :param test_loader: Loader for the real data used in evaluating the generator
            :return: Frechet distance
        """
        self.gen.eval()
        real_features_list = []
        fake_features_list = []
        cur_samples = 0
        n_samples = len(test_loader.dataset)

        with torch.no_grad():  # You don't need to calculate gradients here, so you do this to save memory
            try:
                for real_example, _ in tqdm(test_loader, total=n_samples // test_loader.batch_size):  # Go by batch
                    real_samples = real_example
                    real_features = self.inception_model(real_samples.to(constants.torch_device)).detach().to(constants.torch_device)
                    real_features_list.append(real_features)
                    fake_samples = torch.randn(len(real_example), self.gen.z_dim, device=constants.torch_device)
                    fake_samples = self.gen(fake_samples)
                    fake_features = self.inception_model(fake_samples.to(constants.torch_device)).detach().to(constants.torch_device)
                    fake_features_list.append(fake_features)
                    cur_samples += len(real_samples)
                    constants.log_info(f'Frechet inception has processed {cur_samples} samples')
            except Exception as e:
                constants.log_error(str(e))

        fake_features_all = torch.cat(fake_features_list)
        real_features_all = torch.cat(real_features_list)
        mean_fake = torch.mean(fake_features_all, 0)
        mean_real = torch.mean(real_features_all, 0)
        sigma_fake = FrechetInception.__get_covariance(fake_features_all)
        sigma_real = FrechetInception.__get_covariance(real_features_all)

        # Compute the Frechet distance
        return FrechetDistance(mean_fake, mean_real, sigma_fake, sigma_real)


    @staticmethod
    def __get_covariance(features: torch.Tensor) -> torch.Tensor:
        """
            Computes the co-variance of z-vector features
            :param features: tensor of features in the latent space
            :return: Co-variance matrix/tensor
        """
        import numpy as np
        return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))