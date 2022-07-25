__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
from models.gan.discriminator import Discriminator

"""
    Compute the Gradient Penalty for the Wasserstein generative adversarial model. 
    The loss for critic, C, a generator G, target data noise
        E[C(noise)] - E[C(G(z))] + lambda.(||grad(C(y))|| - 1)^2
        y = eps.noise + (1-eps).G(z)
     
    Note: We used critic and discriminator interchangeably
    
    :param critic: Critic or discriminator of the Wasserstein GAN
    :param gp_lambda: Current weight of the gradient penalty
    :reference https://arxiv.org/abs/1701.07875
"""


class WGanLoss(object):
    def __init__(self, critic: Discriminator, gp_lambda: float):
        self.critic = critic
        self.gp_lambda = gp_lambda

    def critic_loss(self,
                    real: torch.Tensor,
                    predicted: torch.Tensor,
                    batch_size: int,
                    critic_fake_scores: torch.Tensor,
                    critic_real_scores: torch.Tensor) -> torch.Tensor:
        """
            Compute the loss of the critic (discriminator) for the Wasserstein GAN. It uses the critic's scores for
            predicted and target data as well as the gradient penalty, and gradient penalty weight.
                    E[C(noise)] - E[C(G(z))] + lambda.(||grad(C(y))|| - 1)^2

            :param real: Real or labeled data
            :param predicted: Predicted or labeled data
            :param batch_size: Size of the batch processed
            :param critic_fake_scores: Critic fake prediction
            :param critic_real_scores: Critic target prediction
            :return: Loss for for the Wasserstein GAN
        """
        epsilon = torch.rand(batch_size, 1, 1, 1, device=constants.torch_device, requires_grad=True)
        gradient = self.compute_gradient(real, predicted, epsilon)
        num_elements = WGanLoss.__get_num_elements(critic_fake_scores)
        return - critic_real_scores.sum() / num_elements \
               - WGanLoss.generator_loss(critic_fake_scores, num_elements) \
               + self.gp_lambda * gradient


    @staticmethod
    def generator_loss(critic_fake_scores: torch.Tensor, num_elements: int = -1) -> torch.Tensor:
        """
            Compute the normalized loss for the generator as the sum of prediction fakse
            :param critic_fake_scores: Fake prediction from critic
            :param num_elements: Number of elements in the prediction
            :return: Normalized loss for the generator
        """
        if num_elements == -1:
            num_elements = WGanLoss.__get_num_elements(critic_fake_scores)
        return -critic_fake_scores.sum() / num_elements


    def compute_gradient(self, real: torch.Tensor, predicted: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
            Compute the normalized gradient penalty
            :param real: Batch of target images
            :param predicted: A batch of predicted images
            :param epsilon:  Vector for the weighted sum of predicted and target data
            :return: Gradient penalty input_tensor normalized for the size of data
        """
        gradient = self.__get_gradient(real, predicted, epsilon)
        return WGanLoss.__gradient_penalty(gradient)


    # ----------------- Supporting private methods --------------------------


    def __get_gradient(self, real, predicted, epsilon) -> torch.Tensor:
        """
            Compute the gradient
            :param real: Batch of target images
            :param predicted: A batch of predicted images
            :param epsilon: Vector for the weighted sum of predicted and target data
            :return: Gradient
        """
        # Mix the images together
        weighted_real_predicted = WGanLoss.__weighted_aggr(real, predicted, epsilon)

        # Calculate the critic's scores on the mixed images
        critic_weighted_scores = self.critic(weighted_real_predicted)

        # Take the gradient of the scores with respect to the images
        gradient_vector = torch.autograd.grad(
            inputs= weighted_real_predicted,
            outputs= critic_weighted_scores,
            # These other parameters have to do with the pytorch autograd engine works
            grad_outputs=torch.ones_like(critic_weighted_scores),
            create_graph=True,
            retain_graph=True,
        )
        return gradient_vector[0]


    @staticmethod
    def __gradient_penalty(gradient: torch.Tensor) -> torch.Tensor:
        """
            Implement the computation of the Gradient penalty
                y = noise.epsilon + g(z).(1- epsilon)
                (||grad(disc(y)|| -1)^2
            :param gradient: For the critic/discriminator score
            :return: Gradient penalty as a input_tensor
        """
        # Flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)

        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)
        gradient_norm_1 = gradient_norm-1.0
        normalized_penalty = (gradient_norm_1*gradient_norm_1).sum()/len(gradient_norm)
        return normalized_penalty

    @staticmethod
    def __weighted_aggr(real: torch.Tensor, fake: torch.Tensor, epsilon: float) -> float:
        return real * epsilon + fake * (1 - epsilon)


    @staticmethod
    def __get_num_elements(critic_fake_scores: torch.Tensor):
        size_list = list(critic_fake_scores.size())

        assert len(size_list) > 0, f'WGanLoss - size_list is empty'
        return 1 if not len else size_list[0]



