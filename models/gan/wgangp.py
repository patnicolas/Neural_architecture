__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
from torch.utils.data import DataLoader
from models.gan.generator import Generator
from models.gan.wganloss import WGanLoss
from models.gan.discriminator import Discriminator
from models.gan.dcgan import DCGan
from models.nnet.hyperparams import HyperParams


def unsqueeze_noise(fake_noise: torch.Tensor, input_size: int):
    return fake_noise.view(len(fake_noise), input_size, 1, 1)


"""
    Generative adversarial network regularized with wasserstein distance using Gradient Penalty.
    There are two constructors:
    - Direct instantiation from pre-defined generator and discriminator (__init__)
    - Mirrored instantiation for which the generator is a mirror de-convolutional network to the convolutional
        network used in the discriminator
        
    :param model_id: Identifier for the model
    :param gen: Generator for this Wasserstein GAN
    :param disc: Discriminator for this Wasserstein GAN
    :param hyper_params: Hyper parameters used for this Wasserstein GAN
    :param critic_repeat: Number of iteration to compute the discriminator (critic) loss
    :param gp_lambda: Factor for the gradient penalty
    :reference https://arxiv.org/abs/1701.07875
"""


class WGanGp(DCGan):
    def __init__(self,
                 model_id: str,
                 gen: Generator,
                 disc: Discriminator,
                 hyper_params: HyperParams,
                 critic_repeats: int,
                 gp_lambda: float = 10.0,
                 debug=None):
        super(WGanGp, self).__init__(model_id, gen, disc, hyper_params, debug)
        assert 0 < critic_repeats < 10, f'Number of critic repeats {critic_repeats} should be [1, 9]'
        assert 1.0 < gp_lambda < 50.0, f'Gradient penalty lambda {gp_lambda} should be ]1.0, 25.0['

        self.critic_repeats = critic_repeats
        self.w_gan_gp = WGanLoss(disc, gp_lambda)


    @classmethod
    def transposed_init(cls,
                        model_id: str,
                        conv_dimension: int,
                        conv_neural_blocks: list,
                        output_gen_activation: torch.nn.Module,
                        in_channels: int,
                        hyper_params: HyperParams,
                        critic_repeats: int,
                        gp_lambda: float = 10.0,
                        debug=None) -> DCGan:
        """
            Generate a Deep convolutional Wasserstein GAN using mirrors neural blocks (transposition)
            :param model_id: Identifier for this deep convolutional GAN
            :param conv_dimension: Dimension of the convolution (1 or 2)
            :param conv_neural_blocks: Neural block used in the convolutional model of discriminator
            :param output_gen_activation: Activation function for the output to the generator
            :param in_channels: Size of input_tensor to the discriminator
            :param hyper_params: Hyper-parameters used for training and evaluation
            :param critic_repeats: Number of iteration to compute the discriminator (critic) loss
            :param gp_lambda: Factor for the gradient penalty
            :param debug: Debugging function
            :return: deep convolutional Wasserstein GAN
        """
        deep_conv_gan = DCGan.transposed_init(model_id,
                                              conv_dimension,
                                              conv_neural_blocks,
                                              output_gen_activation,
                                              in_channels,
                                              None,
                                              hyper_params,
                                              unsqueeze=True)
        return cls(deep_conv_gan.model_id,
                   deep_conv_gan.gen,
                   deep_conv_gan.disc,
                   hyper_params,
                   critic_repeats,
                   gp_lambda,
                   debug)


    def train(self, epoch: int, train_loader: DataLoader) -> (float, float, int):
        """
            Training method, for convolutional Discriminator and Generator using Wasserstein distance
            :param epoch: Current epoch (starting at 0)
            :param train_loader: PyTorch data loader for training data
            :return: Tuple (mean loss generator, mean loss discriminator, size of dataset
        """
        mean_critic_loss = 0.0
        mean_gen_loss = 0.0

        for real, _ in train_loader:
            cur_batch_size = len(real)
            real = real.to(constants.torch_device)

            mean_iteration_critic_loss = 0
            # -------  Discriminator loss and gradient update -----------
            for _ in range(self.critic_repeats):
                # Reset gradient to zero
                for params in self.disc.parameters():
                    params.grad = None

                fake_noise = self.gen.noise(cur_batch_size, unsqueeze_noise)
                fake = self.gen(fake_noise)
                critic_fake_scores = self.disc(fake.detach())
                critic_real_scores = self.disc(real)

                critic_loss = self.w_gan_gp.critic_loss(real,
                                                        fake.detach(),
                                                        cur_batch_size,
                                                        critic_fake_scores,
                                                        critic_real_scores)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += critic_loss.item() / self.critic_repeats
                # Update gradients
                critic_loss.backward(retain_graph=True)
                # Update optimizer
                self.disc_opt.update_step()
            mean_critic_loss += mean_iteration_critic_loss

            # -------  Generator loss and gradient update -----------
            for params in self.gen.parameters():
                params.grad = None

            fake_noise_2 = self.gen.noise(cur_batch_size, unsqueeze_noise)
            # noise = fake_noise.view(len(fake_noise_2), self.gen.in_channels, 1, 1)
            fake_2 = self.gen(fake_noise_2)
            critic_fake_scores = self.disc(fake_2)

            gen_loss = self.w_gan_gp.generator_loss(critic_fake_scores)
            gen_loss.backward()

            # Update the weights
            self.gen_opt.update_step()
            # Keep track of the average generator loss
            mean_gen_loss += gen_loss.item()
        return mean_gen_loss, mean_critic_loss, len(train_loader.dataset)


    def eval(self, epoch: int, eval_loader: DataLoader) -> (float, float, int):
        """
            evaluation, for convolutional Discriminator and Generator using Wasserstein distance
            :param epoch: Current epoch (starting at 0)
            :param train_loader: PyTorch data loader for training data
            :return: Tuple (mean loss generator, mean loss discriminator, size of dataset
        """
        mean_critic_loss = 0.0
        mean_gen_loss = 0.0

        for real, _ in eval_loader:
            cur_batch_size = len(real)
            real = real.to(constants.torch_device)
            mean_iteration_critic_loss = 0

            for _ in range(self.critic_repeats):
                fake_noise = self.gen.noise(cur_batch_size, unsqueeze_noise)
                fake = self.gen(fake_noise)
                critic_fake_scores = self.disc(fake.detach())
                critic_real_scores = self.disc(real)

                critic_loss = self.w_gan_gp.critic_loss(
                    real,
                    fake.detach(),
                    cur_batch_size,
                    critic_fake_scores,
                    critic_real_scores)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += critic_loss.item() / self.critic_repeats
            mean_critic_loss += mean_iteration_critic_loss

            # -------  Generator loss and gradient update -----------
            fake_noise_2 = self.gen.noise(cur_batch_size, unsqueeze_noise)
            # noise = fake_noise.view(len(fake_noise_2), self.gen.in_channels, 1, 1)
            fake_2 = self.gen(fake_noise_2)
            critic_fake_scores = self.disc(fake_2)

            gen_loss = self.w_gan_gp.generator_loss(critic_fake_scores)
            # Keep track of the average generator loss
            mean_gen_loss += gen_loss.item()
        return mean_gen_loss, mean_critic_loss, len(eval_loader.dataset)
