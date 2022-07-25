from unittest import TestCase

import torch
import constants
import unittest
from models.gan.wgangp import WGanGp
from models.gan.test.mnistdriver import MNISTDriver


class TestWGanGp(TestCase):

    @unittest.skip("Not needed")
    def test_train(self):
        batch_size = 22
        z_dim = 56
        hyper_params = MNISTDriver.create_hyper_params_adam(lr=0.001,
                                                            epochs=10,
                                                            batch_size=batch_size,
                                                            loss_function=torch.nn.BCEWithLogitsLoss())
        critic, generator = MNISTDriver.discriminator_and_generator("discmodel4",
                                                                    "genmodel4",
                                                                    in_channels=1,
                                                                    hidden_dim=16,
                                                                    out_channels=1,
                                                                    z_dim=z_dim,
                                                                    unsqueeze=False)
        wasserstein_gan = WGanGp("Convolutional Wasserstein GAN",
                                 generator,
                                 critic,
                                 hyper_params,
                                 critic_repeats=4,
                                 gp_lambda=10.0)
        constants.log_info(repr(wasserstein_gan))

        train_loader, _ = MNISTDriver.load_data(batch_size, num_samples=8092)
        wasserstein_gan.train(0, train_loader)


    @unittest.skip("Not needed")
    def test_train_and_eval(self):
        batch_size = 30
        z_dim = 25
        hyper_params = MNISTDriver.create_hyper_params_adam(lr=0.001,
                                                            epochs=10,
                                                            batch_size=batch_size,
                                                            loss_function=torch.nn.BCEWithLogitsLoss())
        critic, generator = MNISTDriver.discriminator_and_generator("discmodel4",
                                                                    "genmodel4",
                                                                    in_channels=10,
                                                                    hidden_dim=16,
                                                                    out_channels=1,
                                                                    z_dim=z_dim)

        wasserstein_gan = WGanGp("Wasserstein Convolutional",
                                 generator,
                                 critic,
                                 hyper_params,
                                 critic_repeats=4,
                                 gp_lambda=10.0)
        constants.log_info(repr(wasserstein_gan))

        train_loader, eval_loader = MNISTDriver.load_data(batch_size, num_samples=4098)
        wasserstein_gan.train_and_eval(train_loader, eval_loader)


    def test_train_and_eval_transpose(self):
        conv_dimension = 2
        conv_blocks = MNISTDriver.create_conv_model_auto(conv_dimension=conv_dimension,
                                                         in_channels=1,
                                                         out_channels=1,
                                                         hidden_dim=16)
        batch_size = 30
        hyper_params = MNISTDriver.create_hyper_params_adam(lr=0.001,
                                                            epochs=8,
                                                            batch_size=30,
                                                            loss_function=torch.nn.BCEWithLogitsLoss())

        wasserstein_gan = WGanGp.transposed_init("Wasserstein Convolutional GAN",
                                                 conv_dimension=conv_dimension,
                                                 conv_neural_blocks=conv_blocks,
                                                 output_gen_activation=torch.nn.Sigmoid(),
                                                 in_channels=10,
                                                 hyper_params=hyper_params,
                                                 critic_repeats=4,
                                                 gp_lambda=10.0)
        constants.log_info(repr(wasserstein_gan))
        train_loader, eval_loader = MNISTDriver.load_data(batch_size, num_samples=4098)
        wasserstein_gan.train_and_eval(train_loader, eval_loader)
