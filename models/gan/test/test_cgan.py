from unittest import TestCase

import torch
import constants
import unittest
from models.gan.test.mnistdriver import MNISTDriver
from models.gan.cgan import CGan


class TestCGan(TestCase):

    @unittest.skip("Not needed")
    def test_train(self):
        batch_size = 25
        size_image = 28
        hyper_params = MNISTDriver.create_hyper_params_adam(lr=0.001,
                                                            epochs=8,
                                                            batch_size=batch_size,
                                                            loss_function=torch.nn.BCEWithLogitsLoss())
        z_dim = 10
        in_channels = 1
        num_classes = 10
        # Need to append the classes to the input_tensor to generator and in-channels of the conv. discriminator
        adjusted_z_dim, adjusted_in_channels = CGan.adjust_input_sizes(z_dim, in_channels, num_classes)
        discriminator, generator = MNISTDriver.discriminator_and_generator('disctest2',
                                                                           'gentest2',
                                                                           in_channels=adjusted_in_channels,
                                                                           hidden_dim=16,
                                                                           out_channels=1,
                                                                           z_dim=adjusted_z_dim,
                                                                           unsqueeze=True)

        conditional_conv_gan = CGan("Conditional Convolutional GAN",
                                    generator,
                                    discriminator,
                                    hyper_params,
                                    num_classes,
                                    one_hot_target_func=target_one_hot_labels,
                                    shapes=[size_image, size_image],
                                    z_dim=z_dim)
        constants.log_info(repr(conditional_conv_gan))

        train_loader, _ = MNISTDriver.load_data(batch_size, num_samples=4098)
        conditional_conv_gan.train(0, train_loader)


    def test_train_transpose_init(self):
        batch_size = 30
        hyper_params = MNISTDriver.create_hyper_params_adam(lr=0.001,
                                                            epochs=8,
                                                            batch_size=batch_size,
                                                            loss_function=torch.nn.BCEWithLogitsLoss())
        z_dim = 1
        num_classes = 10
        hidden_dim = 16
        image_size = 32
        # Need to append the classes to the input_tensor to generator and in-channels of the conv. discriminator
        adjusted_z_dim, adjusted_in_channels = CGan.adjust_input_sizes(gen_in_channels=hidden_dim,
                                                                       disc_in_channels=z_dim,
                                                                       num_classes=num_classes)
        conv_dimension = 2
        conv_blocks = MNISTDriver.create_conv_model_auto(conv_dimension=conv_dimension,
                                                         in_channels=adjusted_in_channels,
                                                         out_channels=z_dim,
                                                         hidden_dim=hidden_dim)
        conditional_conv_gan = CGan.transposed_init("Conditional Convolutional GAN",
                                                    conv_dimension=conv_dimension,
                                                    conv_neural_blocks=conv_blocks,
                                                    output_gen_activation=torch.nn.Sigmoid(),
                                                    in_channels=adjusted_z_dim,
                                                    out_channels=z_dim,
                                                    hyper_params=hyper_params,
                                                    num_classes=num_classes,
                                                    one_hot_target_func=target_one_hot_labels,
                                                    shapes=[image_size, image_size],
                                                    z_dim=hidden_dim)
        constants.log_info(repr(conditional_conv_gan))

        train_loader, _ = MNISTDriver.load_data(batch_size, num_samples=4096)
        mean_gen_loss, mean_disc_loss, _ = conditional_conv_gan.train(0, train_loader)
        constants.log_info(f'Mean gen loss: {mean_gen_loss}  Mean disc loss: {mean_disc_loss}')


def target_one_hot_labels(one_hot_labels: torch.Tensor, shapes: list) -> torch.Tensor:
    image_one_hot_labels = one_hot_labels[:, :, None, None]
    return image_one_hot_labels.repeat(1, 1, shapes[0], shapes[1])
