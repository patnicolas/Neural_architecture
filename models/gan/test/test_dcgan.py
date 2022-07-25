from unittest import TestCase
import unittest
import torch
import constants
from models.gan.dcgan import DCGan
from models.gan.test.mnistdriver import MNISTDriver


class TestDCGan(TestCase):
    def test_save_and_load(self):
        dc_gan = TestDCGan.create_dc_gan()
        dc_gan.save()


    @unittest.skip("Not needed")
    def test_train_and_eval(self):
        try:
            batch_size = 32
            dc_gan = TestDCGan.create_dc_gan()
            constants.log_info(f'Constructed gan: {repr(dc_gan)}')

            mnist_training_loader, mnist_validation_loader = MNISTDriver.load_data(batch_size, 512)
            dc_gan.train_and_eval(mnist_training_loader, mnist_validation_loader)
        except Exception as e:
            self.fail(str(e))


    def test_train_and_eval_transpose(self):
        try:
            conv_dimension = 2
            conv_blocks = MNISTDriver.create_conv_model_auto(conv_dimension=conv_dimension,
                                                             in_channels=1,
                                                             out_channels=1,
                                                             hidden_dim=16)
            lr = 0.001
            epochs = 8
            batch_size = 30
            hyper_params = MNISTDriver.create_hyper_params_adam(lr, epochs, batch_size, torch.nn.BCEWithLogitsLoss())

            input_size = 10
            out_channels = None
            dc_gan = DCGan.transposed_init("Convolutional GAN",
                                           conv_dimension=conv_dimension,
                                           conv_neural_blocks=conv_blocks,
                                           output_gen_activation=torch.nn.Sigmoid(),
                                           in_channels=input_size,
                                           out_channels=out_channels,
                                           hyper_params=hyper_params,
                                           unsqueeze=False)

            constants.log_info(f'Semi-automated built GAN: {repr(dc_gan)}')
            num_samples = 512
            mnist_training_loader, mnist_validation_loader = MNISTDriver.load_data(batch_size, num_samples)
            dc_gan.train_and_eval(mnist_training_loader, mnist_validation_loader)
        except Exception as e:
            self.fail(str(e))


    @staticmethod
    def __create_dc_gan():
        z_dim = 56
        batch_size = 22
        hyper_params = MNISTDriver.create_hyper_params_adam(lr=0.001,
                                                            epochs=10,
                                                            batch_size=batch_size,
                                                            loss_function=torch.nn.BCEWithLogitsLoss())
        dc_discriminator, dc_generator = MNISTDriver.discriminator_and_generator("discmodel1",
                                                                                 "genmodel1",
                                                                                 in_channels=1,
                                                                                 hidden_dim=16,
                                                                                 out_channels=1,
                                                                                 z_dim=z_dim,
                                                                                 unsqueeze=True)
        return DCGan("Gan", dc_generator, dc_discriminator, hyper_params)
