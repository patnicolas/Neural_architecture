from unittest import TestCase

import torch
import constants
from models.gan.test.mnistdriver import MNISTDriver
from models.gan.sngan import SNGan


class TestSNGan(TestCase):

    def test_transposed_init(self):
        try:
            conv_dimension = 2
            conv_blocks = MNISTDriver.create_sn_conv_model_auto(conv_dimension=conv_dimension,
                                                                in_channels=1,
                                                                out_channels=1,
                                                                hidden_dim=16)
            batch_size = 30
            hyper_params = MNISTDriver.create_hyper_params_adam(lr=0.001,
                                                                epochs=8,
                                                                batch_size=30,
                                                                loss_function=torch.nn.BCEWithLogitsLoss())
            input_size = 10
            out_channels = None
            sn_gan = SNGan.transposed_init("Spectral Norm GAN",
                                           conv_dimension = conv_dimension,
                                           conv_neural_blocks = conv_blocks,
                                           output_gen_activation=torch.nn.Sigmoid(),
                                           in_channels=input_size,
                                           out_channels=out_channels,
                                           hyper_params=hyper_params,
                                           debug = MNISTDriver.debug)
            constants.log_info(repr(sn_gan))
        except AssertionError as e:
            self.fail(str(e))


    def test_transposed_train_and_eval(self):
        try:
            conv_dimension = 2
            conv_blocks = MNISTDriver.create_sn_conv_model_auto(conv_dimension=conv_dimension,
                                                                in_channels=1,
                                                                out_channels=1,
                                                                hidden_dim=16)
            batch_size = 30
            hyper_params = MNISTDriver.create_hyper_params_adam(lr=0.001,
                                                                epochs=8,
                                                                batch_size=30,
                                                                loss_function=torch.nn.BCEWithLogitsLoss())
            input_size = 10
            out_channels = None
            sn_gan = SNGan.transposed_init("Spectral Norm GAN",
                                           conv_dimension = conv_dimension,
                                           conv_neural_blocks = conv_blocks,
                                           output_gen_activation=torch.nn.Sigmoid(),
                                           in_channels=input_size,
                                           out_channels=out_channels,
                                           hyper_params=hyper_params)
            num_samples = 512
            mnist_training_loader, mnist_validation_loader = MNISTDriver.load_data(batch_size, num_samples)
            sn_gan.train_and_eval(mnist_training_loader, mnist_validation_loader)
        except AssertionError as e:
            self.fail(str(e))