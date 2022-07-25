from unittest import TestCase

import constants
import torch
import unittest
from models.nnet.hyperparams import HyperParams
from models.autoencoder.convvaemodel import ConvVAEModel
from models.autoencoder.convvae import ConvVAE
from models.autoencoder.test.vaemnistdriver import VaeMNISTDriver
from models.autoencoder.variationalneuralblock import VariationalNeuralBlock


class TestConvVAE(TestCase):
    @unittest.skip("No reason")
    def test_train_and_eval(self):
        try:
            z_dim = 56
            batch_size = 22
            hyper_params = VaeMNISTDriver.create_hyper_params_adam(lr=0.001,
                                                                   epochs=10,
                                                                   batch_size=batch_size,
                                                                   loss_function=torch.nn.BCEWithLogitsLoss())
            conv_model, de_conv_model = VaeMNISTDriver.encoder_and_decoder(in_channels=1,
                                                                           hidden_dim=16,
                                                                           out_channels=1,
                                                                           z_dim=z_dim)
            latent_size = 24
            fc_hidden_dim = 32
            flatten_input = 288  # 512
            variational_block = VariationalNeuralBlock(flatten_input, fc_hidden_dim, latent_size)
            conv_vae_model = ConvVAEModel('Conv VAE model', conv_model, de_conv_model, variational_block)

            # Step 3: Load data set
            train_loader, test_loader = VaeMNISTDriver.load_data(batch_size, 1200)
            conv_vae = ConvVAE(conv_vae_model, hyper_params, None)
            constants.log_info(repr(conv_vae))
            conv_vae._train_and_eval(train_loader, test_loader)
        except Exception as e:
            self.fail(str(e))
        except AssertionError as e:
            self.fail(str(e))

    def test_train_and_eval_transpose_init(self):
        conv_dimension = 2
        in_channels = 1
        out_channels = 32
        z_dim = 26
        batch_size = 25
        conv_blocks = VaeMNISTDriver.create_conv_model(conv_dimension, in_channels, out_channels*2)
        conv_vae_model = ConvVAEModel.transposed_init('Convolutional VAE', conv_dimension, conv_blocks)
        constants.log_info(repr(conv_vae_model))

        hyper_params = VaeMNISTDriver.create_hyper_params_adam(lr=0.001,
                                                               epochs=10,
                                                               batch_size=batch_size,
                                                               loss_function=torch.nn.BCEWithLogitsLoss())

        train_loader, test_loader = VaeMNISTDriver.load_data(batch_size, 1200)
        conv_vae = ConvVAE(conv_vae_model, hyper_params, None)
        constants.log_info(repr(conv_vae))
        conv_vae._train_and_eval(train_loader, test_loader)

    @unittest.skip("No reason")
    def test_train_and_eval_old(self):
        try:
            z_dim = 26
            input_channels = 1  # Gray colors 1 feature
            output_channels = 32  # 10 digits => 10 encoder/classes
            conv_2d_model = TestConvVAE.__create_2d_conv_model("Conv2d", input_channels, z_dim, output_channels*2)

            latent_size = 16
            fc_hidden_dim = 32
            flatten_input = 288  # 512
            variational_block = VariationalNeuralBlock(flatten_input, fc_hidden_dim, latent_size)

            de_conv_2d_model = TestConvVAE.__create_de_2d_conv_model('DeConv2d', output_channels, hidden_dim,
                                                                     input_channels)
            conv_vae_model = ConvVAEModel('Conv LinearVAE', conv_2d_model, de_conv_2d_model, variational_block)

            lr = 0.001
            momentum = 0.9
            epochs = 10
            optim_label = 'adam'
            batch_size = 22
            early_stop_patience = 3
            loss_function = torch.nn.BCEWithLogitsLoss()
            hyper_params = HyperParams(lr, momentum, epochs, optim_label, batch_size, early_stop_patience,
                                       loss_function)

            # Step 3: Load data set
            train_loader, test_loader = TestConvVAE.__load_data(batch_size, 1200)
            conv_vae = ConvVAE(conv_vae_model, hyper_params, None)
            print(repr(conv_vae))
            conv_vae._train_and_eval(train_loader, test_loader)
        except Exception as e:
            self.fail(str(e))
        except AssertionError as e:
            self.fail(str(e))
