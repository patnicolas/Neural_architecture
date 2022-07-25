from unittest import TestCase

import torch
from models.gan.gan import Gan
from models.nnet.hyperparams import HyperParams
import constants
from models.gan.dcgenerator import DCGenerator
from models.gan.dcdiscriminator import DCDiscriminator
from models.gan.test.mnistdriver import MNISTDriver


class TestGAN(TestCase):
    def test_train_MNIST(self):
        try:
            input_channels = 10  # Gray colors 1 feature
            hidden_dim = 16
            output_channels = 1  # 10 digits => 10 encoder/classes
            z_dim = 1
            conv_2d_model = MNISTDriver.create_2d_conv_model("Conv2d", z_dim, hidden_dim, output_channels)
            dc_discriminator = DCDiscriminator(conv_2d_model)

            de_conv_2d_model = MNISTDriver.create_de_2d_conv_model('DeConv2d', input_channels, hidden_dim, z_dim)
            dc_generator = DCGenerator(de_conv_2d_model, input_channels)
            lr = 0.001
            momentum = 0.9
            epochs = 4
            optim_label = 'adam'
            batch_size = 36
            early_stop_patience = 3
            loss_function = torch.nn.BCEWithLogitsLoss()
            hyper_params = HyperParams(lr, momentum, epochs, optim_label, batch_size, early_stop_patience, loss_function)
            dc_gan = Gan("Gan", dc_generator, dc_discriminator, hyper_params)
            constants.log_info(repr(dc_gan))

            train_loader, eval_loader = MNISTDriver.load_data(batch_size, 256)
            dc_gan.train_and_eval(train_loader, eval_loader)
        except Exception as e:
            self.fail(str(e))
