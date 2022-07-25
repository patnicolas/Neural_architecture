from unittest import TestCase

import torch
import constants
from models.autoencoder.convaeconfig import ConvVAEBlockConfig, ConvVAEConfig


class TestConvVAEConfig(TestCase):
    def test_build(self):
        try:
            conv_dimension = 2
            in_channel = 1
            hidden_dim = 8
            out_channels = 32
            fc_flatten_input = 288
            fc_hidden_dim = 30
            latent_size = 16

            conv_configs = [
                ConvVAEBlockConfig(2, 4, 2, 0, torch.nn.LeakyReLU(0.2), False),
                ConvVAEBlockConfig(2, 4, 2, 1, torch.nn.LeakyReLU(0.2), False),
                ConvVAEBlockConfig(2, 4, 2, 1, torch.nn.LeakyReLU(0.2), False)
            ]
            de_conv_configs = [
                ConvVAEBlockConfig(2, 4, 1, 0, torch.nn.ReLU(), True),
                ConvVAEBlockConfig(2, 4, 2, 0, torch.nn.ReLU(), True),
                ConvVAEBlockConfig(2, 4, 2, 1, torch.nn.Sigmoid(), True)
            ]
            conv_vae_model = ConvVAEConfig.build(
                'Test model',
                conv_dimension,
                conv_configs,
                de_conv_configs,
                in_channel,
                hidden_dim,
                out_channels,
                fc_flatten_input,
                fc_hidden_dim,
                latent_size
            )
            constants.log_info(repr(conv_vae_model))
        except AssertionError as e:
            self.fail(str(e))
        except Exception as e:
            self.fail(str(e))
