from unittest import TestCase

import torch
import constants
from models.gan.wganloss import WGanLoss
from models.gan.dcdiscriminator import DCDiscriminator
from models.gan.dcgenerator import DCGenerator
from models.gan.test.mnistdriver import MNISTDriver


class TestWGanLoss(TestCase):
    def test_compute_gradient(self):
        gp_lambda = 10.0
        critic = TestWGanLoss.get_critic()
        constants.log_info(repr(critic))
        generator = TestWGanLoss.get_generator()
        constants.log_info(repr(generator))

        wasserstein_loss = WGanLoss(TestWGanLoss.get_critic(), gp_lambda)
        train_loader, _ = MNISTDriver.load_data(30)

        for real, _ in train_loader:
            cur_batch_size = len(real)
            real = real.to(constants.torch_device)

            for params in critic.parameters():
                params.grad = None

            fake_noise = generator.noise(cur_batch_size)
            x = fake_noise.view(len(fake_noise), generator.z_dim, 1, 1)
            fake = generator(x)
            critic_fake_scores = critic(fake.detach())
            critic_real_scores = critic(real)
            epsilon = torch.rand(cur_batch_size, 1, 1, 1, device=constants.torch_device, requires_grad=True)
            loss = wasserstein_loss.compute_gradient(critic_fake_scores, critic_real_scores, epsilon)
            constants.log_size(loss, 'Wasserstein loss')
            constants.log_info(str(loss))

    @staticmethod
    def get_generator() -> DCGenerator:
        input_channels = 10  # Gray colors 1 feature
        hidden_dim = 16
        z_dim = 1
        de_conv_2d_model = MNISTDriver.create_de_2d_conv_model('DeConv2d', input_channels, hidden_dim, z_dim)
        return DCGenerator(de_conv_2d_model, input_channels)

    @staticmethod
    def get_critic() -> DCDiscriminator:
        hidden_dim = 16
        output_channels = 1  # 10 digits => 10 encoder/classes
        z_dim = 1
        conv_2d_model = MNISTDriver.create_2d_conv_model("Conv2d", z_dim, hidden_dim, output_channels)
        return DCDiscriminator(conv_2d_model)


