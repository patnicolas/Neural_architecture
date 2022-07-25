from unittest import TestCase
import torch
import constants
from models.gan.dcgan import DCGan
from torch.utils.data import DataLoader
from models.gan.test.mnistdriver import MNISTDriver
from models.gan.dcgenerator import DCGenerator
from models.gan.frechetinception import FrechetInception


class TestFrechetInception(TestCase):
    def test_eval(self):
        try:
            gen, inception_model, test_loader = TestFrechetInception.train_model()
            frechet_inception = FrechetInception(gen, inception_model)
            frechet_inception.eval(test_loader)
        except Exception as e:
            self.fail(str(e))

    @staticmethod
    def train_model() -> (DCGenerator, torch.nn.Module, DataLoader):
        conv_blocks = MNISTDriver.create_conv_model_auto(conv_dimension=2,
                                                         in_channels=1,
                                                         out_channels=1,
                                                         hidden_dim=16)
        hyper_params = MNISTDriver.create_hyper_params_adam(lr = 0.001,
                                                            epochs = 8,
                                                            batch_size = 32,
                                                            loss_function = torch.nn.BCEWithLogitsLoss())
        input_size = 10
        out_channels = None
        dc_gan = DCGan.transposed_init("Convolutional GAN",
                                       conv_dimension=2,
                                       conv_neural_blocks=conv_blocks,
                                       output_gen_activation=torch.nn.Sigmoid(),
                                       in_channels=input_size,
                                       out_channels=out_channels,
                                       hyper_params=hyper_params,
                                       unsqueeze=False)

        constants.log_info(f'Semi-automated built GAN: {repr(dc_gan)}')
        mnist_training_loader, mnist_validation_loader = MNISTDriver.load_data(batch_size = 32, num_samples = 512)
        dc_gan.train_and_eval(mnist_training_loader, mnist_validation_loader)
        return dc_gan.gen, dc_gan.disc.model.get_model(), mnist_validation_loader
