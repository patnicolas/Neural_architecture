from unittest import TestCase

from models.gan.test.mnistdriver import MNISTDriver
from models.gan.dcdiscriminator import DCDiscriminator
import constants


class TestDCDiscriminator(TestCase):
    def test_save_and_load(self):
        try:
            z_dim = 56
            dc_discriminator, _ = MNISTDriver.discriminator_and_generator('disctest1',
                                                                          'gentest1',
                                                                          in_channels=1,
                                                                          hidden_dim=16,
                                                                          out_channels=1,
                                                                          z_dim=z_dim,
                                                                          unsqueeze=True)
            constants.log_info(repr(dc_discriminator))
            dc_discriminator.save()
            dc_discriminator2 = DCDiscriminator.load('disctest1')
            constants.log_info(repr(dc_discriminator2))
        except Exception as e:
            self.fail(str(e))
