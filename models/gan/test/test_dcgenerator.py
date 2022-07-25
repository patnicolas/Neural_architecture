from unittest import TestCase

from models.gan.dcgenerator import DCGenerator
from models.gan.test.mnistdriver import MNISTDriver
import constants


class TestDCGenerator(TestCase):
    def test_save_and_load(self):
        try:
            z_dim = 56
            _, dc_generator = MNISTDriver.discriminator_and_generator('disctest1',
                                                                      'gentest1',
                                                                      in_channels=1,
                                                                      hidden_dim=16,
                                                                      out_channels=1,
                                                                      z_dim=z_dim,
                                                                      unsqueeze=True)
            constants.log_info(repr(dc_generator))
            dc_generator.save()
            dc_generator2 = DCGenerator.load('gentest1')
            constants.log_info(repr(dc_generator2))
        except Exception as e:
            self.fail(str(e))
