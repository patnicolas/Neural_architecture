
from unittest import TestCase
import unittest

import numpy

from models.encoder.trainingsetencoder import TrainingSetEncoder


class TestTrainingSetEncoder(TestCase):
    @unittest.skip("No reason")
    def test_extract_columns(self):
        input = "71046 26 R05,\"0.0@0.0@0.20833333,0.341@0.11@0.108,0.89@0.2@0.675\""
        output = TrainingSetEncoder.extract_columns(input)
        print(str(output))

    @unittest.skip("No reason")
    def test__init__(self):
        claim_encoder = TrainingSetEncoder('40/7/utrad', 10, 128)
        tensor_iterator = claim_encoder.features_iter
        print(next(tensor_iterator))
        print(next(tensor_iterator))
        claim_encoder.to_image()

    @unittest.skip("No reason")
    def test_broadcast(self):
        import numpy as np
        input = np.array([[1.0, 0.4], [0.7, 3.3]])*120
        output = input.astype(numpy.uint8)
        print(output)

    def test_to_image(self):
        # claim_encoder_128 = TrainingSetEncoder('40/7/utrad', 10, 128)
        # claim_encoder_128.to_image()
        # claim_encoder_64 = TrainingSetEncoder('40/7/utrad', 10, 64)
        # claim_encoder_64.to_image()
        claim_encoder_512 = TrainingSetEncoder('40/7/utrad', 10, 512)
        claim_encoder_512.to_image()