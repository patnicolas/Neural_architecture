from unittest import TestCase

import pandas as pd
import constants
from models.encoder.numericalencoder import NumericalEncoder


class TestNumericalEncoder(TestCase):

    def test_to_numeric(self):
        try:
            df = pd.DataFrame([3, 12, 4, 1, 34, 1, 0, 6, 9])
            numerical_encoder = NumericalEncoder(df)
            np_result = numerical_encoder.to_numeric()
            constants.log_info(str(np_result))
        except AssertionError as e:
            self.fail(str(e))
