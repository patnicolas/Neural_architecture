from unittest import TestCase
import pandas as pd
import unittest
import constants
from models.encoder.categoricalencoder import CategoricalEncoder


class TestCategoricalFeature(TestCase):

    @unittest.skip("Not needed")
    def test_to_numeric(self):
        try:
            df = pd.DataFrame(["AA", "BB", "CC", "BB", "CC", "AA"])
            categorical_feature = CategoricalEncoder("my_col", "numeric")
            numeric_encoded_array = categorical_feature.encode(df)
            constants.log_info(f'{str(numeric_encoded_array)}\nwith shape {numeric_encoded_array.shape}')

        except AssertionError as e:
            self.fail(str(e))


    @unittest.skip("Not needed")
    def test_to_one_encoded(self):
        try:
            df = pd.DataFrame(["AA", "BB", "CC", "BB", "CC", "AA"])
            categorical_feature = CategoricalEncoder('my_col', 'onehot')
            one_hot_encoded_array = categorical_feature.encode(df)
            constants.log_info(f'{str(one_hot_encoded_array)}\nwith shape {one_hot_encoded_array.shape}')
        except AssertionError as e:
            self.fail(str(e))

    @unittest.skip("Not needed")
    def test_to_tfidf(self):
        try:
            df = pd.DataFrame([
                "abbrochment hello not abdominoperitoneum perhaps aboriginal",
                "why not aberrantis and aboriginal and aberrantis"
            ])
            tfidf_encoded_array = CategoricalEncoder.to_tfidf(df, True)
            constants.log_info(f'{str(tfidf_encoded_array)}\nwith shape {tfidf_encoded_array.shape}')
        except AssertionError as e:
            self.fail(str(e))

    def test_to_one_encoded_failed(self):
        try:
            df = pd.DataFrame([("AA", 2), ("BB", 3), ("CC", 0),  ("BB",5),  ("CC", 8), ("AA", 6) ])
            categorical_feature = CategoricalEncoder('col', 'one_hot')
            self.fail('An AssertionError should be thrown')
        except AssertionError as e:
            constants.log_info("Succeeded!")

