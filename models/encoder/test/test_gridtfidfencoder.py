from unittest import TestCase
import constants
from models.encoder.gridtfidfencoder import GridTfIdfEncoder
import pandas as pd


class TestGridTfidfEncoder(TestCase):
    def test_to_tfidf(self):
        try:
            df = pd.DataFrame([
                "abbrochment hello not abdominoperitoneum perhaps aboriginal ok abbrochment",
                "why not aberrantis and aboriginal and aberrantis"
            ])
            num_tfidf_features = 30
            num_context_features = 2
            df = GridTfIdfEncoder.to_tfidf(df, num_tfidf_features + num_context_features)
            constants.log_info(df)
        except AssertionError as e:
            self.fail(str(e))
