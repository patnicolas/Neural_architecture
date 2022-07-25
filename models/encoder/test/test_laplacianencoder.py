from unittest import TestCase
import unittest
from models.encoder.laplacianencoder import LaplacianEncoder
import constants
import pandas as pd
import numpy as np
from models.encoder.vocabulary import Vocabulary
from datasets.s3datasetloader import S3DatasetLoader
from util.profiler import Profiler


class TestLaplacianEncoder(TestCase):
    @unittest.skip("Not needed")
    def test_create_laplacian(self):
        try:
            terms = ["abbrochment", "aboriginal", "aberrantis", "aboriginal", "abdominoperitoneum", "abbrochment"]
            term_index_weights = Vocabulary.get_index_weights()
            max_num_items = 32
            laplacian = LaplacianEncoder('', term_index_weights,  max_num_items)
            laplacian_matrix = laplacian.encoder(terms)
            constants.log_info(f'\n{str(laplacian_matrix)}')
        except Exception as e:
            self.fail(str(e))

    @unittest.skip("Not needed")
    def test_encode(self):
        try:
            TestLaplacianEncoder.encode_laplacian(False)
        except Exception as e:
            self.fail(str(e))

    def test_profile(self):
        try:
            profiler = Profiler('main()')
            profiler.run(200, '../../output/stats-results')
        except Exception as e:
            self.fail(str(e))

    @staticmethod
    def encode_laplacian(store: bool)-> np.array:
        df = TestLaplacianEncoder.__load_text_terms(256)
        max_num_items = 64
        term_index_weights = Vocabulary.get_index_weights()
        laplacian = LaplacianEncoder('termsText', term_index_weights, max_num_items)
        return laplacian.encode(df, store)

    @staticmethod
    def __load_text_terms(num_samples: int) -> pd.DataFrame:
        s3_bucket = constants.s3_config['bucket_name']
        s3_folder = "dl/termsAndContext/utrad-all"
        col_names = ['termsText', 'age', 'gender']
        s3_dataset_loader = S3DatasetLoader(s3_bucket, s3_folder, col_names, False, '.json', num_samples)
        return s3_dataset_loader.df



def profiling_func():
    TestLaplacianEncoder.encode_laplacian(False)

if __name__ == '__main__':
    profiler = Profiler('profiling_func()')
    profiler.run(20, '../output/stats-results')