from unittest import TestCase
import unittest
import constants
from models.encoder.vocabulary import Vocabulary
from models.encoder.cofrequencyencoder import CoFrequencyEncoder
import pandas as pd


class TestCofrequencyEncoder(TestCase):

    def test_load_covariance_indices(self):
        try:
            num_samples = 2
            indices_weights = CoFrequencyEncoder.load_co_frequency_indices(num_samples)
            constants.log_info(str(indices_weights))
        except Exception as e:
            self.fail(str(e))

    @unittest.skip('Not needed')
    def test_encode(self):
        try:
            import pandas as pd

            text_col_name = 'textTerms'
            vocab = Vocabulary(constants.vocab_path)
            max_num_terms = 8
            covariance_encoder = CoFrequencyEncoder(text_col_name, vocab, max_num_terms)
            constants.log_info(repr(covariance_encoder))

            input_indices = [93563, 54560, 66621, 97834,64576, 37289, 97017, 8915, 1145]
            input_terms = [vocab.index_word_dict[index] for index in input_indices]
            df = pd.DataFrame([
                f'{input_terms[0]},{input_terms[1]},{input_terms[3]},{input_terms[5]},{input_terms[1]},{input_terms[4]}',
                f'{input_terms[3]},{input_terms[6]},{input_terms[8]},{input_terms[5]},{input_terms[6]},{input_terms[2]},{input_terms[7]},{input_terms[1]}'
            ], columns=['textTerms'])
            cov_list = covariance_encoder.encode(df)
            constants.log_info(cov_list)
            del vocab
            del covariance_encoder
        except Exception as e:
            self.fail(str(e))

    @unittest.skip('Not needed')
    def test_encode2(self):
        try:
            import pandas as pd

            text_col_name = 'textTerms'
            vocab = Vocabulary(constants.vocab_path)
            max_num_terms = 16
            covariance_encoder = CoFrequencyEncoder(text_col_name, vocab, max_num_terms, 3)
            constants.log_info(repr(covariance_encoder))
            df = TestCofrequencyEncoder.__load_text_terms(3)
            cov_list = covariance_encoder.encode(df)
            constants.log_info(cov_list)
        except Exception as e:
            self.fail(str(e))

    @staticmethod
    def __load_text_terms(num_samples: int) -> pd.DataFrame:
        from datasets.s3datasetloader import S3DatasetLoader

        s3_bucket = constants.s3_config['bucket_name']
        s3_folder = "dl/termsAndContext/utrad-all"
        col_names = ['textTerms', 'age', 'gender']
        s3_dataset_loader = S3DatasetLoader(s3_bucket, s3_folder, col_names, False, '.json', num_samples)
        return s3_dataset_loader.df