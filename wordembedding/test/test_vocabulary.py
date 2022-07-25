from unittest import TestCase
import unittest
from models.encoder.vocabulary import Vocabulary
import constants


class TestVocabulary(TestCase):
    @unittest.skip("no needed")
    def test_vocabulary_fail(self):
        try:
            vocab = Vocabulary()
            print(repr(vocab))
        except Exception as e:
            self.fail(f'Failed vocabulary test with {str(e)}')

    def test_get_index_weights(self):
        vocab = Vocabulary()
        weights_dict = vocab.get_index_weights()
        constants.log_info(str(weights_dict))


