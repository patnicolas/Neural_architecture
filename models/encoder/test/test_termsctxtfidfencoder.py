from unittest import TestCase
import unittest
from models.encoder.termsctxtfidfencoder import TermsCtxTfIdfEncoder
import constants


class TestTermsCtxTfIdfEncoder(TestCase):
    @unittest.skip("No reason")
    def test_to_dense(self):
        label = "79081 26"
        fields = [0.3, 0.6, 0.8, 0.3]
        indices = [2, 6, 7, 9]
        np_array = TermsCtxTfIdfEncoder.to_dense(fields, indices, 12)
        print(str(np_array))


    def test__init__(self):
        data_source = '40/7/test1'
        num_files = 5
        termsCtxTfIdfEncoder = TermsCtxTfIdfEncoder(data_source, num_files)
        count = 0
        running = True
        while running:
            try:
                label, features = termsCtxTfIdfEncoder.next()
                count += 1
                constants.log_info(f'Label: {str(label)}')
            except StopIteration as e:
                constants.log_info(str(e))
                running = False

        print(f'Done after {count} records')

