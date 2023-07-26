import unittest
import constants
from unittest import TestCase
from util.ioutil import IOUtil


class TestIOUtil(TestCase):
    @unittest.skip("No required")
    def test_to_json(self):
        try:
            file_name = '../../data/test.json'
            ioutil = IOUtil(file_name)
            json_content = ioutil.to_json()
            print(json_content)
        except Exception as e:
            self.fail(str(e))

    def test_pickle(self):
        try:
            file_name = '../../input/test1-pickle'
            dict = {"a":1, "b":2}
            lst = [dict, dict]
            ioutil = IOUtil(file_name)
            ioutil.to_pickle(lst)
            new_lst = ioutil.from_pickle()
            constants.log_info(new_lst)
        except Exception as e:
            self.fail(str(e))

