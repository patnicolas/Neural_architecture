import random
from unittest import TestCase

from util.asynchttppost import AsyncHttpPost
import asyncio
import unittest
import time
from util.asynchttppost import execute_request


class TestAsyncHttpPost(TestCase):
    @unittest.skip("Not needed")
    def test_list_partitioning(self):
        lst = [0, 1, 3, 4, 7, 8, 9, 12, 13, 15, 16, 18, 29, 38, 39, 40, 41, 42]
        num_clients = 4
        stride = len(lst) // num_clients
        partitions = [lst[i:i + stride] for i in range(0, len(lst), stride)]
        print(partitions)

    def test_requests_async(self):
        in_file = "../../data/requests/sample-request-allowed-stage.json"
        new_headers = {'Content-type': 'application/json'}
        url = 'http://ip-10-5-35-209.us-east-2.compute.internal:8087/geminiml/predict'
        num_clients = 4

        start_time = time.time()
        asyncio.run(execute_request(url, new_headers, in_file, num_clients))
        duration = time.time() - start_time
        print(f'Async. duration: {duration}')


    @staticmethod
    def __predict_file() -> str:
        # in_file = "../../data/requests/test2.json"
        # in_file = "../../data/requests/sample-streamline-training.json"
        # in_file = "../../data/requests/sample-request-streamline.json"
        # in_file = "../../data/requests/sample-request-test1.json"
        # in_file = "../../data/requests/sample-request-allowed-prod.json"
        # in_file = "../../data/requests/sample-request-disabled-prod.json"
        in_file = "../../data/requests/sample-request-allowed-stage.json"
        # in_file = "../../data/requests/sample-request-disabled-stage.json"
        # in_file = "../../data/requests/sample-request-aliases.json"
        # in_file = "../../data/requests/sample-training.json"
        return in_file