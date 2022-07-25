from unittest import TestCase

import constants
import torch
from util.s3util import S3Util
from models.nnet.snapshot import Snapshot


class TestSnapshot(TestCase):
    def test_put(self):
        try:
            s3_util = S3Util(constants.default_s3_bucket_name, 'temp/models/test1', False)
            snapshot = Snapshot(s3_util)
            linear = torch.nn.Linear(20, 10)
            optimizer = torch.optim.Adam(linear.parameters(), 0.001)
            snapshot.put(linear, optimizer, 11)
            print('test_put completed')
        except Exception as e:
            print(str(e))
            self.fail()

    def test_get(self):
        try:
            s3_util = S3Util(constants.default_s3_bucket_name, 'temp/models/test1', False)
            snapshot = Snapshot(s3_util)
            linear = torch.nn.Linear(20, 10)
            adam = torch.optim.Adam(linear.parameters(), 0.001)
            snapshot.put(linear, adam, 11)

            model, optimizer, epoch = snapshot.get(linear, adam)
            print(str(model))
            print(str(optimizer))
            print(epoch)
            print('test_get completed')
        except Exception as e:
            print(str(e))
            self.fail()


