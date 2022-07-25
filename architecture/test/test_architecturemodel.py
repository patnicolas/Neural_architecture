import unittest

from architecture.tuningfeatures import TuningFeatures, TuningParam
from architecture.architecturemodel import ArchitectureModel
from torch import nn


class TestArchitectureModel(unittest.TestCase):
    def test_load_architecture(self):
        s3_folder_name = 'architecture/'
        tuning_features = ArchitectureModel.load(s3_folder_name)
        print(str(tuning_features))

    @unittest.skip("Not needed")
    def test_forward(self):
        import numpy
        import torch

        tuning_features = TuningFeatures()
        param_1 = TuningParam('F1', 'real', False, 0.0, 1.0)
        param_2 = TuningParam('F4', 'real', True, -2.0, 8.0)
        param_3 = TuningParam('F3', 'ordinal', True, 0.0, 8.0)
        param_4 = TuningParam('F2', 'ordinal', True, 0.0, 8.0)

        tuning_features.add_param(param_1)
        tuning_features.add_param(param_2)
        tuning_features.add_param(param_3)
        tuning_features.add_param(param_4)

        arch_name = "simulator_model"
        architecture_model = ArchitectureModel(arch_name, 0.4, tuning_features)
        tuning_features.sort()
        new_weights = numpy.array([0.6, 0.2, 0.6, 0.9], dtype = numpy.float32)
        torch_weights = nn.Parameter(torch.from_numpy(new_weights))
        score = architecture_model(torch_weights)
        print(score.to(torch.float32))
