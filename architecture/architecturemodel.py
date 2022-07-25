import numpy as np
import torch
from torch import nn
from architecture.tuningfeatures import TuningFeatures, TuningParam
import constants
from util.s3util import S3Util


class ArchitectureModel(nn.Module):
    def __init__(self, model_filename: str, accuracy_latency_ratio: float, tuning_features: TuningFeatures):
        super(ArchitectureModel, self).__init__()
        self.accuracy_latency_ratio = accuracy_latency_ratio
        # self.current_score = 0.0
        self.tuning_features = tuning_features
        self.model_filename = model_filename

    def forward(self, tuning_params: torch.Tensor) -> torch.Tensor:
        # weights = tuning_params.detach().numpy()
        # self.tuning_features.update(weights)

        print('Execute model')
        return torch.tensor(1.0, requires_grad=True) + tuning_params[0]*0.00001

    def update(self, weights: np.array):
        self.tuning_features.update(weights)

    def to_torch(self):
        return self.tuning_features.to_numpy()

    def __str__(self) -> str:
        return f'Model file name: {self.model_filename}, Accuracy-Latency ratio: {self.accuracy_latency_ratio}\n ' \
               f'Model {str(self.tuning_features)}'

    def execute(self, tuning_params: torch.Tensor) -> torch.Tensor:
        # Updated the weights for the architecture
        weights = tuning_params.detach().numpy()
        self.tuning_features.update(weights)
        print('Execute model')
        return torch.from_numpy(np.array(0.6))

    def initialize(self) -> torch.Tensor:
        values = self.tuning_features.get_values()
        return torch.tensor(values, dtype=torch.float64,requires_grad=True)

    @staticmethod
    def load(s3_folder_name: str) -> TuningFeatures:
        default_s3_bucket_name = constants.s3_config['bucket_name']
        s3_util = S3Util(default_s3_bucket_name, s3_folder_name, True, 20)
        config_dict = s3_util.s3_to_list('json')
        kafka_params = config_dict[0]['kafka']
        spark_params = config_dict[0]['spark']

        tuning_features = TuningFeatures()
        for param in ArchitectureModel.__load_dict(kafka_params):
            tuning_features.add_param(param)
        for param in ArchitectureModel.__load_dict(spark_params):
            tuning_features.add_param(param)
        return tuning_features

    @staticmethod
    def __load_dict(params_list: list) -> list:
        return [TuningParam(
            param['param_name'],
            param['param_type'],
            param['to_normalize'],
            param['lower_bound'],
            param['upper_bound']) for param in params_list]
