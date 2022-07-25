from unittest import TestCase
import torch
import constants
from models.gan.frechetdistance import FrechetDistance


class TestFrechetDistance(TestCase):

    def test_equals_distance(self):
        try:
            mean_x = torch.tensor([0.0, 0.0])
            sigma_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            frechet_inception = FrechetDistance(mean_x, mean_x, sigma_x, sigma_x)
            distance = frechet_inception.distance()
            constants.log_info(f'Equals: {str(distance)}')
        except Exception as e:
            self.fail(str(e))

    def test_similarity_distance(self):
        try:
            mean_x = torch.tensor([0.0, 0.0])
            mean_y = torch.tensor([1.0, 1.0])
            sigma_x = torch.tensor([[1.1, -1.0], [-1.0, 1.1]])
            sigma_y = torch.tensor([[0.9, -0.5], [-0.5, 0.9]])
            frechetInception = FrechetDistance(mean_x, mean_y, sigma_x, sigma_y)
            distance = frechetInception.distance()
            constants.log_info(f'Similar {str(distance)}')
        except Exception as e:
            self.fail(str(e))

    def test_non_similarity_distance(self):
        try:
            mean_x = torch.tensor([0.0, 0.0])
            mean_y = torch.tensor([1.0, 1.0])
            sigma_x = torch.tensor([[0.4, -0.2], [-0.2, 0.4]])
            sigma_y = torch.tensor([[7.9, -0.5], [-0.5, 7.9]])
            frechetInception = FrechetDistance(mean_x, mean_y, sigma_x, sigma_y)
            distance = frechetInception.distance()
            constants.log_info(f'Non similar: {str(distance)}')
        except Exception as e:
            self.fail(str(e))