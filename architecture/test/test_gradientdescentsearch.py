
import unittest
import torch
from architecture.gradientdescentsearch import GradientDescentSearch
from architecture.architecturemodel import ArchitectureModel


class GradientDescentSearchTest(unittest.TestCase):
    def test_grad(self):
        x = torch.tensor([2.0, 1.0], requires_grad=True)
        # y = 8 * x ** 4 + 3 * x ** 3 + 7 * x ** 2 + 6 * x + 3
        y = torch.tensor(1.0, requires_grad=True) + x[0]*0.0
        y.backward()
        g = x.grad
        print(g)

    @unittest.skip("Can be skipped")
    def test_init(self):
        learning_rate = 0.001
        momentum = 0.85
        num_iterations = 20
        search_method = GradientDescentSearch(learning_rate, momentum, num_iterations)
        assert search_method.momentum == momentum, 'Search method momentum failed'
        print(str(search_method))

    @unittest.skip("Can be skipped")
    def test_forward(self):
        learning_rate = 0.001
        momentum = 0.85
        num_iterations = 20
        search_method = GradientDescentSearch(learning_rate, momentum, num_iterations)
        architecture_filename = 'architecture/'
        tuning_features = ArchitectureModel.load(architecture_filename)
        architecture_model = ArchitectureModel(architecture_filename, 1.0, tuning_features)
        search_method.search(architecture_model)
