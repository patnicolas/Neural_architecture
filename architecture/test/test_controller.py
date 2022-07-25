
import unittest
from unittest import TestCase
from architecture.gradientdescentsearch import GradientDescentSearch
from architecture.controller import Controller


class TestController(TestCase):
    @unittest.skip("Can be skipped")
    def test_controller_init(self):
        architecture_filename = 'architecture/'
        learning_rate = 0.001
        momentum = 0.85
        num_iterations = 20
        search_method = GradientDescentSearch(learning_rate, momentum, num_iterations)
        controller = Controller(search_method, architecture_filename)
        print(str(controller))

    def test_controller_search(self):
        architecture_filename = 'architecture/'
        learning_rate = 0.001
        momentum = 0.85
        num_iterations = 20
        search_method = GradientDescentSearch(learning_rate, momentum, num_iterations)
        controller = Controller(search_method, architecture_filename)
        print(str(controller))
        controller.search()