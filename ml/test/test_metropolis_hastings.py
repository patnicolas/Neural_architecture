from unittest import TestCase
import unittest
import numpy as np
from util.plottermixin import PlotterParameters, PlotterMixin
from ml.metropolis_hastings import MetropolisHastings
from ml.metropolis_hastings import NormalMcMc
from scipy import stats


class TestMetropolisHastings(TestCase):

    def test_mcmc_sample(self):
        num_iterations = 2500
        num_data_points = 1000

        metropolis_hastings = MetropolisHastings(
            NormalMcMc.markov_model,
            NormalMcMc.log_likelihood_normal,
            NormalMcMc.prior,
            num_iterations)

        accepted, success_rate = metropolis_hastings.sample()
        print(f'Success rate {success_rate}')
        plotting_params_1 = PlotterParameters(num_data_points, "x",  "prior", "Prior distribution")
        x = np.linspace(0, 1, num_data_points)
        posterior = stats.beta(NormalMcMc.m + NormalMcMc.a, NormalMcMc.n - NormalMcMc.h + NormalMcMc.b)
        y = posterior(x)
        PlotterMixin.single_plot_np_array(x, y, plotting_params_1)

       # plotting_params_2 = PlotterParameters(num_data_points, "x",  "accepted", "MC-MC")
       # PlotterMixin.single_plot(accepted, plotting_params_2)

