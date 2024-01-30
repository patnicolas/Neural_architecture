from unittest import TestCase
import numpy as np
from util.plottermixin import PlotterParameters, PlotterMixin
from ml.metropolis_hastings import MetropolisHastings
from ml.metropolis_hastings import NormalMcMc
from scipy import stats


class TestMetropolisHastings(TestCase):

    def test_mcmc_sample(self):
        num_iterations = 2500
        num_data_points = 1000
        burn_in_ratio = 0.1
        sigma_delta = 0.2

        metropolis_hastings = MetropolisHastings(
            NormalMcMc.markov_model,
            NormalMcMc.log_likelihood,
            NormalMcMc.prior,
            num_iterations,
            burn_in_ratio,
            sigma_delta)

        theta_history, success_rate = metropolis_hastings.sample()
        theta_history_str = ' '.join(theta_history)
        print(f'Theta history: {theta_history_str}\nSuccess rate {success_rate}')

        plotting_params_1 = PlotterParameters(num_data_points, "x",  "prior", "Prior distribution")
        x = np.linspace(0, 1, num_data_points)
        posterior = stats.beta(NormalMcMc.n + NormalMcMc.a, NormalMcMc.n - NormalMcMc.h + NormalMcMc.b)
        y = posterior.pdf(x)
        PlotterMixin.single_plot_np_array(x, y, plotting_params_1)

       # plotting_params_2 = PlotterParameters(num_data_points, "x",  "accepted", "MC-MC")
       # PlotterMixin.single_plot(accepted, plotting_params_2)

