from unittest import TestCase
import numpy as np
from util.plottermixin import PlotterParameters, PlotterMixin
from ml.metropolis_hastings import MetropolisHastings
from ml.proposal_distribution import ProposalBeta, ProposalDistribution
from typing import AnyStr
from scipy import stats


class TestMetropolisHastings(TestCase):

    """

    def test_beta_priors(self):
        num_data_points = 1000

        plotting_params_1 = PlotterParameters(num_data_points, "x",  "Prior", "Beta distribution")
        x = np.linspace(0, 1, num_data_points)
        posterior = stats.beta(NormalMcMc.n + NormalMcMc.a, NormalMcMc.n - NormalMcMc.h + NormalMcMc.b)
        y = posterior.pdf(x)
        PlotterMixin.single_plot_np_array(x, y, plotting_params_1)
    """

    def test_mh_sample_high_burn_in(self):
        num_iterations = 5000
        burn_in_ratio = 0.5
        sigma_delta = 0.4
        theta0 = 0.8

        alpha = 12
        beta = 10
        num_trails = 96
        h = 10
        normal_McMc = ProposalBeta(alpha, beta, num_trails, h)
        TestMetropolisHastings.execute_metropolis_hastings(
            normal_McMc,
            num_iterations,
            burn_in_ratio,
            sigma_delta,
            theta0,
            f"Beta({alpha}, {beta}) burn_in ratio:{burn_in_ratio}, Initial theta: {theta0}"
        )

    def test_mh_sample_no_burn_in(self):
        num_iterations = 5000
        burn_in_ratio = 0.0
        sigma_delta = 0.4
        theta0 = 0.8

        alpha = 12
        beta = 10
        num_trails = 96
        h = 10
        normal_McMc = ProposalBeta(alpha, beta, num_trails, h)

        TestMetropolisHastings.execute_metropolis_hastings(
            normal_McMc,
            num_iterations,
            burn_in_ratio,
            sigma_delta,
            theta0,
            f"Beta({alpha}, {beta}) burn_in ratio:{burn_in_ratio}, Initial theta: {theta0}"
        )

    def test_mh_sample_burn_in_beta(self):
        num_iterations = 5000
        burn_in_ratio = 0.1
        sigma_delta = 0.4
        theta0 = 0.8

        alpha = 4
        beta = 2
        num_trails = 96
        h = 10
        normal_McMc = ProposalBeta(alpha, beta, num_trails, h)

        TestMetropolisHastings.execute_metropolis_hastings(
            normal_McMc,
            num_iterations,
            burn_in_ratio,
            sigma_delta,
            theta0,
            f"Beta({alpha}, {beta}) burn_in ratio:{burn_in_ratio}, Initial theta: {theta0}"
        )

    @staticmethod
    def execute_metropolis_hastings(
            proposed_distribution: ProposalDistribution,
            num_iterations: int,
            burn_in_ratio: float,
            sigma_delta: float,
            theta0: float,
            description: AnyStr):
        metropolis_hastings = MetropolisHastings(proposed_distribution, num_iterations, burn_in_ratio, sigma_delta)

        theta_history, success_rate = metropolis_hastings.sample(theta0)
        theta_history_str = str(theta_history)
        print(f'Theta history: {theta_history_str}\nSuccess rate {success_rate}')

        plotting_params_1 = PlotterParameters(len(theta_history), "iterations", "Theta", description)
        x = range(metropolis_hastings.burn_ins, num_iterations)
        PlotterMixin.single_plot_np_array(x, list(theta_history), plotting_params_1)




